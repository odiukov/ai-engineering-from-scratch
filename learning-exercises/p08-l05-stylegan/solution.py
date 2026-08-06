"""
StyleGAN — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def leaky_relu(x, slope=0.2):
    """LeakyReLU: положительное проходит как есть, отрицательное умножается на slope.

    leaky_relu(3.0)   ->  3.0
    leaky_relu(-3.0)  ->  -0.6
    leaky_relu(0.0)   ->  0.0

    В отличие от ReLU здесь нет мёртвой зоны: на отрицательной стороне
    градиент равен slope, а не нулю. И mapping-сеть StyleGAN, и почти все
    дискриминаторы GAN построены именно на ней.
    """
    return x if x > 0 else slope * x


def mapping_network(z, layers):
    """Mapping-сеть StyleGAN, f: Z -> W. MLP из leaky_relu-слоёв.

    layers — список пар (W, b). На каждом слое поэлементно:
      h = leaky_relu(W @ h + b)

    mapping_network([1.0], [([[2.0]], [0.0])])   ->  [2.0]
    mapping_network([-1.0], [([[2.0]], [0.0])])  ->  [-0.4]

    Зачем вообще эта сеть. z взят из N(0, I), его геометрия навязана
    руками, и любое направление в нём тянет за собой всё сразу: позу,
    свет, личность. w сеть выбирает сама под форму данных, и его оси
    получаются куда менее перепутанными. В настоящем StyleGAN здесь восемь
    слоёв.

    Пустой список слоёв — вернуть z как есть (копией, не тем же списком).
    """
    h = list(z)
    for W, b in layers:
        if len(W) != len(b):
            raise ValueError(f"layer has {len(W)} rows but {len(b)} biases")
        pre = []
        for row, bi in zip(W, b):
            if len(row) != len(h):
                raise ValueError(
                    f"row of length {len(row)} does not match input of length {len(h)}"
                )
            pre.append(sum(w * v for w, v in zip(row, h)) + bi)
        h = [leaky_relu(v) for v in pre]
    return h


def adain(features, scale, bias):
    """AdaIN: нормализовать вектор признаков и заново задать ему стиль.

    adain(x, scale, bias) = scale * (x - mean(x)) / std(x) + bias

    adain([1.0, 3.0], 1.0, 0.0)  ->  [-1.0, 1.0]
    adain([0.0, 0.0], 2.0, 5.0)  ->  [5.0, 5.0]

    std считается по формуле для населения (делить на n, а не на n-1) и с
    добавкой 1e-8 под корнем — иначе постоянный вход даст деление на ноль.

    «Стиль» в StyleGAN — это ровно первые два момента карты признаков.
    После нормализации от собственных mean и std входа не остаётся ничего,
    и всё, что видит следующий слой, приходит из w. Отсюда и вся
    распутанность W: слою просто нечего унаследовать, кроме стиля.

    Пустой вектор — ValueError.
    """
    if not features:
        raise ValueError("features must not be empty")
    n = len(features)
    m = sum(features) / n
    var = sum((f - m) ** 2 for f in features) / n
    sd = math.sqrt(var + 1e-8)
    return [scale * (f - m) / sd + bias for f in features]


def modulate(w, scale_w, bias_w):
    """Аффинная проекция w в пару (scale, bias) для одного слоя AdaIN.

    scale = скалярное произведение scale_w на w
    bias  = скалярное произведение bias_w на w

    modulate([1.0, 2.0], [1.0, 0.0], [0.0, 3.0])  ->  (1.0, 6.0)

    В настоящем StyleGAN это обучаемый линейный слой на каждое разрешение;
    именно через него один и тот же w впрыскивается во ВСЕ слои сразу,
    вместо того чтобы подмешиваться на входе один раз.

    Несовпадение длин — ValueError.
    """
    if not (len(w) == len(scale_w) == len(bias_w)):
        raise ValueError("w, scale_w and bias_w must have the same length")
    scale = sum(a * b for a, b in zip(scale_w, w))
    bias = sum(a * b for a, b in zip(bias_w, w))
    return scale, bias


def average_w(ws):
    """Средний вектор w по выборке: тот самый w_bar для truncation trick.

    average_w([[0.0, 2.0], [2.0, 0.0]])  ->  [1.0, 1.0]

    На практике его считают один раз по десяткам тысяч случайных z и кладут
    рядом с весами: он нужен на каждом инференсе.

    Пустой список — ValueError. Векторы разной длины — тоже ValueError:
    zip обрезал бы молча.
    """
    if not ws:
        raise ValueError("ws must not be empty")
    dim = len(ws[0])
    if any(len(w) != dim for w in ws):
        raise ValueError("all w vectors must have the same length")
    return [sum(w[i] for w in ws) / len(ws) for i in range(dim)]


def truncate_w(w, w_mean, psi):
    """Truncation trick: w' = w_mean + psi * (w - w_mean).

    truncate_w([3.0], [1.0], 1.0)  ->  [3.0]   (ничего не меняется)
    truncate_w([3.0], [1.0], 0.0)  ->  [1.0]   (всё схлопнулось в среднее)
    truncate_w([3.0], [1.0], 0.5)  ->  [2.0]

    psi < 1 — обмен разнообразия на качество: сэмплы берутся из узкого
    конуса вокруг среднего, артефактов меньше и лиц тоже меньше. Почти все
    демки StyleGAN идут с psi ≈ 0.7. На проде это единственная ручка, до
    которой достаёт слой сервинга.

    Несовпадение длин — ValueError.
    """
    if len(w) != len(w_mean):
        raise ValueError("w and w_mean must have the same length")
    return [m + psi * (wi - m) for wi, m in zip(w, w_mean)]


def style_mixing(w_a, w_b, num_layers, crossover):
    """Список из num_layers векторов w: первые crossover — w_a, остальные — w_b.

    style_mixing([1.0], [2.0], 3, 1)  ->  [[1.0], [2.0], [2.0]]
    style_mixing([1.0], [2.0], 3, 3)  ->  [[1.0], [1.0], [1.0]]
    style_mixing([1.0], [2.0], 3, 0)  ->  [[2.0], [2.0], [2.0]]

    Это и mixing regularization при обучении, и способ смешивать стили при
    инференсе. Низкие разрешения (первые слои) держат позу и форму лица,
    высокие — свет и цвет. Отсюда «лицо человека A в освещении человека B».

    crossover вне 0..num_layers — ValueError. num_layers <= 0 — тоже.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 <= crossover <= num_layers:
        raise ValueError(f"crossover must be within 0..{num_layers}")
    # копии, а не один и тот же объект: иначе правка одного слоя тихо
    # изменит все остальные
    return [list(w_a) if i < crossover else list(w_b) for i in range(num_layers)]


def synthesis(const, blocks, w_per_layer, noise_sigma=0.0, rng=None, adain_on=True):
    """Синтез-сеть StyleGAN в миниатюре. Вернуть вектор признаков после всех блоков.

    const — обучаемая константа на входе (в настоящей сети это 4x4x512).
    blocks — список словарей с ключами W, b, scale_w, bias_w.
    w_per_layer — список векторов w, по одному на блок (см. style_mixing).

    Каждый блок по порядку:
      h = leaky_relu поэлементно от (W @ h + b)
      если adain_on: scale, bias = modulate(w, scale_w, bias_w)
                     h = adain(h, scale, bias)
      если noise_sigma > 0: к каждому h_i прибавить noise_sigma * rng.gauss(0, 1)

    z в синтез-сеть не подаётся вообще — вход постоянный, вся информация
    приезжает через AdaIN. Прямое следствие, которое стоит проверить
    тестом: при adain_on=True масштаб const на выход не влияет, потому что
    первый же AdaIN его нормализует.

    Шум даёт стохастическую деталь (поры, волоски) и не меняет глобальную
    структуру — статистики выхода остаются на месте.

    noise_sigma > 0 без rng — ValueError: молчаливый глобальный random
    сделал бы прогон невоспроизводимым.
    Длина w_per_layer не равна числу блоков — ValueError.
    """
    if len(w_per_layer) != len(blocks):
        raise ValueError(
            f"got {len(w_per_layer)} w vectors for {len(blocks)} blocks"
        )
    if noise_sigma > 0 and rng is None:
        raise ValueError("noise_sigma > 0 requires an rng")
    h = list(const)
    for block, w in zip(blocks, w_per_layer):
        pre = [
            sum(a * b for a, b in zip(row, h)) + bi
            for row, bi in zip(block["W"], block["b"])
        ]
        h = [leaky_relu(v) for v in pre]
        if adain_on:
            scale, bias = modulate(w, block["scale_w"], block["bias_w"])
            h = adain(h, scale, bias)
        if noise_sigma > 0:
            h = [v + noise_sigma * rng.gauss(0.0, 1.0) for v in h]
    return h
