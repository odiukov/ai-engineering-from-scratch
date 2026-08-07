"""
Vision Transformers — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def patchify(image, patch_size):
    """Разрезать картинку HxW на квадратные патчи и вернуть их ПЛОСКИМИ списками.

    Картинка — список из H строк, каждая строка — список из W чисел (один канал).
    Патчи не перекрываются, обход сетки патчей идёт построчно: слева направо,
    потом вниз. Каждый патч возвращается уже развёрнутым в вектор длины
    patch_size**2 (сначала первая строка патча, потом вторая, и так далее).

    patchify([[1, 2, 3, 4],
              [5, 6, 7, 8]], 2)        ->  [[1, 2, 5, 6], [3, 4, 7, 8]]

    patchify([[1, 2], [3, 4]], 2)      ->  [[1, 2, 3, 4]]     (один патч)

    Картинка 64x64 при patch_size=16 даёт (64/16) * (64/16) = 16 патчей —
    ровно та сетка, что в уроке.

    Ловушка: если H или W не делится на patch_size нацело — брось ValueError.
    В ViT это `assert image_size % patch_size == 0`; молча отрезать хвост
    нельзя, иначе часть картинки исчезнет без предупреждения.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height = len(image)
    width = len(image[0]) if height else 0
    # проверяем ДО начала работы: лучше упасть сразу, чем вернуть половину картинки
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image {height}x{width} is not divisible by patch_size {patch_size}"
        )
    patches = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = []
            for row in range(top, top + patch_size):
                # срез копирует кусок строки одним вызовом — быстрее поэлементного цикла
                patch.extend(image[row][left : left + patch_size])
            patches.append(patch)
    return patches


def unpatchify(patches, patch_size, height, width):
    """Собрать картинку HxW обратно из плоских патчей. Обратна patchify.

    unpatchify([[1, 2, 5, 6], [3, 4, 7, 8]], 2, 2, 4)
        ->  [[1, 2, 3, 4],
             [5, 6, 7, 8]]

    unpatchify(patchify(img, p), p, H, W) == img для любой картинки, которая
    делится на p. Это тот самый инвариант, которым проверяют MAE-декодер:
    он реконструирует патчи, а собирать из них картинку — вот этой функцией.

    Порядок патчей тот же, что у patchify: построчно по сетке патчей.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image {height}x{width} is not divisible by patch_size {patch_size}"
        )
    cols = width // patch_size
    rows = height // patch_size
    if len(patches) != rows * cols:
        raise ValueError(f"expected {rows * cols} patches, got {len(patches)}")
    # заранее выделяем пустой холст: писать по индексу дешевле, чем склеивать строки
    image = [[0.0] * width for _ in range(height)]
    for index, patch in enumerate(patches):
        top = (index // cols) * patch_size
        left = (index % cols) * patch_size
        for row in range(patch_size):
            # присваивание срезу кладёт целую строку патча за один раз
            image[top + row][left : left + patch_size] = patch[
                row * patch_size : (row + 1) * patch_size
            ]
    return image


def patch_embed(image, patch_size, W, b):
    """Патчи -> токены: разрезать картинку и линейно спроецировать каждый патч.

    token = W @ patch + b, где W имеет форму (dim x patch_size**2), b — длины dim.
    Вернуть список токенов, по одному на патч, в порядке patchify.

    patch_embed([[1.0, 2.0],
                 [3.0, 4.0]], 2, [[1, 0, 0, 0], [0, 0, 0, 1]], [0.0, 10.0])
        ->  [[1.0, 14.0]]        (патч [1,2,3,4]: первая строка W берёт 1,
                                  вторая берёт 4 и прибавляет 10)

    Для картинки 64x64 при patch_size=16 и dim=192 получится 16 токенов
    длины 192 — вход первого блока энкодера в уроке.

    Это в точности `nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)`:
    свёртка с шагом, равным ядру, никуда не скользит внахлёст — она просто
    нарезает картинку на патчи и проецирует каждый. Отсюда фраза «первый conv
    и есть patch embedding».

    Разрезать заново не надо — позови patchify.
    """
    patches = patchify(image, patch_size)
    tokens = []
    for patch in patches:
        # одна строка W — один выходной канал свёртки; скалярное произведение
        # строки на плоский патч и есть значение этого канала
        tokens.append(
            [sum(w * x for w, x in zip(row, patch)) + bias for row, bias in zip(W, b)]
        )
    return tokens


def add_cls_and_positions(tokens, cls_token, pos_embed):
    """Приписать [CLS] спереди и ПОТОМ прибавить позиционные эмбеддинги.

    Порядок важен: сначала последовательность становится длиннее на один
    токен, и только потом к КАЖДОМУ токену (включая [CLS]) поэлементно
    прибавляется свой позиционный вектор.

    add_cls_and_positions([[1.0, 2.0]], [0.0, 0.0], [[10.0, 20.0], [0.5, 0.5]])
        ->  [[10.0, 20.0], [1.5, 2.5]]

    Длина результата = len(tokens) + 1.

    Ловушка: len(pos_embed) обязан равняться len(tokens) + 1, иначе ValueError.
    Это самый частый баг при переносе ViT: картинку подали другого размера,
    патчей стало больше, а pos_embed остался от старой длины. Ровно это ловит
    skill из урока (vit-patch-and-pos-embed-inspector).

    Входные tokens не мутировать: наверху их ещё используют.
    """
    # список новых списков: ни tokens, ни cls_token наружу не протекают
    sequence = [list(cls_token)] + [list(token) for token in tokens]
    if len(pos_embed) != len(sequence):
        raise ValueError(
            f"pos_embed has length {len(pos_embed)}, "
            f"but sequence length is {len(sequence)}"
        )
    return [
        [t + p for t, p in zip(token, pos)] for token, pos in zip(sequence, pos_embed)
    ]


def softmax(scores):
    """Softmax одного вектора: превратить логиты в вероятности, сумма = 1.

    softmax([0.0, 0.0, 0.0])  ->  [0.3333..., 0.3333..., 0.3333...]
    softmax([0.0, 1.0])       ->  [0.26894142..., 0.73105857...]

    Ловушка: math.exp(1000) — это OverflowError. Поэтому перед exp из всех
    чисел вычитают максимум вектора: результат от этого не меняется
    (общий множитель сокращается в числителе и знаменателе), а exp считается
    от неположительных чисел и уже не переполняется.

    В PyTorch это `x.softmax(-1)`, и он делает ровно тот же трюк с максимумом.
    """
    shift = max(scores)
    # вычитание максимума: самый большой exp теперь ровно 1.0, переполнения нет
    exps = [math.exp(s - shift) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def scaled_dot_product_attention(Q, K, V):
    """Внимание. Вернуть кортеж (out, weights).

    weights[i] = softmax( Q[i] . K[j] / sqrt(d_k) по всем j ),
    out[i]     = сумма weights[i][j] * V[j].

    d_k — длина одного вектора запроса, то есть len(Q[0]).

    Q = [[1.0, 0.0]], K = [[1.0, 0.0], [1.0, 0.0]], V = [[1.0, 1.0], [3.0, 3.0]]
        ->  weights = [[0.5, 0.5]],  out = [[2.0, 2.0]]
            (ключи одинаковые — внимание не может их различить и делит поровну)

    Каждая строка weights суммируется в 1, поэтому out[i] — выпуклая
    комбинация строк V: любая координата out лежит между min и max той же
    координаты по V. Внимание НИЧЕГО не придумывает, оно только взвешивает.

    Ловушка: делить надо на sqrt(d_k), а не на d_k. Смысл делителя — вернуть
    скалярному произведению единичный масштаб (дисперсия суммы d_k слагаемых
    растёт как d_k, значит стандартное отклонение — как sqrt(d_k)). Деление
    на d_k пересглаживает распределение, и тесты это видят.

    Softmax считай уже написанной функцией, не переписывай формулу заново.
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    weights = []
    for q in Q:
        # строка сырых похожестей запроса на все ключи, сразу поделённая на sqrt(d_k)
        scores = [sum(qi * ki for qi, ki in zip(q, k)) / scale for k in K]
        weights.append(softmax(scores))
    d_v = len(V[0])
    out = []
    for row in weights:
        # взвешенная сумма строк V; идём по координатам, чтобы не строить транспонированную V
        out.append([sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)])
    return out, weights


def layer_norm(vec, gamma, beta, eps=1e-5):
    """LayerNorm одного токена: нормировать ПО ПРИЗНАКАМ, потом gamma и beta.

    (x - mean) / sqrt(var + eps) * gamma + beta, где mean и var считаются по
    координатам самого вектора vec.

    layer_norm([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        ->  примерно [-1.2247, 0.0, 1.2247]   (среднее 0, дисперсия 1)

    layer_norm([5.0, 5.0], [1.0, 1.0], [0.0, 0.0])  ->  [0.0, 0.0]

    Второй пример объясняет eps: у постоянного вектора дисперсия ровно 0,
    и без eps здесь было бы деление на ноль.

    Ловушка: дисперсия ПОПУЛЯЦИОННАЯ — делить на n, а не на n-1. Именно так
    считает `nn.LayerNorm`; statistics.variance даст n-1 и разойдётся с PyTorch.

    В отличие от BatchNorm, здесь нет никакой зависимости от батча: каждый
    токен нормируется сам по себе. Поэтому LayerNorm одинаково работает при
    батче 1 и при батче 1024 — и поэтому он стоит во всех трансформерах.
    """
    n = len(vec)
    mean = sum(vec) / n
    # делим на n, а не на n-1: это population variance, как в nn.LayerNorm
    var = sum((x - mean) ** 2 for x in vec) / n
    denom = math.sqrt(var + eps)
    return [(x - mean) / denom * g + b for x, g, b in zip(vec, gamma, beta)]


def prenorm_residual(x, gamma, beta, sublayer):
    """Блок pre-LN: x + sublayer(layer_norm(x)). sublayer — функция вектор->вектор.

    prenorm_residual([1.0, 2.0, 3.0], [1.0] * 3, [0.0] * 3, lambda v: [0.0] * len(v))
        ->  [1.0, 2.0, 3.0]        (пустой под-слой не трогает x ВООБЩЕ)

    prenorm_residual([1.0, 2.0, 3.0], [1.0] * 3, [0.0] * 3, lambda v: v)
        ->  примерно [-0.2247, 2.0, 4.2247]

    Так устроены ОБА под-блока энкодера ViT:
        x = x + MSA(LN(x))
        x = x + MLP(LN(x))

    Чем pre-LN отличается от post-LN. Post-LN — это layer_norm(x + sublayer(x)):
    нормировка стоит СНАРУЖИ, поэтому она разрушает чистый residual-путь.
    Даже если под-слой вернул нули, post-LN всё равно отнормирует x и изменит
    его. У pre-LN нормировка спрятана ВНУТРЬ ветки, а от входа к выходу идёт
    ничем не тронутая прямая: градиент из последнего блока доходит до первого
    без масштабирования на каждом слое.

    Практическое следствие: post-LN не обучался глубже 6-8 слоёв без warmup
    (в начале обучения градиенты у верхних слоёв взрывались, и learning rate
    приходилось поднимать медленно и вручную). Pre-LN обучается на десятках
    слоёв стабильно и без warmup — поэтому его используют все ViT и все
    современные LLM.

    Нормировку не переписывай — позови layer_norm.
    """
    normed = layer_norm(x, gamma, beta)
    # x складывается с выходом ветки, а не подменяется им — это и есть residual
    return [xi + si for xi, si in zip(x, sublayer(normed))]
