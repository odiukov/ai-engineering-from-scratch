"""
Диффузионные трансформеры и rectified flow — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def patchify(image, patch_size):
    """Порезать картинку на патчи-токены — вход любого DiT.

    image — список каналов, каждый канал это список строк:
    image[c][y][x]. Возвращается список токенов в порядке обхода
    патчей слева направо, сверху вниз. Внутри токена значения идут
    channel-major: сначала весь первый канал патча (по строкам),
    потом второй, и так далее.

    patchify([[[1, 2], [3, 4]]], 1)  ->  [[1], [2], [3], [4]]
    patchify([[[1, 2], [3, 4]]], 2)  ->  [[1, 2, 3, 4]]

    Если высота или ширина не делится на patch_size — ValueError.
    Реальные DiT решают это паддингом, но молчаливо обрезать нельзя:
    так теряется край картинки.

    Аналог: nn.Conv2d(C, dim, kernel_size=p, stride=p) с последующим
    .flatten(2).transpose(1, 2) — свёртка с шагом в размер ядра это и есть
    патчификация плюс линейная проекция. Здесь мы делаем только резку.

    Зачем: U-Net смотрит на пиксели через свёртки, DiT — на патчи через
    self-attention. Патчификация это единственное место, где картинка
    превращается в последовательность.
    """
    if not image or not image[0]:
        raise ValueError("image must have at least one channel and one row")
    height = len(image[0])
    width = len(image[0][0])
    if height % patch_size or width % patch_size:
        raise ValueError("image size must be divisible by patch_size")

    tokens = []
    # внешние два цикла бегут по СЕТКЕ патчей, внутренние — внутри патча.
    # Порядок обхода патчей задаёт порядок токенов, и unpatchify обязан
    # использовать ровно его же, иначе картинка соберётся перемешанной.
    for patch_y in range(0, height, patch_size):
        for patch_x in range(0, width, patch_size):
            token = []
            for channel in image:
                for dy in range(patch_size):
                    for dx in range(patch_size):
                        token.append(channel[patch_y + dy][patch_x + dx])
            tokens.append(token)
    return tokens


def unpatchify(tokens, channels, patch_size):
    """Собрать картинку обратно из токенов. Точная обратная к patchify.

    Картинка считается квадратной: число токенов обязано быть полным
    квадратом, иначе ValueError. Именно так устроен _unpatchify в DiT —
    он берёт h = w = sqrt(num_patches).

    unpatchify([[1], [2], [3], [4]], 1, 1)  ->  [[[1, 2], [3, 4]]]
    unpatchify([[1, 2, 3, 4]], 1, 2)        ->  [[[1, 2], [3, 4]]]

    Ловушка: длина токена обязана быть channels * patch_size ** 2. Если
    не сходится — ValueError, а не «возьмём сколько есть».

    Зачем: голова DiT выдаёт по одному вектору на патч, а на выходе нужна
    картинка. Unpatchify — последний слой любой диффузионной модели.
    """
    grid = int(round(math.sqrt(len(tokens))))
    if grid * grid != len(tokens):
        raise ValueError("number of tokens must be a perfect square")
    expected = channels * patch_size * patch_size
    for token in tokens:
        if len(token) != expected:
            raise ValueError("token length must be channels * patch_size ** 2")

    size = grid * patch_size
    # заготовка нужного размера: заполняем по месту, а не аппендим, потому
    # что токены приходят по патчам, а картинка хранится по строкам
    image = [[[0] * size for _ in range(size)] for _ in range(channels)]
    for index, token in enumerate(tokens):
        patch_y = (index // grid) * patch_size
        patch_x = (index % grid) * patch_size
        pos = 0
        for c in range(channels):
            for dy in range(patch_size):
                for dx in range(patch_size):
                    image[c][patch_y + dy][patch_x + dx] = token[pos]
                    pos += 1
    return image


def adaln_zero_block(x, branch, scale, shift, gate):
    """Один шаг DiT-блока: AdaLN-Zero модуляция плюс gated-residual.

    Считает LayerNorm без обучаемых параметров, домножает на (1 + scale),
    прибавляет shift, прогоняет через branch (attention или MLP) и
    возвращает x + gate * branch(...). scale и shift — списки по длине x,
    gate — одно число.

    adaln_zero_block([1.0, -1.0], lambda h: h, [0.0, 0.0], [0.0, 0.0], 0.0)
        ->  [1.0, -1.0]          (gate = 0 — блок это тождество)
    adaln_zero_block([1.0, -1.0], lambda h: [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], 0.5)
        ->  [1.5, -0.5]

    Ловушка: gate умножает ТОЛЬКО выход ветки, а не сумму. Если умножить
    сумму, при gate = 0 обнулится и сам x, и сеть на старте выдаст нули
    вместо входа — ровно то, чего zero-init должен избежать.

    Аналог: nn.LayerNorm(dim, elementwise_affine=False) плюс
    modulate(x, scale, shift) из DiT, где (scale, shift, gate) выдаёт
    nn.Linear(cond_dim, dim * 3) с нулевой инициализацией.

    Зачем: обусловливание на timestep и текст в DiT идёт не через
    конкатенацию, а через эти три числа. Нулевая инициализация делает
    свежий блок тождественным, и глубокая диффузионная сеть учится стабильно.
    """
    if len(scale) != len(x) or len(shift) != len(x):
        raise ValueError("scale and shift must match the length of x")

    mean = sum(x) / len(x)
    # дисперсия смещённая (делим на n, не на n-1) — так считает nn.LayerNorm
    variance = sum((value - mean) ** 2 for value in x) / len(x)
    inv_std = 1.0 / math.sqrt(variance + 1e-5)
    normed = [(value - mean) * inv_std for value in x]
    modulated = [n * (1.0 + s) + b for n, s, b in zip(normed, scale, shift)]

    out = branch(modulated)
    if len(out) != len(x):
        raise ValueError("branch must return a vector of the same length")
    # residual: gate гасит ветку, но не сам x
    return [value + gate * delta for value, delta in zip(x, out)]


def rectified_flow_path(x0, eps, t):
    """Точка на прямой между данными и шумом: x_t = (1 - t) * x0 + t * eps.

    rectified_flow_path([0.0, 0.0], [2.0, 4.0], 0.0)  ->  [0.0, 0.0]
    rectified_flow_path([0.0, 0.0], [2.0, 4.0], 0.5)  ->  [1.0, 2.0]
    rectified_flow_path([0.0, 0.0], [2.0, 4.0], 1.0)  ->  [2.0, 4.0]

    Ловушка: направление. При t = 0 получаются ЧИСТЫЕ данные, при t = 1 —
    чистый шум. Перепутанные концы дают модель, которая учится идти не в ту
    сторону, и сэмплер уходит от данных.

    Зачем: в DDPM траектория от данных к шуму кривая, и её разгибание стоит
    тысячи шагов. Rectified flow объявляет её отрезком прямой — отсюда
    20 шагов вместо 1000.
    """
    if len(x0) != len(eps):
        raise ValueError("x0 and eps must have the same length")
    return [(1.0 - t) * a + t * e for a, e in zip(x0, eps)]


def velocity_target(x0, eps):
    """Цель регрессии в rectified flow: скорость v = eps - x0.

    velocity_target([0.0, 0.0], [2.0, 4.0])  ->  [2.0, 4.0]
    velocity_target([1.0], [1.0])            ->  [0.0]

    Это производная rectified_flow_path по t. Она НЕ зависит от t: вдоль
    прямой скорость постоянна, поэтому модель учит одно и то же поле
    в каждой точке пути.

    Ловушка: знак. Это eps - x0, «от данных к шуму». В DDPM цель была сам
    eps, здесь — разность. Перепутанный знак разворачивает сэмплер.

    Зачем: постоянная скорость и есть причина, по которой шагов нужно мало.
    Прямую можно пройти одним шагом Эйлера без ошибки.
    """
    if len(x0) != len(eps):
        raise ValueError("x0 and eps must have the same length")
    return [e - a for a, e in zip(x0, eps)]


def flow_matching_loss(velocity_fn, x0_batch, rng):
    """Средний квадрат ошибки предсказания скорости на случайных t и eps.

    Для каждого x0 из батча: тянем t равномерно из [0, 1), тянем eps из
    стандартного нормального, строим x_t и сравниваем velocity_fn(x_t, t)
    с velocity_target(x0, eps). Возвращается MSE по всем скалярам батча.

    flow_matching_loss(lambda x, t: [0.0], [[0.0]], random.Random(0))  ->  число >= 0

    Все случайные числа берутся ТОЛЬКО из rng — глобальный random запрещён,
    иначе одинаковый seed даст разные лоссы и тесты станут нестабильны.
    Порядок важен: сначала t, потом весь eps для этого примера.

    Ловушка: делить надо на общее число скаляров (батч * длину вектора),
    а не на размер батча. Иначе лосс поедет от длины вектора.

    Аналог: rectified_flow_train_step из урока без шага оптимизатора —
    F.mse_loss(model(x_t, t), epsilon - x0).
    """
    if not x0_batch:
        raise ValueError("x0_batch must not be empty")

    total = 0.0
    count = 0
    for x0 in x0_batch:
        t = rng.random()
        eps = [rng.gauss(0.0, 1.0) for _ in x0]
        x_t = rectified_flow_path(x0, eps, t)
        target = velocity_target(x0, eps)
        pred = velocity_fn(x_t, t)
        if len(pred) != len(target):
            raise ValueError("velocity_fn must return a vector of the same length")
        for p, v in zip(pred, target):
            total += (p - v) ** 2
            count += 1
    return total / count


def classifier_free_guidance(v_uncond, v_cond, scale):
    """Смешать безусловное и обусловленное предсказания скорости.

    v = v_uncond + scale * (v_cond - v_uncond)

    classifier_free_guidance([1.0], [3.0], 0.0)  ->  [1.0]   (чистый uncond)
    classifier_free_guidance([1.0], [3.0], 1.0)  ->  [3.0]   (чистый cond)
    classifier_free_guidance([1.0], [3.0], 3.5)  ->  [8.0]   (экстраполяция)

    Ловушка: scale = 1 это НЕ «выключено», это ровно обусловленное
    предсказание. Выключено — при scale = 0. Значения больше единицы
    экстраполируют за пределы отрезка, и именно поэтому CFG усиливает
    промпт, а на больших scale пережигает картинку.

    Зачем: rectified flow меняет сэмплер, но не обусловливание. Модели
    2026 года ходят на scale 3.5-5, а schnell-варианты обучены вообще без
    CFG и работают на scale 0.
    """
    if len(v_uncond) != len(v_cond):
        raise ValueError("both velocity vectors must have the same length")
    return [u + scale * (c - u) for u, c in zip(v_uncond, v_cond)]


def euler_sample(velocity_fn, x_init, steps):
    """Сэмплер: проинтегрировать ODE от t = 1 (шум) до t = 0 (данные).

    Шаг: x = x - dt * velocity_fn(x, t), затем t = t - dt, где dt = 1 / steps.
    Начинаем с t = 1.0 и делаем ровно steps шагов.

    euler_sample(lambda x, t: [1.0], [1.0], 4)  ->  [0.0]
    euler_sample(lambda x, t: [0.0], [5.0], 20) ->  [5.0]

    steps < 1 — ValueError: интегрировать ноль шагов бессмысленно.

    Ловушка: МИНУС. Скорость смотрит от данных к шуму, а идём мы против
    неё. Плюс вместо минуса уводит сэмпл в чистый шум.

    Зачем: если поле скоростей и правда постоянно вдоль пути (а в идеально
    выпрямленном flow это так), любое число шагов даёт один и тот же
    ответ. Отсюда 4-шаговые schnell-модели.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")

    x = list(x_init)
    dt = 1.0 / steps
    t = 1.0
    for _ in range(steps):
        v = velocity_fn(x, t)
        if len(v) != len(x):
            raise ValueError("velocity_fn must return a vector of the same length")
        # список пересобираем целиком: править x на месте нельзя, иначе
        # velocity_fn на следующем шаге увидит частично обновлённый вектор
        x = [value - dt * speed for value, speed in zip(x, v)]
        t -= dt
    return x
