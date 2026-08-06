"""
Visual Autoregressive (VAR): предсказание следующего масштаба — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def downsample(img, target):
    """Усреднить квадратное изображение до размера target x target.

    downsample([[1, 3], [5, 7]], 1)  ->  [[4.0]]          среднее всех
    downsample([[1, 3], [5, 7]], 2)  ->  [[1, 3], [5, 7]]  ничего не меняем

    Сторона исходного изображения обязана делиться на target нацело —
    иначе бросай ValueError, а не режь остаток молча.

    В VAR это построение пирамиды: тот же контент на всё более грубой сетке.
    """
    h = len(img)
    if target <= 0 or h % target != 0:
        raise ValueError(f"downsample: {h} не делится на {target}")
    factor = h // target
    area = factor * factor
    out = []
    for by in range(target):
        row = []
        for bx in range(target):
            # среднее по блоку factor x factor, а не выборка одного пикселя:
            # выборка потеряла бы половину энергии сигнала
            total = 0.0
            for y in range(by * factor, (by + 1) * factor):
                for x in range(bx * factor, (bx + 1) * factor):
                    total += img[y][x]
            row.append(total / area)
        out.append(row)
    return out


def upsample(grid, target):
    """Растянуть сетку до target x target ближайшим соседом.

    upsample([[2.0]], 2)              ->  [[2.0, 2.0], [2.0, 2.0]]
    upsample([[1, 2], [3, 4]], 2)     ->  [[1, 2], [3, 4]]

    target обязан делиться на сторону сетки нацело, иначе ValueError.

    Именно так декодер VAR поднимает эмбеддинги каждого масштаба до полного
    разрешения перед суммированием.
    """
    h = len(grid)
    if h <= 0 or target % h != 0:
        raise ValueError(f"upsample: {target} не делится на {h}")
    factor = target // h
    out = []
    for row in grid:
        # строку собираем один раз и повторяем factor раз ссылкой на КОПИЮ,
        # иначе все строки блока окажутся одним и тем же списком
        wide = [v for v in row for _ in range(factor)]
        for _ in range(factor):
            out.append(list(wide))
    return out


def encode_grid(grid, codebook):
    """Заменить каждое значение сетки индексом ближайшего кода.

    encode_grid([[0.1, 0.9]], [0.0, 0.5, 1.0])  ->  [[0, 2]]
    encode_grid([[0.4]], [0.0, 0.5, 1.0])       ->  [[1]]

    Возвращает сетку той же формы, но из целых индексов. При равном
    расстоянии до двух кодов бери меньший индекс — иначе тесты на
    воспроизводимость начнут мигать.

    Это квантование VQ-VAE: непрерывный латент становится дискретным токеном.
    """
    return [[min(range(len(codebook)), key=lambda i: abs(codebook[i] - v))
             for v in row] for row in grid]


def tokenize_multiscale(img, codebooks, scales):
    """Residual VQ: список токен-сеток, по одной на масштаб из scales.

    Масштаб k кодирует ОСТАТОК, который не смогли объяснить масштабы 1..k-1:
    огрубить остаток до scale x scale, квантовать, вычесть из остатка.

    tokenize_multiscale(img8x8, [book, book], (1, 2))
        ->  [сетка 1x1, сетка 2x2]

    Ловушка: если на каждом шаге огрублять исходное изображение вместо
    остатка, масштабы начнут дублировать друг друга и сумма разъедется.
    """
    size = len(img)
    residual = [list(row) for row in img]
    tokens = []
    for scale, book in zip(scales, codebooks):
        coarse = downsample(residual, scale)
        tok = encode_grid(coarse, book)
        # то, что этот масштаб реально объяснил, поднимаем до полного
        # разрешения и вычитаем — дальше работает уже следующий масштаб
        approx = upsample([[book[i] for i in row] for row in tok], size)
        residual = [[residual[y][x] - approx[y][x] for x in range(size)]
                    for y in range(size)]
        tokens.append(tok)
    return tokens


def detokenize_multiscale(tokens, codebooks, size):
    """Декодер VAR: сумма поднятых до size эмбеддингов всех масштабов.

    detokenize_multiscale([[[1]]], [[0.0, 0.5]], 2)  ->  [[0.5, 0.5], [0.5, 0.5]]

    Обратная к tokenize_multiscale. Никакой хитрости: подняли каждый
    масштаб до полного разрешения и сложили.

    Проверять надо так: чем больше масштабов подано, тем ближе результат к
    исходному изображению — ошибка обязана падать, а не колебаться.
    """
    out = [[0.0] * size for _ in range(size)]
    for tok, book in zip(tokens, codebooks):
        approx = upsample([[book[i] for i in row] for row in tok], size)
        for y in range(size):
            for x in range(size):
                out[y][x] += approx[y][x]
    return out


def scale_positions(scales):
    """Плоская последовательность позиций (индекс масштаба, строка, столбец).

    scale_positions((1, 2))
        ->  [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    Всего sum(s * s for s in scales) позиций, порядок — масштаб за масштабом,
    внутри масштаба построчно.

    Это и есть позиционное кодирование VAR: тройка (масштаб, строка, столбец)
    вместо одного индекса, как у обычного GPT.
    """
    out = []
    for k, scale in enumerate(scales):
        for row in range(scale):
            for col in range(scale):
                out.append((k, row, col))
    return out


def scale_causal_mask(scales):
    """Маска внимания VAR: mask[i][j] — можно ли токену i смотреть на токен j.

    Правило: токен масштаба k видит ВСЕ токены масштабов меньше k и НИ ОДНОГО
    из своего масштаба — все позиции масштаба k предсказываются параллельно,
    одним проходом, поэтому значений соседей у них ещё нет.

    scale_causal_mask((1, 2))
        ->  строка 0 целиком False; строки 1..4 — [True, False, False, False, False]

    Ловушка: обычная треугольная causal-маска из GPT здесь неверна — она
    разрешила бы смотреть на соседей внутри масштаба и сломала бы
    параллельную генерацию.
    """
    positions = scale_positions(scales)
    # сравниваем только индекс масштаба: пространственный порядок внутри
    # масштаба на видимость не влияет вообще
    return [[pj[0] < pi[0] for pj in positions] for pi in positions]


def generate_scales(predictor, scales, rng):
    """Сэмплировать пирамиду токенов: ровно один вызов predictor на масштаб.

    predictor(k, prev_tokens) возвращает список вероятностей по словарю для
    масштаба k, видя только уже сгенерированные масштабы. Все scale * scale
    позиций текущего масштаба сэмплируются из этого одного распределения —
    параллельно внутри масштаба.

    generate_scales(lambda k, prev: [0.0, 1.0], (1, 2), random.Random(0))
        ->  [[[1]], [[1, 1], [1, 1]]]

    rng — экземпляр random.Random, чтобы результат был воспроизводим.

    Здесь и вся выгода VAR: для K=10 масштабов это 10 проходов трансформера
    вместо 28-50 шагов диффузии и вместо сотен шагов пословной AR-генерации.
    """
    drawn = []
    for k, scale in enumerate(scales):
        probs = predictor(k, drawn)
        grid = []
        for _ in range(scale):
            row = []
            for _ in range(scale):
                # обратное преобразование по кумулятивной сумме
                r = rng.random()
                acc = 0.0
                pick = len(probs) - 1
                for i, p in enumerate(probs):
                    acc += p
                    if r <= acc:
                        pick = i
                        break
                row.append(pick)
            grid.append(row)
        drawn.append(grid)
    return drawn
