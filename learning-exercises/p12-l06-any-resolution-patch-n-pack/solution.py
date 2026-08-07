"""
Any-resolution: patch-n'-pack и NaFlex — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def patch_count(height, width, patch):
    """Сколько патч-токенов даст картинка на её РОДНОМ разрешении.

    patch_count(224, 224, 14)   ->  256
    patch_count(600, 1500, 14)  ->  4494   (чек 600x1500: 42 * 107)

    Деление нацело, а не с округлением вверх: остаточная полоска шириной
    меньше патча просто обрезается. Родное разрешение почти никогда не
    делится на 14 ровно, и «дотянуть» его паддингом дороже, чем срезать
    тринадцать пикселей.

    Любой неположительный аргумент — ValueError.
    """
    if height <= 0 or width <= 0 or patch <= 0:
        raise ValueError("height, width and patch must be positive")
    return (height // patch) * (width // patch)


def pack_batch(sizes, patch):
    """Склеить батч разноразмерных картинок в одну последовательность.

    sizes — список пар (height, width). Вернуть список полуинтервалов
    (start, end) — куда попали патчи каждой картинки в общей склейке.

    pack_batch([(224, 224), (336, 336)], 14)  ->  [(0, 256), (256, 832)]

    Это и есть patch-n'-pack из NaViT: никакого паддинга, никаких
    выброшенных на ветер токенов. Куски идут встык, а разделять их будет
    маска внимания, а не пустые позиции.

    Общая длина — это end последнего интервала.

    Пустой батч — ValueError. Картинка мельче одного патча — тоже ValueError:
    ноль токенов в упаковке ломает и маску, и статистику.
    """
    if not sizes:
        raise ValueError("batch must not be empty")
    spans = []
    offset = 0
    for height, width in sizes:
        n = patch_count(height, width, patch)
        if n == 0:
            raise ValueError("image is smaller than a single patch")
        spans.append((offset, offset + n))
        offset += n
    return spans


def block_diagonal_mask(spans):
    """Блочно-диагональная маска внимания для упакованного батча.

    Матрица total x total из 0 и 1: единица там, где обе позиции
    принадлежат одной картинке.

    block_diagonal_mask([(0, 2), (2, 3)])
        ->  [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 1]]

    Без этой маски патчи соседней картинки в склейке становятся
    полноправным контекстом, и модель начинает «дорисовывать» одну картинку
    по другой. Симптом на проде — галлюцинации, которые пропадают, стоит
    прогнать картинку в батче размером один.

    Матрица симметрична, диагональ вся единичная, а число единиц равно
    сумме квадратов длин блоков — три бесплатные самопроверки.
    """
    total = spans[-1][1] if spans else 0
    mask = [[0] * total for _ in range(total)]
    for start, end in spans:
        # заполняем только свой квадрат: O(sum n_i^2) вместо O(total^2)
        for i in range(start, end):
            row = mask[i]
            for j in range(start, end):
                row[j] = 1
    return mask


def mask_density(spans):
    """Доля разрешённых клеток маски: sum(n_i^2) / total^2.

    mask_density([(0, 2), (2, 4)])  ->  0.5
    mask_density([(0, 5)])          ->  1.0

    Она же — доля работы, которую внимание реально делает. Батч из
    восьми одинаковых картинок даёт плотность 1/8: семь восьмых клеток
    плотной маски посчитаны зря. Ровно поэтому FlashAttention ходит по
    varlen-пути с cu_seqlens и денсовую маску вообще не строит.

    Пустой список интервалов — ValueError.
    """
    if not spans:
        raise ValueError("need at least one span")
    total = spans[-1][1]
    inside = sum((end - start) ** 2 for start, end in spans)
    return inside / (total * total)


def padded_batch_cost(sizes, patch):
    """Сколько позиций съел бы батч, если добивать всё до самой длинной.

    padded_batch_cost([(224, 224), (336, 336)], 14)  ->  1152   (2 * 576)

    Сравни с суммой длин при упаковке (832 для того же батча): 320 позиций
    ушли бы в паддинг, и внимание честно посчитало бы их квадрат.

    Паддинг никогда не дешевле упаковки и равен ей ровно тогда, когда все
    картинки одного размера.
    """
    if not sizes:
        raise ValueError("batch must not be empty")
    lengths = [patch_count(h, w, patch) for h, w in sizes]
    return len(lengths) * max(lengths)


def square_resize_cost(sizes, side, patch):
    """Сколько токенов даст батч после ресайза всего в квадрат side x side.

    square_resize_cost([(600, 1500), (224, 224)], 336, 14)  ->  1152

    Обрати внимание: исходные размеры в ответ не входят вообще. В этом и
    беда подхода — расписка 600x1500 и квадратный кадр стоят одинаково,
    хотя у первой текст после сжатия в квадрат перестаёт читаться.

    Функция всё равно принимает sizes: так её видно рядом с pack_batch и
    padded_batch_cost, и разница в аргументах говорит сама за себя.
    """
    if not sizes:
        raise ValueError("batch must not be empty")
    if side <= 0 or patch <= 0:
        raise ValueError("side and patch must be positive")
    return len(sizes) * (side // patch) ** 2


def drop_patches(spans, keep, rng):
    """Случайно выбросить часть патчей каждой картинки. Новые интервалы.

    Каждый патч сохраняется независимо с вероятностью keep (бросок
    rng.random() < keep). Интервалы пересобираются встык, без дыр.

    rng = random.Random(0); drop_patches([(0, 100)], 0.5, rng)  ->  [(0, 37)]

    Это fractional patch dropping из NaViT: и регуляризация, и ускорение
    обучения даром. SigLIP 2 его унаследовал.

    rng передаётся ЯВНО, глобальный random использовать нельзя: иначе
    прогон не воспроизводится, и отладить расхождение метрик невозможно.

    keep вне отрезка [0, 1] — ValueError.
    """
    if not 0.0 <= keep <= 1.0:
        raise ValueError("keep must be within [0, 1]")
    out = []
    offset = 0
    for start, end in spans:
        # бросаем монетку на каждый патч отдельно, а не режем ровно долю:
        # именно так делает NaViT, и длина получается случайной
        kept = sum(1 for _ in range(end - start) if rng.random() < keep)
        out.append((offset, offset + kept))
        offset += kept
    return out


def fit_to_token_budget(height, width, patch, max_tokens):
    """Ужать картинку под потолок токенов, сохранив пропорции. Пара (h, w).

    Обе стороны на выходе кратны patch и не меньше одного патча, а
    patch_count(h, w, patch) гарантированно не превышает max_tokens.

    fit_to_token_budget(1920, 1080, 14, 1024)  ->  (588, 336)   1008 токенов
    fit_to_token_budget(224, 224, 14, 4096)    ->  (224, 224)   уже влезает

    Это ровно knob max_pixels из Qwen2.5-VL. Картинку, которая и так
    влезает, УВЕЛИЧИВАТЬ нельзя: апскейл не добавляет ни пикселя информации,
    зато платить за него придётся квадратично.

    Порядок действий: сначала общий масштаб sqrt(бюджет / нативных токенов),
    потом привязка обеих сторон вниз к кратному patch, потом — пока токенов
    всё ещё много — откусывать по одному патчу от длинной стороны.

    max_tokens < 1 или неположительные размеры — ValueError.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if height <= 0 or width <= 0 or patch <= 0:
        raise ValueError("height, width and patch must be positive")
    # min(1.0, ...) — запрет на апскейл
    scale = min(1.0, math.sqrt(max_tokens * patch * patch / (height * width)))
    h = max(patch, int(height * scale) // patch * patch)
    w = max(patch, int(width * scale) // patch * patch)
    # округления вниз могли оставить лишний ряд: добираем по одному патчу
    while (h // patch) * (w // patch) > max_tokens:
        if h >= w and h > patch:
            h -= patch
        elif w > patch:
            w -= patch
        else:
            break  # уже один патч на один патч, меньше некуда
    return (h, w)
