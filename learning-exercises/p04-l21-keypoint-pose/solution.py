"""
Ключевые точки и оценка позы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def gaussian_heatmap(size, cx, cy, sigma=2.0):
    """Целевая тепловая карта size x size: гауссов холм с центром в (cx, cy).

    Значение в пикселе: exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2)).
    Возвращается список строк, heatmap[y][x].

    gaussian_heatmap(3, 1, 1, sigma=1.0)[1][1]  ->  1.0
    gaussian_heatmap(3, 1, 1, sigma=1.0)[0][1]  ->  примерно 0.6065

    Ловушка: индексация. Первый индекс — строка, то есть y. Перепутать
    порядок легко, а тесты на несимметричном центре это ловят.

    Почему сеть учат на карты, а не сразу на (x, y): свёрточная карта
    признаков пространственная, и цель тоже пространственная. Небольшая
    ошибка локализации даёт небольшой лосс, а не нулевой градиент.
    """
    denom = 2.0 * sigma * sigma
    return [
        [math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / denom)) for x in range(size)]
        for y in range(size)
    ]


def argmax_coords(heatmap):
    """Координаты (x, y) максимума карты. Целые числа.

    argmax_coords([[0.0, 0.0], [0.0, 1.0]])  ->  (1, 1)

    Возвращаем именно (x, y), а не (row, col) — так принято во всех
    keypoint-API, и так же это уедет в разметку.

    Это весь инференс позы: argmax по каждому каналу.
    """
    best = (float("-inf"), 0, 0)
    for y, row in enumerate(heatmap):
        for x, v in enumerate(row):
            if v > best[0]:
                best = (v, x, y)
    return (best[1], best[2])


def subpixel_offset(heatmap, x, y):
    """Субпиксельная поправка (dx, dy) к целочисленному максимуму (x, y).

    По трём соседним значениям вдоль оси строим параболу и берём её вершину:
        d = 0.5 * (left - right) / (left - 2 * center + right)

    subpixel_offset([[0.0, 0.0, 0.0],
                     [1.0, 4.0, 3.0],
                     [0.0, 0.0, 0.0]], 1, 1)  ->  примерно (0.25, 0.0)

    На краю карты соседа нет — поправка 0.0. Если знаменатель нулевой
    (три равных значения, плато), поправка тоже 0.0, иначе деление на ноль.

    Зачем: argmax даёт целые пиксели, а на карте 64x64 один пиксель это
    несколько сантиметров реального сустава. Параболическая поправка —
    то, что делает каждая production-модель позы после argmax.
    """
    h, w = len(heatmap), len(heatmap[0])

    def fit(left, center, right):
        den = left - 2.0 * center + right
        if den == 0.0:
            return 0.0
        return 0.5 * (left - right) / den

    dx = fit(heatmap[y][x - 1], heatmap[y][x], heatmap[y][x + 1]) if 0 < x < w - 1 else 0.0
    dy = fit(heatmap[y - 1][x], heatmap[y][x], heatmap[y + 1][x]) if 0 < y < h - 1 else 0.0
    return (dx, dy)


def heatmaps_to_keypoints(heatmaps):
    """Список карт (по одной на точку) -> список координат (x, y) с субпикселем.

    Порядок карт = порядок ключевых точек. Именно поэтому поза это
    УПОРЯДОЧЕННЫЙ набор: канал 5 всегда левое плечо, и никак иначе.

    heatmaps_to_keypoints([gaussian_heatmap(9, 4, 4)])  ->  [(4.0, 4.0)]
    """
    out = []
    for hm in heatmaps:
        x, y = argmax_coords(hm)
        dx, dy = subpixel_offset(hm, x, y)
        out.append((x + dx, y + dy))
    return out


def mean_l2_error(predicted, target):
    """Средняя евклидова ошибка по всем точкам позы, в пикселях.

    mean_l2_error([(0.0, 0.0), (0.0, 0.0)], [(3.0, 4.0), (0.0, 0.0)])  ->  2.5

    Простая и честная метрика для отладки, но у неё нет единиц измерения,
    привязанных к размеру человека: 5 пикселей на портрете крупным планом
    и 5 пикселей на человеке в толпе — совершенно разные ошибки. Отсюда
    PCK и OKS ниже.
    """
    if not predicted:
        return 0.0
    total = sum(math.dist(p, t) for p, t in zip(predicted, target))
    return total / len(predicted)


def pck(predicted, target, threshold=0.2, normalizer=1.0):
    """PCK: доля точек, попавших в радиус threshold * normalizer.

    normalizer — размер объекта в тех же пикселях (диагональ бокса, размер
    головы, расстояние между плечами). Отсюда и смысл: "точка считается
    верной, если ошибка меньше 20% размера человека".

    pck([(0.0, 0.0)], [(1.0, 0.0)], threshold=0.5, normalizer=10.0)  ->  1.0
    pck([(0.0, 0.0)], [(9.0, 0.0)], threshold=0.5, normalizer=10.0)  ->  0.0

    Ключевое свойство: PCK не меняется при масштабировании картинки, если
    normalizer масштабируется вместе с ней. Метрика без normalizer врала бы
    в зависимости от разрешения — ровно та ошибка, которую делают, сравнивая
    модели на разных ресайзах.
    """
    if not predicted:
        return 0.0
    radius = threshold * normalizer
    hits = sum(1 for p, t in zip(predicted, target) if math.dist(p, t) <= radius)
    return hits / len(predicted)


def oks(predicted, target, scale, kappas=None):
    """OKS — Object Keypoint Similarity, аналог IoU для позы.

    OKS = среднее по точкам exp(-d^2 / (2 * scale^2 * kappa^2)),
    где scale — размер объекта, kappa — допуск конкретного сустава.

    Локоть размечают точнее, чем бедро, поэтому у каждого сустава свой
    kappa. Если kappas не передан, берём 0.05 для всех.

    oks([(0.0, 0.0)], [(0.0, 0.0)], scale=10.0)  ->  1.0
    oks([(0.0, 0.0)], [(5.0, 0.0)], scale=10.0)  ->  примерно 0.0

    Это то, что COCO усредняет в mAP@OKS. Как и PCK, метрика безразмерна:
    увеличь картинку вдвое вместе со scale — OKS не изменится.
    """
    if not predicted:
        return 0.0
    if kappas is None:
        kappas = [0.05] * len(predicted)
    total = 0.0
    for p, t, k in zip(predicted, target, kappas):
        d2 = (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
        total += math.exp(-d2 / (2.0 * scale * scale * k * k))
    return total / len(predicted)


def paf_line_integral(paf, p1, p2, samples=10):
    """Интеграл Part Affinity Field вдоль отрезка p1 -> p2.

    paf — сетка, paf[y][x] это пара (vx, vy). В каждой точке берём скалярное
    произведение вектора поля с единичным направлением отрезка и усредняем.

    Поле, целиком равное направлению конечности, даёт 1.0. Поле, направленное
    навстречу — -1.0. Перпендикулярное — 0.0.

    paf_line_integral([[(1.0, 0.0), (1.0, 0.0)]], (0, 0), (1, 0))  ->  1.0

    Так OpenPose снизу вверх собирает скелет: для каждой пары кандидатов
    "плечо-локоть" считаем интеграл, и жадно берём пары с максимальным.

    Ловушки: точки вне сетки нужно прижать к границе, а p1 == p2 не имеет
    направления — верни 0.0.
    """
    h, w = len(paf), len(paf[0])
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return 0.0
    ux, uy = dx / length, dy / length

    total = 0.0
    for i in range(samples):
        # samples == 1 не должен делить на ноль: берём середину отрезка
        t = 0.5 if samples == 1 else i / (samples - 1)
        # round + прижатие к границе: сетка дискретная, отрезок — нет
        gx = min(w - 1, max(0, int(round(p1[0] + t * dx))))
        gy = min(h - 1, max(0, int(round(p1[1] + t * dy))))
        vx, vy = paf[gy][gx]
        total += vx * ux + vy * uy
    return total / samples
