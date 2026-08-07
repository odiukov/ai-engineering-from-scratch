"""
Монокулярная глубина и геометрия — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def flatten_valid(pred, target):
    """Развернуть две карты глубины в плоские списки, выкинув битые пиксели.

    Пиксель считается валидным, когда глубина в target строго положительна и
    оба значения конечны (не inf и не nan). Порядок обхода — построчно.

    flatten_valid([[1.0, 2.0]], [[1.0, 0.0]])  ->  ([1.0], [1.0])
    flatten_valid([[1.0]], [[3.0]])            ->  ([1.0], [3.0])

    Зачем: у датчиков глубины дырки — ноль там, где луч не вернулся, и
    насыщение там, где вернулся слишком яркий. Если не отфильтровать, любая
    метрика поделит на ноль и покажет бессмыслицу.

    Ловушка: разные размеры карт — это ValueError, а не «сравним по минимуму».
    """
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same height")
    flat_pred, flat_target = [], []
    for row_p, row_t in zip(pred, target):
        if len(row_p) != len(row_t):
            raise ValueError("pred and target must have the same width")
        for p, t in zip(row_p, row_t):
            # math.isfinite ловит и inf, и nan одной проверкой
            if t > 0 and math.isfinite(t) and math.isfinite(p):
                flat_pred.append(p)
                flat_target.append(t)
    return flat_pred, flat_target


def abs_rel_error(pred, target):
    """AbsRel — средняя относительная ошибка глубины. Меньше значит лучше.

    Формула: mean(|d_pred - d_gt| / d_gt) по валидным пикселям.

    abs_rel_error([[2.0]], [[1.0]])            ->  1.0
    abs_rel_error([[1.0, 4.0]], [[1.0, 2.0]])  ->  0.5
    abs_rel_error([[5.0]], [[5.0]])            ->  0.0

    У продакшн-моделей 0.05-0.1.

    Ошибка ДЕЛИТСЯ на ground truth, а не на предсказание — метрика
    несимметрична: перепутав аргументы, получишь другое число.

    Ловушка: если валидных пикселей не осталось, среднее считать не от чего —
    брось ValueError.
    """
    p, t = flatten_valid(pred, target)
    if not t:
        raise ValueError("no valid pixels to compare")
    return sum(abs(pi - ti) / ti for pi, ti in zip(p, t)) / len(t)


def delta_accuracy(pred, target, threshold=1.25):
    """delta < 1.25 — доля пикселей, где предсказание в пределах 25% от истины.

    Для каждого пикселя берём max(d_pred/d_gt, d_gt/d_pred) и сравниваем с
    threshold. Больше значит лучше, у SOTA 0.9+.

    delta_accuracy([[1.0, 1.0]], [[1.0, 10.0]])  ->  0.5
    delta_accuracy([[1.0]], [[1.0]])             ->  1.0

    Отношение через max делает метрику СИММЕТРИЧНОЙ: перестановка аргументов
    ответа не меняет. С AbsRel так не выйдет.

    Ловушка: неположительное предсказание (модель выдала 0 или минус) — деления
    не будет, такой пиксель просто не попадает в число точных.
    """
    p, t = flatten_valid(pred, target)
    if not t:
        raise ValueError("no valid pixels to compare")
    good = 0
    for pi, ti in zip(p, t):
        if pi <= 0:
            continue  # отношение не определено, считаем пиксель промахом
        if max(pi / ti, ti / pi) < threshold:
            good += 1
    return good / len(t)


def align_scale_shift(pred, target):
    """Подогнать относительную глубину под метрическую: найти (a, b) в a*pred+b.

    Метод наименьших квадратов: минимизируем sum((a*p + b - t)^2) по валидным
    пикселям. Возвращаем кортеж (a, b).

    align_scale_shift([[1.0, 2.0]], [[3.0, 5.0]])  ->  (2.0, 1.0)
    align_scale_shift([[1.0, 2.0]], [[1.0, 2.0]])  ->  (1.0, 0.0)

    Зачем: MiDaS и Depth Anything выдают глубину БЕЗ единиц измерения. Пока
    её не выровняли по ground truth, AbsRel считать бессмысленно — он померит
    произвол масштаба, а не качество модели.

    Замкнутая формула: a = (n*Spt - Sp*St) / (n*Spp - Sp*Sp), b = mean(t) - a*mean(p).

    Ловушка: если все предсказания одинаковые, знаменатель ноль — прямую через
    одну точку не провести, брось ValueError.
    """
    p, t = flatten_valid(pred, target)
    if not p:
        raise ValueError("no valid pixels to align")
    n = len(p)
    sum_p = sum(p)
    sum_t = sum(t)
    sum_pt = sum(pi * ti for pi, ti in zip(p, t))
    sum_pp = sum(pi * pi for pi in p)
    denom = n * sum_pp - sum_p * sum_p
    # denom это n^2 * дисперсия предсказания: ноль ровно тогда,
    # когда предсказание постоянно
    if abs(denom) < 1e-12:
        raise ValueError("prediction has no variance, scale is not identifiable")
    a = (n * sum_pt - sum_p * sum_t) / denom
    b = (sum_t - a * sum_p) / n
    return a, b


def aligned_abs_rel(pred, target):
    """AbsRel после выравнивания масштаба и сдвига — метрика для MiDaS и Depth
    Anything.

    Сначала align_scale_shift, потом применить (a, b) ко ВСЕЙ карте, потом
    обычный abs_rel_error.

    aligned_abs_rel([[1.0, 2.0]], [[3.0, 5.0]])    ->  0.0
    aligned_abs_rel([[10.0, 20.0]], [[3.0, 5.0]])  ->  0.0

    Ключевое свойство: результат не меняется, если предсказание умножить на
    любую положительную константу или сдвинуть на любую константу. Именно это
    и значит «scale-and-shift invariant».
    """
    a, b = align_scale_shift(pred, target)
    aligned = [[a * value + b for value in row] for row in pred]
    return abs_rel_error(aligned, target)


def pixel_to_camera(u, v, d, intrinsics):
    """Поднять один пиксель (u, v) с глубиной d в 3D-точку камеры (X, Y, Z).

    intrinsics — кортеж (fx, fy, cx, cy) модели дырочной камеры.

    pixel_to_camera(160, 120, 2.0, (320.0, 320.0, 160.0, 120.0))  ->  (0.0, 0.0, 2.0)
    pixel_to_camera(320, 120, 2.0, (320.0, 320.0, 160.0, 120.0))  ->  (1.0, 0.0, 2.0)

    Формулы: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d.

    Смысл: пиксель задаёт ЛУЧ из центра камеры, глубина говорит, как далеко по
    этому лучу лежит точка. В главной точке (cx, cy) луч смотрит строго вперёд,
    поэтому X и Y там нулевые при любой глубине.

    Ловушка: нулевое фокусное расстояние — ValueError, а не ZeroDivisionError
    из середины формулы.
    """
    fx, fy, cx, cy = intrinsics
    if fx == 0 or fy == 0:
        raise ValueError("focal length must be non-zero")
    return ((u - cx) * d / fx, (v - cy) * d / fy, d)


def depth_to_point_cloud(depth, intrinsics):
    """Развернуть всю карту глубины в облако точек: (H, W) -> (H, W, 3).

    Порядок обхода тот же, что у карты: строка v (сверху вниз), столбец u
    (слева направо). depth[v][u] — глубина пикселя со столбцом u в строке v.

    depth_to_point_cloud([[2.0]], (1.0, 1.0, 0.0, 0.0))  ->  [[(0.0, 0.0, 2.0)]]

    Одна функция — и она стоит под каждым AR-приложением, каждым обходом
    препятствий и каждым «возьми чашку».

    Ловушка: v это номер СТРОКИ, u — номер СТОЛБЦА. Перепутанные местами, они
    дают зеркально вывернутое облако, которое на глаз выглядит правдоподобно.
    """
    # переиспользуем pixel_to_camera вместо копипасты формулы: правка
    # интринсик-логики в одном месте гарантированно доедет сюда
    return [[pixel_to_camera(u, v, depth[v][u], intrinsics)
             for u in range(len(depth[v]))]
            for v in range(len(depth))]


def lift_box_to_3d(depth, box, intrinsics):
    """Поднять 2D-детекцию в 3D-точку: центр бокса на медианной глубине бокса.

    box — кортеж (x1, y1, x2, y2), границы в пикселях, правая и нижняя НЕ
    включаются (как срез в питоне). Центр берём как ((x1+x2)/2, (y1+y2)/2).
    Глубину — медиану валидных (положительных и конечных) глубин внутри бокса.

    lift_box_to_3d([[2.0, 2.0], [2.0, 2.0]], (0, 0, 2, 2), (1.0, 1.0, 1.0, 1.0))
        ->  (0.0, 0.0, 2.0)

    Почему медиана, а не среднее: внутри бокса всегда есть пиксели фона и
    бликов. Одно значение «100 метров» на краю утаскивает среднее, медиане же
    нужно испортить половину пикселей.

    Ловушки: пустой бокс и бокс, целиком попавший на битые пиксели — оба это
    ValueError.
    """
    x1, y1, x2, y2 = box
    values = []
    for v in range(max(0, y1), min(len(depth), y2)):
        for u in range(max(0, x1), min(len(depth[v]), x2)):
            d = depth[v][u]
            if d > 0 and math.isfinite(d):
                values.append(d)
    if not values:
        raise ValueError("box contains no valid depth pixels")
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return pixel_to_camera((x1 + x2) / 2, (y1 + y2) / 2, median, intrinsics)
