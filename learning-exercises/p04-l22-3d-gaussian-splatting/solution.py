"""
3D Gaussian Splatting своими руками — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def covariance_2d(scale_x, scale_y, angle):
    """Ковариация одного сплата: Sigma = R S S^T R^T, матрица 2x2.

    R — поворот на angle радиан, S = diag(scale_x, scale_y).

    covariance_2d(2.0, 1.0, 0.0)         ->  [[4.0, 0.0], [0.0, 1.0]]
    covariance_2d(2.0, 1.0, math.pi / 2) ->  [[1.0, 0.0], [0.0, 4.0]]

    Обрати внимание: в Sigma входят КВАДРАТЫ масштабов. Поворот на 90 градусов
    меняет оси местами — это и есть проверка, что R и S перемножены в правильном
    порядке. Ошибка `S R R^T S^T` даёт то же самое при isotropic scale и
    молча неверный результат при anisotropic, то есть на реальной сцене.

    Неположительный масштаб -> ValueError: вырожденный сплат нельзя обратить,
    а именно обратная ковариация нужна на каждом пикселе.

    В 3D всё то же самое, только R строится из кватерниона, а S — из
    exp(log_scale) по трём осям.
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"scales must be positive, got ({scale_x}, {scale_y})")
    c, s = math.cos(angle), math.sin(angle)
    sx2, sy2 = scale_x * scale_x, scale_y * scale_y
    # раскрытое R diag(sx2, sy2) R^T: три уникальных числа вместо двух
    # матричных умножений, и симметрия гарантирована конструкцией
    a = sx2 * c * c + sy2 * s * s
    b = (sx2 - sy2) * s * c
    d = sx2 * s * s + sy2 * c * c
    return [[a, b], [b, d]]


def inverse_2x2(matrix):
    """Обратная матрица 2x2: [[a, b], [c, d]] -> [[d, -b], [-c, a]] / (ad - bc).

    inverse_2x2([[2.0, 0.0], [0.0, 4.0]])  ->  [[0.5, 0.0], [0.0, 0.25]]
    inverse_2x2([[1.0, 0.0], [0.0, 1.0]])  ->  [[1.0, 0.0], [0.0, 1.0]]

    Нулевой определитель -> ValueError. Для сплата это значит «эллипс схлопнулся
    в отрезок»: плотность в такой точке уходит в бесконечность, и вся картинка
    после этого превращается в NaN. Лучше упасть здесь.

    Именно эта матрица стоит в квадратичной форме exp(-0.5 d^T Sigma^-1 d),
    поэтому её считают один раз на сплат, а не один раз на пиксель.
    """
    (a, b), (c, d) = matrix[0], matrix[1]
    det = a * d - b * c
    if det == 0.0:
        raise ValueError("singular covariance: the splat has collapsed to a line")
    return [[d / det, -b / det], [-c / det, a / det]]


def gaussian_density(mean, cov, point):
    """Плотность сплата в пикселе: exp(-0.5 * (p - mu)^T Sigma^-1 (p - mu)).

    gaussian_density((0.0, 0.0), [[1.0, 0.0], [0.0, 1.0]], (0.0, 0.0))  ->  1.0
    gaussian_density((0.0, 0.0), [[4.0, 0.0], [0.0, 4.0]], (2.0, 0.0))  ->  exp(-0.5)

    Второй пример — ровно одна сигма: расстояние 2 при sigma = 2. Значение в
    центре всегда 1.0, нормировочный множитель 1/(2 pi sqrt(det)) НЕ нужен —
    его роль играет обучаемая opacity.

    Величина в экспоненте — квадрат расстояния Махаланобиса. Для вытянутого
    сплата один и тот же пиксель по короткой оси даёт заметно меньшую
    плотность, чем по длинной; на этом и держится анизотропия сцены.
    """
    inv = inverse_2x2(cov)
    dx = point[0] - mean[0]
    dy = point[1] - mean[1]
    # квадратичная форма руками: для 2x2 это дешевле любого общего кода
    q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy)
    return math.exp(-0.5 * q)


def alpha_composite(layers):
    """Alpha-композитинг спереди назад. Вернуть (цвет, остаточную прозрачность).

    layers — список (alpha, (r, g, b)) УЖЕ отсортированный от ближнего к дальнему.

    C = sum_i alpha_i * T_i * c_i,  T_i = prod_{j < i} (1 - alpha_j)

    alpha_composite([(1.0, (1.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 1.0))])
        ->  ([1.0, 0.0, 0.0], 0.0)          дальний слой полностью скрыт
    alpha_composite([])  ->  ([0.0, 0.0, 0.0], 1.0)

    Возвращаемая T — это доля пикселя, через которую виден фон. Сумма всех
    вкладов alpha_i * T_i плюс T всегда ровно 1: единица непрозрачности
    делится между слоями и фоном, ничего не теряется и не появляется.

    Порядок принципиален и не переставим: два перекрывающихся сплата,
    поменянные местами, дают РАЗНЫЙ цвет. Именно поэтому в 3DGS есть
    отдельная стадия depth-sort на каждый тайл.

    alpha вне [0, 1] -> ValueError.

    Это ровно то же уравнение, что интегрирует NeRF вдоль луча, только по
    явному разреженному набору гауссиан вместо плотной выборки.
    """
    colour = [0.0, 0.0, 0.0]
    transmittance = 1.0
    for alpha, rgb in layers:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        weight = alpha * transmittance
        for k in range(3):
            colour[k] += weight * rgb[k]
        transmittance *= 1.0 - alpha
    return colour, transmittance


def render_pixel(splats, point, max_alpha=0.99):
    """Отрисовать один пиксель: отсортировать сплаты по глубине и скомпозитить.

    Сплат — dict с ключами mean, cov, colour, opacity, depth.
    Меньшая depth — ближе к камере.

    Вклад сплата: alpha = min(opacity * density, max_alpha).

    render_pixel([{"mean": (0.0, 0.0), "cov": [[1.0, 0.0], [0.0, 1.0]],
                   "colour": (1.0, 1.0, 1.0), "opacity": 1.0, "depth": 0.0}],
                 (0.0, 0.0))
        ->  ([0.99, 0.99, 0.99], 0.01)

    Потолок max_alpha — не косметика: при alpha ровно 1.0 множитель (1 - alpha)
    обнуляет градиент всем сплатам за спиной, и они навсегда перестают учиться.
    0.99 оставляет им щель.

    Пустой список сплатов — валидный чёрный пиксель с T = 1.0, а не ошибка.
    """
    ordered = sorted(splats, key=lambda s: s["depth"])
    layers = []
    for s in ordered:
        density = gaussian_density(s["mean"], s["cov"], point)
        layers.append((min(s["opacity"] * density, max_alpha), s["colour"]))
    return alpha_composite(layers)


def eval_sh_degree_1(coeffs, direction):
    """Цвет сплата под заданным углом обзора: SH до степени 1 (4 коэффициента).

    coeffs — 4 тройки (r, g, b). direction — ЕДИНИЧНЫЙ вектор (x, y, z).

    result = C0*c0 - C1*y*c1 + C1*z*c2 - C1*x*c3,
    где C0 = 0.2820947917738781, C1 = 0.4886025119029199.

    eval_sh_degree_1([(1.0, 1.0, 1.0), (0, 0, 0), (0, 0, 0), (0, 0, 0)], (1.0, 0.0, 0.0))
        ->  [0.28209..., 0.28209..., 0.28209...]

    Нулевая степень — одно число на канал, цвет не зависит от направления
    (ламбертова поверхность). Первая степень добавляет линейную по направлению
    поправку: блик, который виден с одной стороны и не виден с другой.
    В production берут степень 3 — это 16 коэффициентов на канал, 48 float
    на сплат только под цвет.

    Ловушка: direction обязан быть единичным. Базис SH определён на сфере, и
    ненормированное направление тихо масштабирует цвет -> ValueError.
    Ровно 4 коэффициента, иначе ValueError.
    """
    if len(coeffs) != 4:
        raise ValueError(f"degree 1 needs exactly 4 SH coefficients, got {len(coeffs)}")
    x, y, z = direction
    if abs(math.sqrt(x * x + y * y + z * z) - 1.0) > 1e-6:
        raise ValueError("direction must be a unit vector")
    c0 = 0.2820947917738781
    c1 = 0.4886025119029199
    # знаки минус у y и x — не опечатка, это стандартный порядок базиса
    # (Y_1-1, Y_10, Y_11), тот же, что в gsplat и в inria/gaussian-splatting
    return [
        c0 * coeffs[0][k] - c1 * y * coeffs[1][k] + c1 * z * coeffs[2][k] - c1 * x * coeffs[3][k]
        for k in range(3)
    ]


def densify_decision(grad_norm, scale, opacity, grad_threshold=0.0002,
                     scale_threshold=0.01, opacity_threshold=0.005):
    """Что сделать со сплатом на шаге денсификации: prune / split / clone / keep.

    densify_decision(0.001, 0.5, 0.0001)  ->  "prune"   (прозрачный, не влияет)
    densify_decision(0.001, 0.5, 0.9)     ->  "split"   (большой и недообученный)
    densify_decision(0.001, 0.001, 0.9)   ->  "clone"   (мелкий, деталей не хватает)
    densify_decision(0.0, 0.5, 0.9)       ->  "keep"

    Порядок проверок важен: сначала prune. Сплат с нулевой прозрачностью не
    надо ни делить, ни клонировать — из него получатся два одинаково
    бесполезных сплата, и сцена будет расти впустую.

    Дальше большой градиент означает «здесь реконструкция не сходится».
    Крупный сплат слишком гладкий для этого места — делим. Мелкий сплат просто
    не покрывает область — клонируем.

    Так сцена растёт со ~100k точек SfM до 1-5M сплатов за обучение.
    """
    if opacity < opacity_threshold:
        return "prune"
    if grad_norm > grad_threshold:
        return "split" if scale > scale_threshold else "clone"
    return "keep"


def gaussian_float_count(sh_degree):
    """Сколько float занимает один сплат при заданной степени SH.

    position 3 + rotation 4 + scale 3 + opacity 1 + 3 * (L + 1)^2 на цвет.

    gaussian_float_count(0)  ->  14
    gaussian_float_count(3)  ->  59

    Цвет — самая дорогая часть: при L=3 это 48 из 59 float. Поэтому первое,
    что делают при экспорте в .splat или glTF, — квантуют или срезают SH.

    Сцена в 5M сплатов при L=3: 5e6 * 59 * 4 байта = 1.18 ГБ в float32,
    а int8 занимает 295 МБ (десятичных), то есть примерно 281 МиБ.
    Отрицательная степень -> ValueError.
    """
    if sh_degree < 0:
        raise ValueError(f"sh_degree must be non-negative, got {sh_degree}")
    return 3 + 4 + 3 + 1 + 3 * (sh_degree + 1) ** 2
