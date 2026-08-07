"""
Воплощённые VLA: RT-2, OpenVLA, pi0, GR00T — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Сколько токенов на один шаг действия у каждого формата вывода.
# "flow" — это pi0: голова flow-matching выдаёт непрерывный вектор, токенов
# в словаре LLM она не тратит вообще.
FORMATS = ("discrete_bin", "fast", "flow")


def discretize(action, bins=256, lo=-1.0, hi=1.0):
    """Квантование непрерывного действия в номера бинов (трюк RT-2).

    Возвращает список целых из диапазона 0..bins-1.

    discretize([-1.0, 0.0, 1.0], bins=4)  ->  [0, 2, 3]
    discretize([5.0], bins=256)           ->  [255]   (за границей — клип)

    Отрезок [lo, hi] режется на bins равных корзин, значение попадает в ту,
    внутри которой оно лежит. Ровно hi попадает в последнюю корзину, а не
    в несуществующую bins-ю — это и есть главная ловушка формулы
    int((v - lo) / (hi - lo) * bins).

    Так RT-2 превращает целевое положение сустава в обычный токен словаря:
    10-DOF рука — это 10 токенов на каждый шаг управления.
    """
    span = hi - lo
    out = []
    for v in action:
        idx = int((v - lo) / span * bins)
        # клип с двух сторон: значение ровно hi даёт idx == bins, а вход
        # за пределами калибровки прилетает регулярно
        out.append(min(bins - 1, max(0, idx)))
    return out


def undiscretize(tokens, bins=256, lo=-1.0, hi=1.0):
    """Обратное преобразование: номера бинов -> ЦЕНТРЫ соответствующих корзин.

    undiscretize([0, 2, 3], bins=4)  ->  [-0.75, 0.25, 0.75]

    Центр, а не левый край: середина корзины минимизирует максимальную
    ошибку восстановления. Она равна половине ширины корзины,
    (hi - lo) / (2 * bins) — для 256 бинов на [-1, 1] это 0.0039.

    Ловушка: точное значение уже не вернуть. Квантование необратимо, и
    именно эту потерю pi0 убирает непрерывной flow-matching головой.
    """
    span = hi - lo
    width = span / bins
    return [lo + (t + 0.5) * width for t in tokens]


def dct(x):
    """Дискретное косинусное преобразование DCT-II — основа FAST-токенизатора.

    dct([1.0, 1.0, 1.0, 1.0])  ->  [4.0, 0.0, 0.0, 0.0]

    X_k = sum_n x_n * cos(pi / N * (n + 0.5) * k), k = 0..N-1.
    Это scipy.fft.dct(x, type=2, norm=None), только руками.

    Зачем: гладкая траектория руки — это почти постоянный сигнал, и вся её
    энергия садится в первые пару коэффициентов. Оставив их, FAST жмёт
    30 шагов в ~10 токенов.
    """
    n = len(x)
    out = []
    for k in range(n):
        # шаг угла на отсчёт считаем один раз на коэффициент
        step = math.pi * k / n
        out.append(sum(v * math.cos(step * (i + 0.5)) for i, v in enumerate(x)))
    return out


def idct(coeffs):
    """Обратное преобразование к dct: DCT-III с нормировкой.

    idct(dct(x))  ->  x   (с точностью до float)

    x_n = X_0 / N + (2 / N) * sum_{k=1}^{N-1} X_k * cos(pi / N * (n + 0.5) * k)

    Нулевой коэффициент входит с весом 1/N, остальные с 2/N — асимметрия
    формулы, на которой спотыкаются все, кто пишет DCT впервые.
    """
    n = len(coeffs)
    out = []
    for i in range(n):
        acc = coeffs[0] / n
        for k in range(1, n):
            acc += 2.0 / n * coeffs[k] * math.cos(math.pi * k / n * (i + 0.5))
        out.append(acc)
    return out


def fast_tokens(trajectory, keep_coeff=4, bins=256):
    """FAST-токенизация траектории: DCT по каждой оси, первые keep_coeff, квантование.

    trajectory — список шагов, шаг — список из dof чисел в [-1, 1].
    Возвращает плоский список из dof * keep_coeff целых токенов.

    len(fast_tokens([[0.0, 0.0]] * 30, keep_coeff=4))  ->  8

    Порядок: сначала все коэффициенты оси 0, потом оси 1 и так далее.

    Коэффициенты DCT растут с длиной горизонта (X_0 — это сумма), поэтому
    перед квантованием дели их на len(trajectory): тогда они снова лежат в
    [-1, 1] и в тот же discretize укладываются без изменений.

    Смысл: 30 шагов * 10 DOF = 300 discrete-bin токенов против 40 FAST —
    отсюда ускорение декодирования в 3-5 раз.
    """
    horizon = len(trajectory)
    dof = len(trajectory[0])
    tokens = []
    for d in range(dof):
        series = [step[d] for step in trajectory]
        coeffs = dct(series)[:keep_coeff]
        # нормировка на горизонт возвращает коэффициенты в [-1, 1],
        # чтобы работал тот же самый discretize
        tokens.extend(discretize([c / horizon for c in coeffs], bins))
    return tokens


def fast_reconstruct(tokens, horizon, dof, keep_coeff=4, bins=256):
    """Восстановление траектории из FAST-токенов. Обратная к fast_tokens.

    Возвращает horizon шагов по dof чисел.

    fast_reconstruct(fast_tokens([[0.5]] * 8), 8, 1)  ->  примерно [[0.5]] * 8

    Разворот: токены -> центры корзин -> умножить обратно на horizon ->
    дополнить нулями до horizon коэффициентов -> idct.

    Ловушка: отброшенные коэффициенты — это НЕ ноль в исходном сигнале, а
    выброшенные высокие частоты. Гладкая траектория восстановится почти
    точно, дрожь барабанной палочки — превратится в кашу.
    """
    out_dims = []
    for d in range(dof):
        chunk = tokens[d * keep_coeff:(d + 1) * keep_coeff]
        coeffs = [c * horizon for c in undiscretize(chunk, bins)]
        # добиваем нулями: отброшенные высокие частоты просто не участвуют
        coeffs = coeffs + [0.0] * (horizon - len(coeffs))
        out_dims.append(idct(coeffs))
    return [[out_dims[d][i] for d in range(dof)] for i in range(horizon)]


def format_token_budget(dof, horizon, keep_coeff=4):
    """Сколько токенов стоит один горизонт действий в каждом формате вывода.

    format_token_budget(10, 30)
        ->  {"discrete_bin": 300, "fast": 40, "flow": 0}

    discrete_bin: по токену на каждую степень свободы каждого шага.
    fast:         dof * keep_coeff, длина горизонта роли не играет.
    flow:         0 — голова pi0 выдаёт вектор чисел, а не токены словаря.

    Отсюда считается частота управления: 10-DOF рука на 30 Гц в
    discrete-bin требует 300 токенов в секунду, и авторегрессионный 7B VLM
    столько не выдаёт — потому RT-2 и работает на 3-5 Гц.
    """
    return {
        "discrete_bin": dof * horizon,
        "fast": dof * keep_coeff,
        "flow": 0,
    }


def cofinetune_mix(web, robot, web_to_robot, size, rng):
    """Батч для co-fine-tuning: смесь web-данных и роботных демонстраций.

    Возвращает список из size элементов, выбранных с вероятностью
    web_to_robot / (1 + web_to_robot) из web и иначе из robot.

    cofinetune_mix(["w"], ["r"], 0.0, 3, random.Random(0))  ->  ["r", "r", "r"]

    web_to_robot — то самое соотношение из урока: у RT-2 примерно 1:1, у
    OpenVLA примерно 0.5:1. Слишком много web — модель забывает действия,
    слишком мало — теряет общие знания и ломается на новой формулировке
    инструкции.

    rng передаётся аргументом, глобальный random использовать нельзя:
    состав батча обязан воспроизводиться по сиду.
    """
    p_web = web_to_robot / (1.0 + web_to_robot)
    batch = []
    for _ in range(size):
        # один вызов random() на решение и один choice на выбор: так
        # последовательность зависит только от rng, а не от длин корпусов
        pool = web if rng.random() < p_web else robot
        batch.append(rng.choice(pool))
    return batch
