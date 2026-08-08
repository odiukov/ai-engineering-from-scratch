"""
Инициализация весов и устойчивость обучения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def zero_init(fan_in, fan_out, seed=0):
    """Матрица нулей fan_out строк на fan_in столбцов.

    zero_init(2, 3)  ->  [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    Параметр seed не используется — он есть только ради единого интерфейса
    со всеми остальными init-функциями, чтобы их можно было подставлять
    друг вместо друга в forward_magnitudes.

    Это худшая инициализация из возможных: все нейроны слоя считают одно и
    то же, получают один и тот же градиент и обновляются одинаково. Слой
    из 512 нейронов работает как один.
    """
    return [[0.0] * fan_in for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0, seed=0):
    """Матрица fan_out x fan_in из нормального шума N(0, scale).

    len(random_init(4, 3))     ->  3     (строк = fan_out)
    len(random_init(4, 3)[0])  ->  4     (столбцов = fan_in)

    Один и тот же seed обязан давать один и тот же результат — заведи
    локальный random.Random(seed), а не трогай глобальный random: иначе
    два вызова подряд дадут разные матрицы и тесты станут невоспроизводимы.

    Симметрию такая инициализация ломает, но масштаб выбран наугад: при
    scale=1 сигнал взрывается за десяток слоёв, при scale=0.01 затухает.
    """
    rng = random.Random(seed)
    return [[rng.gauss(0.0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out, seed=0):
    """Инициализация Xavier/Glorot: N(0, sqrt(2 / (fan_in + fan_out))).

    xavier_init(2, 2) — веса примерно из N(0, 0.707), т.к. sqrt(2/4) = 0.707

    Дисперсия Var(w) = 2 / (fan_in + fan_out) выбрана так, чтобы дисперсия
    сигнала не менялась ни в прямом проходе (там важен fan_in), ни в
    обратном (там fan_out) — отсюда сумма в знаменателе.

    Правильный выбор для sigmoid и tanh: около нуля они почти линейны.
    """
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return random_init(fan_in, fan_out, scale=std, seed=seed)


def kaiming_init(fan_in, fan_out, seed=0):
    """Инициализация Kaiming/He: N(0, sqrt(2 / fan_in)).

    kaiming_init(8, 4) — веса примерно из N(0, 0.5), т.к. sqrt(2/8) = 0.5

    Двойка в числителе компенсирует ReLU: она зануляет примерно половину
    выходов, и без этой поправки дисперсия падала бы вдвое на каждом слое
    (0.5^50 = 8.8e-16 к пятидесятому).

    fan_out в формуле нет намеренно — He считал только прямой проход.
    Это не дефолт nn.Linear в PyTorch: тот использует равномерное
    распределение с границами +/-1/sqrt(fan_in) (через специальный вызов
    kaiming_uniform_ с a=sqrt(5)), а не He normal для ReLU.
    """
    std = math.sqrt(2.0 / fan_in)
    return random_init(fan_in, fan_out, scale=std, seed=seed)


def variance(values):
    """Дисперсия списка чисел относительно его собственного среднего.

    variance([1.0, 1.0, 1.0])       ->  0.0
    variance([2.0, 4.0, 4.0, 6.0])  ->  2.0

    Делим на len(values), а не на len - 1: нас интересует дисперсия
    выборки как она есть, а не несмещённая оценка генеральной.

    Ради этой величины и затевается вся возня с инициализацией:
    цель — чтобы Var(выход слоя) равнялась Var(входа слоя).
    """
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def matvec(matrix, vector):
    """Умножение матрицы на вектор: строка i даёт выход нейрона i.

    matvec([[1, 0], [0, 1]], [3, 4])  ->  [3.0, 4.0]
    matvec([[1, 1]], [3, 4])          ->  [7.0]

    Длина vector обязана совпадать с числом столбцов (fan_in).
    Результат длиной в число строк (fan_out).
    """
    return [
        sum(w * x for w, x in zip(row, vector))
        for row in matrix
    ]


def is_symmetry_broken(matrix):
    """Различаются ли строки матрицы, то есть разные ли нейроны в слое.

    is_symmetry_broken([[1.0, 2.0], [3.0, 4.0]])  ->  True
    is_symmetry_broken([[1.0, 2.0], [1.0, 2.0]])  ->  False
    is_symmetry_broken(zero_init(3, 5))           ->  False

    Матрица из одной строки — вырожденный случай: сравнивать не с чем,
    считаем симметрию сломанной (True).

    Пока строки одинаковы, нейроны получают одинаковый градиент и вечно
    остаются копиями друг друга, сколько бы эпох ни прошло.
    """
    if len(matrix) <= 1:
        return True
    first = matrix[0]
    return any(row != first for row in matrix[1:])


def forward_magnitudes(init_fn, activation, n_layers=20, width=32, seed=0):
    """Средний модуль активации на каждом из n_layers слоёв подряд.

    Вход — вектор длины width из N(0, 1). На каждом слое: свежая матрица
    init_fn(width, width, seed + номер слоя), matvec, поэлементно activation.
    Возвращает список из n_layers чисел — по одному на слой.

    len(forward_magnitudes(kaiming_init, lambda t: max(0.0, t)))  ->  20

    Каждому слою даётся свой seed (seed + i), иначе все слои получат
    одинаковые веса и эксперимент выродится в возведение матрицы в степень.

    Именно этот список показывает, взорвался сигнал или затух: у Kaiming
    с ReLU он держится около единицы все 50 слоёв, у random(scale=1)
    улетает за 1e10, у random(scale=0.01) падает ниже 1e-6.
    """
    rng = random.Random(seed)
    x = [rng.gauss(0.0, 1.0) for _ in range(width)]

    magnitudes = []
    for i in range(n_layers):
        weights = init_fn(width, width, seed + i)
        # activation применяем поэлементно: слой это matvec + нелинейность
        x = [activation(z) for z in matvec(weights, x)]
        magnitudes.append(sum(abs(v) for v in x) / width)
    return magnitudes


def recommend_init(activation_name):
    """Какую инициализацию брать под данную функцию активации.

    recommend_init("relu")     ->  "kaiming"
    recommend_init("sigmoid")  ->  "xavier"
    recommend_init("softsign") ->  "xavier"   (неизвестное — берём Xavier)

    Имя нечувствительно к регистру: "ReLU" и "relu" — одно и то же.

    Для ReLU половина симметричных входов действительно зануляется — отсюда
    точная поправка Kaiming. GELU и Swish/SiLU не обнуляют отрицательную
    половину буквально, но тоже работают как мягкие гейты; Kaiming для них —
    распространённая эвристика, а не вывод с тем же коэффициентом sqrt(2).
    Для насыщающихся sigmoid/tanh берём Xavier; он же запасной вариант.
    """
    relu_family = {"relu", "leaky_relu", "gelu", "swish", "silu", "elu"}
    return "kaiming" if activation_name.lower() in relu_family else "xavier"
