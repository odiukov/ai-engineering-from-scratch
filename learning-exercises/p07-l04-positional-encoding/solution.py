"""
Позиционное кодирование — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def sinusoidal_encoding(n, d):
    """Абсолютное синусоидальное кодирование Vaswani 2017. Форма (n, d).

    PE[pos][2i]     = sin(pos / 10000^(2i / d))
    PE[pos][2i + 1] = cos(pos / 10000^(2i / d))

    sinusoidal_encoding(1, 4)   ->  [[0.0, 1.0, 0.0, 1.0]]
    sinusoidal_encoding(2, 2)   ->  [[0.0, 1.0], [sin(1), cos(1)]]

    Обрати внимание: d обязано быть чётным, иначе последней паре не хватит
    косинуса — это ValueError.

    Пара (2i, 2i+1) — это одна частота. Чем больше i, тем медленнее
    синусоида: у последней пары период порядка 10000 * 2 * pi позиций.
    Поэтому кодирование и не экстраполируется: за пределами обученной длины
    модель видит фазы, которых никогда не встречала.
    """
    if d % 2 != 0:
        raise ValueError("d must be even: each frequency needs a sin/cos pair")
    pe = []
    for pos in range(n):
        row = [0.0] * d
        for i in range(d // 2):
            # частота считается один раз на пару, а не два раза
            theta = pos / (10000 ** (2 * i / d))
            row[2 * i] = math.sin(theta)
            row[2 * i + 1] = math.cos(theta)
        pe.append(row)
    return pe


def add_positional_encoding(X):
    """X + sinusoidal_encoding(len(X), d). Форма не меняется.

    add_positional_encoding([[0.0, 0.0], [0.0, 0.0]])
        ->  [[0.0, 1.0], [sin(1), cos(1)]]

    Ровно это и делают с эмбеддингами перед первым слоем внимания.

    Проверь на себе главное: два одинаковых токена на разных позициях после
    этой операции перестают быть одинаковыми. До неё внимание не могло их
    различить вообще никак.
    """
    if not X:
        return []
    pe = sinusoidal_encoding(len(X), len(X[0]))
    return [[x + p for x, p in zip(row, pe_row)] for row, pe_row in zip(X, pe)]


def apply_rope(x, pos, base=10000):
    """RoPE: повернуть каждую пару координат (2i, 2i+1) на угол pos * theta_i.

    theta_i = base ** (-2i / d), где d — длина x.

    apply_rope([1.0, 0.0], 0)  ->  [1.0, 0.0]        (нулевая позиция — тождество)
    apply_rope([1.0, 0.0], 1)  ->  [cos(1), sin(1)]

    Поворот на угол a: (a_x, a_y) -> (x*cos - y*sin, x*sin + y*cos).
    Ловушка: знаки. Перепутаешь — получишь поворот в обратную сторону, и
    относительное свойство ниже сломается на нечётных сдвигах.

    Длина x обязана быть чётной (пары!) — иначе ValueError.

    Это поворот, а не растяжение: норма вектора обязана сохраниться.
    """
    d = len(x)
    if d % 2 != 0:
        raise ValueError("RoPE rotates pairs of coordinates: d must be even")
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def rope_dot(q, k, pos_q, pos_k, base=10000):
    """Скалярное произведение q и k ПОСЛЕ поворота каждого на свою позицию.

    Это тот самый скор внимания, который увидит softmax.

    rope_dot([1.0, 0.0], [1.0, 0.0], 0, 0)  ->  1.0
    rope_dot([1.0, 0.0], [1.0, 0.0], 5, 5)  ->  1.0

    Главное свойство RoPE: результат зависит только от РАЗНОСТИ pos_q - pos_k.
    Сдвинь обе позиции на одно и то же число — скор не изменится. Абсолютные
    позиции входят в формулу, а из скора выпадают.

    При pos_q == pos_k это обычное скалярное произведение q и k: повороты
    одинаковы и взаимно сокращаются.
    """
    rq = apply_rope(q, pos_q, base)
    rk = apply_rope(k, pos_k, base)
    return sum(a * b for a, b in zip(rq, rk))


def scale_rope_base(base, factor, d_head):
    """NTK-aware растяжение контекста: новый base = base * factor^(d/(d-2)).

    scale_rope_base(10000, 1.0, 128)  ->  10000.0   (растягивать нечего)
    scale_rope_base(10000, 32.0, 128) ->  примерно 338000

    factor — во сколько раз хотим удлинить контекст (например, 8K -> 128K
    это factor=16). Эта функция показывает именно dynamic NTK-aware формулу.
    Больший base означает меньшие углы поворота на той же позиции, то есть
    «медленные» размерности перестают убегать за пределы виденных фаз.

    d_head обязано быть больше 2, иначе показатель d/(d-2) взрывается —
    ValueError.

    Не приписывай ей любой длинный контекст автоматически: например, урок
    связывает 128K-контекст Llama 3.1 с YaRN, где интерполяция зависит от
    размерности и дополнительно корректируется температура внимания.
    """
    if d_head <= 2:
        raise ValueError("d_head must exceed 2 for NTK-aware scaling")
    if factor <= 0:
        raise ValueError("factor must be positive")
    return base * factor ** (d_head / (d_head - 2))


def alibi_slopes(n_heads):
    """Наклоны ALiBi по головам: slope[h] = 2 ** (-8 * (h + 1) / n_heads).

    alibi_slopes(8)  ->  [0.5, 0.25, 0.125, ..., 1/256]
    alibi_slopes(1)  ->  [1/256]

    Геометрическая прогрессия: голова 0 наказывает расстояние сильнее всех
    (смотрит близко), последняя — слабее всех (смотрит далеко). Ни одного
    обучаемого параметра здесь нет, это константы.

    n_heads должно быть положительным — иначе ValueError.
    """
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    return [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]


def alibi_bias(n_heads, seq_len):
    """Матрицы штрафов ALiBi: bias[h][i][j] = -slope[h] * |i - j|.

    Возвращает список из n_heads матриц (seq_len, seq_len).

    alibi_bias(1, 2)  ->  [[[0.0, -1/256], [-1/256, 0.0]]]

    Это прибавляется к скорам внимания ДО softmax. Никаких эмбеддингов
    позиции при этом не нужно вообще — отсюда и хорошая экстраполяция:
    формула -m*|i-j| определена на любой длине.

    Диагональ — нули: расстояние до себя нулевое, штрафа нет.
    """
    slopes = alibi_slopes(n_heads)
    # |i - j| считается один раз на пару и переиспользуется всеми головами
    distance = [[abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
    return [[[-slope * gap for gap in row] for row in distance] for slope in slopes]
