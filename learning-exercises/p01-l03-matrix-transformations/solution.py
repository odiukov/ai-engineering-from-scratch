"""
Матричные преобразования и собственные значения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def rotation_matrix(degrees):
    """Матрица поворота против часовой стрелки на заданный угол.

    rotation_matrix(90)  ->  [[0, -1], [1, 0]]   (с точностью до округления)
    rotation_matrix(0)   ->  [[1, 0], [0, 1]]

    Вид: [[cos, -sin], [sin, cos]]. Проверь на 90°: cos=0, sin=1.
    """
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    return [[c, -s], [s, c]]


def scaling_matrix(sx, sy):
    """Матрица растяжения: x умножается на sx, y на sy.

    scaling_matrix(2, 3)  ->  [[2, 0], [0, 3]]

    Нули вне диагонали означают "оси не смешиваются".
    """
    return [[sx, 0], [0, sy]]


def apply(M, v):
    """Применить преобразование к вектору. Каждая строка даёт одну координату.

    apply([[0, -1], [1, 0]], [3, 1])  ->  [-1, 3]
    """
    return [sum(a * b for a, b in zip(row, v)) for row in M]


def compose(A, B):
    """Композиция: сначала B, потом A. Возвращает одну матрицу.

    compose(A, B) применённая к v даёт то же, что apply(A, apply(B, v)).

    Порядок обратный записи — это не опечатка, а следствие того, что
    матрицы пишутся слева от вектора: A(Bv) = (AB)v.
    """
    Bt = [list(col) for col in zip(*B)]
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def determinant_2x2(M):
    """Определитель матрицы 2x2: во сколько раз меняется площадь.

    determinant_2x2([[1, 0], [0, 1]])  ->  1     (площадь не изменилась)
    determinant_2x2([[2, 0], [0, 3]])  ->  6     (площадь выросла в 6 раз)
    determinant_2x2([[1, 2], [2, 4]])  ->  0     (площадь схлопнулась)

    Формула: a*d - b*c для [[a, b], [c, d]].
    Отрицательный определитель означает переворот ориентации.
    """
    (a, b), (c, d) = M
    return a * d - b * c


def eigenvalues_2x2(M):
    """Собственные значения матрицы 2x2, отсортированные по возрастанию.

    Возвращает кортеж из двух чисел. Гарантируется, что корни вещественные.

    eigenvalues_2x2([[2, 0], [0, 3]])  ->  (2.0, 3.0)
    eigenvalues_2x2([[1, 2], [2, 4]])  ->  (0.0, 5.0)

    Решается квадратное уравнение lambda^2 - trace*lambda + det = 0,
    где trace = a + d, det = a*d - b*c.
    """
    (a, b), (c, d) = M
    tr = a + d
    det = a * d - b * c
    disc = math.sqrt(tr * tr - 4 * det)
    # деление на 2.0, а не на 2 — чтобы результат всегда был float
    return ((tr - disc) / 2.0, (tr + disc) / 2.0)


def is_eigenvector(M, v, tol=1e-9):
    """Является ли v собственным вектором M: остаётся ли он на своей прямой.

    is_eigenvector([[2, 0], [0, 3]], [1, 0])  ->  True   (растянулся в 2 раза)
    is_eigenvector([[2, 0], [0, 3]], [1, 1])  ->  False  (повернулся)

    Собственный вектор после преобразования лишь масштабируется:
    M @ v = lambda * v. Направление не меняется.
    Нулевой вектор собственным не считается.
    """
    if all(abs(x) < tol for x in v):
        return False
    Mv = apply(M, v)
    # ищем масштаб по первой ненулевой компоненте, потом проверяем остальные
    scale = None
    for orig, new in zip(v, Mv):
        if abs(orig) > tol:
            scale = new / orig
            break
    if scale is None:
        return False
    return all(abs(new - scale * orig) < tol for orig, new in zip(v, Mv))
