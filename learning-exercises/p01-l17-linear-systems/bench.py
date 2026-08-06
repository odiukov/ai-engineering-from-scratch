"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# --- треугольные системы: O(n^2), поэтому n берём заметно больше ---------
_TRI_N = 600

_UPPER = [
    [random.uniform(-1.0, 1.0) if j > i else 0.0 for j in range(_TRI_N)]
    for i in range(_TRI_N)
]
_LOWER = [
    [random.uniform(-1.0, 1.0) if j < i else 0.0 for j in range(_TRI_N)]
    for i in range(_TRI_N)
]
# диагональ доминирующая: без неё длинная подстановка расходится к inf
for _i in range(_TRI_N):
    _UPPER[_i][_i] = float(_TRI_N)
    _LOWER[_i][_i] = float(_TRI_N)

_TRI_RHS = [random.uniform(-1.0, 1.0) for _ in range(_TRI_N)]

# --- квадратные системы: O(n^3), один вызов должен влезать в десятки мс ---
_SQ_N = 90

_SQUARE = [[random.uniform(-1.0, 1.0) for _ in range(_SQ_N)] for _ in range(_SQ_N)]
_SQ_RHS = [random.uniform(-1.0, 1.0) for _ in range(_SQ_N)]

# A = M^T M + n*I — заведомо симметрична и положительно определена.
# Верхний треугольник считаем один раз и зеркалим, чтобы симметрия была точной.
_M = [[random.uniform(-1.0, 1.0) for _ in range(_SQ_N)] for _ in range(_SQ_N)]
_SPD = [[0.0] * _SQ_N for _ in range(_SQ_N)]
for _i in range(_SQ_N):
    for _j in range(_i, _SQ_N):
        _s = sum(_M[_k][_i] * _M[_k][_j] for _k in range(_SQ_N))
        _SPD[_i][_j] = _s
        _SPD[_j][_i] = _s
for _i in range(_SQ_N):
    _SPD[_i][_i] += _SQ_N

# --- регрессия: много строк, мало столбцов ------------------------------
_LS_M, _LS_N = 300, 40

_X = [[random.uniform(-1.0, 1.0) for _ in range(_LS_N)] for _ in range(_LS_M)]
_W_TRUE = [random.uniform(-2.0, 2.0) for _ in range(_LS_N)]
_Y = [
    sum(_X[_k][_j] * _W_TRUE[_j] for _j in range(_LS_N)) + random.gauss(0.0, 0.1)
    for _k in range(_LS_M)
]

BENCH = {
    "back_substitution": (_UPPER, _TRI_RHS),
    "forward_substitution": (_LOWER, _TRI_RHS),
    "gaussian_elimination": (_SQUARE, _SQ_RHS),
    "cholesky": (_SPD,),
    "solve_cholesky": (_SPD, _SQ_RHS),
    "normal_equations": (_X, _Y),
    "least_squares": (_X, _Y),
    "ridge_regression": (_X, _Y, 0.5),
}
