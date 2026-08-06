"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_x = [random.uniform(-5.0, 5.0) for _ in range(20000)]
_y = [random.uniform(-5.0, 5.0) for _ in range(20000)]

_sum_sq = lambda p: sum(v * v for v in p)

# в этой f один вызов сам по себе стоит заметных денег: гессиан считает
# её 7 раз, иначе замер утонул бы в накладных расходах
_wavy = lambda p: sum(math.sin(p[0] * k) + p[1] * p[1] for k in range(2000))

_H = [[7.0, 2.0], [2.0, 5.0]]
_grad_quad = lambda p: [10 * p[0], 2 * p[1]]
_hess_quad = lambda p: [[10.0, 0.0], [0.0, 2.0]]

# 2000 однотипных линейных ограничений: KKT-невязки считаются по всему списку
_constraints = [
    (lambda p, k=k: p[0] + p[1] - k, lambda p: [1.0, 1.0]) for k in range(1, 2001)
]
_lambdas = [0.0] * 2000
_grad_sq = lambda p: [2 * p[0], 2 * p[1]]

BENCH = {
    "convex_combination": (_x, _y, 0.5),
    "segment_violation": (_sum_sq, _x, _y, 0.5),
    "check_convexity": (_sum_sq, 20, (-5.0, 5.0), 3000),
    "hessian_2x2": (_wavy, [0.3, -0.7]),
    "eigenvalues_2x2": (_H,),
    "newton_step": ([1.0, 1.0], [3.0, -4.0], _H),
    "newton_minimize": (_grad_quad, _hess_quad, [10.0, 10.0], 2000, 0.0),
    "kkt_violations": (_grad_sq, _constraints, [1.0, 1.0], _lambdas),
}
