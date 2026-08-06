"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 10x6: одного вызова svd хватает на десятки миллисекунд, дефляция успевает
# показать разницу между аккуратной реализацией и лобовой
_A = [[random.uniform(-1.0, 1.0) for _ in range(6)] for _ in range(10)]
_triples = [(1.0 + i, [random.random() for _ in range(10)], [random.random() for _ in range(6)]) for i in range(6)]
_u = [random.random() for _ in range(60)]
_v = [random.random() for _ in range(60)]

BENCH = {
    "outer": (_u, _v),
    "frobenius_norm": (_A,),
    "power_iteration": (_A,),
    "top_singular_triple": (_A,),
    "svd": (_A, 2),
    "reconstruct": (_triples,),
    "condition_number": (_A,),
    "pseudoinverse": (_A,),
}
