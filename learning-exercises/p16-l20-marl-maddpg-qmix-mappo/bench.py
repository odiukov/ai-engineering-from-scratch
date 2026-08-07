"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# 7 агентов и 7 целей: 5040 перестановок — централизованный перебор
# уже заметно дороже независимого выбора, но укладывается в десятки мс.
_starts = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(7)]
_pellets = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(7)]
_assignment = list(range(7))

# 6 агентов по 6 действий: 46 656 совместных действий против 36 локальных
_q_tables = [[random.uniform(-5, 5) for _ in range(6)] for _ in range(6)]
_weights = [random.uniform(-3, 3) for _ in range(6)]
_q_values = [random.uniform(-5, 5) for _ in range(6)]
_returns = [random.gauss(0, 1) for _ in range(5000)]

BENCH = {
    "assignment_cost": (_starts, _pellets, _assignment),
    "independent_assignment": (_starts, _pellets),
    "centralized_assignment": (_starts, _pellets),
    "mix": (_q_values, _weights, 0.5),
    "mix_gradient": (_q_values, _weights),
    "joint_argmax": (_q_tables, _weights, 0.5),
    "decentralized_argmax": (_q_tables,),
    "centralized_advantage": (_returns,),
}
