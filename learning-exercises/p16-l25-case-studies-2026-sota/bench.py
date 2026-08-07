"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_xs = [random.uniform(0, 1000) for _ in range(20000)]
_ys = [0.003 * x + random.gauss(0, 1) for x in _xs]

# DAG на 2000 узлов: MacNet-масштаб. Наивный поиск «кто готов» перебором
# всех узлов на каждом шаге тут даёт квадрат и заметно просядет.
_dag = {}
for _i in range(2000):
    _succ = ["n%04d" % (_i + k) for k in (1, 7, 23) if _i + k < 2000]
    _dag["n%04d" % _i] = _succ

_runs = [
    {"version": "v%d" % random.randint(1, 12), "done": random.random() < 0.9}
    for _ in range(20000)
]

BENCH = {
    "linear_fit": (_xs, _ys),
    "r_squared": (_xs, _ys),
    "relative_improvement": (0.30, 0.57),
    "verification_budget": (1_000_000, 0.25),
    "subagent_budget": ("complex",),
    "topological_order": (_dag,),
    "critical_path": (_dag,),
    "retirable_versions": (_runs, "v12"),
}
