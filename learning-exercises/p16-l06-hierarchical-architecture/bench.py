"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# Широкое дерево ровно из двух уровней: 200 sub-manager'ов по 30 работников.
_edges = []
for s in range(200):
    _edges.append(("top", f"sub-{s}"))
    for w in range(30):
        _edges.append((f"sub-{s}", f"w-{s}-{w}"))

_org = {}
for _manager, _report in _edges:
    _org.setdefault(_manager, []).append(_report)
    _org.setdefault(_report, [])

_answers = {f"w-{s}-{w}": f"finding {random.randint(0, 9)}"
            for s in range(200) for w in range(30)}
_labels = [f"sub-{s}" for s in range(200)]

BENCH = {
    "build_org": (_edges,),
    "validate_org": (_org, "top"),
    "depth": (_org, "top"),
    "leaves": (_org, "top"),
    "provenance": (_org, "top", "w-199-29"),
    "too_deep": (_org, "top"),
    "delegate": (_org, "top", _labels, _labels),
    "aggregate": (_org, "top", _answers),
}
