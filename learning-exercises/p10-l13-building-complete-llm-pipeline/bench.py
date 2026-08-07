"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 120  # длинная линейная цепочка стадий: топосорт должен остаться быстрым
_names = [f"s{i:03d}" for i in range(_N)]
_deps = {name: ([] if i == 0 else [_names[i - 1]]) for i, name in enumerate(_names)}

_stages = {}
for _i, _name in enumerate(_names):
    _stages[_name] = {
        "deps": _deps[_name],
        "inputs": {} if _i == 0 else {_names[_i - 1]: f"h{_i - 1}"},
        "output": f"h{_i}",
    }

_gates = {f"m{i}": (">=", 0.5) for i in range(40)}
_metrics = {f"m{i}": random.random() for i in range(40)}

_payload = {f"k{i}": [random.random() for _ in range(8)] for i in range(60)}

_manifest = {
    "stages": _stages,
    "gates": _gates,
    "metrics": _metrics,
    "budget_usd": 1e12,
    "pretrain": {
        "params": 7e9,
        "tokens": 2e12,
        "peak_flops": 989e12,
        "mfu": 0.4,
        "usd_per_gpu_hour": 2.5,
    },
}

BENCH = {
    "stable_hash": (_payload,),
    "topological_order": (_deps,),
    "descendants": (_deps, _names[0]),
    "rollback_set": (_deps, _names[0]),
    "chain_violations": (_stages,),
    "gate_failures": (_gates, _metrics),
    "estimate_cost_usd": (7e9, 2e12, 989e12, 0.4, 2.5),
    "plan": (_manifest,),
}
