"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_seconds = [random.uniform(0.05, 2.0) for _ in range(5000)]
_sub_questions = [f"q -- aspect {i}" for i in range(5000)]

_results = [
    {"sub_question": f"q{i}", "answer": "ok", "seconds": 0.1,
     "claims": {f"claim-{i % 50}": random.choice(["yes", "no"])}}
    for i in range(5000)
]
_conflicts = {f"claim-{i}": ["no", "yes"] for i in range(50)}

BENCH = {
    "scale_effort": (137,),
    "plan": ("what changed in multi-agent systems?", 10),
    "run_workers": (_sub_questions, lambda sq: {"answer": "ok", "seconds": 0.1}),
    "sequential_seconds": (_seconds, 0.05, 0.05),
    "parallel_seconds": (_seconds, 0.05, 0.05, 0.02),
    "detect_conflicts": (_results,),
    "synthesize": ("what changed?", _results, _conflicts),
}
