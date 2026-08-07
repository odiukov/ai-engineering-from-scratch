"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_LAYERS = ("benchmark", "custom", "online")
_N = 400

_cases = [
    {
        "id": f"case_{i:04d}",
        "layer": _LAYERS[i % len(_LAYERS)],
        "prompt": f"задача номер {i}",
        "expect": "ok" if i % 3 else "нет такого",
        "topics": (f"topic_{i % 40}",),
    }
    for i in range(_N)
]

_agent = lambda prompt: f"ok, обработал {prompt}"

_results = [
    {
        "id": case["id"],
        "layer": case["layer"],
        "passed": bool(random.getrandbits(1)),
        "answer": "",
        "error": None,
    }
    for case in _cases
]
_baseline = {case["id"]: bool(random.getrandbits(1)) for case in _cases}
_runs = [
    {case["id"]: bool(random.getrandbits(1)) for case in _cases} for _ in range(20)
]
_topics = tuple(f"topic_{i}" for i in range(200))

BENCH = {
    "run_case": (_cases[0], _agent),
    "run_suite": (_cases, _agent),
    "summarize": (_results,),
    "detect_regression": (_baseline, _results),
    "ci_gate": (_results, _baseline, 0.05),
    "evaluator_optimizer": (lambda fb: "мимо", lambda c: (False, "мимо"), 200),
    "flaky_cases": (_runs,),
    "coverage_gaps": (_cases, _topics),
}
