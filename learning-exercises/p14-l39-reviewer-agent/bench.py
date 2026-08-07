"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_BEHAVIORS = [f"behavior-{i:04d}" for i in range(2000)]
_ACCEPT = [f"pytest -q tests/test_{i:04d}.py" for i in range(500)]

_touched = [f"app/module_{i:04d}.py" for i in range(4000)]
_touched += [f"vendor/lib_{i:04d}.py" for i in range(1000)]

_inputs = {
    "task": {"required_behaviors": _BEHAVIORS, "acceptance": _ACCEPT},
    "diff": {
        "touched_files": _touched,
        "added_tests": [f"tests/test_{i:04d}.py" for i in range(500)],
        "behaviors_covered": _BEHAVIORS,
    },
    "scope": {
        "allowed_files": ["app/*.py", "tests/*.py"],
        "declared_growth": [f"vendor/lib_{i:04d}.py" for i in range(1000)],
    },
    "assumptions": [
        {"text": f"допущение {i}", "recorded_in": "docs/notes.md"} for i in range(3000)
    ],
    "feedback": [
        {"command": random.choice(_ACCEPT), "exit_code": 0} for _ in range(20_000)
    ]
    + [{"command": cmd, "exit_code": 0} for cmd in _ACCEPT],
    "handoff": {"next_action": "прогнать нагрузочный тест", "clean_state": []},
}

_rejected = dict(_inputs)
_rejected["diff"] = dict(_inputs["diff"], behaviors_covered=[])

_scores = {
    name: {"score": 2, "confidence": 0.9, "reason": "", "evidence": ()}
    for name in (
        "problem_fit",
        "scope_discipline",
        "assumptions",
        "verification_quality",
        "handoff_readiness",
    )
}

_previous = {
    "scores": {
        name: {"score": 0, "confidence": 0.9, "reason": "", "evidence": ()}
        for name in _scores
    },
    "grounds": sorted(_scores),
}

_cases = [
    {"id": f"case-{i}", "inputs": _inputs if i % 2 else _rejected,
     "verdict": "pass" if i % 2 else "hard_fail"}
    for i in range(40)
]


def _judge(x, y):
    return "first" if x["id"] < y["id"] else "second"


BENCH = {
    "reviewer_view": (_inputs,),
    "score_rubric": (_inputs,),
    "verdict_from_scores": (_scores,),
    "review_report": (_inputs,),
    "re_review": (_previous, _inputs),
    "consistent_pairwise_winner": (_judge, {"id": "A"}, {"id": "B"}),
    "calibration_agreement": (lambda i: {"verdict": "pass"}, _cases),
}
