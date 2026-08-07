"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ACCEPT = [f"pytest -q tests/test_{i:04d}.py" for i in range(400)]

# журнал с повторными прогонами: наивный поиск "последнего запуска команды"
# линейным сканом по всему журналу даёт квадрат
_feedback = [
    {"command": random.choice(_ACCEPT), "exit_code": random.choice((0, 0, 0, 1))}
    for _ in range(20_000)
] + [{"command": cmd, "exit_code": 0} for cmd in _ACCEPT]

_touched = [f"app/module_{i:04d}.py" for i in range(3000)] + [
    f"docs/page_{i:04d}.md" for i in range(3000)
]

_rules = [
    {"id": f"rule-{i:04d}", "severity": "block" if i % 3 else "warn", "passed": i % 7 != 0}
    for i in range(5000)
]

_artifacts = {
    "scope": {
        "acceptance": _ACCEPT,
        "allowed_files": ["app/*.py", "tests/*.py"],
        "forbidden_files": ["scripts/*", ".github/*"],
    },
    "feedback": _feedback,
    "diff": {"touched_files": _touched},
    "rules": _rules,
    "coverage": {"measured": 84.0, "previous": 84.0},
}

_clean = {
    "scope": _artifacts["scope"],
    "feedback": [{"command": cmd, "exit_code": 0} for cmd in _ACCEPT],
    "diff": {"touched_files": [f"app/module_{i:04d}.py" for i in range(3000)]},
    "rules": [{"id": f"r-{i}", "severity": "block", "passed": True} for i in range(5000)],
    "coverage": {"measured": 99.0, "previous": 99.0},
}

_blocked_report = {
    "task_id": "T-bench",
    "path": "outputs/verification/T-bench.json",
    "passed": False,
    "findings": [
        {
            "code": "RULE_FAILED",
            "severity": "block",
            "message": f"правило не выполнено: rule-{i}",
            "source": "rules",
            "overridden": False,
        }
        for i in range(5000)
    ],
    "gates_ran": ["feedback", "scope", "rules"],
    "gates_skipped": ["coverage"],
    "strict": False,
    "generated_at": 0,
    "overrides": [],
}

BENCH = {
    "finding": ("NULL_EXIT", "block", "нет exit_code", "feedback"),
    "gate_feedback": (_artifacts,),
    "gate_scope": (_artifacts,),
    "gate_rules": (_artifacts,),
    "gate_coverage": (_artifacts,),
    "run_gates": (_clean,),
    "verification_report": ("T-bench", _clean),
    "apply_override": (_blocked_report, "RULE_FAILED", "долг оформлен", "u-1", "abc123", 1),
}
