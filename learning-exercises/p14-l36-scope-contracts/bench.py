"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_N = 400

_touched = [
    f"{random.choice(('app', 'lib', 'docs', 'scripts', 'migrations'))}/"
    f"{random.choice(('a', 'b', 'c'))}/mod_{i}.{random.choice(('py', 'md', 'sql'))}"
    for i in range(_N)
]

_contract = {
    "task_id": "T-BENCH",
    "goal": "замер",
    "allowed_files": [f"app/**/mod_{i}.py" for i in range(60)] + ["lib/**/*.py"],
    "forbidden_files": ["migrations/**", "scripts/**"],
    "soft_files": ["docs/**", "**/*.md"],
    "acceptance_criteria": [f"pytest -q tests/test_{i}.py" for i in range(40)],
    "rollback_plan": "откатить коммит",
    "approvals_required": [f"approval_{i}" for i in range(40)],
    "time_budget_minutes": 30,
    "network_egress": [f"host{i}.example" for i in range(200)],
    "violation_budget": 5,
}
_other = {
    **_contract,
    "allowed_files": [f"app/**/mod_{i}.py" for i in range(30, 90)],
    "network_egress": [f"host{i}.example" for i in range(100, 300)],
}
_run = {
    "touched_files": _touched,
    "commands_run": _contract["acceptance_criteria"][:20],
    "elapsed_minutes": 12.0,
    "network_hosts": [f"host{i}.example" for i in range(150, 250)],
}
_features = {
    "project": "bench",
    "active": "",
    "features": [
        {"id": f"f{i:04d}", "status": "done" if i < _N - 1 else "todo", "goal": "g"}
        for i in range(_N)
    ],
}

BENCH = {
    "path_matches": ("app/a/b/c/mod_17.py", "app/**/*.py"),
    "classify_write": ("app/a/mod_17.py", _contract),
    "contract_gaps": (_contract,),
    "merge_egress": (_contract["network_egress"], _other["network_egress"]),
    "merge_contracts": (_contract, _other),
    "scope_check": (_contract, _run),
    "pick_feature": (_features,),
}
