"""Входные данные для замера скорости."""

import random

random.seed(0)

_tree = ["app.py", "README.md", "scripts/release.sh", "test_app.py"]
_tree += ["src/mod%d.py" % i for i in range(400)]
_tree += ["docs/page%d.md" % i for i in range(400)]

_contract = {
    "allowed_globs": ["*.py"] + ["src/*.py"],
    "forbidden_globs": ["scripts/*", "README.md", "*.lock"],
    "acceptance_command": "python3 -m pytest -q",
}

_touched = [random.choice(_tree) for _ in range(3000)]

_repo = {"src/mod%d.py" % i: "def f():\n    return 422\n" for i in range(400)}
_checks = [
    {"name": "t%d" % i, "file": "src/mod%d.py" % i, "requires": ["422", "def f"]}
    for i in range(400)
]

_packet = {
    "next_action": "починить t17",
    "changed_files": ["src/mod0.py"],
    "verdict_pointer": {"verification": "v.json", "review": "r.json"},
}

_run = {
    "touched": _touched,
    "repo_after": _repo,
    "checks": _checks,
    "acceptance_test": "t0",
    "commands": ["python3 -m pytest -q"],
    "handoff": _packet,
    "reviewer_scores": {"dim%d" % i: i % 3 for i in range(50)},
}

_baseline = {
    "tests_actually_run": False,
    "acceptance_met": False,
    "files_outside_scope": 7,
    "handoff_quality": 0,
    "reviewer_total": 1,
}
_candidate = {
    "tests_actually_run": True,
    "acceptance_met": True,
    "files_outside_scope": 0,
    "handoff_quality": 3,
    "reviewer_total": 9,
}
_comparison = [
    {"outcome": k, "baseline": _baseline[k], "candidate": _candidate[k], "winner": "candidate"}
    for k in _baseline
]

BENCH = {
    "adapt_scope_contract": (_tree, ("secrets/*",)),
    "classify_touched_files": (_touched, _contract),
    "simulate_test_run": (_repo, _checks),
    "handoff_quality": (_packet,),
    "measure_run": (_run, _contract),
    "compare_pipelines": (_baseline, _candidate),
    "render_before_after": (_comparison,),
    "false_negative_reason": ({"kind": "formatter", "steps": 1},),
}
