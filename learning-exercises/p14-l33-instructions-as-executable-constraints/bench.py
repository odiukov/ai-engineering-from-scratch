"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_CATS = ("startup", "forbidden", "definition_of_done", "uncertainty", "approval")
_CHECKS = (
    "must_read_state",
    "no_edits_to",
    "tests_exit_zero",
    "ask_when_unsure",
    "approve_new_dependency",
)

_blocks = []
for i in range(400):
    check = _CHECKS[i % len(_CHECKS)]
    _blocks.append(
        f"## {_CATS[i % len(_CATS)]}/rule-{i:03d}\n"
        f"- category: {_CATS[i % len(_CATS)]}\n"
        f"- check: {check}\n"
        + (f"- arg: scripts/dir_{i}/*.sh\n" if check == "no_edits_to" else "")
        + f"- severity: {random.choice(('block', 'warn', 'info'))}\n"
        f"- expires_at: 202{random.randint(6, 9)}-0{random.randint(1, 9)}-15\n"
        f"Описание правила номер {i}.\n"
    )

_markdown = "# Agent Rules\n\n" + "\n".join(_blocks)

# разбор один раз здесь, чтобы замер check_rules не мерил заодно и парсер
_rules = [
    {
        "slug": f"cat-{i}/rule-{i}",
        "category": _CATS[i % len(_CATS)],
        "check": _CHECKS[i % len(_CHECKS)],
        "arg": f"scripts/dir_{i}/*.sh",
        "severity": random.choice(("block", "warn", "info")),
        "expires_at": f"2027-0{random.randint(1, 9)}-15",
        "description": "",
    }
    for i in range(400)
]

_trace = {
    "read_state": True,
    "edited_files": [f"pkg/mod_{i}.py" for i in range(200)],
    "tests_exit_code": 0,
    "confidence": 0.95,
    "asked_for_help": False,
    "added_dependencies": [],
    "approvals": [],
}

_results = [
    {"slug": r["slug"], "severity": r["severity"], "status": "pass"} for r in _rules
]

BENCH = {
    "parse_rules": (_markdown,),
    "is_operational": (_rules[0],),
    "compile_rule": (_rules[1],),
    "check_rules": (_rules, _trace),
    "severity_verdict": (_results,),
    "expired_rules": (_rules, "2027-06-01"),
    "rules_lock": (_rules,),
}
