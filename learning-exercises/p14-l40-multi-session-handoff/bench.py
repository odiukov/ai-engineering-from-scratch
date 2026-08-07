"""Входные данные для замера скорости."""

import random

random.seed(0)

_feedback = [
    {"cmd": "pytest -q", "exit_code": random.choice([0, 0, 0, 0, 1])}
    for _ in range(4000)
]

_verdict = {
    "status": "pass",
    "report_path": "outputs/verification_report.json",
    "findings": [
        {"severity": random.choice(["info", "warn", "block"]), "detail": "f%d" % i}
        for i in range(400)
    ],
}
_review = {
    "report_path": "outputs/review_report.json",
    "findings": [
        {"severity": random.choice(["info", "warn"]), "detail": "r%d" % i}
        for i in range(400)
    ],
}

_board = [
    {"id": "F%d" % i, "status": "done", "actual_done": True, "title": "фича %d" % i}
    for i in range(300)
]

_workbench = {
    "uncommitted_files": [],
    "stash_note": None,
    "temp_artifacts": [],
    "tests": {"status": "green", "failure": ""},
    "feature_board": _board,
    "branch": "feat/signup",
    "expected_branch": "feat/signup",
    "orphan_branches": [],
}

_snapshot = {
    "task_id": "T-17",
    "topic": "signup-validation",
    "last_known_good_commit": "abc1234",
    "state": {
        "summary": "Добавил валидацию пароля.",
        "commands_run": ["pytest -q"] * 50,
        "failed_attempts": ["pydantic v1"] * 20,
    },
    "verdict": _verdict,
    "review": _review,
    "feedback": _feedback,
    "diff_summary": {"changed": ["f%d.py" % i for i in range(300)]},
}

_risks = [
    {"severity": "warn", "detail": "d%d" % i, "source": "verification"} for i in range(400)
]

_packets = [
    {
        "task_id": "T-%04d" % i,
        "generated_at": "2026-08-%02d" % (i % 28 + 1),
        "status": "active",
        "branch": "main",
        "topic": "auth",
    }
    for i in range(3000)
]

_payload = {
    "task_id": "T-17",
    "branch": "main",
    "status": "active",
    "generated_at": "2026-08-07",
    "last_known_good_commit": "abc1234",
    "summary": "s",
    "changed_files": ["f%d.py" % i for i in range(300)],
    "commands_run": ["pytest -q"] * 50,
    "failed_attempts": ["x"] * 20,
    "open_risks": _risks,
    "next_action": "начать фичу F1",
    "verdict_pointer": {"verification": "v.json", "review": "r.json"},
}

BENCH = {
    "trim_feedback": (_feedback, 5),
    "derive_open_risks": (_verdict, _review),
    "choose_next_action": (_verdict, _risks, _board),
    "clean_state_issues": (_workbench, _risks),
    "build_handoff": (_snapshot, _workbench, "2026-08-07T10:00:00"),
    "render_markdown": (_payload,),
    "resume_blockers": (_payload,),
    "select_active_handoff": (_packets, "main", "auth"),
}
