"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SKILLS = ("plan", "code", "review", "test")
_MATRIX = {
    f"role-{i}": {s: random.uniform(0.0, 1.0) for s in _SKILLS} for i in range(40)
}
_TASKS = [(f"task-{i}", _SKILLS[i % len(_SKILLS)]) for i in range(300)]

_SPEC = {
    "name": "add_two",
    "must_contain": ["def ", "return"],
    "tests": [((a, a + 1), 2 * a + 1) for a in range(200)],
}
_ARTIFACT = {"code": "def add_two(a, b):\n    return a + b\n", "fn": lambda a, b: a + b}


def _executor(spec, feedback):
    return _ARTIFACT


BENCH = {
    "competence": (_MATRIX, "role-0", "code"),
    "best_role": (_MATRIX, "code"),
    "assign_tasks": (_MATRIX, _TASKS),
    "team_quality": (_MATRIX, _TASKS),
    "generalist": (_SKILLS, 0.6),
    "critic_review": (_SPEC, _ARTIFACT),
    "verifier_run": (_SPEC, _ARTIFACT),
    "run_pipeline": (_SPEC, _executor),
}
