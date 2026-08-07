"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# длинный лог: наивное усечение, склеивающее строки по одной, заметно
# медленнее среза списка
_long_log = "\n".join(f"line-{i} value={i}" for i in range(50_000))

_dirty_log = "\n".join(
    [
        "Authorization: Bearer sk-live-%s" % random.getrandbits(48),
        "api_key=%032x" % random.getrandbits(128),
        "AKIA%016X" % random.getrandbits(60),
    ]
    * 2000
)

# длинная цепочка попыток: наивный поиск родителя линейным проходом по
# списку даёт квадрат, словарь — линию
_chain = []
for _i in range(5000):
    _chain.append(
        {
            "command": ("pytest",),
            "command_id": f"attempt-{_i}",
            "parent_command_id": f"attempt-{_i - 1}" if _i else None,
            "exit_code": 1,
            "stdout_tail": "",
            "stderr_tail": "",
            "dropped_stdout_lines": 0,
            "dropped_stderr_lines": 0,
            "redacted": 0,
            "duration_ms": 1,
            "started_at": _i,
            "agent_note": "чиню",
            "error": None,
        }
    )

_store = {i: 900 for i in range(5)}


def _red_runner(turn, state):
    return {"command": ["pytest"], "exit_code": 1, "stdout": _long_log, "duration_ms": 1}


def _clock():
    return 1


BENCH = {
    "deterministic_tail": (_long_log,),
    "redact": (_dirty_log,),
    "make_record": (["pytest", "-q"], 1, _long_log, _dirty_log, "чиню", 0, 5, "a-0"),
    "loop_can_advance": (_chain[-1],),
    "retry_chain": (_chain, "attempt-4999"),
    "rotate": (_store, 500),
    "run_feedback_loop": (_red_runner, lambda r, s: "again", "чиню", _clock),
}
