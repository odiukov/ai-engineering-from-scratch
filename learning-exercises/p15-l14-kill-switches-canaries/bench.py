"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_switch = {
    "engaged": False, "reason": None, "engaged_at": None,
    "released_by": None, "released_at": None,
}
_breaker = {"state": "closed", "recent": (), "fails": 0,
            "opened_at": None, "probes_left": 0}

# Длинная траектория: разнообразная работа, потом повторяющийся цикл.
_actions = [
    {"kind": "tool", "payload": f"read:file{i % 37}.py"} for i in range(4000)
] + [{"kind": "read", "payload": "~/.env.canary"}]

# Ряд частот вызовов с медленным дрейфом и шумом.
_rates = [1.0 + 0.002 * i + _rng.uniform(-0.3, 0.3) for i in range(4000)]

# Журнал таймстемпов, ускоряющийся к концу.
_times = [0.0]
for _i in range(3999):
    _times.append(_times[-1] + max(0.05, 1.0 - 0.0002 * _i))

BENCH = {
    "engage_kill_switch": (_switch, "runaway loop", 10.0),
    "release_kill_switch": (dict(_switch, engaged=True), "alice", "fixed", 20.0),
    "breaker_step": (_breaker, "read:app.log", True, 1.0),
    "canary_hits": (_actions,),
    "ewma": (_rates, 0.3),
    "ewma_alarm": (_rates, 0.3, 12.0),
    "hard_limit_breach": (_times, 100_000, 10.0),
    "run_trajectory": (_actions,),
}
