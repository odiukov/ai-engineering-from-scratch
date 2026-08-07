"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

# 4000 ходов: обычная работа, потом срыв в polling loop с шумом по токенам.
_turn_tokens = [
    _rng.randint(2000, 3000) if i < 800 else _rng.randint(7000, 12000)
    for i in range(4000)
]

_limits = {
    "max_tokens_per_request": 10_000,
    "max_turns": 4000,
    "max_budget_usd": 1e9,
    "velocity_usd_per_min": 1e9,
    "velocity_window_min": 10.0,
}

_ledger = {"turns": 0, "tokens": 0, "usd": 0.0, "history": [], "stopped_by": None}
_usd = 0.0
for _i, _tok in enumerate(_turn_tokens, 1):
    _usd += (_tok / 1000.0) * 0.003
    _ledger["turns"] = _i
    _ledger["tokens"] += _tok
    _ledger["usd"] = _usd
    _ledger["history"].append((_i * 0.5, _usd))

BENCH = {
    "tokens_to_usd": (12_345,),
    "cap_request_tokens": (80_000, 10_000),
    "new_ledger": (),
    "record_turn": (_ledger, 2500, 2000.5),
    "window_velocity": (_ledger["history"], 2000.0, 10.0),
    "first_breached_cap": (_ledger, _limits, 2000.0),
    "run_session": (_turn_tokens, _limits),
    "budget_warnings": (_ledger, _limits),
}
