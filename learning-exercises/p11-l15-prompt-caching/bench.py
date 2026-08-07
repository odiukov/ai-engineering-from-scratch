"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_vocab = [f"tok{i}" for i in range(500)]
_system = [random.choice(_vocab) for _ in range(4000)]
_tail = [random.choice(_vocab) for _ in range(600)]

_prompt = _system + _tail
_other = ["ts=2026-08-07"] + _prompt[1:]

_sections = [
    ("system", _system[:2000], True),
    ("history", _tail[:300], False),
    ("tools", _system[2000:], True),
    ("user", _tail[300:], False),
]

_cache = [_system[:n] for n in (1024, 2048, 4000)]
_session = [_prompt] * 40

BENCH = {
    "common_prefix_len": (_prompt, _other),
    "cache_friendly_layout": (_sections,),
    "cache_lookup": (_cache, _prompt, 1024),
    "split_tokens": (_prompt, _cache, 4000, 1024),
    "request_cost": ({"read": 4000, "write": 0, "fresh": 600}, 3.0),
    "simulate_session": (_session, 4000, 3.0, 1024),
    "break_even_reads": (1.25, 0.1),
}
