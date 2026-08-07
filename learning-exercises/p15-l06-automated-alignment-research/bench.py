"""Входные данные для замера скорости."""

import hashlib
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_tasks = [(f"task-{i:03d}", 0.2 + (i % 7) / 10) for i in range(300)]
_agents = [f"AAR-{c}" for c in "ABCDEFGH"]

# Собираем цепочку заранее, чтобы verify_forum/regime_summary мерились на
# готовом форуме, а не на его постройке.
_forum = []
_prev = "0" * 16
for _i, (_name, _base) in enumerate(_tasks):
    _rec = {
        "author": _agents[_i % len(_agents)],
        "task": _name,
        "regime": "free",
        "result": _rng.random(),
        "prev_hash": _prev,
    }
    _payload = "{author}|{task}|{regime}|{result:.3f}|{prev}".format(prev=_prev, **{
        k: _rec[k] for k in ("author", "task", "regime", "result")})
    _rec["my_hash"] = hashlib.sha256(_payload.encode("utf-8")).hexdigest()[:16]
    _prev = _rec["my_hash"]
    _forum.append(_rec)

BENCH = {
    "record_hash": (_forum[0], "0" * 16),
    "append_record": (_forum, {"author": "AAR-A", "task": "extra",
                               "regime": "free", "result": 0.5}),
    "verify_forum": (_forum,),
    "tamper_record": (_forum, 10, 0.5),
    "allocate": ([name for name, _ in _tasks], _agents),
    "solve_task": (_rng, 0.4, "free"),
    "run_forum": (_rng, _tasks, _agents, "free"),
    "regime_summary": (_forum,),
}
