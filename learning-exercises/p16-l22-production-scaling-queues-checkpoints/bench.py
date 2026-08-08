"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_steps = [(lambda s, k=k: {"n": s["n"] + k}) for k in range(1, 41)]

# 200 тредов по 40 супершагов: 8000 записей в общем журнале. Наивный
# last_checkpoint, сканирующий весь журнал на каждом шаге, тут заметно
# просядет против варианта с индексом по thread_id.
_log = []
for _t in range(200):
    for _s in range(40):
        _log.append({"thread_id": "t-%d" % _t, "step": _s, "state": {"n": _s}})

_tasks = [
    {"thread_id": "q-%d" % i, "state": {"n": 0}, "worker": None,
     "leased_at": None, "done": False}
    for i in range(60)
]
class _BenchSink:
    def __init__(self):
        self.committed = {}

    def apply(self, key, payload, crash_after_commit=False):
        if key in self.committed:
            return False
        self.committed[key] = payload
        return True


_sink = _BenchSink()
for _i in range(5000):
    _sink.apply("k-%d" % _i, {"amount": 1})

BENCH = {
    "append_checkpoint": (list(_log), "t-1", 41, {"n": 0}),
    "last_checkpoint": (_log, "t-150"),
    "run_thread": (_steps, "fresh", [], {"n": 0}),
    "resume_until_done": (_steps, "fresh", [], {"n": 0}, [5, 17, 33]),
    "queue_transition": ("idle", "take"),
    "claim_task": (_tasks, "w1", 0, 5),
    "dedup_effect": (_sink, "k-4999", {"amount": 1}),
    "process_queue": (list(_tasks), _steps, []),
}
