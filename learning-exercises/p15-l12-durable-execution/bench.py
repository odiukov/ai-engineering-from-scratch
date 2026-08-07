"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _step(x):
    return (x * 31 + 7) % 1000


def _done(thread_id, name, args, result, at=0.0):
    return {
        "thread_id": thread_id,
        "key": f"{name}|{tuple(args)!r}",
        "name": name,
        "args": list(args),
        "status": "done",
        "result": result,
        "at": at,
    }


# Журнал из 600 завершённых активностей на каждый из двух тредов. Линейный
# поиск по нему делается на каждом шаге replay, поэтому именно он определяет
# скорость восстановления после сбоя.
_NAMES = [f"act{i}" for i in range(600)]
_LOG = []
for _i, _name in enumerate(_NAMES):
    for _tid in ("t-1", "t-2"):
        _LOG.append(_done(_tid, _name, (_i,), _i * 2, float(_i)))

# Журналы ниже заполнены заранее, чтобы замер был чистым реплеем: иначе первый
# прогон мутирует список, и второй участник сравнения окажется в других
# условиях.
_ACTIVITIES = tuple((name, _step) for name in _NAMES[:200])
_REPLAY_LOG = []
_value = 3
for _name, _fn in _ACTIVITIES:
    _REPLAY_LOG.append(_done("t-1", _name, (_value,), _fn(_value)))
    _value = _fn(_value)

_ACTIVITY_LOG = [_done("t-1", "double", (21,), 42)]
_CLOCK_LOG = [_done("t-1", "clock", (), 100.0)]

BENCH = {
    "activity_key": ("fetch", ("hello", 2)),
    "find_completed": (_LOG, "t-2", "act599|(599,)"),
    "run_activity": (_ACTIVITY_LOG, "t-1", "double", (21,), _step),
    "deterministic_value": (_CLOCK_LOG, "t-1", "clock", lambda: 100.0),
    "run_workflow": (_REPLAY_LOG, "t-1", 3, _ACTIVITIES),
    "execution_count": (_LOG, "t-1"),
    "replay_state": (_LOG, "t-1"),
    "needs_fresh_approval": (_LOG, "t-1", 10_000.0, 60.0),
}
