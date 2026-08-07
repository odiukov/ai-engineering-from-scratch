"""Входные данные для замера скорости."""

_P = 8
_M = 32


def _one_f_one_b(p, m):
    """Копия рецепта 1F1B: bench не должен зависеть от твоего exercise.py."""
    queues = []
    for r in range(p):
        warm = min(p - 1 - r, m)
        ops = [("F", i, 1) for i in range(warm)]
        pending = 0
        for i in range(warm, m):
            ops.append(("F", i, 1))
            ops.append(("B", pending, 1))
            pending += 1
        while pending < m:
            ops.append(("B", pending, 1))
            pending += 1
        queues.append(ops)
    return queues


_order = _one_f_one_b(_P, _M)

# события ровно того расписания, что выдаст симулятор на _order
_events = []
for _rank, _ops in enumerate(_order):
    for _step, (_kind, _mb, _d) in enumerate(_ops):
        _events.append((_rank, _step + _rank, _kind, _mb, _d))

BENCH = {
    "gpipe_order": (_P, _M),
    "one_f_one_b_order": (_P, _M),
    "dualpipe_order": (_P, _M),
    "simulate_pipeline": (_order,),
    "makespan": (_events,),
    "bubble_slots": (_events,),
    "bubble_fraction": (_events,),
    "peak_activation_memory": (_order,),
}
