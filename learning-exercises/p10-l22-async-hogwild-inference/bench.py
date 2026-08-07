"""Входные данные для замера скорости."""

_CATEGORIES = 32
_ROUNDS = 400

# кеш от прогона четырёх воркеров: сюда попадают и дубли, и перебор квоты
_cache = [
    (_round, _worker, (_round + _worker) % _CATEGORIES, _round // _CATEGORIES)
    for _round in range(_ROUNDS)
    for _worker in range(4)
]

_counts = [_round % 7 for _round in range(_CATEGORIES)]

BENCH = {
    "amdahl_time": (10000.0, 0.7, 4),
    "hogwild_time": (10000.0, 0.7, 4, 200.0),
    "hogwild_speedup": (10000.0, 0.7, 4, 200.0),
    "best_worker_count": (10000.0, 0.7, 200.0, 64),
    "visible_counts": (_cache, _CATEGORIES, 3),
    "next_category": (_counts, 1.0),
    "run_hogwild": (4, 200, _CATEGORIES, 1.0, 0),
    "useful_work": (_cache, 10),
}
