"""Входные данные для замера скорости."""

_TOPICS = ("prices", "orders", "alerts")


def _build(n):
    """Лог из n записей: корень с источником, остальные цитируют предыдущую."""
    pool = []
    for i in range(n):
        entry = {
            "seq": i,
            "writer": f"agent-{i % 7}",
            "topic": _TOPICS[i % 3],
            "value": 42.0,
            "ts": i,
            "source": "page-1" if i == 0 else None,
            "cites": () if i == 0 else (i - 1,),
            "supersedes": None,
        }
        pool.append(entry)
    return pool


_pool = _build(400)
_truth = {"page-1": 4.2}

BENCH = {
    # append_entry в замер не входит: он одноразовый по контракту (второй
    # вызов на той же записи обязан упасть), а bench гоняет функцию в цикле.
    "make_entry": ("A", "prices", 4.2, 10),
    "subscribe": (_pool, ["prices", "alerts"]),
    "active_entries": (_pool,),
    "provenance_chain": (_pool, 399),
    "verify": (_pool, _truth),
}
