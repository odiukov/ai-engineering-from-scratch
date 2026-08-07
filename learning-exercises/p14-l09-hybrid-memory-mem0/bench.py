"""Входные данные для замера скорости."""

_WORDS = ("ava", "bob", "lives", "berlin", "lisbon", "refund", "invoice",
          "curriculum", "agent", "memory", "drift", "tool")

_records = [
    {
        "rid": f"m{i + 1:03d}",
        "text": " ".join(_WORDS[(i * 3 + j) % len(_WORDS)] for j in range(7)),
        "user_id": "ava" if i % 2 == 0 else "bob",
        "session_id": f"s{i % 5:03d}",
        "scope": "user",
        "importance": 0.1 + (i % 9) / 10.0,
        "ts": float(i * 3600),
        "kv": {"city": "Lisbon"} if i % 11 == 0 else {},
    }
    for i in range(300)
]

_edges = []
for _i in range(200):
    _edges.append({
        "subject": f"user{_i % 20}",
        "relation": "lives_in",
        "obj": f"city{_i}",
        "valid": _i >= 180,
        "valid_from": float(_i),
        "invalid_from": None if _i >= 180 else float(_i + 1),
    })

_vector = [float(i % 7) for i in range(64)]

BENCH = {
    "embed": ("ava lives in Berlin and ships agents for a living",),
    "cosine": (_vector, [float((i + 3) % 7) for i in range(64)]),
    "vector_search": (_records, "ava lives berlin curriculum", 5),
    "kv_lookup": (_records, "ava", "city"),
    "graph_add_edge": (_edges, "user3", "lives_in", "Porto", 1000.0),
    "graph_neighbors": (_edges, "user3", 190.0),
    "fuse_score": (_records[0], 0.7, 1000000.0),
    "hybrid_search": (_records, "ava lives berlin city", "ava", 1000000.0),
}
