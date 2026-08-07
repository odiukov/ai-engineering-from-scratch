"""Входные данные для замера скорости."""

_WORDS = ("agent", "memory", "context", "tool", "drift", "archival",
          "recall", "session", "budget", "prompt")

_store = [
    {
        "rid": f"a{i + 1:03d}",
        "text": " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(8)),
        "tags": ("bench",),
        "session_id": "s0",
        "turn_id": i,
    }
    for i in range(400)
]

_main = {
    "core": {"persona": "terse", "user": "name=ava", "task": "bench"},
    "messages": [("user", f"turn {i} about agent memory") for i in range(40)],
    "evicted": [("user", f"old turn {i} about tool drift") for i in range(200)],
    "max_messages": 40,
}

BENCH = {
    "core_memory_append": ({"user": "name=ava"}, "user", "city=Lisbon"),
    "core_memory_replace": ({"user": "city=Berlin"}, "user", "Berlin", "Lisbon"),
    "append_message": (_main, "user", "one more turn"),
    "render_main_context": (_main,),
    "archival_insert": (_store, "one more fact about agent memory"),
    "archival_search": (_store, "agent memory context budget", 5),
    "conversation_search": (_main, "old turn 3 about tool drift"),
    "page_in": (_main, _store[:3]),
}
