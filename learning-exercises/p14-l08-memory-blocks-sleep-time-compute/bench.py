"""Входные данные для замера скорости."""

_SENTENCE = "ava ships agents and prefers terse citation heavy writing"

_block = {
    "label": "human",
    "value": ". ".join(f"{_SENTENCE} {i}" for i in range(30)) + ".",
    "limit": 4000,
    "description": "facts about the user",
    "version": 7,
    "history": (),
}

_records = [
    {"rid": f"a{i + 1:03d}",
     "text": f"{_SENTENCE} variant {i % 40}",
     "valid": i % 7 != 0}
    for i in range(300)
]

_blocks = {
    "human": _block,
    "persona": {"label": "persona", "value": "terse. cites arXiv.", "limit": 160,
                "description": "self-concept", "version": 2, "history": ()},
    "task": {"label": "task", "value": "plan a curriculum. audience senior.",
             "limit": 40, "description": "current task", "version": 3,
             "history": ()},
}

BENCH = {
    "make_block": ("human", "facts about the user", 180),
    "block_append": (_block, "one more fact"),
    "block_replace": (_block, "ava", "eva"),
    "near_limit": (_block, 0.8),
    "summarize_block": (_block, 500),
    "dedup_archival": (_records, 0.9),
    "invalidate_contradicted": (_records, "variant 3"),
    "sleep_time_pass": (_blocks, _records, ("variant 3",)),
}
