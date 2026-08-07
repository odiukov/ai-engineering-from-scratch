"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_roots = [f"file:///Users/alice/project-{i}/notes" for i in range(40)]

_open = [
    f"file:///Users/alice/project-{random.randrange(60)}/notes/file-{i}.md"
    for i in range(400)
]

_state = {"roots": _roots, "open": _open}

_ids = [f"note-{i}" for i in range(60)]

_schema = {
    "type": "object",
    "properties": {"note_id": {"type": "string", "enum": _ids},
                   "confirm": {"type": "boolean"}},
    "required": ["note_id", "confirm"],
}

_store = {
    f"note-{i}": {"title": "TPS report" if i % 2 else f"note {i}",
                  "uri": f"file:///Users/alice/project-0/notes/file-{i}.md"}
    for i in range(60)
}


def _ask(request):
    return {"action": "accept", "content": {"note_id": "note-1", "confirm": True}}


BENCH = {
    "normalize_root": ("file:///Users/alice/../alice/Notes/sub/../sub/",),
    "within_roots": ("file:///Users/alice/project-39/notes/a.md", _roots),
    "update_roots": (_state, _roots),
    "elicitation_request": (1, "Pick one", _schema),
    "handle_elicitation_response": (
        {"action": "accept", "content": {"note_id": "note-1", "confirm": True}}, _schema),
    "disambiguate": (1, _ids),
    "delete_note": (dict(_store), "TPS report", ["file:///Users/alice/project-0"], _ask),
}
