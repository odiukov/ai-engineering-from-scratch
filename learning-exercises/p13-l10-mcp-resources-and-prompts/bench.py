"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_store = {
    f"notes://note-{i}": {"mimeType": "text/markdown", "text": "x" * 200}
    for i in range(300)
}
_store["img://logo"] = {"mimeType": "image/png", "data": bytes(range(256)) * 40}

_templates = [
    {"uriTemplate": f"db{i}://" + "{table}/{row}", "read": lambda p: {"text": p["table"]}}
    for i in range(20)
]

_subscriptions = {
    f"notes://note-{i}": [f"session-{j}" for j in range(random.randint(1, 8))]
    for i in range(200)
}

_prompts = {
    "review_note": {
        "description": "Review one note.",
        "arguments": [{"name": "note_id", "required": True}, {"name": "tone", "required": False}],
        "messages": [
            {"role": "assistant", "content": {"type": "text", "text": "Tone {tone}. " * 40}},
            {"role": "user", "content": {"type": "text", "text": "Review {note_id}. " * 40}},
        ],
    }
}

BENCH = {
    "pick_primitive": ({"attachable": True},),
    "resource_entry": ("notes://note-1", "MCP overview", "text/markdown", "Заметка"),
    "read_resource": (_store, "img://logo"),
    "expand_template": ("db19://{table}/{row}", "db19://users/7"),
    "resolve": (_store, _templates, "db19://users/7"),
    "subscribe": (_subscriptions, "notes://note-199", "session-new"),
    "updated_notifications": (_subscriptions, "notes://note-7"),
    "render_prompt": (_prompts, "review_note", {"note_id": "note-14", "tone": "dry"}),
}
