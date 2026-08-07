"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# Плотный UI: 800 элементов, как на настоящей странице бронирования
_elements = []
for _i in range(800):
    _x = random.randrange(0, 2400)
    _y = random.randrange(0, 1300)
    _elements.append({
        "desc": f"element {_i}",
        "bbox": (_x, _y, _x + 120, _y + 36),
        "goto": f"https://example/{_i}" if _i % 7 == 0 else None,
    })

_state = {
    "url": "https://example/start",
    "elements": _elements,
    "fields": {f"f{i}": f"v{i}" for i in range(50)},
    "error": None,
}

_click = {"action": "click", "x": 2399, "y": 1299, "element_desc": "element 799"}
_reply = (
    "Мне нужно нажать кнопку поиска, чтобы увидеть рейсы.\n"
    '```json\n{"action": "click", "x": 384, "y": 220, '
    '"element_desc": "Search button"}\n```\n'
)
_history = [
    {"screenshot": f"img{i}", "action": {"action": "wait", "ms": i}}
    for i in range(500)
]
_results = [{"success": random.random() < 0.27} for _ in range(1000)]

BENCH = {
    "validate_action": (_click,),
    "parse_action": (_reply,),
    "scale_click": (_click, (1120, 1120), (2560, 1440)),
    "apply_action": (_state, _click),
    "recover": (_click, _state),
    "compress_history": (_history, 4),
    "agent_loop": (_state, lambda s: {"action": "click", "x": 1, "y": 1}, 200),
    "success_rate": (_results,),
}
