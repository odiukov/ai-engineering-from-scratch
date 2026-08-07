"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_UI = "https://ui.example.com"

_csp = {
    "defaultSrc": "'self'",
    "scriptSrc": "'self' 'unsafe-inline'",
    "connectSrc": "'self'",
    "styleSrc": "'self'",
    "imgSrc": "'self' data:",
    "frameAncestors": "'none'",
}

_html = "<!doctype html><svg>" + "<circle r='1'/>" * 500 + "</svg>"

_permissions = [random.choice(("camera", "microphone", "geolocation", "gpu"))
                for _ in range(200)]

_message = {"jsonrpc": "2.0", "id": 1, "method": "host.callTool", "params": {"n": 1}}

BENCH = {
    "is_ui_uri": ("ui://notes/timeline",),
    "ui_resource_contents": ("ui://notes/timeline", _html),
    "csp_header": (_csp,),
    "csp_findings": (_csp,),
    "review_permissions": (_permissions,),
    "tool_result_with_ui": ("Вот таймлайн:", "ui://notes/timeline", _csp, ["camera"]),
    "accept_message": (_UI, _UI, _message),
    "dispatch_host_call": (_message, _UI, _UI, {"host.callTool": lambda p: p}),
}
