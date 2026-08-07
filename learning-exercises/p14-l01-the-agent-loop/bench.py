"""Входные данные для замера скорости."""


def _calc(expr):
    left, right = expr.split("+")
    return str(int(left) + int(right))


_registry = {"calc": _calc}
_long = "x" * 5000
_history = [{"kind": "user", "content": "2+3"}] + [
    {"kind": "action", "content": "calc"} for _ in range(400)
]
_calls = [{"tool_use_id": f"u{i}"} for i in range(400)]
_results = [{"tool_use_id": f"u{i}"} for i in reversed(range(400))]

BENCH = {
    "dispatch_tool": (_registry, "calc", {"expr": "2+3"}),
    "format_observation": ("calc", _long, 200),
    "flag_injection": (_long,),
    "stop_reason": ({"kind": "action", "action": "calc"}, 0, 8),
    "toy_llm": (_history,),
    "run_agent_loop": ("2+3", _registry),
    "tool_usage": (_history,),
    "correlate_results": (_calls, _results),
}
