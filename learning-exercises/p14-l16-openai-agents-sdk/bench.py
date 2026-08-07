"""Входные данные для замера скорости."""

_N = 200


def _issue_refund(amount):
    return f"refunded {amount}"


def _final(user_input):
    return {"kind": "final", "text": f"handled: {user_input}"}


def _relay(user_input):
    """Каждый агент передаёт запрос следующему: длинная цепочка хендоффов."""
    return {"kind": "handoff", "to": "hop1"}


_TOOLS = tuple(
    {
        "name": f"tool{i}",
        "description": f"Tool {i}. Use when topic{i} comes up.",
        "fn": _issue_refund,
    }
    for i in range(_N)
)

_LEAF = {
    "name": "leaf",
    "instructions": "answer directly",
    "policy": _final,
    "tools": _TOOLS,
    "handoffs": (),
}

_HOPS = [_LEAF]
for _index in range(1, 12):
    _HOPS.append(
        {
            "name": f"hop{_index}",
            "instructions": f"relay {_index}",
            "policy": _final if _index == 1 else _relay,
            "tools": (),
            "handoffs": (_HOPS[-1],),
        }
    )

_ROOT = {
    "name": "root",
    "instructions": "route everything down the chain",
    "policy": lambda user_input: {"kind": "handoff", "to": _HOPS[-1]["name"]},
    "tools": _TOOLS,
    "handoffs": (_HOPS[-1],),
}


def _pass(text):
    return (True, "ok")


_GUARDRAILS = tuple((f"check{i}", _pass) for i in range(_N))

_SESSION = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(_N)]

_SPANS = [
    {"name": f"llm.agent{i}", "attributes": {"output": f"secret {i}", "passed": True}}
    for i in range(_N)
]

BENCH = {
    "handoff_tool_name": ("Billing Agent " * 20,),
    "make_agent": ("root", "route everything", _final, _TOOLS, (_LEAF,)),
    "visible_tools": (_ROOT,),
    "run_guardrails": (_GUARDRAILS, "a fairly long user request " * 20, "input"),
    "session_prompt": (_SESSION, "now"),
    "run_turn": (_ROOT, "please help", 20),
    "run_guarded": (_ROOT, "please help", _GUARDRAILS, _GUARDRAILS, 20),
    "redact_spans": (_SPANS,),
}
