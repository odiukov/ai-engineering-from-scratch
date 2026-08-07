"""Входные данные для замера скорости."""

import hashlib
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# много серверов по много инструментов: наивное «для каждого вызова пройти
# все серверы списком» заметно медленнее словаря
_servers = {
    f"server-{s:02d}": [
        {"name": f"tool_{s:02d}_{t:03d}", "description": f"Use when case {s}.{t}."}
        for t in range(60)
    ]
    for s in range(20)
}

_manifest = {
    f"{server}::{tool['name']}": hashlib.sha256(
        tool["description"].encode("utf-8")
    ).hexdigest()
    for server, tools in _servers.items()
    for tool in tools
}

# половина серверов объявляет одинаковые имена -> массовые коллизии
_colliding = {
    f"server-{s:02d}": [
        {"name": f"tool_{t:03d}", "description": f"desc {s}.{t}"} for t in range(60)
    ]
    for s in range(20)
}

_papers = [
    {"arxiv_id": f"26{i:05d}", "title": f"Paper number {i} on agent protocols"}
    for i in range(2000)
]

_task = {
    "id": "task_bench",
    "skillId": "summarize_papers",
    "state": "completed",
    "artifact": {
        "name": "summary",
        "mimeType": "text/markdown",
        "parts": [{"kind": "text", "text": "x" * 20000}],
    },
    "_internal": {"steps": tuple(f"step-{i}" for i in range(500))},
}

_TRACE_ID = format(random.getrandbits(128), "032x")

_spans = [
    {
        "name": "agent.invoke_agent",
        "kind": "INTERNAL",
        "traceId": _TRACE_ID,
        "spanId": "0" * 15 + "1",
        "parentSpanId": None,
        "startTimeUnixNano": 0,
        "endTimeUnixNano": 10_000_000,
        "attributes": {"gen_ai.operation.name": "invoke_agent"},
    }
] + [
    {
        "name": f"mcp.call.{i}",
        "kind": "CLIENT",
        "traceId": _TRACE_ID,
        "spanId": format(i + 2, "016x"),
        "parentSpanId": "0" * 15 + "1",
        "startTimeUnixNano": 1 + i,
        "endTimeUnixNano": 2 + i,
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": f"tool_{i}",
        },
    }
    for i in range(5000)
]

_users = {"tok_alice": {"id": "alice", "scopes": ("research:read", "research:write")}}
_scopes = {
    tool["name"]: "research:read" for tools in _servers.values() for tool in tools
}

# сценарий капстоуна поверх того же мира
_servers["research"] = [
    {"name": "arxiv_search", "description": "Use when the user searches arXiv."},
    {"name": "generate_report", "description": "Use when the user wants a report."},
]
for _tool in _servers["research"]:
    _manifest[f"research::{_tool['name']}"] = hashlib.sha256(
        _tool["description"].encode("utf-8")
    ).hexdigest()
_scopes["arxiv_search"] = "research:read"
_scopes["generate_report"] = "research:write"

_world = {
    "servers": _servers,
    "handlers": {name: (lambda args: {"ok": True}) for name in _scopes},
    "users": _users,
    "required_scopes": _scopes,
    "manifest": _manifest,
}


def _clock():
    return 1


_ctx = {"spans": [], "audit": [], "clock": _clock, "rng": random.Random(1)}
_run_ctx = {"spans": [], "audit": [], "clock": _clock, "rng": random.Random(2)}

BENCH = {
    "pin_manifest": (_servers,),
    "verify_pins": (_servers, _manifest),
    "merge_tools": (_colliding,),
    "authorize": (_world, "tok_alice", "tool_19_059"),
    "emit_span": (_ctx, "mcp.call", "CLIENT", _TRACE_ID, None, {"gen_ai.a": 1}),
    "delegate_task": ("task_bench", "summarize_papers", {"papers": _papers}),
    "opaque_result": (_task,),
    "gateway_call": (_world, _ctx, "tok_alice", "tool_19_059", {}, _TRACE_ID, None),
    "run_research": (_world, _run_ctx, "tok_alice", "agent protocol"),
    "trace_report": (_spans,),
}
