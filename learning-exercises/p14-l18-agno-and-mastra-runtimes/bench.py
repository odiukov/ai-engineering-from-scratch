"""Входные данные для замера скорости."""

_router = {
    "models": {f"p{i % 23}/model-{i}": f"p{i % 23}" for i in range(500)},
    "aliases": {f"alias-{i}": f"p{i % 23}/model-{i}" for i in range(500)},
}

_tool = {
    "name": "search",
    "schema": {"query": str, "limit": int},
    "handler": lambda p: f"{p['query'].upper()}x{p['limit']}",
}

_factory = lambda: {"instructions": "be brief", "tools": ["search"]}

_steps = [(f"s{i}", lambda x: x + 1) for i in range(500)]

_records = [
    {"kind": ("memory", "workflows", "observability")[i % 3], "data": i} for i in range(3000)
]
_routing = {"memory": "postgres", "workflows": "postgres", "observability": "clickhouse"}

BENCH = {
    "stub_model": ("a long prompt " * 200, "anthropic/claude-x"),
    "route_model": (_router, "alias-499"),
    "typed_tool_call": (_tool, {"query": "ai", "limit": 3}),
    "handle_request": ({}, "s1", "hi", _factory, _router, "alias-1"),
    "run_workflow": (_steps, 0),
    "split_records": (_records, _routing),
    "estimate_runtime_cost": (100000, 800.0),
    "pick_runtime": ({"language": "python", "needs_durable_graph_state": False},),
}
