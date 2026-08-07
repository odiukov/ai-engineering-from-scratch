"""Входные данные для замера скорости."""

_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_REMOTE_SPAN = "00f067aa0ba902b7"
_HEADER = f"00-{_TRACE_ID}-{_REMOTE_SPAN}-01"

# Плоский трейс на 800 спанов: агентский корень и много tool-спанов под ним.
# Наивный span_tree с поиском родителя линейным проходом станет квадратичным.
_deep_trace = {
    "trace_id": _TRACE_ID,
    "remote_parent": None,
    "stack": [],
    "spans": [
        {
            "trace_id": _TRACE_ID,
            "span_id": f"{i:016x}",
            "parent_id": None if i == 0 else f"{0:016x}",
            "name": f"tool_call t{i}",
            "kind": "INTERNAL",
            "attributes": {},
            "start_ns": i * 10,
            "end_ns": i * 10 + 5,
            "duration_ns": 5,
        }
        for i in range(800)
    ],
}

_open_trace = {"trace_id": _TRACE_ID, "remote_parent": None, "stack": [], "spans": []}
_bench_span = {"attributes": {}}

BENCH = {
    "describe_span": ("invoke_agent", "planner"),
    "genai_attributes": ("anthropic", "chat", "claude-x", "claude-x-0301", "planner", "kb-1"),
    "format_traceparent": (_TRACE_ID, _REMOTE_SPAN),
    "continue_trace": (_HEADER,),
    "start_span": (_open_trace, f"{999999:016x}", "tool_call s", "INTERNAL", {}, 0),
    "end_span": (
        {
            "trace_id": _TRACE_ID,
            "remote_parent": None,
            "stack": ["a" * 16],
            "spans": [{"span_id": "a" * 16, "start_ns": 0, "end_ns": None}],
        },
        "a" * 16,
        100,
    ),
    "span_tree": (_deep_trace,),
    "capture_content": ({}, _bench_span, ["a message"] * 20, "reference"),
}
