"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_TRACE_ID = format(_rng.getrandbits(128), "032x")

_CHAT_ATTRS = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-sonnet",
    "gen_ai.response.model": "claude-sonnet-2026-02-01",
    "gen_ai.response.id": "resp_1",
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 45,
}


def _span(name, parent_id, start, end):
    return {
        "name": name,
        "kind": "CLIENT",
        "traceId": _TRACE_ID,
        "spanId": format(_rng.getrandbits(64), "016x"),
        "parentSpanId": parent_id,
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "attributes": dict(_CHAT_ATTRS),
        "events": [],
    }


_root = _span("agent.invoke_agent", None, 0, 10_000_000)
_root["kind"] = "INTERNAL"
_root["attributes"] = {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.agent.name": "bench-agent",
    "gen_ai.agent.id": "agent_bench",
}

# широкое дерево: наивная сборка "для каждого спана пройти весь список"
# заметно медленнее словаря по spanId
_flat = [_root] + [
    _span(f"llm.chat.{i}", _root["spanId"], 1 + i, 2 + i) for i in range(3000)
]

# finish_span в замер не входит: он одноразовый по замыслу (повторное
# закрытие спана — ValueError), а повторный прогон на одном входе — как раз
# то, чем меряют скорость.
BENCH = {
    "new_span": ("llm.chat", "CLIENT", 1000, random.Random(1), _root, _CHAT_ATTRS),
    "traceparent": (_root,),
    "parse_traceparent": (f"00-{_TRACE_ID}-{_root['spanId']}-01",),
    "missing_gen_ai_attributes": (_flat[1],),
    "capture_content_event": (_span("x", None, 0, 1), "gen_ai.content.prompt", "hi", 5, True),
    "span_tree": (_flat,),
    "trace_problems": (_flat,),
}
