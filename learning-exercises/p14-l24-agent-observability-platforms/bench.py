"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_RUBRIC = {
    "has_final_answer": 1.0,
    "no_tool_errors": 1.0,
    "within_step_budget": 1.0,
    "no_pii_in_output": 1.0,
}


def _span(session_id, name, start_ms, end_ms, status="ok", output=None):
    return {
        "session_id": session_id,
        "name": name,
        "start_ns": int(start_ms * 1e6),
        "end_ns": int(end_ms * 1e6),
        "status": status,
        "output": output,
    }


# 400 сессий по 6 спанов, порядок поступления перемешан — как из сети.
_spans = []
for _i in range(400):
    _sid = f"s{_i:04d}"
    _end = random.uniform(5.0, 4000.0)
    _spans.append(_span(_sid, "invoke_agent", 0.0, _end))
    for _k in range(4):
        _spans.append(
            _span(
                _sid,
                f"tool_call t{_k}",
                1.0 + _k,
                2.0 + _k,
                status="error" if _i % 7 == 0 and _k == 0 else "ok",
            )
        )
    _spans.append(_span(_sid, "final_answer", _end - 1.0, _end, output=f"done {_i} of 400"))
random.shuffle(_spans)

_sessions = {}
for _s in _spans:
    _sessions.setdefault(_s["session_id"], []).append(_s)
for _key in _sessions:
    _sessions[_key].sort(key=lambda s: (s["start_ns"], s["name"]))
_sessions = {_key: _sessions[_key] for _key in sorted(_sessions)}

_one_session = _sessions["s0000"]
_latencies = [random.uniform(5.0, 4000.0) for _ in range(5000)]
_text = "mail a.b@x.io and card 4111 1111 1111 1111, step 3 of 10. " * 100

BENCH = {
    "ingest_spans": (_spans,),
    "session_latency_ms": (_one_session,),
    "latency_percentile": (_latencies, 95),
    "redact_pii": (_text,),
    "judge_session": (_one_session, _RUBRIC),
    "categorize_failures": (_sessions, _RUBRIC),
    "summarize": (_sessions, _RUBRIC, 1000.0),
    "worst_session": (_sessions, _RUBRIC),
}
