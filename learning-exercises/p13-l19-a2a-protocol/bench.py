"""Входные данные для замера скорости."""

_skills = [
    {
        "id": f"skill_{i:03d}", "name": f"Skill {i}", "description": "d", "tags": ["bench"],
        "inputModes": ["text/plain", "application/pdf", "application/json"][: 1 + i % 3],
        "outputModes": ["text/plain", "text/markdown"], "requiredData": ["targetLength"] if i % 2 else [],
    }
    for i in range(200)
]
_card = {
    "name": "bench-agent", "description": "Benchmark card.",
    "supportedInterfaces": [{"url": "https://bench.example/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    "version": "1.0.0", "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"], "skills": _skills,
}
_signed = dict(_card)
_signed["signatures"] = [{"protected": "x", "signature": "y"}]
_chunks = [f"chunk-{i} " for i in range(2000)]
_messages = [
    {"messageId": f"msg-{i}", "role": "ROLE_USER", "parts": [{"text": f"line {i}"}]}
    for i in range(300)
] + [{"messageId": "msg-data", "role": "ROLE_USER", "parts": [{"data": {"targetLength": "short"}}]}]

BENCH = {
    "build_agent_card": ("bench-agent", "d", "u", "1.0.0", _skills, None),
    "select_skill": (_card, ["text/plain", "application/pdf"], "text/markdown"),
    "canonical_json": (_card,), "sign_agent_card": (_card, "s3cret"),
    "verify_agent_card": (_signed, "s3cret"),
    "next_task_state": ("TASK_STATE_WORKING", "finish"),
    "make_artifact": ("art-bench", "summary", "text/markdown", _chunks),
    "run_task": ("bench_task", _card, "skill_001", _messages),
}
