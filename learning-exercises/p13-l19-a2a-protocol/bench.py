"""Входные данные для замера скорости."""

_skills = [
    {
        "id": f"skill_{i:03d}",
        "name": f"Skill {i}",
        "inputModes": ["text", "file", "data"][: 1 + i % 3],
        "outputModes": ["text", "artifact"],
        "requiredData": ["targetLength"] if i % 2 else [],
    }
    for i in range(200)
]

_card = {
    "schemaVersion": "1.0",
    "name": "bench-agent",
    "description": "Benchmark card.",
    "url": "https://bench.example.com/a2a",
    "version": "1.0.0",
    "skills": _skills,
    "capabilities": {"streaming": True, "pushNotifications": False},
}

_signed = {"alg": "HS256", "card": _card, "signature": "0" * 64}

_chunks = [f"chunk-{i} " for i in range(2000)]

_messages = [
    {"role": "user", "parts": [{"kind": "text", "text": f"line {i}"}]}
    for i in range(300)
] + [{"role": "user", "parts": [{"kind": "data", "data": {"targetLength": "short"}}]}]

BENCH = {
    "build_agent_card": ("bench-agent", "d", "u", "1.0.0", _skills, None),
    "select_skill": (_card, ["text", "file"], "text"),
    "canonical_json": (_card,),
    "sign_agent_card": (_card, "s3cret"),
    "verify_agent_card": (_signed, "s3cret"),
    "next_task_state": ("working", "finish"),
    "make_artifact": ("summary", "text/markdown", _chunks),
    "run_task": ("bench_task", _card, "skill_001", _messages),
}
