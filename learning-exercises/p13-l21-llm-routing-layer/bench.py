"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_words = ["mcp", "a2a", "otel", "router", "gateway", "cache", "budget"]

_long_text = " ".join(random.choice(_words) for _ in range(4000))
_long_text += " ssn 123-45-6789 card 1234567890123456 mail dev@example.com"

_messages = [
    {"role": "user", "content": " ".join(random.choice(_words) for _ in range(200))}
    for _ in range(50)
]


def _provider(model, messages):
    return {
        "status": 200,
        "model": model,
        "usage": {"input_tokens": 1000, "output_tokens": 500},
        "content": "ok",
    }


_invocations = [
    {
        "alias": "smart",
        "model": "openai/gpt-4o" if i % 2 else "anthropic/claude-haiku",
        "cached": i % 5 == 0,
        "input_tokens": 1000,
        "output_tokens": 500,
        "cost_usd": 0.0125,
    }
    for i in range(20000)
]

_candidates = list(
    ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-sonnet",
     "anthropic/claude-haiku", "google/gemini-pro"]
) * 4000

BENCH = {
    "redact_pii": (_long_text,),
    "estimate_cost": ("openai/gpt-4o", 1000, 1000),
    "cheapest_model": (_candidates, 0.80, 1200),
    "cache_key": ("smart", _messages),
    "resolve_chain": ("smart", None),
    "route": ("smart", _messages, _provider),
    "charge": ({"search": 1.0}, "search", 0.01, 100.0),
    "spend_report": (_invocations,),
}
