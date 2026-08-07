"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_keyring = {
    f"sk-app-{i}": {
        "tenant": f"team-{i % 50}",
        "tier": ("free", "trial", "paid")[i % 3],
        "secret_ref": "vault:openai_prod",
        "active": True,
    }
    for i in range(5000)
}
_vault = {"openai_prod": "sk-openai-real"}

_events = tuple(sorted(random.uniform(0.0, 600.0) for _ in range(20000)))

_state = {
    "keyring": _keyring,
    "vault": _vault,
    "buckets": {},
    "quota_used": {},
}


def _provider(model, request):
    return {"status": 200, "usage": {"input_tokens": 500, "output_tokens": 200}}


BENCH = {
    "resolve_key": ("sk-app-7", _keyring, _vault),
    "token_bucket": ({"tokens": 3.0, "at": 0.0}, 1.0, 1, 100, 50.0),
    "sliding_window": (_events, 610.0, 60.0, 10000),
    "backoff_delays": (200000,),
    "call_with_fallback": (("openai/gpt-4o", "anthropic/claude"), {}, _provider),
    "handle_request": ("sk-app-7", {"estimated_tokens": 700}, 0.0, _state, _provider, ("openai/gpt-4o",)),
    "latency_budget": ("Kong", 300, 400),
    "pick_gateway": (100, 300, 500, True, False),
}
