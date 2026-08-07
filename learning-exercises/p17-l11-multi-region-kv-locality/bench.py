"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_regions = ("us-east-1", "us-west-2", "eu-west-1")
_replicas = [
    {"name": f"{r}-{i}", "region": r} for r in _regions for i in range(4)
]
_system_prompt = list(range(600))
_requests = [
    {
        "origin": random.choice(_regions),
        "tokens": _system_prompt + [random.randrange(40)],
    }
    for _ in range(4000)
]
_ttfts = [random.choice((80.0, 155.0, 800.0)) for _ in range(20000)]

BENCH = {
    "prefix_key": (_system_prompt, 512),
    "rtt_ms": ("us-east-1", "eu-west-1"),
    "expected_ttft_ms": (True, "us-east-1", "eu-west-1"),
    "route_round_robin": (7, _replicas),
    "route_cache_aware": ("abc", "us-east-1", _replicas, [[] for _ in _replicas]),
    "percentile": (_ttfts, 99),
    "simulate": (_requests, _replicas, "cache_aware"),
    "dr_manifest_gaps": (["model.safetensors", "config.json"],),
}
