"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_TOPICS = [
    "What is the return policy?",
    "What are your store hours?",
    "Analyze the trade-offs of a sharded write path",
    "Summarize this quarterly earnings report for the board",
    "Write code for a binary search tree with deletion",
]
_QUERIES = [random.choice(_TOPICS) for _ in range(200)]

_MESSAGES = [{"role": "user", "content": "What is the return policy?" * 20}]
_CACHE = {"entries": {}, "hits": 0, "misses": 0, "ttl": 3600.0, "max_size": 512}
_BUCKET = {"tokens": 1e9, "capacity": 1e9, "refill_rate": 1e6, "last_refill": 0.0}

_LOGS = [
    {
        "query": q,
        "model": random.choice(["gpt-4o", "gpt-4o-mini", "claude-sonnet-4"]),
        "complexity": "medium",
        "response": "...",
        "cache_status": random.choice(["hit", "miss"]),
        "input_tokens": 500,
        "output_tokens": 150,
        "cost": random.random() / 1000,
        "saved_cost": random.random() / 1000,
    }
    for q in _QUERIES
]

BENCH = {
    "calculate_cost": ("gpt-4o", 1000, 500, 800),
    "cache_key": ("gpt-4o", _MESSAGES, 0.0),
    "cache_lookup": (_CACHE, "gpt-4o", _MESSAGES, 0.0, 0.0),
    "cache_store": (_CACHE, "gpt-4o", _MESSAGES, 0.0, "cached answer", 0.0),
    "route_model": ("Analyze the trade-offs between microservices and monoliths", "pro"),
    "token_bucket_take": (_BUCKET, 100, 1.0),
    "serve_query": (_CACHE, _TOPICS[0], "pro", 0.0),
    "summarize_usage": (_LOGS,),
}
