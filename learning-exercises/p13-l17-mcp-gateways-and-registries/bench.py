"""Входные данные для замера скорости."""

import hashlib
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_TOOLS = ("search", "read", "write", "list", "diff", "sync", "export", "render")

# 40 бэкендов, у многих имена инструментов совпадают — как в реальном слиянии.
_backends = {
    f"srv{i}": random.sample(_TOOLS, k=4) for i in range(40)
}

_tools = [
    {"server": s, "name": t, "description": f"Tool {t} of {s}"}
    for s, names in _backends.items()
    for t in names
]
_manifest = {
    f"{t['server']}::{t['name']}": hashlib.sha256(
        t["description"].encode("utf-8")
    ).hexdigest()
    for t in _tools
}

_policy = {f"u{i}": [f"srv{i}::*"] for i in range(500)}

_candidates = [
    {
        "name": f"io.github.dev{i}/notes",
        "source": random.choice(("official", "metaregistry", "unlisted")),
        "verified": i % 3 != 0,
    }
    for i in range(1000)
]

_gateway = {
    "routes": {"search": "srv0::search"},
    "policy": {"alice": ["srv0::*"]},
    "sessions": {"tok_alice": "alice"},
    "buckets": {"alice": {"tokens": 1000.0, "updated": 0}},
    "limit": {"capacity": 1000, "refill_per_second": 1000.0},
    "audit": [],
}

BENCH = {
    "merge_tool_namespaces": (_backends,),
    "rbac_allows": (_policy, "u499", "srv499", "search"),
    "pin_filter": (_tools, _manifest),
    "token_bucket_take": ({"tokens": 5.0, "updated": 0}, 1, 10, 0.5),
    "audit_event": ("alice", "search", "ok", 1000),
    "registry_rank": ("official",),
    "choose_server": (_candidates,),
    "handle_call": (_gateway, "tok_alice", "search", 1),
}
