"""Входные данные для замера скорости."""

_key = "sk-proj-A1b2C3d4E5f6G7h8I9j0KLMN"
_line = (
    "2026-08-07T10:00:00Z gateway=eu-1 user=bob@example.com ssn=123-45-6789 "
    "key=" + _key + " status=200 latency_ms=412 tokens=1834"
)
_log = " || ".join(_line for _ in range(200))

_table = {}
_records = [{"n": i, "user": "u%d" % (i % 17), "cost": i * 0.001} for i in range(2000)]

_chain = []
_prev = "0" * 64


def _build():
    import hashlib
    import json

    chain = []
    prev = "0" * 64
    for r in _records:
        payload = json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256((prev + "|" + payload).encode("utf-8")).hexdigest()
        chain.append({"record": r, "prev": prev, "hash": h})
        prev = h
    return chain


_chain = _build()

BENCH = {
    "find_secrets": (_log,),
    "placeholder_for": ("SSN", "123-45-6789", _table),
    "redact": (_log, {}),
    "is_allowed": ("engineer", "call_model"),
    "chain_hash": ("0" * 64, _records[0]),
    "append_audit": (_chain[:500], {"n": 999}),
    "verify_chain": (_chain,),
    "audit_llm_call": ([], {}, "2026-08-07T10:00:00Z", "u1", "engineer", "t1",
                       "claude", _line, "ok"),
}
