"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_PAYLOAD = {f"field_{i}": random.randint(0, 10_000) for i in range(200)}

_PROPOSAL = {
    "thread_id": "t-bench",
    "action": "db.update",
    "payload": _PAYLOAD,
    "intent": "bulk status refresh from the nightly scan",
    "lineage": "stale-issue dashboard, run 2026-08-07",
    "blast_radius": "200 rows; reversible within the 1h backup window",
    "rollback": "restore the affected rows from the nightly backup",
}

# Записи заранее приведены в нужный статус, чтобы замер не зависел от порядка
# вызовов: и эталон, и решение учащегося получают одинаковые входы.
_PENDING = {
    "key": "0" * 16,
    "status": "pending",
    "created_at": 0.0,
    "expires_at": 900.0,
    "ack": None,
    **_PROPOSAL,
}
_COMMITTED = dict(_PENDING, status="committed")

_APPROVE_STORE = {"k": dict(_PENDING, status="approved")}
_COMMIT_STORE = {"k": _COMMITTED}
_VERIFY_STORE = {"k": dict(_COMMITTED)}

_ANSWERS = {
    "understood_resource": True,
    "verified_blast_radius": True,
    "rollback_ready": True,
}

BENCH = {
    "idempotency_key": ("t-bench", "db.update", _PAYLOAD),
    "missing_metadata": (_PROPOSAL,),
    "is_expired": (_PENDING, 100.0),
    "propose": ({"already": 1}, _PROPOSAL, 0.0),
    "approve": (_APPROVE_STORE, "k", _ANSWERS, 1.0),
    "commit": (_COMMIT_STORE, "k", lambda key, action, payload: None, 2.0),
    "verify": (_VERIFY_STORE, "k", lambda action, payload: True),
}
