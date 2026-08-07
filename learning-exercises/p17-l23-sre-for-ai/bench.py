"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_vocab = ["cache", "oom", "spike", "gateway", "tenant", "deploy", "latency",
          "pool", "vllm", "gpu", "kv_cache", "burst", "concurrency", "timeout"]

_symptom = " ".join(random.choice(_vocab) for _ in range(200))

_runbooks = {
    f"RB-{i:03d}": {
        "symptom": " ".join(random.choice(_vocab) for _ in range(40)),
        "action": "restart_pod",
    }
    for i in range(4000)
}

_hypotheses = [
    {
        "agent": f"agent-{i}",
        "root_cause": " ".join(random.choice(_vocab) for _ in range(12)),
        "confidence": random.random(),
    }
    for i in range(5000)
]

_agents = [
    (lambda h: (lambda incident: h))(dict(_hypotheses[i]))
    for i in range(200)
]

_outages = [(i * 137.0, 45.0) for i in range(20000)]

_budget = {"allowed": 43.2, "spent": 5.0, "remaining": 38.2,
           "consumed_fraction": 0.1157, "burn_rate": 0.1157}

BENCH = {
    "normalize_cause": (_symptom,),
    "retrieve_runbook": ("kv_cache oom spike under burst concurrency", _runbooks),
    "adversarial_review": (_hypotheses,),
    "is_safe_action": ("restart_pod",),
    "triage": ("high error rate in checkout", _agents, _runbooks),
    "bad_minutes": (_outages,),
    "error_budget": (0.999, 30 * 24 * 3600, _outages),
    "release_decision": (_budget, 1.0, 1.0),
}
