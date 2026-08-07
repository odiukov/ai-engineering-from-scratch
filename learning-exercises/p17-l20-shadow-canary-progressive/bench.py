"""Входные данные для замера скорости."""

_users = [f"user-{i}" for i in range(30000)]

_baseline = {
    "latency_p99_ms": 900.0,
    "cost_per_req": 0.02,
    "error_rate": 0.02,
    "output_len_p99": 450.0,
    "thumbs_down_rate": 0.03,
}
_gates = {
    "latency_p99_ms": 1.5,
    "cost_per_req": 1.2,
    "error_rate": 2.0,
    "output_len_p99": 1.4,
    "thumbs_down_rate": 1.5,
}
_healthy = dict(_baseline)


def _base_provider(request):
    return {"text": "prod answer", "tokens": 120}


def _candidate_provider(request):
    return {"text": "candidate answer", "tokens": 170}


def _measure(share):
    return _healthy


BENCH = {
    "bucket_of": ("user-12345",),
    "assign_variant": ("user-12345", 0.25),
    "split_traffic": (_users, 0.25),
    "shadow_call": ({"q": "hi"}, _base_provider, _candidate_provider),
    "gate_breaches": (_healthy, _baseline, _gates),
    "widen_gates_for_noise": (_gates, 0.07),
    "run_canary": (_measure,),
    "rollback_policy": ({"canary_share": 0.5, "model_digest": "sha256:new"}, "sha256:old"),
}
