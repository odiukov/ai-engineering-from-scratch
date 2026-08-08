"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_DEPLOYMENT = {
    "public_facing": True,
    "harmful_capability_labels": ("cyber", "cbrn", "disinformation"),
    "competitive_pressure": True,
    "independent_audit": False,
    "multi_layer_defense": False,
    "information_security": False,
    "agent_autonomy_hours": 48.0,
    "training_compute_ops": 10**27,
    "annual_gross_revenue_usd": 1_000_000_000.0,
}

# Портфель показателей: типичный вход агрегата — не четыре подрычага одного
# сервиса, а весь реестр рисков организации.
_MANY_SCORES = {f"indicator_{i}": _rng.uniform(0.0, 1.0) for i in range(5000)}
_MANY_WEIGHTS = {name: _rng.uniform(0.1, 5.0) for name in _MANY_SCORES}

_FULL_STACK = {
    "lab_scaling_policy": 0.9,
    "external_evaluation": 0.8,
    "civil_society_tracking": 0.7,
    "government_baseline": 0.6,
    "practitioner_controls": 0.85,
}

BENCH = {
    "tag_risks": (_DEPLOYMENT,),
    "mitigation_checklist": (_DEPLOYMENT,),
    "aggregate_risk": (_MANY_SCORES, _MANY_WEIGHTS),
    "stack_assessment": (_FULL_STACK,),
    "identify_organization": ("https://www.nist.gov/caisi",),
    "sb53_obligations": (_DEPLOYMENT,),
    "incident_report_status": (100.0, 110.0),
}
