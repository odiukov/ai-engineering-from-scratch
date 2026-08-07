"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# «Флот» моделей: типичный вход отчёта — не одна модель, а весь список
_FLEET = [
    {
        "rd_automation_share": _rng.uniform(0.0, 0.9),
        "metr_horizon_hours": _rng.uniform(0.5, 60.0),
        "cyber_uplift": _rng.uniform(0.0, 0.8),
    }
    for _ in range(2000)
]

_IN_PLACE = ["model_card", "usage_policy", "weights_security"]

_BIG_OLD = {f"commitment_{i}": ("unilateral" if i % 2 else "industry")
            for i in range(3000)}
_BIG_NEW = {f"commitment_{i}": ("industry" if i % 3 else "unilateral")
            for i in range(1500, 4500)}

_SATISFIED = ["declared_cadence", "published_risk_reports",
              "frontier_safety_roadmap"] * 300

BENCH = {
    "capability_level": (_FLEET[0],),
    "required_safeguards": ("ASL-4",),
    "missing_safeguards": ("ASL-4", _IN_PLACE),
    "deployment_decision": (_FLEET[0], _IN_PLACE),
    "unilateral_commitments": (_BIG_OLD,),
    "diff_policies": (_BIG_OLD, _BIG_NEW),
    "affirmative_case_sections": ("ASL-4", 0.28),
    "policy_score": (_SATISFIED,),
}
