"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# Широкая ось сравнения: политика, разобранная на тысячи возможностей —
# ровно тот случай, когда наивный перебор «для каждой оси пройди все
# политики целиком» становится заметно медленнее.
_WIDE_AXES = tuple(f"cap_{i}" for i in range(4000))

_WIDE_POLICY = {
    "name": "wide policy",
    "table": {
        name: ("Tracked" if i % 3 else "Research", "synthetic action")
        for i, name in enumerate(_WIDE_AXES)
        if i % 5  # каждая пятая ось не покрыта: пробелы должны считаться тоже
    },
    "gating": frozenset({"Tracked"}),
    "artifacts": ("capabilities_report", "safeguards_report", "sag_review"),
}

_OTHER_POLICY = {
    "name": "other policy",
    "table": {
        name: ("CCL" if i % 2 else "monitoring", "synthetic action")
        for i, name in enumerate(_WIDE_AXES)
    },
    "gating": frozenset({"CCL"}),
    "artifacts": ("fsf_risk_report",),
}

_WIDE_POLICIES = (_WIDE_POLICY, _OTHER_POLICY)

# Много доменов с порогами — вход отчёта по флоту, а не по одной модели.
_MANY_DOMAINS = {
    f"domain_{i}": {
        "at_least": {f"metric_{i}": _rng.uniform(0.0, 1.0)},
        "at_most": {},
    }
    for i in range(3000)
}
_MANY_MEASUREMENTS = {f"metric_{i}": _rng.uniform(0.0, 1.0) for i in range(3000)}

# Имён метрик в реестре всего четыре — поправка по определению короткая,
# и замер здесь ловит не сложность, а лишние аллокации на каждый ключ.
_REAL_MEASUREMENTS = {
    "rnd_pipeline_automation_share": _rng.uniform(0.0, 1.2),
    "cyber_uplift": _rng.uniform(0.0, 1.0),
    "bio_uplift": _rng.uniform(0.0, 1.0),
    "cost_ratio_vs_human": _rng.uniform(0.3, 3.0),
}

BENCH = {
    "classify": (_WIDE_POLICY, "cap_1", _WIDE_AXES),
    "is_gated": (_WIDE_POLICY, "cap_1", _WIDE_AXES),
    "required_artifacts": (_WIDE_POLICY, "cap_1", _WIDE_AXES),
    "compare": ("cap_1", _WIDE_POLICIES, _WIDE_AXES),
    "coverage_report": (_WIDE_POLICY, _WIDE_AXES),
    "gating_divergence": (_WIDE_POLICIES, _WIDE_AXES),
    "ccl_reached": (_MANY_MEASUREMENTS, _MANY_DOMAINS),
    "sandbagging_correction": (_REAL_MEASUREMENTS, 0.3),
}
