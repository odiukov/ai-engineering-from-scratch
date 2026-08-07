"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_config = {
    "novelty_mislabel": 0.25,
    "experiment_failure": 0.42,
    "retry_recovery": 0.55,
    "polish_masks_weakness": 0.70,
    "writeup_success": 0.85,
    "internal_review_accept": 0.50,
}

_paper = {
    "claim": {"effect_observed": True},
    "novelty": "novel",
    "experiment": {"ok": True, "flawed": True, "retried": True},
    "masked": True,
}

_checks = {"experiment_reproduced": True, "novelty_verified": True,
           "human_signoff": False}

_outcomes = [
    {"submitted": i % 3 == 0, "stage": "" if i % 3 == 0 else "experiment",
     "paper": _paper, "clean": i % 6 == 0}
    for i in range(5000)
]

BENCH = {
    "novelty_check": (_rng, True, 0.25),
    "run_experiment": (_rng, 0.42, 0.55),
    "polish_figures": (_rng, _paper, 0.7),
    "supports_conclusion": (_paper,),
    "review": (_paper, False),
    "release_gate": (_paper, _checks),
    "run_loop": (_rng, _config),
    "summarize": (_outcomes,),
}
