"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"word{i}" for i in range(120)]
_REFERENCE = " ".join(random.choice(_WORDS) for _ in range(180))
_HYPOTHESIS = " ".join(random.choice(_WORDS) for _ in range(180))

_REF_TOKENS = _REFERENCE.split()
_HYP_TOKENS = _HYPOTHESIS.split()

_SCORES = [random.randint(1, 5) for _ in range(200)]
_CRITERIA = ["relevance", "correctness", "helpfulness", "safety"]
_BASELINE = {c: [random.randint(1, 5) for _ in range(200)] for c in _CRITERIA}
_NEW = {c: [random.randint(1, 5) for _ in range(200)] for c in _CRITERIA}

BENCH = {
    "normalize_tokens": (_REFERENCE,),
    "lcs_length": (_REF_TOKENS, _HYP_TOKENS),
    "rouge_l": (_REFERENCE, _HYPOTHESIS),
    "jaccard_overlap": (_REFERENCE, _HYPOTHESIS),
    "wilson_interval": (180, 200, 1.96),
    "bootstrap_interval": (_SCORES, 0, 200, 0.95),
    "compare_runs": (_BASELINE, _NEW, 0.3),
}
