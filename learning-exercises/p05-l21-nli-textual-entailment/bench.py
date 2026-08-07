"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["cat", "dog", "sleeping", "couch", "the", "a", "is", "on", "park",
          "ball", "chased", "no", "quietly", "loudly", "man", "woman"]

# длинный premise: наивная проверка `t in list(premise_tokens)` на нём
# превращается в квадрат, вариант через set остаётся линейным
_PREMISE = " ".join(random.choice(_WORDS) for _ in range(4000)) + "."
_HYPOTHESIS = " ".join(random.choice(_WORDS) for _ in range(60)) + "."
_ANSWER = " ".join(
    " ".join(random.choice(_WORDS) for _ in range(12)) + "." for _ in range(40)
)
_LABELS = ["finance", "sports", "politics", "technology", "health", "science"]

BENCH = {
    "tokenize": (_PREMISE,),
    "has_negation": (_PREMISE.split(),),
    "lexical_overlap": (_PREMISE, _HYPOTHESIS),
    "softmax": ({"entailment": 1.5, "contradiction": -2.5, "neutral": 0.0},),
    "nli_scores": (_PREMISE, _HYPOTHESIS),
    "hypothesis_only_label": (_HYPOTHESIS,),
    "zero_shot_classify": (_PREMISE, _LABELS),
    "is_faithful": (_ANSWER, _PREMISE),
}
