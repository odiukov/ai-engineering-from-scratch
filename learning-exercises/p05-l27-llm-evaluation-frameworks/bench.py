"""Входные данные для замера скорости."""

import random
import re

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["apple", "iphone", "released", "june", "2007", "cupertino", "android",
          "launched", "moon", "landing", "1969", "google", "pixel", "camera"]


def _sentence():
    return " ".join(random.choice(_WORDS) for _ in range(8)) + "."


_ANSWER = " ".join(_sentence() for _ in range(200))
_CONTEXT = " ".join(_sentence() for _ in range(200))
_CHUNKS = [_sentence() for _ in range(200)]
_RELEVANT = _CHUNKS[:40]
_GOLD_CLAIMS = [_sentence() for _ in range(200)]


def _judge(claim, context):
    tokens = re.findall(r"[a-z0-9]+", claim.lower())
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in ctx) / len(tokens)


def _similarity(a, b):
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _question_generator(answer):
    return [_sentence() for _ in range(3)]


_RAW = ['{"score": %s}' % round(random.random(), 3) for _ in range(2000)]
_SCORES = [random.random() if random.random() > 0.05 else None for _ in range(5000)]
_JUDGE_SCORES = [random.random() for _ in range(2000)]
_HUMAN_SCORES = [random.random() for _ in range(2000)]

BENCH = {
    "split_claims": (_ANSWER,),
    "faithfulness": (_ANSWER, _CONTEXT, _judge),
    "answer_relevance": ("when was it released", _ANSWER, _question_generator, _similarity),
    "context_precision": (_CHUNKS, _RELEVANT),
    "context_recall": (_GOLD_CLAIMS, _CHUNKS, _judge),
    "parse_judge_score": (_RAW[0],),
    "aggregate_scores": (_SCORES,),
    "spearman_rho": (_JUDGE_SCORES, _HUMAN_SCORES),
}
