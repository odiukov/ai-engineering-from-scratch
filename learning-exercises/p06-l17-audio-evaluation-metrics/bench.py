"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_vocab = ["раз", "два", "три", "четыре", "пять", "шесть", "семь"]
# 400 слов — примерно три минуты речи, реальный размер одного файла в eval-наборе
_ref_words = [_rng.choice(_vocab) for _ in range(400)]
_hyp_words = [w if _rng.random() > 0.1 else _rng.choice(_vocab) for w in _ref_words]
_reference = " ".join(_ref_words)
_hypothesis = " ".join(_hyp_words)
_embedding_a = [_rng.gauss(0, 1) for _ in range(192)]
_embedding_b = [_rng.gauss(0, 1) for _ in range(192)]
_latencies = [_rng.lognormvariate(5.0, 0.6) for _ in range(50000)]
_real = [_rng.gauss(0.0, 1.0) for _ in range(50000)]
_generated = [_rng.gauss(0.3, 1.2) for _ in range(50000)]

BENCH = {
    "normalize_text": (_reference,),
    "edit_ops": (_ref_words, _hyp_words),
    "wer": (_reference, _hypothesis),
    "cer": (_reference[:400], _hypothesis[:400]),
    "cosine_similarity": (_embedding_a, _embedding_b),
    "percentile": (_latencies, 95),
    "der": (1.0, 2.0, 3.0, 100.0),
    "frechet_distance_1d": (_real, _generated),
}
