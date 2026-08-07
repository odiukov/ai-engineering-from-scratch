"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_ids = [f"doc-{i:04d}" for i in range(3000)]
_text = {d: random.uniform(0, 40) for d in _ids}          # шкала BM25
_image = {d: random.uniform(-1, 1) for d in random.sample(_ids, 2000)}
_audio = {d: random.uniform(0, 1) for d in random.sample(_ids, 1500)}
_fused = {d: random.random() for d in _ids}
_ranked = sorted(_fused, key=_fused.get, reverse=True)
_relevant = set(random.sample(_ids, 20))
_top = [(d, _fused[d]) for d in _ranked[:20]]
_evidence = {
    d: (random.choice(("text", "image", "audio")), f"snippet {d}") for d in _ids
}

BENCH = {
    "min_max_normalize": (_text,),
    "score_fusion": ([_text, _image, _audio], [0.5, 0.3, 0.2]),
    "top_k": (_fused, 10),
    "moe_gate": ("quiet vegan brunch with natural light and a short menu",),
    "recall_at_k": (_ranked, _relevant, 100),
    "grounded_answer": (_top, _evidence),
    "needs_another_hop": (_fused,),
    "agentic_retrieve": (
        "quiet vegan brunch",
        lambda q: [_text, _image, _audio],
        lambda q: q + " under 40 dB",
        [0.5, 0.3, 0.2],
    ),
}
