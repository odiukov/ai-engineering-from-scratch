"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 32
_PATCHES = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(576)]
_LATENTS = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(64)]
_HIDDEN = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(128)]
_VISUAL = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(64)]
_SEQ = ["image" if i % 9 == 0 else "text" for i in range(180)]
_EXAMPLES = [(f"img_{i}.jpg", f"caption number {i}") for i in range(64)]

BENCH = {
    "cross_attention": (_LATENTS, _PATCHES, _PATCHES),
    "perceiver_resampler": (_PATCHES, _LATENTS, 6),
    "gated_residual": (_HIDDEN, _HIDDEN, 0.5),
    "gated_cross_attention_step": (_HIDDEN, _VISUAL, 0.5),
    "most_recent_image": (_SEQ,),
    "interleaved_cross_mask": (_SEQ, 64),
    "build_few_shot_prompt": (_EXAMPLES, "query.jpg"),
}
