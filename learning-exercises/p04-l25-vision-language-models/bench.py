"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_VIT_DIM = 64
_HIDDEN = 96
_LLM_DIM = 48
_N_TOKENS = 256

_tokens = [[_rng.uniform(-1.0, 1.0) for _ in range(_VIT_DIM)] for _ in range(_N_TOKENS)]
_w1 = [[_rng.uniform(-0.1, 0.1) for _ in range(_VIT_DIM)] for _ in range(_HIDDEN)]
_b1 = [0.0] * _HIDDEN
_w2 = [[_rng.uniform(-0.1, 0.1) for _ in range(_HIDDEN)] for _ in range(_LLM_DIM)]
_b2 = [0.0] * _LLM_DIM

_levels = [
    [[_rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(_N_TOKENS)]
    for _ in range(4)
]

_img_embs = [[_rng.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(2000)]
_txt_embs = [[_rng.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(2000)]
_confs = [_rng.uniform(0.0, 1.0) for _ in range(2000)]

_text_embeds = [[0.0] * 8 for _ in range(4096)]
_input_ids = [99 if i % 4 == 0 else 7 for i in range(4096)]
_vision_embeds = [[1.0] * 8 for _ in range(1024)]

BENCH = {
    "gelu": (0.7,),
    "linear": (_tokens, _w1, _b1),
    "projector_forward": (_tokens, _w1, _b1, _w2, _b2),
    "count_projector_params": (768, 4096, 4096),
    "deepstack_concat": (_levels,),
    "cosine_similarity": (_img_embs[0], _txt_embs[0]),
    "cross_modal_error_rate": (_img_embs, _txt_embs, _confs),
    "merge_image_tokens": (_text_embeds, _vision_embeds, _input_ids, 99),
}
