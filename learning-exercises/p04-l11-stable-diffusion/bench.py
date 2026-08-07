"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_latent = [_rng.gauss(0.0, 5.5) for _ in range(4 * 64 * 64)]
_eps_uncond = [_rng.gauss(0.0, 1.0) for _ in range(4 * 64 * 64)]
_eps_cond = [_rng.gauss(0.0, 1.0) for _ in range(4 * 64 * 64)]
_mask = [_rng.random() for _ in range(4 * 64 * 64)]

_query = [_rng.gauss(0.0, 1.0) for _ in range(64)]
_keys = [[_rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(77)]
_values = [[_rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(77)]
_scores = [_rng.gauss(0.0, 3.0) for _ in range(5000)]

_W = [[_rng.gauss(0.0, 1.0) for _ in range(128)] for _ in range(128)]
_A = [[_rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(128)]
_B = [[_rng.gauss(0.0, 1.0) for _ in range(128)] for _ in range(8)]

BENCH = {
    "latent_compression_factor": ((3, 512, 512), (4, 64, 64)),
    "scale_latents": (_latent, 0.18215),
    "softmax": (_scores,),
    "cross_attention": (_query, _keys, _values),
    "classifier_free_guidance": (_eps_uncond, _eps_cond, 7.5),
    "img2img_timesteps": (50, 0.6),
    "inpaint_blend": (_eps_cond, _latent, _mask),
    "lora_update": (_W, _A, _B, 0.8),
}
