"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SEQ = 4000
_xs = [random.gauss(0.0, 1.0) for _ in range(_SEQ)]


def _plan(num_layers, attn_ratio=8, moe_every=2):
    """Копия рецепта Jamba: bench не должен зависеть от твоего exercise.py."""
    out = []
    for i in range(num_layers):
        is_attention = attn_ratio > 0 and (i + 1) % attn_ratio == 0
        out.append(("attention" if is_attention else "mamba", (i + 1) % moe_every == 0))
    return out


_jamba = _plan(32)

BENCH = {
    "ssm_step": (0.5, 1.0, 0.9, 0.6),
    "ssm_scan": (_xs, 0.9, 0.6, 1.2),
    "layer_plan": (32, 8, 2),
    "count_layer_types": (_jamba,),
    "kv_cache_bytes": (_jamba, 32, 128, 262144),
    "ssm_state_bytes": (_jamba, 4096),
    "inference_memory": (_jamba, 4096, 32, 128, 262144),
    "kv_cache_advantage": (32, 8, 32, 128, 262144),
}
