"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_D = 128  # ширина «головы» игрушечной модели
_FF = 256  # ширина MLP

_vec = [random.gauss(0.0, 1.0) for _ in range(_D)]
_gamma = [1.0] * _D


def _matrix(rows, cols):
    return [[random.gauss(0.0, 0.05) for _ in range(cols)] for _ in range(rows)]


_W_gate = _matrix(_FF, _D)
_W_up = _matrix(_FF, _D)
_W_down = _matrix(_D, _FF)

_N_EXPERTS = 8
_experts = [(_matrix(_FF, _D), _matrix(_FF, _D), _matrix(_D, _FF)) for _ in range(_N_EXPERTS)]
_router = [random.gauss(0.0, 1.0) for _ in range(_N_EXPERTS)]

_logits = [random.gauss(0.0, 1.0) for _ in range(1024)]

_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}

BENCH = {
    "rms_norm": (_vec, _gamma),
    "rope_rotate": (_vec, 4096),
    "softmax": (_logits,),
    "swiglu_mlp": (_vec, _W_gate, _W_up, _W_down),
    "top_k_route": (_router, 2),
    "moe_block": (_vec, _experts, _router, 2),
    "param_count": (_CONFIG,),
    "kv_cache_bytes": (_CONFIG, 131072),
}
