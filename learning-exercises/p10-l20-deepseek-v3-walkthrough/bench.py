"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_EXPERTS = 64
_TOKENS = 300

_logits = [random.gauss(0.0, 1.0) for _ in range(_EXPERTS)]
_rows = [[random.gauss(0.0, 1.0) for _ in range(_EXPERTS)] for _ in range(_TOKENS)]
_bias = [0.0] * _EXPERTS
_load = [random.randrange(0, 20) for _ in range(_EXPERTS)]

_config = {
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 61,
    "first_k_dense_layers": 3,
    "num_attention_heads": 128,
    "head_dim": 128,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "shared_experts": 1,
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
}

BENCH = {
    "mla_kv_cache_bytes": (61, 512, 131072, 64, 2),
    "gqa_kv_cache_bytes": (61, 8, 128, 131072),
    "expert_params": (7168, 2048),
    "attention_params": (7168, 128, 128, 512),
    "model_parameters": (_config,),
    "route_topk": (_logits, 8),
    "expert_load": (_rows, _EXPERTS, 8),
    "balance_bias_step": (_bias, _load, 0.05),
}
