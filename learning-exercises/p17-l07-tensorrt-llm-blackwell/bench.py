"""Входные данные для замера скорости."""

_shape = {"params_b": 405.0, "active_b": 405.0, "layers": 126, "kv_heads": 8, "head_dim": 128}
_stacks = tuple(
    {"name": f"stack-{i}", "hbm_gb": 80 + i, "bw_tb_s": 3.0 + i * 0.01,
     "weight_bits": 4 + (i % 3) * 4, "kv_bits": 8, "mtp": 1.0 + i * 0.01,
     "disagg": 1.0 + i * 0.02, "usd_per_gpu_hour": 2.5 + i * 0.05}
    for i in range(200)
)
_factors = [1.0 + i / 1000.0 for i in range(500)]

BENCH = {
    "weights_gb": (405.0, 4),
    "kv_cache_gb": (126, 8, 128, 131072, 256, 8),
    "hbm_footprint_gb": (_shape, 4, 8, 131072, 256),
    "gpus_needed": (275.0, 192),
    "decode_tokens_per_s": (405.0, 4, 8.0),
    "stack_speedup": (_factors,),
    "cost_per_million_tokens": (1028.6, 6.20),
    "choose_stack": (_stacks, _shape, 8192, 128, "chat", 8),
}
