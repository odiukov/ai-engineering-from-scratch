"""Входные данные для замера скорости."""

_model = {"params_b": 8.0, "layers": 32, "kv_heads": 8, "head_dim": 128}
_device = {"bandwidth_gb_s": 60.0, "tops": 38.0, "ram_gb": 8.0,
           "os_overhead_gb": 2.5, "battery_wh": 14.5}

BENCH = {
    "weights_gb": (8.0, 4),
    "decode_ceiling_tps": (4.0, 60.0),
    "kv_cache_gb": (131072, 32, 8, 128, 8),
    "roofline_times": (4096, _model, _device, 4),
    "roofline_regime": (4096, _model, _device, 4),
    "max_context_tokens": (1.5, _model, 16),
    "fits_on_device": (_device, _model, 4, 4096, 16),
    "energy_per_token_j": (_model, 4, 40.0),
}
