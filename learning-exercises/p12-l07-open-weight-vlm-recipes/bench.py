"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ENCODERS = ("clip", "siglip", "dinov2", "internvit")
_CONNECTORS = ("mlp", "qformer", "perceiver")
_LLMS = ("qwen7b", "llama8b", "llama70b")
_DATA = ("llava150k", "sharegpt4v", "pixmo", "cauldron")
_RESOLUTIONS = (336, 384, 448, 980)
_TOKENS = (144, 576, 1024, 2304)

# Полный перебор дал бы 4*3*3*4*4*4 = 2304 строки; берём случайную выборку,
# чтобы controlled_pairs (квадратичный по строкам) успевал за десятки мс.
_rows = []
for _ in range(300):
    _rows.append(
        {
            "encoder": random.choice(_ENCODERS),
            "connector": random.choice(_CONNECTORS),
            "llm": random.choice(_LLMS),
            "data": random.choice(_DATA),
            "resolution": random.choice(_RESOLUTIONS),
            "tokens": random.choice(_TOKENS),
            "mmmu": round(random.uniform(30.0, 50.0), 2),
        }
    )

_line = "encoder=siglip;connector=mlp;tokens=576 | mmmu=41.2;docvqa=88.0"
_deltas = {("encoder", "clip", "siglip"): 3.0, ("resolution", 384, 448): 1.5}
_swaps = [("encoder", "clip", "siglip"), ("resolution", 384, 448)] * 50

BENCH = {
    "parse_ablation_row": (_line,),
    "controlled_pairs": (_rows, "encoder"),
    "axis_delta": (_rows, "encoder", "clip", "siglip", "mmmu"),
    "explained_variance": (_rows, "encoder", "mmmu"),
    "rank_axes_by_impact": (_rows, "mmmu"),
    "expected_score": (38.0, _swaps, _deltas),
    "pick_recipe": (_rows, "mmmu", 1024),
}
