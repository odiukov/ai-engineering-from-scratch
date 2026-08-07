"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VIT_DIM, _HIDDEN, _LLM_DIM = 32, 64, 48
_PATCHES = [[random.gauss(0, 1) for _ in range(_VIT_DIM)] for _ in range(576)]
_W1 = [[random.gauss(0, 0.1) for _ in range(_VIT_DIM)] for _ in range(_HIDDEN)]
_B1 = [random.gauss(0, 0.01) for _ in range(_HIDDEN)]
_W2 = [[random.gauss(0, 0.1) for _ in range(_HIDDEN)] for _ in range(_LLM_DIM)]
_B2 = [random.gauss(0, 0.01) for _ in range(_LLM_DIM)]
_PROMPT = "A chat between a human and an assistant. USER: <image> Describe it. ASSISTANT:"

BENCH = {
    "gelu": (0.7,),
    "mlp_projector": (_PATCHES, _W1, _B1, _W2, _B2),
    "projector_param_count": (1024, 4096, 4096),
    "build_llava_prompt": ("A chat.", "Describe this image in detail."),
    "expand_image_placeholder": (_PROMPT, 2880),
    "pick_anyres_grid": (1170, 2532),
    "anyres_token_count": (1170, 2532),
    "context_usage": (2880, 500, 32768),
}
