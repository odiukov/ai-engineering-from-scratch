"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# батч CLIP: 128 пар (картинка, подпись) по 256 измерений -> матрица 128x128
_images = [[_rng.gauss(0.0, 1.0) for _ in range(256)] for _ in range(128)]
_texts = [[_rng.gauss(0.0, 1.0) for _ in range(256)] for _ in range(128)]

# zero-shot на 1000 классов ImageNet
_classes = [[_rng.gauss(0.0, 1.0) for _ in range(256)] for _ in range(1000)]
_names = [f"class_{i}" for i in range(1000)]

# 80 шаблонов OpenAI на каждый из 200 классов
_templates = [f"a {word} photo of a {{}}" for word in ("blurry", "cropped", "bright")] * 27
_per_class = [
    [[_rng.gauss(0.0, 1.0) for _ in range(256)] for _ in range(80)] for _ in range(200)
]

BENCH = {
    "normalize_rows": (_images,),
    "similarity_matrix": (_images, _classes),
    "clip_loss": (_images, _texts, 14.2857),
    "siglip_loss": (_images, _texts, 14.2857, -10.0),
    "build_prompts": (_names, _templates),
    "average_class_embeddings": (_per_class,),
    "zero_shot_probabilities": (_images, _classes, 100.0),
    "zero_shot_classify": (_images, _classes, _names),
}
