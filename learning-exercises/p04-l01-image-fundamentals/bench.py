"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_H, _W = 64, 64
_img = [[[_rng.randrange(256) for _ in range(3)] for _ in range(_W)] for _ in range(_H)]
_chw = [[[_rng.gauss(0.0, 1.0) for _ in range(_W)] for _ in range(_H)] for _ in range(3)]
_gray = [[float(_rng.randrange(256)) for _ in range(_W)] for _ in range(_H)]

BENCH = {
    "hwc_to_chw": (_img,),
    "chw_to_hwc": (_chw,),
    "rgb_to_grayscale": (_img,),
    "rgb_to_hsv": ((200, 100, 50),),
    "preprocess_imagenet": (_img,),
    "deprocess_imagenet": (_chw,),
    "resize_nearest": (_gray, 128, 128),
    "resize_bilinear": (_gray, 128, 128),
}
