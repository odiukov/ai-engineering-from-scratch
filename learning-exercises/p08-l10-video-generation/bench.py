"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_N_FRAMES = 2400                         # сто секунд при 24 fps
_video = []
_base = 0.0
for _t in range(_N_FRAMES):
    _base += _rng.gauss(0, 0.05)
    _video.append(_base)

BENCH = {
    "position_embedding": (137, 256),
    "patchify": (_video, 2),
    "patch_tokens": (_video, 2, 16),
    "attention_pairs": (240, 4096, True),
    "frame_deltas": (_video,),
    "flicker_score": (_video,),
    "sample_frames": (_N_FRAMES, _rng, 0.7),
    "condition_on_first_frame": (_video, 5.0),
}
