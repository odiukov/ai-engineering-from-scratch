"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_signal = [random.uniform(-1.0, 1.0) for _ in range(16000 * 3)]
_frames = [[random.uniform(-1.0, 1.0) for _ in range(400)] for _ in range(300)]

BENCH = {
    "sine_wave": (440, 3.0),
    "n_frames": (16000 * 30,),
    "frame_signal": (_signal,),
    "frame_energy": (_frames[0],),
    "pad_or_clip": (_frames, 3000),
    "conv_stem_length": (3000, 3, 2, 1),
    "whisper_prompt": ("fr", "translate", False),
    "parse_whisper_prompt": (["<|startoftranscript|>", "<|fr|>", "<|translate|>"],),
}
