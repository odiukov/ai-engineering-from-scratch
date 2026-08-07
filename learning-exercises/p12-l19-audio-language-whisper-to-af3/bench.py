"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_sr = 8000
_wave = [0.4 * math.sin(2 * math.pi * 440 * i / _sr) for i in range(_sr // 4)]
_frame = _wave[:200]
_fbank_args = (64, 20, _sr)
_frames = [[random.gauss(0, 1) for _ in range(32)] for _ in range(300)]
_queries = [[random.gauss(0, 1) for _ in range(32)] for _ in range(64)]

BENCH = {
    "hz_to_mel": (4000.0,),
    "mel_to_hz": (2000.0,),
    "mel_filterbank": _fbank_args,
    "frame_signal": (_wave, _sr),
    "dft_magnitude": (_frame, 64),
    "log_mel_spectrogram": (_wave, _sr, 20, 64),
    "qformer_attend": (_queries, _frames),
    "pick_pipeline": (["transcription", "music", "emotion"],),
}
