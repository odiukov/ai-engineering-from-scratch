"""Входные данные для замера скорости."""

import math

_SR = 16000
_N = 256

# чистый тон ровно на бине 32: пик известен заранее, замер воспроизводим
_tone = [0.5 * math.sin(2 * math.pi * 2000 * i / _SR) for i in range(_N)]
_pcm = [int(round(v * 32767)) for v in _tone]

BENCH = {
    "sine": (440, _SR, 0.05, 0.5),
    "nyquist": (_SR,),
    "alias_frequency": (23456, _SR),
    "bin_to_hz": (17, _SR, 1024),
    "dft_magnitudes": (_tone,),
    "dominant_frequency": (_tone, _SR),
    "float_to_pcm16": (_tone,),
    "pcm16_to_float": (_pcm,),
}
