"""Входные данные для замера скорости."""

import math

_SR = 16000
_N_FFT = 128
_HOP = 64

# чирп: частота едет от 300 до 6000 Гц, спектр меняется от кадра к кадру
_signal = []
_phase = 0.0
for _i in range(1024):
    _f = 300.0 + (6000.0 - 300.0) * _i / 1024
    _phase += 2 * math.pi * _f / _SR
    _signal.append(math.sin(_phase))

_frame = _signal[:_N_FFT]
_win = [0.5 * (1 - math.cos(2 * math.pi * i / (_N_FFT - 1))) for i in range(_N_FFT)]
_filterbank = [[1.0 / (_N_FFT // 2 + 1)] * (_N_FFT // 2 + 1) for _ in range(16)]
_spec = [[abs(v) for v in _signal[i : i + _N_FFT // 2 + 1]] for i in range(0, 512, 8)]

BENCH = {
    "hann": (_N_FFT,),
    "frame_signal": (_signal, _N_FFT, _HOP),
    "rfft_magnitudes": (_frame,),
    "stft_magnitude": (_signal, _N_FFT, _HOP),
    "hz_to_mel": (4000.0,),
    "mel_to_hz": (2000.0,),
    "mel_filterbank": (16, _N_FFT, _SR),
    "log_mel_spectrogram": (_spec, _filterbank),
}
