"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 128 отсчётов для квадратичного DFT: 128^2 = 16384 итераций внутреннего
# цикла — десятки миллисекунд, ровно та планка, где O(N^2) уже больно
_signal = [
    math.sin(2 * math.pi * 5 * n / 128) + 0.3 * random.uniform(-1, 1)
    for n in range(128)
]

# 1024 отсчёта для FFT: тот же порядок времени, но данных в 8 раз больше —
# в этом и виден переход O(N^2) -> O(N log N)
_long = [
    math.sin(2 * math.pi * 17 * n / 1024) + 0.1 * random.uniform(-1, 1)
    for n in range(1024)
]

# спектр той же длины, чтобы idft и magnitude_spectrum получали пары
_spectrum = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(128)]
_long_spectrum = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(1024)]

# свёртка идёт через fft, поэтому длина степень двойки; 256, а не 1024,
# потому что обратный ход у неё всё ещё квадратичный (idft), и на 1024
# один вызов вылезал бы за сотню миллисекунд
_kernel = [1.0, 0.5, 0.25, 0.125] + [0.0] * 252

BENCH = {
    "c_add": ((1.0, 2.0), (3.0, -4.0)),
    "c_mul": ((1.0, 2.0), (3.0, -4.0)),
    "twiddle": (37, 1024),
    "dft": (_signal,),
    "idft": (_spectrum,),
    "magnitude_spectrum": (_long_spectrum,),
    "fft": (_long,),
    "circular_convolution": (_long[:256], _kernel),
}
