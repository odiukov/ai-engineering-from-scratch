"""
Квантование: как уместить модель — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

BYTES_PER_GB = 1024 ** 3


def quantize_symmetric(values, num_bits=8):
    """Симметричное квантование списка чисел. Вернуть (список целых, scale).

    quantize_symmetric([0.0, 1.0, -1.0], 8)  ->  ([0, 127, -127], 1/127)
    quantize_symmetric([0.0, 0.0], 8)        ->  ([0, 0], 1.0)

    Схема:
        qmax  = 2^(bits-1) - 1        для 8 бит это 127
        qmin  = -2^(bits-1)           для 8 бит это -128
        scale = max(|x|) / qmax
        q     = clamp(round(x / scale), qmin, qmax)

    Тензор из одних нулей: scale = 1.0, а не деление на ноль. Такой слой
    в модели вполне бывает.

    Ловушка диапазона: qmin на единицу больше по модулю, чем qmax. Масштаб
    считается по qmax — иначе самое большое положительное значение вылезет
    за 127 и обрежется.
    """
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    abs_max = max((abs(v) for v in values), default=0.0)
    if abs_max == 0.0:
        return ([0] * len(values), 1.0)
    scale = abs_max / qmax
    return ([min(max(round(v / scale), qmin), qmax) for v in values], scale)


def dequantize(quantized, scale):
    """Обратное преобразование: целые обратно в числа с плавающей точкой.

    dequantize([0, 127, -127], 1 / 127)  ->  [0.0, 1.0, -1.0]

    Одно умножение. Вся информация о величинах живёт в scale — целые сами
    по себе не значат ничего, и хранить их без scale бессмысленно.
    """
    return [q * scale for q in quantized]


def quantize_asymmetric(values, num_bits=8):
    """Асимметричное квантование со сдвигом. Вернуть (целые, scale, zero_point).

    quantize_asymmetric([0.0, 1.0], 8)  ->  ([0, 255], 1/255, 0)
    quantize_asymmetric([0.0, 0.0], 8)  ->  ([0, 0], 1.0, 0)

    Здесь целые БЕЗ знака: от 0 до 2^bits - 1. Реальный диапазон
    натягивается на весь этот отрезок:
        lo, hi     = min(0, min(x)), max(0, max(x))
        scale      = (hi - lo) / (qmax - qmin)
        zero_point = clamp(qmin - round(lo / scale), qmin, qmax)
        q          = clamp(round(x / scale) + zero_point, qmin, qmax)

    Обратно: x = (q - zero_point) * scale.

    Зачем: активации после ReLU неотрицательны. Симметричная схема отдаёт
    половину целочисленного диапазона отрицательным значениям, которых
    там не бывает, и теряет на этом ровно бит точности.

    Ловушка, из-за которой схема ломается молча: ноль ОБЯЗАН быть
    представим. Поэтому диапазон расширяют до нуля (min(0, ...) и
    max(0, ...)). Возьмёшь буквальные min и max по данным, где все значения
    положительны, — zero_point уедет за границу [qmin, qmax], зажмётся, и
    после сдвига все q упрутся в qmax. Тензор превратится в константу.

    Вырожденный случай (все нули): scale = 1.0, zero_point = 0.
    """
    qmin, qmax = 0, 2 ** num_bits - 1
    # ноль должен попадать в диапазон: иначе zero_point невыразим целым
    lo = min(0.0, min(values))
    hi = max(0.0, max(values))
    if hi == lo:
        return ([0] * len(values), 1.0, 0)
    scale = (hi - lo) / (qmax - qmin)
    zero_point = min(max(qmin - round(lo / scale), qmin), qmax)
    q = [min(max(round(v / scale) + zero_point, qmin), qmax) for v in values]
    return (q, scale, zero_point)


def dequantize_asymmetric(quantized, scale, zero_point):
    """Обратное преобразование для асимметричной схемы.

    dequantize_asymmetric([0, 255], 1 / 255, 0)  ->  [0.0, 1.0]

    Ловушка порядка: сначала ВЫЧЕСТЬ zero_point, потом умножить на scale.
    Перепутаешь — получишь сдвинутый на zero_point * scale тензор, и модель
    поедет вся целиком, без единого исключения по дороге.
    """
    return [(q - zero_point) * scale for q in quantized]


def quantize_per_channel(matrix, num_bits=8):
    """Поканальное квантование: свой scale на каждую строку матрицы.

    Вернуть (матрица целых, список scale по строкам).

    quantize_per_channel([[1.0, -1.0], [100.0, -100.0]], 8)
        ->  ([[127, -127], [127, -127]], [1/127, 100/127])

    Одна строка с огромными весами больше не портит остальные: у каждой
    свой масштаб. Платим за это N числами scale вместо одного — копейки
    против выигрыша в качестве. Любой продовый метод квантования работает
    поканально или ещё мельче (по группам весов внутри канала).

    Собери из quantize_symmetric, не переписывай формулу заново.
    """
    rows, scales = [], []
    for row in matrix:
        q, scale = quantize_symmetric(row, num_bits)
        rows.append(q)
        scales.append(scale)
    return (rows, scales)


def dequantize_per_channel(quantized, scales):
    """Обратное поканальное преобразование: своя scale на свою строку.

    dequantize_per_channel([[127, -127]], [1 / 127])  ->  [[1.0, -1.0]]

    Ловушка: scales — это список по строкам, а не одно число. Применишь
    первый scale ко всей матрице — вернёшь ровно per-tensor квантование
    и не заметишь.
    """
    return [dequantize(row, scale) for row, scale in zip(quantized, scales)]


def quantization_error(original, reconstructed):
    """Насколько квантование испортило тензор. Оба входа — ПЛОСКИЕ списки.

    Вернуть словарь с ключами "mse", "rmse", "max_error", "snr_db",
    "cosine_similarity".

    quantization_error([1.0, 2.0], [1.0, 2.0])
        ->  mse 0.0, cosine_similarity 1.0, snr_db очень большой

    Как считать:
        mse    = среднее квадратов разностей
        rmse   = sqrt(mse)
        snr_db = 10 * log10(средний квадрат оригинала / mse)
        cosine = скалярное произведение / (норма * норма)

    SNR в децибелах — рабочая единица: каждый лишний бит добавляет примерно
    6 dB. Косинусная близость ловит другое: она игнорирует общий масштаб и
    показывает, сохранилось ли НАПРАВЛЕНИЕ вектора весов.

    Ловушки: mse == 0 даёт деление на ноль в SNR — подставь нижнюю границу
    1e-20; нулевая норма любого из векторов делает косинус неопределённым,
    возвращай 0.0.
    """
    n = len(original)
    diff = [a - b for a, b in zip(original, reconstructed)]
    mse = sum(d * d for d in diff) / n
    signal = sum(a * a for a in original) / n
    # нижняя граница на mse: идеальное восстановление иначе даёт log10(inf)
    snr_db = 10.0 * math.log10(signal / max(mse, 1e-20)) if signal > 0 else 0.0

    norm_a = math.sqrt(sum(a * a for a in original))
    norm_b = math.sqrt(sum(b * b for b in reconstructed))
    if norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = sum(a * b for a, b in zip(original, reconstructed)) / (norm_a * norm_b)

    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_error": max((abs(d) for d in diff), default=0.0),
        "snr_db": snr_db,
        "cosine_similarity": cosine,
    }


def model_memory_gb(num_params_billions, bits_per_param):
    """Сколько гигабайт занимают веса модели при данной разрядности.

    model_memory_gb(70, 16)  ->  130.38...   (Llama 3 70B в FP16)
    model_memory_gb(70, 4)   ->  32.59...    (та же модель в INT4)
    model_memory_gb(7, 8)    ->  6.51...

    Формула: параметры * (биты / 8) байт, потом делим на 1024^3.

    Гигабайт здесь двоичный (1024^3), как его показывает nvidia-smi. Маркетинг
    считает по 10^9 и получает числа побольше — отсюда и расхождение с
    «140GB», которые пишут про 70B в FP16 (на самом деле 130.4 GiB).

    Это только веса. KV-кэш, активации и фрагментация аллокатора сверху —
    планировать VRAM по одному этому числу нельзя.
    """
    return num_params_billions * 1e9 * (bits_per_param / 8) / BYTES_PER_GB
