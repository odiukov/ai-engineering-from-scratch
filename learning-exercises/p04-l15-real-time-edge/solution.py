"""
Real-time вывод на edge — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def drop_warmup(times, warmup):
    """Выкинуть первые warmup замеров: это прогрев, а не измерение.

    drop_warmup([5.0, 4.9, 1.2, 1.1, 1.3], 2)  ->  [1.2, 1.1, 1.3]
    drop_warmup([1.0, 2.0], 0)                 ->  [1.0, 2.0]

    Первые проходы идут по холодным кэшам и через JIT-компиляцию, они в разы
    медленнее установившегося режима. Без прогрева профиль врёт в худшую сторону.

    Ловушки: не мутируй входной список (замеры могут понадобиться ещё раз);
    warmup >= len(times) -> ValueError, мерить стало нечего.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if warmup >= len(times):
        raise ValueError("warmup drops every measurement, nothing left to report")
    # срез уже создаёт новый список — вход остаётся нетронутым
    return list(times[warmup:])


def latency_stats(times):
    """Перцентили латентности: dict с p50_ms, p95_ms, p99_ms, mean_ms.

    latency_stats([1.0, 2.0, 3.0, 4.0])
        ->  {"p50_ms": 2.0, "p95_ms": 4.0, "p99_ms": 4.0, "mean_ms": 2.5}
    latency_stats([7.0])
        ->  {"p50_ms": 7.0, "p95_ms": 7.0, "p99_ms": 7.0, "mean_ms": 7.0}

    Перцентиль считай методом nearest-rank по отсортированной копии:
    индекс = ceil(p / 100 * n) - 1, зажатый в [0, n - 1].

    Ловушка из урока: `times[int(len(times) * 0.99)]` — это не p99. На 100
    замерах такой индекс равен 99, то есть максимуму выборки, а настоящий
    p99 стоит на индексе 98. Ошибка тем заметнее, чем длиннее хвост.

    Средним отчитываться нельзя: у real-time системы бюджет съедает хвост,
    а не типичный кадр. Пустой список -> ValueError.
    """
    if not times:
        raise ValueError("cannot compute latency stats of an empty sample")
    ordered = sorted(times)
    n = len(ordered)

    def rank(p):
        # nearest-rank: наименьшее значение, ниже которого лежит >= p% выборки
        idx = math.ceil(p / 100.0 * n) - 1
        return ordered[min(max(idx, 0), n - 1)]

    return {
        "p50_ms": rank(50),
        "p95_ms": rank(95),
        "p99_ms": rank(99),
        "mean_ms": sum(ordered) / n,
    }


def throughput_fps(latency_ms, batch_size=1):
    """Пропускная способность в кадрах в секунду.

    throughput_fps(10.0)                 ->  100.0
    throughput_fps(10.0, batch_size=4)   ->  400.0

    Латентность и пропускная способность — разные бюджеты. Батч из 4 кадров за
    10 мс даёт 400 fps при тех же 10 мс задержки на кадр: пропускная выросла
    вчетверо, латентность не изменилась.

    latency_ms <= 0 -> ValueError, иначе получишь деление на ноль.
    """
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return batch_size * 1000.0 / latency_ms


def conv2d_flops(c_in, c_out, k, h_out, w_out, groups=1):
    """FLOPs одной свёртки k x k, посчитанные как умножение + сложение.

    conv2d_flops(3, 64, 3, 224, 224)             ->  173408256
    conv2d_flops(64, 64, 3, 56, 56, groups=64)   ->  3612672

    Формула: 2 * (c_in / groups) * c_out * k * k * h_out * w_out.
    Двойка — потому что каждая точка ядра стоит одно умножение и одно сложение.

    groups — вот где живёт весь MobileNet: при groups == c_in == c_out свёртка
    становится depthwise, каждый канал обрабатывается своим ядром, и FLOPs
    падают ровно в c_in раз. Обычная свёртка 64->64 стоит 231 MFLOPs,
    depthwise — 3.6 MFLOPs.

    Ловушка: c_in должно делиться на groups, иначе ValueError.
    FLOPs — device-independent прокси для латентности, годный для сравнения
    архитектур и негодный как абсолютное время: depthwise-свёртки хорошо
    компилируются, а большие 7x7 — нет.
    """
    if groups <= 0:
        raise ValueError("groups must be positive")
    if c_in % groups != 0:
        raise ValueError("c_in must be divisible by groups")
    return 2 * (c_in // groups) * c_out * k * k * h_out * w_out


def linear_flops(in_features, out_features):
    """FLOPs полносвязного слоя на один вектор.

    linear_flops(1024, 1000)  ->  2048000
    linear_flops(512, 10)     ->  10240

    Та же двойка: умножение плюс сложение на каждый вес.
    """
    if in_features < 0 or out_features < 0:
        raise ValueError("feature counts must be non-negative")
    return 2 * in_features * out_features


def model_flops(layers):
    """Суммарные FLOPs модели, описанной списком слоёв.

    Слой — dict одного из двух видов:
      {"type": "conv",   "c_in":..., "c_out":..., "k":..., "h_out":...,
       "w_out":..., "groups":... (необязательно, по умолчанию 1)}
      {"type": "linear", "in_features":..., "out_features":...}

    model_flops([{"type": "linear", "in_features": 4, "out_features": 5}])  ->  40
    model_flops([])                                                        ->  0

    Неизвестный "type" -> ValueError: молча пропустить слой хуже, чем упасть,
    иначе бюджет окажется занижен и модель не влезет в устройство.
    """
    total = 0
    for layer in layers:
        kind = layer.get("type")
        if kind == "conv":
            total += conv2d_flops(
                layer["c_in"],
                layer["c_out"],
                layer["k"],
                layer["h_out"],
                layer["w_out"],
                layer.get("groups", 1),
            )
        elif kind == "linear":
            total += linear_flops(layer["in_features"], layer["out_features"])
        else:
            raise ValueError(f"unknown layer type: {kind!r}")
    return total


def quantize_int8(values):
    """Аффинная асимметричная квантизация FP32 -> INT8.

    Вернуть кортеж (qs, scale, zero_point), где qs — целые в [-128, 127].

    quantize_int8([0.0, 1.0, 2.0, 3.0])
        ->  ([-128, -43, 42, 127], 0.011764705882352941, -128)
    quantize_int8([2.0, 4.0, 6.0])
        ->  ([-43, 42, 127], 0.023529411764705882, -128)

    Как считать:
      lo = min(min(values), 0.0)      # диапазон ОБЯЗАН включать ноль
      hi = max(max(values), 0.0)
      scale = (hi - lo) / 255
      zero_point = clamp(round(-128 - lo / scale), -128, 127)
      q = clamp(round(x / scale) + zero_point, -128, 127)

    Зачем расширять диапазон до нуля: ноль должен быть представим ТОЧНО.
    Паддинг свёрток и выходы ReLU — это буквально нули, и если ноль после
    квантизации станет 0.04, ошибка потечёт по всей карте признаков.
    Заодно это гарантирует, что zero_point сам влезает в int8.

    Ловушка: hi == lo (все значения нулевые) — делить на ноль нельзя,
    отдельно ставь scale = 1.0. Пустой список -> ValueError.

    В PyTorch этому соответствует torch.quantize_per_tensor(x, scale,
    zero_point, torch.qint8), а подбор scale/zero_point делает observer.
    """
    if not values:
        raise ValueError("cannot quantize an empty tensor")

    lo = min(min(values), 0.0)
    hi = max(max(values), 0.0)
    # scale = 1.0 в вырожденном случае: делить на ноль нельзя, а вся выборка
    # всё равно нулевая и восстановится точно
    scale = (hi - lo) / 255.0 if hi > lo else 1.0

    def clamp(q):
        return min(max(q, -128), 127)

    zero_point = clamp(round(-128.0 - lo / scale))
    qs = [clamp(round(x / scale) + zero_point) for x in values]
    return qs, scale, zero_point


def dequantize_int8(qs, scale, zero_point):
    """Обратное преобразование INT8 -> FP32: (q - zero_point) * scale.

    dequantize_int8([-128, -43, 42, 127], 3.0 / 255, -128)
        ->  [0.0, 1.0, 2.0, 3.0]  (с точностью до шага сетки)

    Round-trip восстанавливает исходные числа с ошибкой не больше scale —
    это и есть цена четырёхкратной экономии памяти. На vision-задачах
    post-training static quantization обычно стоит 0.1–1 п.п. точности.
    """
    return [(q - zero_point) * scale for q in qs]
