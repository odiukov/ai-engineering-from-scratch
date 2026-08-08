"""
Квантование в проде: AWQ, GPTQ, GGUF, FP8, NVFP4 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

BYTES_IN_GB = 10 ** 9

# Форматы из урока: сколько бит на вес и сколько на элемент KV-кэша.
# KV-кэш квантуется ОТДЕЛЬНО от весов — это два разных решения.
FORMATS = {
    "BF16": {"weight_bits": 16, "kv_bits": 16, "engine": "vLLM"},
    "GGUF Q5_K_M": {"weight_bits": 5, "kv_bits": 16, "engine": "llama.cpp"},
    "GGUF Q4_K_M": {"weight_bits": 4, "kv_bits": 16, "engine": "llama.cpp"},
    "GPTQ-Int4": {"weight_bits": 4, "kv_bits": 16, "engine": "vLLM"},
    "AWQ-Int4": {"weight_bits": 4, "kv_bits": 16, "engine": "vLLM"},
    "FP8": {"weight_bits": 8, "kv_bits": 8, "engine": "vLLM/TRT-LLM"},
    "NVFP4 + FP8 KV": {"weight_bits": 4, "kv_bits": 8, "engine": "TRT-LLM"},
}

# LoRA поддерживают vLLM-форматы BF16/FP8/GPTQ/AWQ. GGUF в этой модели
# относится к llama.cpp-пути, а NVFP4 — к TRT-LLM без LoRA-adapter serving.
FORMATS_WITH_LORA = ("BF16", "FP8", "GPTQ-Int4", "AWQ-Int4")

# Геометрия Llama-3-70B: 80 слоёв, GQA с 8 KV-головами по 128.
LLAMA70B_LAYERS = 80
LLAMA70B_KV_HEADS = 8
LLAMA70B_HEAD_DIM = 128


class FormatUnsupportedError(Exception):
    """Формат не умеет того, что от него требуют (например, LoRA поверх NVFP4).

    Свой класс исключения: NotImplementedError — это RuntimeError, и тест на
    RuntimeError позеленел бы на пустой заготовке.
    """


def quant_params(values, bits):
    """Подобрать (scale, zero_point) для аффинного целочисленного квантования.

    Коды занимают диапазон [0, 2**bits - 1]. Диапазон значений РАСШИРЯЕТСЯ до
    нуля: lo = min(0, min(values)), hi = max(0, max(values)). Так ноль
    остаётся представимым точно, а zero_point гарантированно попадает внутрь
    сетки кодов.

        scale = (hi - lo) / (2**bits - 1)
        zero_point = round(-lo / scale)

    quant_params([0.0, 15.0], 4)   ->  (1.0, 0)
    quant_params([-8.0, 7.0], 4)   ->  (1.0, 8)

    Ловушка: у постоянного вектора hi == lo, и scale выходит нулевым — делить
    на него нельзя. В этом случае возвращаем scale = 1.0, zero_point = 0.
    """
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if not values:
        raise ValueError("cannot calibrate on an empty tensor")
    qmax = 2 ** bits - 1
    lo = min(0.0, min(values))
    hi = max(0.0, max(values))
    span = hi - lo
    if span == 0:
        # вырожденный тензор: любая шкала подойдёт, лишь бы не ноль
        return 1.0, 0
    scale = span / qmax
    return scale, int(round(-lo / scale))


def quantize(values, scale, zero_point, bits):
    """Значения -> целые коды. Всё, что вылезло за сетку, прижимается к краю.

    quantize([-8.0, 0.0, 7.0], 1.0, 8, 4)  ->  [0, 8, 15]
    quantize([-99.0, 99.0], 1.0, 8, 4)     ->  [0, 15]   (saturating clamp)

    Формула: q = round(v / scale) + zero_point, потом clamp в [0, 2**bits-1].

    Без clamp коды вылезают за разрядность и при записи в 4-битный тензор
    молча переполняются — вес +99 превращается в мусор вместо максимума.
    """
    qmax = 2 ** bits - 1
    codes = []
    for v in values:
        q = int(round(v / scale)) + zero_point
        codes.append(min(qmax, max(0, q)))
    return codes


def dequantize(codes, scale, zero_point):
    """Коды обратно в числа: (q - zero_point) * scale.

    dequantize([0, 8, 15], 1.0, 8)  ->  [-8.0, 0.0, 7.0]

    Обратно ровно те же числа получаются только если они изначально ложились
    на сетку. В общем случае это приближение — на нём и меряют потери.
    """
    return [(q - zero_point) * scale for q in codes]


def roundtrip(values, bits):
    """Квантовать и сразу разжать: что реально увидит матричное умножение.

    roundtrip([-8.0, 0.0, 7.0], 4)  ->  [-8.0, 0.0, 7.0]  (легло на сетку)
    roundtrip([0.0, 1.0], 2)        ->  [0.0, 1.0]

    Именно эту пару «сжали — разжали» и надо сравнивать с оригиналом:
    сами коды ничего не говорят о потере качества.
    """
    scale, zero_point = quant_params(values, bits)
    return dequantize(quantize(values, scale, zero_point, bits), scale, zero_point)


def quantization_error(values, bits):
    """Ошибка round-trip: {"max_abs", "mean_abs", "scale"}.

    quantization_error([-8.0, 0.0, 7.0], 4)["max_abs"]  ->  0.0
    quantization_error([0.0, 1.0, 100.0], 4)["max_abs"] ->  1.0
        (шкалу растянул выброс 100, и единица целиком ушла в ноль)

    scale возвращается специально: максимальная ошибка честного округления
    не превышает половины шага сетки, и по ней проверяют, что реализация не
    промахнулась мимо scale.
    """
    scale, _ = quant_params(values, bits)
    approx = roundtrip(values, bits)
    errors = [abs(a - v) for a, v in zip(approx, values)]
    return {
        "max_abs": max(errors),
        "mean_abs": sum(errors) / len(errors),
        "scale": scale,
    }


def blockwise_roundtrip(values, bits, block_size):
    """Микромасштабирование: своя шкала на каждый блок из block_size весов.

    Так устроены NVFP4 и MXFP4, и так же по сути работают K-кванты GGUF.

    blockwise_roundtrip([0.0, 1.0, 0.0, 100.0], 4, 2)  ->  [0.0, 1.0, 0.0, 100.0]

    Смысл: один выброс в тензоре растягивает общую шкалу и обнуляет
    разрешение у всех остальных весов. Локальная шкала запирает выброс
    внутри его блока.

    block_size <= 0 — ValueError. Хвост короче блока квантуется как есть.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out = []
    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        out.extend(roundtrip(block, bits))
    return out


def format_memory_gb(params_b, weight_bits, kv_bits, concurrency, ctx_tokens,
                     layers=LLAMA70B_LAYERS, kv_heads=LLAMA70B_KV_HEADS,
                     head_dim=LLAMA70B_HEAD_DIM, activation_gb=5.0):
    """Бюджет HBM целиком: {"weights", "kv", "activations", "total"} в GB.

    Веса:  params_b * weight_bits / 8
    KV:    layers * 2 * kv_heads * head_dim * ctx_tokens * concurrency * kv_bits/8

    format_memory_gb(70, 4, 16, 1, 2048)["weights"]     ->  35.0
    format_memory_gb(70, 4, 16, 128, 2048)["kv"]        ->  85.9...

    Главная ловушка урока: «я ужал модель до 35 GB» считает только первое
    слагаемое. На боевом батче KV-кэш больше весов, и общий бюджет решает
    он, а не формат весов.
    """
    weights = params_b * weight_bits / 8.0
    kv_bytes = layers * 2 * kv_heads * head_dim * ctx_tokens * concurrency * (kv_bits / 8.0)
    kv = kv_bytes / BYTES_IN_GB
    return {"weights": weights, "kv": kv, "activations": activation_gb,
            "total": weights + kv + activation_gb}


def pick_format(target, needs_lora=False, reasoning_heavy=False, forced=None):
    """Выбрать формат по железу и требованиям. Вернуть ключ из FORMATS.

    target: "cpu" | "edge" | "hopper" | "blackwell".

    pick_format("edge")                        ->  "GGUF Q4_K_M"
    pick_format("hopper", needs_lora=True)     ->  "AWQ-Int4"
    pick_format("blackwell")                   ->  "NVFP4 + FP8 KV"
    pick_format("blackwell", reasoning_heavy=True)  ->  "FP8"

    Если формат передан явно через forced — проверить его и вернуть.
    Форматы вне FORMATS_WITH_LORA с needs_lora=True — FormatUnsupportedError
    (свой класс), незнакомое имя формата или target — ValueError.

    Правило урока: рассуждающие нагрузки первыми разваливаются на четырёх
    битах, поэтому им FP8, несмотря на память.
    """
    if forced is not None:
        if forced not in FORMATS:
            raise ValueError(f"unknown format: {forced}")
        if needs_lora and forced not in FORMATS_WITH_LORA:
            raise FormatUnsupportedError(f"{forced} cannot serve LoRA adapters")
        return forced
    if target in ("cpu", "edge"):
        if needs_lora:
            raise FormatUnsupportedError("multi-LoRA serving is a datacenter path, not GGUF")
        return "GGUF Q5_K_M" if reasoning_heavy else "GGUF Q4_K_M"
    if target == "hopper":
        if reasoning_heavy:
            return "FP8"
        return "AWQ-Int4"
    if target == "blackwell":
        if needs_lora:
            return "AWQ-Int4"
        return "FP8" if reasoning_heavy else "NVFP4 + FP8 KV"
    raise ValueError(f"unknown target: {target}")
