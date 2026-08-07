"""
Инференс на устройстве: ANE, Hexagon, WebGPU, Jetson — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

BYTES_IN_GB = 10 ** 9

# Устройства из урока. tops — пиковая производительность в 10^12 операций в
# секунду (NPU меряют именно в TOPS); в модели считаем 2 операции на
# параметр на токен, как обычно для matmul.
EDGE_TARGETS = {
    "iphone-16": {"bandwidth_gb_s": 60.0, "tops": 38.0, "ram_gb": 8.0,
                  "os_overhead_gb": 2.5, "battery_wh": 14.5},
    "snapdragon-8-gen-3": {"bandwidth_gb_s": 77.0, "tops": 45.0, "ram_gb": 12.0,
                           "os_overhead_gb": 3.0, "battery_wh": 18.0},
    "m3-max": {"bandwidth_gb_s": 400.0, "tops": 38.0, "ram_gb": 48.0,
               "os_overhead_gb": 8.0, "battery_wh": 100.0},
    "jetson-agx-orin": {"bandwidth_gb_s": 205.0, "tops": 275.0, "ram_gb": 64.0,
                        "os_overhead_gb": 6.0, "battery_wh": 0.0},
    "h100": {"bandwidth_gb_s": 3350.0, "tops": 989.0, "ram_gb": 80.0,
             "os_overhead_gb": 0.0, "battery_wh": 0.0},
}

# Модели, которые реально гоняют на устройстве. GQA: 8 KV-голов по 128.
EDGE_MODELS = {
    "llama-3.1-8b": {"params_b": 8.0, "layers": 32, "kv_heads": 8, "head_dim": 128},
    "llama-3.2-3b": {"params_b": 3.2, "layers": 28, "kv_heads": 8, "head_dim": 128},
}

# Энергия на байт, вытащенный из LPDDR5, в пикоджоулях. Порядок величины;
# точное число зависит от техпроцесса и частоты.
LPDDR5_PJ_PER_BYTE = 40.0


class EdgeBudgetError(Exception):
    """Модель с кэшем не влезает в память устройства.

    Свой класс исключения: MemoryError означал бы, что кончилась память у
    самого Python, а NotImplementedError — это RuntimeError, и тест на
    RuntimeError позеленел бы на пустой заготовке.
    """


def weights_gb(params_b, bits):
    """Размер весов в GB (10^9 байт) при заданной разрядности.

    weights_gb(8.0, 16)  ->  16.0   (BF16)
    weights_gb(8.0, 4)   ->   4.0   (Q4)

    На устройстве это первое, что упирается в потолок: телефон с 8 GB RAM
    физически не поднимет 8B в BF16, сколько бы TOPS ни было у NPU.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    return params_b * bits / 8.0


def decode_ceiling_tps(model_gb, bandwidth_gb_s):
    """Потолок decode: сколько токенов в секунду позволяет ПАМЯТЬ.

    На каждый токен веса читаются целиком, значит потолок — это
    bandwidth / model_gb, и никакой NPU его не поднимет.

    decode_ceiling_tps(3.5, 50.0)    ->  14.28...  (7B Q4 на мобильной DRAM)
    decode_ceiling_tps(3.5, 3000.0)  ->  857.1     (та же модель на HBM3)

    Разрыв в 30-50 раз между HBM датацентра и DRAM телефона — это и есть
    вся разница в tok/s. Модель одна и та же.
    """
    if model_gb <= 0:
        raise ValueError("model_gb must be positive")
    return bandwidth_gb_s / model_gb


def kv_cache_gb(ctx_tokens, layers, kv_heads, head_dim, bits):
    """Размер KV-кэша в GB для одной сессии длиной ctx_tokens.

    байт = layers * 2 * kv_heads * head_dim * ctx_tokens * bits/8

    kv_cache_gb(4096, 32, 8, 128, 16)   ->  0.536...
    kv_cache_gb(32768, 32, 8, 128, 16)  ->  4.294...

    Вот на этом и ломается «у Llama же 128K контекста»: 128K — фича
    датацентра, а на телефоне 32K KV-кэша больше самой Q4-модели.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    return layers * 2 * kv_heads * head_dim * ctx_tokens * (bits / 8.0) / BYTES_IN_GB


def roofline_times(tokens_per_forward, model, device, weight_bits):
    """Два времени одного прохода: {"memory_s", "compute_s"}.

    memory_s  = байты весов / пропускная способность  (веса читаются один
                раз за проход, сколько бы токенов в нём ни было)
    compute_s = 2 * params * tokens / (tops * 1e12)

    roofline_times(1, EDGE_MODELS["llama-3.1-8b"], EDGE_TARGETS["iphone-16"], 4)
        ->  {"memory_s": 0.0666..., "compute_s": 0.000421...}

    Один токен (decode) читает те же гигабайты, что и пятьсот (prefill), —
    отсюда вся асимметрия. Именно поэтому batching помогает prefill и почти
    не помогает decode.
    """
    if tokens_per_forward < 1:
        raise ValueError("tokens_per_forward must be at least 1")
    memory_s = weights_gb(model["params_b"], weight_bits) / device["bandwidth_gb_s"]
    flops = 2.0 * model["params_b"] * 1e9 * tokens_per_forward
    compute_s = flops / (device["tops"] * 1e12)
    return {"memory_s": memory_s, "compute_s": compute_s}


def roofline_regime(tokens_per_forward, model, device, weight_bits):
    """Во что упёрся проход: "memory" или "compute".

    roofline_regime(1, EDGE_MODELS["llama-3.1-8b"], EDGE_TARGETS["iphone-16"], 4)
        ->  "memory"    (decode)
    roofline_regime(512, EDGE_MODELS["llama-3.1-8b"], EDGE_TARGETS["iphone-16"], 4)
        ->  "compute"   (prefill)

    Считается через roofline_times: узкое место — то, что дольше. Ничья
    засчитывается памяти, потому что decode на границе всё равно ждёт байты.
    """
    times = roofline_times(tokens_per_forward, model, device, weight_bits)
    return "compute" if times["compute_s"] > times["memory_s"] else "memory"


def max_context_tokens(free_gb, model, kv_bits):
    """Сколько токенов контекста влезает в free_gb памяти под KV-кэш.

    max_context_tokens(1.5, EDGE_MODELS["llama-3.1-8b"], 16)  ->  11444
    max_context_tokens(1.5, EDGE_MODELS["llama-3.1-8b"], 8)   ->  22888

    Округление ВНИЗ: половина токена в кэше не живёт. Квантование KV в FP8
    ровно удваивает окно — это и есть цена вопроса «4K или 8K на телефоне».
    """
    if free_gb < 0:
        raise ValueError("free_gb must not be negative")
    per_token_gb = kv_cache_gb(1, model["layers"], model["kv_heads"],
                               model["head_dim"], kv_bits)
    return int(free_gb / per_token_gb)


def fits_on_device(device, model, weight_bits, ctx_tokens, kv_bits):
    """Бюджет памяти устройства: {"model_gb", "kv_gb", "free_gb"}.

    free_gb = ram_gb - os_overhead_gb - model_gb - kv_gb.

    fits_on_device(EDGE_TARGETS["iphone-16"], EDGE_MODELS["llama-3.1-8b"], 4, 4096, 16)
        ->  free_gb около 0.96

    Если остаток отрицательный — EdgeBudgetError, а не отрицательное число:
    на устройстве это не «чуть-чуть не хватило», а падение приложения.

    Порядок вычитания важен: сначала ОС забирает своё, и только остаток
    делится между весами и KV-кэшем.
    """
    m = weights_gb(model["params_b"], weight_bits)
    kv = kv_cache_gb(ctx_tokens, model["layers"], model["kv_heads"],
                     model["head_dim"], kv_bits)
    free = device["ram_gb"] - device["os_overhead_gb"] - m - kv
    if free < 0:
        raise EdgeBudgetError(
            f"needs {m + kv:.2f} GB, only "
            f"{device['ram_gb'] - device['os_overhead_gb']:.2f} GB available"
        )
    return {"model_gb": m, "kv_gb": kv, "free_gb": free}


def energy_per_token_j(model, weight_bits, pj_per_byte, overhead_j=0.0):
    """Энергия на один сгенерированный токен, в джоулях.

    Считаем только чтение весов из DRAM — на edge это доминирующая статья.

    energy_per_token_j(EDGE_MODELS["llama-3.1-8b"], 4, 40.0)  ->  0.16
    energy_per_token_j(EDGE_MODELS["llama-3.1-8b"], 8, 40.0)  ->  0.32

    Отсюда считается автономность: батарея на 14.5 Вт·ч это 52200 Дж,
    и делённые на 0.16 они дают порядка трёхсот тысяч токенов.

    Квантование экономит не только память и время, но и заряд — байты,
    которые не прочитали, не стоят ничего.
    """
    if pj_per_byte < 0:
        raise ValueError("pj_per_byte must not be negative")
    read_bytes = weights_gb(model["params_b"], weight_bits) * BYTES_IN_GB
    return read_bytes * pj_per_byte * 1e-12 + overhead_j
