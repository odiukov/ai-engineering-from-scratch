"""
Инференс на устройстве: ANE, Hexagon, WebGPU, Jetson

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l12-edge-inference
Разбор:  /check-code p17-l12-edge-inference
"""

BYTES_IN_GB = 10 ** 9
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
EDGE_MODELS = {
    "llama-3.1-8b": {"params_b": 8.0, "layers": 32, "kv_heads": 8, "head_dim": 128},
    "llama-3.2-3b": {"params_b": 3.2, "layers": 28, "kv_heads": 8, "head_dim": 128},
}
LPDDR5_PJ_PER_BYTE = 40.0


class EdgeBudgetError(Exception):
    """Модель с кэшем не влезает в память устройства.

    Свой класс исключения: MemoryError означал бы, что кончилась память у
    самого Python, а NotImplementedError — это RuntimeError, и тест на
    RuntimeError позеленел бы на пустой заготовке.
    """
    pass


def weights_gb(params_b, bits):
    """Размер весов в GB (10^9 байт) при заданной разрядности.

    weights_gb(8.0, 16)  ->  16.0   (BF16)
    weights_gb(8.0, 4)   ->   4.0   (Q4)

    На устройстве это первое, что упирается в потолок: телефон с 8 GB RAM
    физически не поднимет 8B в BF16, сколько бы TOPS ни было у NPU.
    """
    raise NotImplementedError


def decode_ceiling_tps(model_gb, bandwidth_gb_s):
    """Потолок decode: сколько токенов в секунду позволяет ПАМЯТЬ.

    На каждый токен веса читаются целиком, значит потолок — это
    bandwidth / model_gb, и никакой NPU его не поднимет.

    decode_ceiling_tps(3.5, 50.0)    ->  14.28...  (7B Q4 на мобильной DRAM)
    decode_ceiling_tps(3.5, 3000.0)  ->  857.1     (та же модель на HBM3)

    Разрыв в 30-50 раз между HBM датацентра и DRAM телефона — это и есть
    вся разница в tok/s. Модель одна и та же.
    """
    raise NotImplementedError


def kv_cache_gb(ctx_tokens, layers, kv_heads, head_dim, bits):
    """Размер KV-кэша в GB для одной сессии длиной ctx_tokens.

    байт = layers * 2 * kv_heads * head_dim * ctx_tokens * bits/8

    kv_cache_gb(4096, 32, 8, 128, 16)   ->  0.536...
    kv_cache_gb(32768, 32, 8, 128, 16)  ->  4.294...

    Вот на этом и ломается «у Llama же 128K контекста»: 128K — фича
    датацентра, а на телефоне 32K KV-кэша больше самой Q4-модели.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def roofline_regime(tokens_per_forward, model, device, weight_bits):
    """Во что упёрся проход: "memory" или "compute".

    roofline_regime(1, EDGE_MODELS["llama-3.1-8b"], EDGE_TARGETS["iphone-16"], 4)
        ->  "memory"    (decode)
    roofline_regime(512, EDGE_MODELS["llama-3.1-8b"], EDGE_TARGETS["iphone-16"], 4)
        ->  "compute"   (prefill)

    Считается через roofline_times: узкое место — то, что дольше. Ничья
    засчитывается памяти, потому что decode на границе всё равно ждёт байты.
    """
    raise NotImplementedError


def max_context_tokens(free_gb, model, kv_bits):
    """Сколько токенов контекста влезает в free_gb памяти под KV-кэш.

    max_context_tokens(1.5, EDGE_MODELS["llama-3.1-8b"], 16)  ->  11444
    max_context_tokens(1.5, EDGE_MODELS["llama-3.1-8b"], 8)   ->  22888

    Округление ВНИЗ: половина токена в кэше не живёт. Квантование KV в FP8
    ровно удваивает окно — это и есть цена вопроса «4K или 8K на телефоне».
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
