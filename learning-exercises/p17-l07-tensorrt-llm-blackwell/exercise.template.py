"""
Компиляция под железо: FP8 и NVFP4 на Blackwell

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l07-tensorrt-llm-blackwell
Разбор:  /check-code p17-l07-tensorrt-llm-blackwell
"""

import math

BYTES_IN_GB = 10 ** 9
MODEL_SHAPES = {
    "llama-70b": {"params_b": 70.0, "active_b": 70.0, "layers": 80, "kv_heads": 8, "head_dim": 128},
    "gpt-oss-120b": {"params_b": 120.0, "active_b": 36.0, "layers": 36, "kv_heads": 8, "head_dim": 128},
    "llama-405b": {"params_b": 405.0, "active_b": 405.0, "layers": 126, "kv_heads": 8, "head_dim": 128},
}
STACKS = (
    {"name": "H100 + BF16 + vLLM", "hbm_gb": 80, "bw_tb_s": 3.35,
     "weight_bits": 16, "kv_bits": 16, "mtp": 1.0, "disagg": 1.0, "usd_per_gpu_hour": 2.50},
    {"name": "H100 + FP8 + vLLM", "hbm_gb": 80, "bw_tb_s": 3.35,
     "weight_bits": 8, "kv_bits": 8, "mtp": 1.0, "disagg": 1.0, "usd_per_gpu_hour": 2.50},
    {"name": "H200 + FP8 + vLLM", "hbm_gb": 141, "bw_tb_s": 4.80,
     "weight_bits": 8, "kv_bits": 8, "mtp": 1.0, "disagg": 1.0, "usd_per_gpu_hour": 3.50},
    {"name": "B200 + NVFP4 + FP8 KV + TRT-LLM", "hbm_gb": 192, "bw_tb_s": 8.00,
     "weight_bits": 4, "kv_bits": 8, "mtp": 1.8, "disagg": 1.6, "usd_per_gpu_hour": 4.80},
    {"name": "GB200 NVL72 + TRT-LLM + Dynamo", "hbm_gb": 192, "bw_tb_s": 8.00,
     "weight_bits": 4, "kv_bits": 8, "mtp": 1.8, "disagg": 2.5, "usd_per_gpu_hour": 6.20},
)
MIN_BITS_FOR_REASONING = 8


class NoStackFitsError(Exception):
    """Ни один стек не прошёл по ограничениям: памяти, числу GPU или качеству.

    Свой класс исключения, а не голый RuntimeError: NotImplementedError —
    это тоже RuntimeError, и тест на RuntimeError позеленел бы на пустой
    заготовке, ничего не проверив.
    """
    pass


def weights_gb(params_b, bits):
    """Сколько GB занимают веса модели при заданной разрядности.

    weights_gb(70, 16)  ->  140.0   (BF16: 2 байта на параметр)
    weights_gb(70, 8)   ->   70.0   (FP8)
    weights_gb(70, 4)   ->   35.0   (NVFP4)

    params_b — миллиарды параметров, bits — бит на один вес.
    bits <= 0 — это не «модель без весов», а ошибка вызова: ValueError.

    Это первая половина бюджета HBM. Вторая половина — KV-кэш, и он от
    разрядности весов не зависит совсем.
    """
    raise NotImplementedError


def kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch, bits):
    """Размер KV-кэша в GB для batch последовательностей длины seq_len.

    На каждый токен каждый слой хранит K и V — отсюда двойка:
        байт = layers * 2 * kv_heads * head_dim * seq_len * batch * bits/8

    kv_cache_gb(80, 8, 128, 2048, 1, 16)    ->  0.671...  (одна сессия)
    kv_cache_gb(80, 8, 128, 2048, 128, 16)  ->  85.9...   (128 сессий)

    Ловушка урока: «я сжал модель до 35 GB» забывает вот эту величину.
    При батче в сотню сессий KV-кэш легко больше самих весов.
    """
    raise NotImplementedError


def hbm_footprint_gb(shape, weight_bits, kv_bits, seq_len, batch, activation_gb=5.0):
    """Полный бюджет HBM: словарь с ключами weights, kv, activations, total.

    shape — запись из MODEL_SHAPES.

    hbm_footprint_gb(MODEL_SHAPES["llama-70b"], 4, 8, 2048, 128)["weights"] -> 35.0
    hbm_footprint_gb(MODEL_SHAPES["llama-70b"], 4, 8, 2048, 128)["kv"]      -> 42.9...

    Обрати внимание: квантование весов с 16 бит до 4 уменьшает первое
    слагаемое вчетверо, а второе не трогает вообще. Именно поэтому KV-кэш
    на Blackwell остаётся в FP8, а не уезжает в FP4 вслед за весами.
    """
    raise NotImplementedError


def gpus_needed(total_gb, gpu_hbm_gb):
    """Сколько GPU нужно, чтобы разложить total_gb по картам ёмкости gpu_hbm_gb.

    gpus_needed(230.0, 80)  ->  3
    gpus_needed(80.0, 80)   ->  1   (ровно влезло — это ещё одна карта, не две)
    gpus_needed(0.0, 80)    ->  0

    Округление только вверх: половины карты не бывает.
    """
    raise NotImplementedError


def decode_tokens_per_s(active_b, weight_bits, hbm_bw_tb_s):
    """Потолок decode: сколько токенов в секунду выдаёт одна карта.

    Модель roofline для decode: на каждый токен читаются ВСЕ активные веса,
    значит скорость упирается в пропускную способность памяти, а не в TFLOPS.

        tok/s = bandwidth_bytes_per_s / (active_b * 1e9 * weight_bits / 8)

    decode_tokens_per_s(70, 16, 3.35)  ->  23.9...   (H100, BF16)
    decode_tokens_per_s(70, 4, 8.0)    ->  228.5...  (B200, NVFP4)

    Для MoE важно active_b, а не общее число параметров: неактивные эксперты
    на этом токене не читаются.
    """
    raise NotImplementedError


def stack_speedup(factors, efficiency=1.0):
    """Перемножить множители ускорения и приземлить их коэффициентом efficiency.

    stack_speedup([2.0, 1.8])            ->  3.6
    stack_speedup([2.0, 1.8], 0.5)       ->  1.8
    stack_speedup([])                    ->  1.0

    Пустое произведение равно единице, а не нулю.

    Урок обещает 7x, хотя произведение всех четырёх множителей даёт около
    14x. Разница — это и есть efficiency: реальный трафик, промахи драфта и
    накладные расходы съедают половину бумажного выигрыша.
    """
    raise NotImplementedError


def cost_per_million_tokens(tokens_per_s, usd_per_gpu_hour):
    """Цена миллиона выходных токенов при данной скорости и цене часа GPU.

    cost_per_million_tokens(1000.0, 3.60)  ->  1.0
    cost_per_million_tokens(2000.0, 3.60)  ->  0.5

    За час карта отдаёт tokens_per_s * 3600 токенов; делим цену часа на это
    число и умножаем на миллион.

    Скорость нулевая — цена бесконечная, а не ноль: это ValueError.
    """
    raise NotImplementedError


def choose_stack(stacks, shape, seq_len, batch, workload="chat", max_gpus=8):
    """Выбрать самый дешёвый стек, проходящий по памяти и по качеству.

    Правила отбора:
      * модель со всеми слагаемыми должна лечь не более чем в max_gpus карт;
      * при workload == "reasoning" веса тоньше MIN_BITS_FOR_REASONING
        запрещены — цепочки рассуждений первыми разваливаются от FP4.

    Вернуть словарь: name, gpus, cost_per_million, tokens_per_s.
    Если не подошёл ни один стек — NoStackFitsError (свой класс, не ValueError
    и не RuntimeError).

    choose_stack(STACKS, MODEL_SHAPES["llama-70b"], 2048, 128)["name"]
        ->  "GB200 NVL72 + TRT-LLM + Dynamo"
    choose_stack(STACKS, MODEL_SHAPES["llama-70b"], 2048, 128, "reasoning")["name"]
        ->  "H200 + FP8 + vLLM"

    Вот здесь и живёт решение «стоит ли NVIDIA-lock своих денег»: если
    качество не пускает тебя на 4 бита, весь выигрыш Blackwell схлопывается.
    """
    raise NotImplementedError
