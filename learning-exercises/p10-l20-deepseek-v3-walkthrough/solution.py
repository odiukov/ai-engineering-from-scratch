"""
Разбор архитектуры DeepSeek-V3 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок читает конфиг DeepSeek-V3 сверху вниз и выводит из него всё
остальное: сколько весов в модели, сколько из них включается на один токен,
сколько памяти съедает KV-кеш при MLA против обычного GQA и как работает
роутинг MoE без вспомогательного loss.

Модель, в которой мы работаем. Весов здесь нет — есть их количество.
Все функции считают ЧИСЛА параметров и БАЙТЫ, а не тензоры: 671 млрд
чисел в память ноутбука не поместятся, а арифметика от этого не меняется.
Роутинг же настоящий: логиты роутера — список float, эксперты выбираются
top-k, балансировочный bias правится по факту нагрузки.

Упрощения, о которых надо знать:
  * MLP считается по SwiGLU — три матрицы (gate, up, down), а не две;
  * у attention считаются только большие матрицы (W_q, сжатие и разжатие
    KV, W_o); нормы, bias-ы и RoPE-хвосты пропущены;
  * эмбеддинг считается один раз (веса привязаны к выходному слою).
Из-за этого итог сходится с опубликованными 671B примерно на процент, а не
до последнего параметра.
"""

DEEPSEEK_V3 = {
    "hidden_size": 7168,
    "intermediate_size": 18432,       # dense MLP на первых слоях
    "moe_intermediate_size": 2048,    # MLP одного эксперта
    "num_hidden_layers": 61,
    "first_k_dense_layers": 3,        # первые 3 блока без роутера
    "num_attention_heads": 128,
    "head_dim": 128,
    "kv_lora_rank": 512,              # латентная размерность MLA
    "qk_rope_head_dim": 64,            # несжатая RoPE-часть ключа
    "num_experts": 256,
    "num_experts_per_tok": 8,         # top-8 роутинг
    "shared_experts": 1,              # всегда включённый эксперт
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
}


def mla_kv_cache_bytes(
    num_layers, kv_lora_rank, seq_len, qk_rope_head_dim=64, bytes_per_element=2
):
    """Размер KV-кеша при Multi-Head Latent Attention, в байтах.

    mla_kv_cache_bytes(61, 512, 131072, 64)  ->  9210691584   (8.6 GiB)
    mla_kv_cache_bytes(61, 512, 1, 64)       ->  70272

    MLA хранит на токен и слой один KV-латент длины
    kv_lora_rank и несжатую RoPE-часть ключа длины qk_rope_head_dim.
    Содержательные K и V разворачиваются из латента на лету, но RoPE-
    компонент нельзя восстановить без позиционной информации.
    Двойки за «K и V» всё ещё нет: это один латент плюс RoPE-хвост.

    bytes_per_element по умолчанию 2: BF16.
    """
    elements_per_token = kv_lora_rank + qk_rope_head_dim
    return num_layers * elements_per_token * seq_len * bytes_per_element


def gqa_kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, bytes_per_element=2):
    """Размер KV-кеша при обычном Grouped-Query Attention, в байтах.

    gqa_kv_cache_bytes(61, 8, 128, 131072)  ->  32749125632   (30.5 GiB)
    gqa_kv_cache_bytes(61, 8, 128, 1)       ->  249856

    Двойка в формуле обязательна: GQA хранит отдельно K и отдельно V.
    Забыть её — ровно вдвое занизить память и решить, что 128k контекста
    влезет туда, куда не влезет.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_element


def expert_params(hidden, intermediate):
    """Сколько весов в одном SwiGLU-MLP.

    expert_params(7168, 2048)   ->  44040192    (один эксперт DeepSeek-V3)
    expert_params(7168, 18432)  ->  396361728   (dense MLP первых слоёв)

    SwiGLU — это ТРИ матрицы: gate (hidden -> intermediate),
    up (hidden -> intermediate), down (intermediate -> hidden). Считать за
    две — классическая ошибка, она занижает MoE-блок на треть.
    """
    return 3 * hidden * intermediate


def attention_params(hidden, num_heads, head_dim, kv_lora_rank):
    """Сколько весов в одном блоке MLA.

    attention_params(7168, 128, 128, 512)  ->  255328256

    Четыре матрицы:
      W_q     hidden        -> num_heads * head_dim
      W_dkv   hidden        -> kv_lora_rank          (сжатие в латент)
      W_ukv   kv_lora_rank  -> 2 * num_heads * head_dim  (разжатие K и V)
      W_o     num_heads * head_dim -> hidden

    Двойка у W_ukv — потому что из одного латента разворачиваются и K, и V.
    Обрати внимание: сжатие-разжатие добавляет ВЕСОВ, а экономит ПАМЯТЬ
    под кеш. Это размен, а не бесплатный обед.
    """
    inner = num_heads * head_dim
    return hidden * inner + hidden * kv_lora_rank + kv_lora_rank * 2 * inner + inner * hidden


def model_parameters(config):
    """Всего весов, активных весов и доля активных. Словарь из трёх ключей.

    model_parameters(DEEPSEEK_V3)["total"]      ->  примерно 6.7e11
    model_parameters(DEEPSEEK_V3)["sparsity"]   ->  примерно 0.06

    Что складывается:
      * эмбеддинг vocab_size * hidden_size — считаем один раз;
      * attention на КАЖДОМ слое — включается всегда, значит активен весь;
      * первые first_k_dense_layers слоёв — обычный MLP размера
        intermediate_size, тоже активен весь;
      * остальные слои — (num_experts + shared_experts) экспертов в total,
        но лишь (num_experts_per_tok + shared_experts) в active.

    "sparsity" — это active / total. У DeepSeek-V3 около 6%: за счёт
    роутинга модель на 671B весов считает как модель на ~40B.

    Использует expert_params и attention_params.
    """
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    dense_layers = config["first_k_dense_layers"]
    moe_layers = layers - dense_layers

    attn = attention_params(
        hidden, config["num_attention_heads"], config["head_dim"], config["kv_lora_rank"]
    )
    dense_mlp = expert_params(hidden, config["intermediate_size"])
    one_expert = expert_params(hidden, config["moe_intermediate_size"])

    embedding = config["vocab_size"] * hidden
    # общая часть: она одинакова в total и в active, роутинг её не трогает
    shared = embedding + layers * attn + dense_layers * dense_mlp

    moe_total = (config["num_experts"] + config["shared_experts"]) * one_expert
    moe_active = (config["num_experts_per_tok"] + config["shared_experts"]) * one_expert

    total = shared + moe_layers * moe_total
    active = shared + moe_layers * moe_active
    return {"total": total, "active": active, "sparsity": active / total}


def route_topk(logits, k, bias=None):
    """Какие k экспертов берут токен. Индексы, от лучшего к худшему.

    route_topk([0.1, 0.9, 0.5], 2)                  ->  [1, 2]
    route_topk([1.0, 1.0, 0.0], 2)                  ->  [0, 1]
    route_topk([0.1, 0.9, 0.5], 2, [1.0, 0.0, 0.0]) ->  [0, 1]

    bias складывается с логитами ТОЛЬКО для выбора. Это и есть
    auxiliary-loss-free балансировка: смещение двигает, кому достанется
    токен, но не участвует в loss и не портит основную задачу.

    Ничьи разруливаются меньшим индексом — иначе тест на воспроизводимость
    начнёт мигать. k вне 1..len(logits) — ValueError.
    """
    if not 1 <= k <= len(logits):
        raise ValueError(f"k={k} не помещается в {len(logits)} экспертов")
    scored = [(logits[i] + (bias[i] if bias else 0.0), i) for i in range(len(logits))]
    # сортируем по убыванию score, при равенстве — по возрастанию индекса
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [i for _, i in scored[:k]]


def expert_load(rows, num_experts, k, bias=None):
    """Сколько токенов достанется каждому эксперту. Список длины num_experts.

    expert_load([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], 3, 1)  ->  [1, 0, 1]
    expert_load([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], 3, 2)  ->  [1, 2, 1]

    rows — по строке логитов на токен. Сумма нагрузок всегда равна
    len(rows) * k: каждый токен уходит ровно k экспертам.

    Неравномерная нагрузка — главная болезнь MoE: перегруженный эксперт
    становится узким местом шага, а недогруженный просто не учится.

    Использует route_topk.
    """
    load = [0] * num_experts
    for row in rows:
        for expert in route_topk(row, k, bias):
            load[expert] += 1
    return load


def balance_bias_step(bias, load, lr):
    """Шаг балансировки без вспомогательного loss. Новый список bias.

    balance_bias_step([0.0, 0.0, 0.0], [10, 2, 0], 0.1)  ->  [-0.1, 0.1, 0.1]
    balance_bias_step([0.0, 0.0], [5, 5], 0.1)           ->  [0.0, 0.0]

    Правило DeepSeek-V3: перегружен относительно средней нагрузки — bias
    вниз, недогружен — вверх, ровно на средней — не трогаем. Шаг
    фиксированный (по знаку), а не пропорциональный перекосу: так
    балансировка не раскачивается на редких выбросах.

    Возвращай НОВЫЙ список: править входной bias на месте нельзя, иначе
    сравнить «до» и «после» уже не получится.
    """
    if not load:
        return list(bias)
    target = sum(load) / len(load)
    updated = []
    for b, value in zip(bias, load):
        if value > target:
            updated.append(b - lr)
        elif value < target:
            updated.append(b + lr)
        else:
            updated.append(b)
    return updated
