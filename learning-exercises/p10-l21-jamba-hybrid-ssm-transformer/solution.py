"""
Jamba: гибрид SSM и трансформера — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок про то, как AI21 смешала два вида слоёв в одной модели: 7 слоёв Mamba
на 1 слой attention, MoE через слой, 256k контекста на одной 80-гигабайтной
карте. Считаем две вещи — саму рекуррентность SSM и бюджет памяти при
инференсе.

Модель, в которой мы работаем:

  * SSM здесь СКАЛЯРНЫЙ: h_t = a * h_{t-1} + b * x_t, y_t = c * h_t.
    В настоящей Mamba a, b, c — матрицы, да ещё и зависящие от входа
    («selective»). Скаляр сохраняет главное свойство — состояние
    фиксированного размера — и позволяет увидеть затухание памяти
    руками, без линейной алгебры;
  * ПЛАН СЛОЁВ — список кортежей (kind, uses_moe), где kind это "mamba"
    или "attention". План и есть архитектура: из него выводится и KV-кеш,
    и размер состояния SSM;
  * ПАМЯТЬ считается в байтах по формулам, тензоров нет.

Главное, что должно вылезти из тестов: KV-кеш растёт вместе с длиной
контекста, а состояние SSM — нет. Отсюда и «256k на одной карте».
"""


def ssm_step(h, x, a, b):
    """Один шаг рекуррентности SSM: новое состояние.

    ssm_step(0.0, 1.0, 0.9, 0.5)  ->  0.5
    ssm_step(2.0, 1.0, 0.9, 0.5)  ->  2.3

    h_next = a * h + b * x. Ровно это заменяет attention в слое Mamba:
    чтобы посчитать следующий шаг, нужны только текущее состояние и
    текущий вход. Ни одного прошлого токена в памяти держать не надо —
    поэтому инференс идёт за O(1) на токен, а не за O(длина контекста).
    """
    return a * h + b * x


def ssm_scan(xs, a, b, c, h0=0.0):
    """Прогон SSM по всей последовательности. Список выходов y_t.

    ssm_scan([1.0, 0.0, 0.0, 0.0], 0.5, 1.0, 1.0)  ->  [1.0, 0.5, 0.25, 0.125]
    ssm_scan([1.0, 2.0, 3.0], 1.0, 1.0, 1.0)       ->  [1.0, 3.0, 6.0]
    ssm_scan([1.0, 2.0, 3.0], 0.0, 1.0, 2.0)       ->  [2.0, 4.0, 6.0]

    Первый пример — импульсный отклик: подали единицу и смотрим, как долго
    её помнят. При a = 0.5 память тает вдвое за шаг. При a = 1 состояние
    становится накопительной суммой — модель помнит всё. При a = 0 памяти
    нет вовсе, слой вырождается в поэлементное умножение.

    h0 позволяет продолжить прогон с уже накопленного состояния: скорми
    первую половину, забери h, скорми вторую — результат тот же, что за
    один проход. На этом и держится потоковый инференс.

    Использует ssm_step.
    """
    h = h0
    out = []
    for x in xs:
        h = ssm_step(h, x, a, b)
        out.append(c * h)
    return out


def layer_plan(num_layers, attn_ratio=8, moe_every=2):
    """План слоёв гибрида: список кортежей (kind, uses_moe).

    layer_plan(8)             ->  семь ('mamba', ...) и восьмой ('attention', True)
    layer_plan(4, 1)[0]       ->  ('attention', False)
    layer_plan(4, 0)[3]       ->  ('mamba', True)

    Рецепт Jamba: каждый attn_ratio-й слой — attention, остальные Mamba;
    каждый moe_every-й слой считает MLP через MoE. Нумерация с нуля, поэтому
    условие смотрит на (i + 1).

    attn_ratio = 1 даёт чистый трансформер, attn_ratio = 0 — чистый SSM
    (attention нет вообще). Отрицательные значения бессмысленны — ValueError,
    как и moe_every < 1.
    """
    if attn_ratio < 0:
        raise ValueError("attn_ratio не может быть отрицательным")
    if moe_every < 1:
        raise ValueError("moe_every должен быть не меньше 1")
    plan = []
    for i in range(num_layers):
        is_attention = attn_ratio > 0 and (i + 1) % attn_ratio == 0
        plan.append(("attention" if is_attention else "mamba", (i + 1) % moe_every == 0))
    return plan


def count_layer_types(plan):
    """Сколько слоёв какого вида. Словарь с ключами attention, mamba, moe.

    count_layer_types(layer_plan(32))  ->  {'attention': 4, 'mamba': 28, 'moe': 16}
    count_layer_types([])              ->  {'attention': 0, 'mamba': 0, 'moe': 0}

    Ключ 'moe' считает слои с MoE ОТДЕЛЬНО: MoE — это про MLP, он
    навешивается и на Mamba, и на attention. Сумма attention и mamba даёт
    длину плана, а moe в неё не входит.
    """
    counts = {"attention": 0, "mamba": 0, "moe": 0}
    for kind, uses_moe in plan:
        counts[kind] += 1
        if uses_moe:
            counts["moe"] += 1
    return counts


def kv_cache_bytes(plan, num_kv_heads, head_dim, seq_len, bytes_per_element=2):
    """Размер KV-кеша по плану слоёв, в байтах.

    kv_cache_bytes(layer_plan(32), 32, 128, 262144)  ->  17179869184   (16 GiB)
    kv_cache_bytes(layer_plan(32, 1), 32, 128, 262144)  ->  137438953472  (128 GiB)

    Платят ТОЛЬКО слои attention: у Mamba кеша нет вообще. Формула на слой
    привычная: 2 (отдельно K, отдельно V) * головы * head_dim * длина
    контекста * байт на число.

    Отсюда и весь фокус Jamba: слоёв 32, а платящих — 4.

    Использует count_layer_types.
    """
    attention_layers = count_layer_types(plan)["attention"]
    return 2 * attention_layers * num_kv_heads * head_dim * seq_len * bytes_per_element


def ssm_state_bytes(plan, hidden, state_size=16, bytes_per_element=2):
    """Размер состояния всех слоёв Mamba, в байтах.

    ssm_state_bytes(layer_plan(32), 4096)  ->  3670016   (3.5 MiB)
    ssm_state_bytes(layer_plan(32, 1), 4096)  ->  0

    Обрати внимание, чего в сигнатуре НЕТ: длины контекста. Состояние SSM
    фиксировано — hidden * state_size чисел на слой, хоть 1k токенов, хоть
    1M. Это и есть «constant memory» из статьи.

    Использует count_layer_types.
    """
    mamba_layers = count_layer_types(plan)["mamba"]
    return mamba_layers * hidden * state_size * bytes_per_element


def inference_memory(plan, hidden, num_kv_heads, head_dim, seq_len,
                     state_size=16, bytes_per_element=2):
    """Весь бюджет памяти под контекст. Словарь kv, ssm, total.

    inference_memory(layer_plan(32), 4096, 32, 128, 262144)["total"]
        ->  17183539200

    Складывает KV-кеш слоёв attention и состояние слоёв Mamba. Про веса
    модели тут ничего нет — только то, что зависит от длины контекста
    (или демонстративно от неё не зависит).

    Использует kv_cache_bytes и ssm_state_bytes.
    """
    kv = kv_cache_bytes(plan, num_kv_heads, head_dim, seq_len, bytes_per_element)
    ssm = ssm_state_bytes(plan, hidden, state_size, bytes_per_element)
    return {"kv": kv, "ssm": ssm, "total": kv + ssm}


def kv_cache_advantage(num_layers, attn_ratio, num_kv_heads, head_dim, seq_len,
                       bytes_per_element=2):
    """Во сколько раз гибрид экономит KV-кеш против чистого трансформера.

    kv_cache_advantage(32, 8, 32, 128, 262144)  ->  8.0
    kv_cache_advantage(32, 1, 32, 128, 262144)  ->  1.0

    Считает два плана — attn_ratio=1 (каждый слой attention) и заданный —
    и делит их KV-кеши. Ответ не зависит ни от длины контекста, ни от числа
    голов: они сокращаются. По сути это просто отношение числа attention-слоёв.

    Чистый SSM (attn_ratio = 0) даёт кеш ноль, делить на него нельзя —
    ValueError вместо ZeroDivisionError.

    Использует layer_plan и kv_cache_bytes.
    """
    hybrid = kv_cache_bytes(
        layer_plan(num_layers, attn_ratio), num_kv_heads, head_dim, seq_len, bytes_per_element
    )
    if hybrid == 0:
        raise ValueError("в плане нет ни одного слоя attention: делить не на что")
    pure = kv_cache_bytes(
        layer_plan(num_layers, 1), num_kv_heads, head_dim, seq_len, bytes_per_element
    )
    return pure / hybrid
