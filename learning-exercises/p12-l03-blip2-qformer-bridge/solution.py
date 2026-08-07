"""
BLIP-2 и Q-Former как мост между модальностями — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(xs):
    """Превратить список чисел в распределение: неотрицательные, сумма 1.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([2.0, 1.0, 0.0])  ->  [0.6652, 0.2447, 0.0900]

    Ловушка: math.exp(1000) — OverflowError. Вычти максимум перед exp,
    результат от этого не меняется (softmax сдвиго-инвариантен), а
    переполнение исчезает.

    Пустой список — ValueError.
    """
    if not xs:
        raise ValueError("softmax of an empty list is undefined")
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    total = sum(e)
    return [x / total for x in e]


def scaled_dot_attention(query, keys, values):
    """Одна голова внимания для ОДНОГО запроса. Вернуть (context, weights).

    scores[j] = dot(query, keys[j]) / sqrt(dim_key)
    weights   = softmax(scores)
    context   = сумма weights[j] * values[j]

    scaled_dot_attention([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]])
        ->  ([0.6698, 0.3302], [0.6698, 0.3302])

    Деление на sqrt(dim_key) — не украшение: без него при большой
    размерности скалярные произведения растут как sqrt(dim), softmax
    насыщается и градиент умирает.

    context — выпуклая комбинация values: он не может выйти за их диапазон
    ни по одной координате. Это хорошая самопроверка.

    Разное число keys и values, пустые списки, несовпадение размерностей
    query и key — ValueError.
    """
    if not keys or not values:
        raise ValueError("keys and values must not be empty")
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    dim_key = len(query)
    if any(len(k) != dim_key for k in keys):
        raise ValueError("query and keys must have the same dim")
    scale = math.sqrt(dim_key)
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    weights = softmax(scores)
    dim_val = len(values[0])
    context = [0.0] * dim_val
    for w, v in zip(weights, values):
        if len(v) != dim_val:
            raise ValueError("all values must have the same dim")
        for d in range(dim_val):
            context[d] += w * v[d]
    return (context, weights)


def cross_attention(queries, keys, values):
    """Cross-attention для НАБОРА запросов. Вернуть (outputs, attn).

    outputs[i] — контекст i-го запроса, attn[i] — его веса по всем keys.

    Главное свойство, ради которого Q-Former вообще существует:
    len(outputs) == len(queries), сколько бы ни было keys. 256 патчей на
    входе, 32 запроса — на выходе 32 токена. Это и есть сжатие модальности.

    cross_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]],
                    [[4.0], [0.0]])  ->  ([[2.679]], [[0.6698, 0.3302]])

    Q берётся из обучаемых запросов, K и V — из замороженного ViT. В этом
    и разница между cross-attention и self-attention.

    Пустой список запросов — ValueError.
    """
    if not queries:
        raise ValueError("need at least one query")
    outputs, attn = [], []
    for q in queries:
        context, weights = scaled_dot_attention(q, keys, values)
        outputs.append(context)
        attn.append(weights)
    return (outputs, attn)


def linear_project(tokens, W, bias=None):
    """Линейная проекция каждого токена: out[d] = dot(W[d], token) + bias[d].

    W — список из out_dim строк длиной in_dim.

    linear_project([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ->  [[1.0, 2.0, 3.0]]

    В BLIP-2 это последний кусочек моста: 768-мерные выходы Q-Former
    проецируются в embedding-размерность LLM (4096 у OPT-6.7B) и просто
    подставляются в начало входной последовательности.

    Несовпадение размерностей — ValueError.
    """
    if not W:
        raise ValueError("W must not be empty")
    in_dim = len(W[0])
    if bias is not None and len(bias) != len(W):
        raise ValueError("bias length must match output dim")
    out = []
    for t in tokens:
        if len(t) != in_dim:
            raise ValueError("token length does not match W")
        row = []
        for d, w_row in enumerate(W):
            s = sum(w * x for w, x in zip(w_row, t))
            row.append(s + (bias[d] if bias is not None else 0.0))
        out.append(row)
    return out


def qformer_forward(patches, queries, W_proj, bias=None):
    """Весь мост целиком: патчи ViT -> обучаемые запросы -> токены для LLM.

    qformer_forward(patches_256, queries_32, W_4096x768)  ->  32 токена по 4096

    Три шага, и все три уже написаны выше:
      1. cross-attention: Q из queries, K и V из patches;
      2. взять выходы (веса внимания тут не нужны);
      3. linear_project в размерность LLM.

    Форма результата задаётся ЗАПРОСАМИ, а не картинкой: замени 256 патчей
    на 1024 — на выходе всё те же 32 токена. Именно поэтому Q-Former
    выигрывает там, где бюджет контекста жмёт (много кадров видео).
    """
    outputs, _attn = cross_attention(queries, patches, patches)
    return linear_project(outputs, W_proj, bias)


def top_patches_per_query(attn, k=3):
    """Для каждого запроса — индексы k патчей с наибольшим весом внимания.

    Внутри строки индексы идут по убыванию веса; при равных весах первым
    остаётся меньший индекс.

    top_patches_per_query([[0.1, 0.7, 0.2]], k=2)  ->  [[1, 2]]

    Это отладочный инструмент: если все 32 запроса тянут из одних и тех же
    трёх патчей, значит они схлопнулись и сжатие ничего не даёт.

    k <= 0 или k больше числа патчей — ValueError.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    out = []
    for row in attn:
        if k > len(row):
            raise ValueError("k is larger than the number of patches")
        # сортируем по (-вес, индекс): минус на весе даёт убывание,
        # индекс вторым ключом делает порядок при ничьей детерминированным
        order = sorted(range(len(row)), key=lambda j: (-row[j], j))
        out.append(order[:k])
    return out


def visual_token_budget(num_images, patch_tokens, num_queries):
    """Сколько токенов контекста съест картинка при двух вариантах моста.

    Словарь с ключами mlp, qformer, compression:
      mlp         = num_images * patch_tokens     (LLaVA: все патчи в LLM)
      qformer     = num_images * num_queries      (BLIP-2: только запросы)
      compression = patch_tokens / num_queries    (во сколько раз короче)

    visual_token_budget(1, 256, 32)
        ->  {"mlp": 256, "qformer": 32, "compression": 8.0}

    Пример из урока: 60 кадров видео дают 34560 токенов через MLP-проектор
    и 1920 через Q-Former. Первое не влезает в 32k контекст, второе влезает
    с запасом.

    Любой неположительный аргумент — ValueError.
    """
    if num_images <= 0 or patch_tokens <= 0 or num_queries <= 0:
        raise ValueError("all counts must be positive")
    return {
        "mlp": num_images * patch_tokens,
        "qformer": num_images * num_queries,
        "compression": patch_tokens / num_queries,
    }


def pick_bridge(num_images, patch_tokens, num_queries, context_budget):
    """Выбрать мост под заданный бюджет контекста. Вернуть "mlp" или "qformer".

    Правило простое и намеренно жадное: MLP-проектор отдаёт LLM сырые патчи
    и потому качественнее на токен, поэтому берём его, если он ВЛЕЗАЕТ.
    Не влезает — отступаем к Q-Former. Не влезает и он — ValueError, потому
    что тихо отдать заведомо переполненный промпт хуже, чем упасть.

    pick_bridge(1, 256, 32, 4096)    ->  "mlp"
    pick_bridge(60, 576, 32, 32768)  ->  "qformer"

    Границу считаем нестрого: ровно заполненный контекст — это влезло.
    """
    budget = visual_token_budget(num_images, patch_tokens, num_queries)
    if budget["mlp"] <= context_budget:
        return "mlp"
    if budget["qformer"] <= context_budget:
        return "qformer"
    raise ValueError("neither bridge fits into the context budget")
