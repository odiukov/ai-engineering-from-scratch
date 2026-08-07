"""
BLIP-2 и Q-Former как мост между модальностями

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l03-blip2-qformer-bridge
Разбор:  /check-code p12-l03-blip2-qformer-bridge
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def top_patches_per_query(attn, k=3):
    """Для каждого запроса — индексы k патчей с наибольшим весом внимания.

    Внутри строки индексы идут по убыванию веса; при равных весах первым
    остаётся меньший индекс.

    top_patches_per_query([[0.1, 0.7, 0.2]], k=2)  ->  [[1, 2]]

    Это отладочный инструмент: если все 32 запроса тянут из одних и тех же
    трёх патчей, значит они схлопнулись и сжатие ничего не даёт.

    k <= 0 или k больше числа патчей — ValueError.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
