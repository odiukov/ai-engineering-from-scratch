"""
Vision-Language Models: паттерн ViT-MLP-LLM — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def gelu(x):
    """GELU — гладкая активация, которая стоит внутри projector-а VLM.

    gelu(0.0)   ->  0.0
    gelu(1.0)   ->  0.8413447460685429
    gelu(-1.0)  ->  -0.15865525393145707

    Точная формула: x * 0.5 * (1 + erf(x / sqrt(2))). `math.erf` есть в
    стандартной библиотеке, приближение через tanh здесь не нужно.

    Ловушка: GELU НЕ обнуляет отрицательные вход полностью, как ReLU. Слегка
    отрицательный вход даёт слегка отрицательный выход — именно эта «утечка»
    и делает градиент гладким.

    Соответствует torch.nn.GELU().
    """
    # erf уже в стандартной библиотеке, tanh-приближение нужно было только
    # ради скорости на GPU — здесь смысла в нём нет
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def linear(vectors, weights, bias):
    """Линейный слой: y = W @ x + b, применённый к каждому вектору списка.

    `weights` — список СТРОК: weights[i] это i-я строка длины in_dim, всего
    out_dim строк. Форма (out_dim, in_dim), ровно как torch.nn.Linear.weight.

    linear([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, 0.0])
        ->  [[1.0, 2.0, 3.0]]
    linear([[1.0], [2.0]], [[3.0]], [1.0])  ->  [[4.0], [7.0]]

    Ловушка: перепутанный порядок индексов. Одна строка weights — это ОДИН
    выходной нейрон, а не одна входная координата. Если длина вектора не
    совпадает с длиной строки — брось ValueError, молча обрезать нельзя.

    Соответствует torch.nn.functional.linear(x, W, b).
    """
    out_dim = len(weights)
    if out_dim != len(bias):
        raise ValueError("bias length must match number of weight rows")
    in_dim = len(weights[0]) if out_dim else 0
    result = []
    for v in vectors:
        if len(v) != in_dim:
            raise ValueError(f"vector of length {len(v)} does not fit in_dim={in_dim}")
        # zip по строке и вектору читается лучше, чем индексный цикл,
        # и не даёт перепутать местами out_dim и in_dim
        result.append([sum(wi * vi for wi, vi in zip(row, v)) + b
                       for row, b in zip(weights, bias)])
    return result


def projector_forward(tokens, w1, b1, w2, b2):
    """Projector целиком: Linear -> GELU -> Linear.

    Это тот самый «мост» из ViT-MLP-LLM: на входе патч-токены ViT формы
    (N, d_vit), на выходе такие же по счёту токены, но уже в пространстве
    эмбеддингов LLM — форма (N, d_llm).

    projector_forward([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0],
                      [[1.0, 1.0]], [0.0])
        ->  [[0.8413447460685429]]

    Число токенов projector НЕ меняет: сколько патчей дал ViT, столько токенов
    увидит LLM. Меняется только размерность каждого токена. (Q-former устроен
    иначе — он как раз сжимает N патчей в фиксированное число запросов.)

    Соответствует nn.Sequential(nn.Linear, nn.GELU, nn.Linear).
    """
    hidden = linear(tokens, w1, b1)
    activated = [[gelu(x) for x in row] for row in hidden]
    return linear(activated, w2, b2)


def count_projector_params(vit_dim, hidden, llm_dim):
    """Сколько обучаемых параметров в двухслойном projector-е (bias считаем).

    count_projector_params(768, 4096, 4096)  ->  19931136
    count_projector_params(2, 2, 1)          ->  9

    Слой Linear(in, out) весит in*out весов плюс out смещений.

    Зачем: на первой стадии обучения VLM ViT и LLM заморожены, учится ТОЛЬКО
    projector. Полезно понимать, что «дешёвая» стадия — это всё равно десятки
    миллионов параметров.
    """
    return vit_dim * hidden + hidden + hidden * llm_dim + llm_dim


def deepstack_concat(levels):
    """DeepStack: признаки с нескольких глубин ViT склеиваются по каналам.

    `levels` — список уровней. Каждый уровень это список из N токенов, токен —
    вектор своей размерности d_l. Результат: те же N токенов, каждый длиной
    d_1 + d_2 + ... + d_L. Порядок уровней сохраняется.

    deepstack_concat([[[1.0, 2.0]], [[3.0]]])  ->  [[1.0, 2.0, 3.0]]

    Идея урока: последний слой ViT знает «что на картинке» (семантика), ранние
    слои знают «где именно» (текстура и геометрия). Ванильный projector берёт
    только последний слой и теряет вторую половину.

    Ловушки: пустой список уровней и разное число токенов на уровнях — оба
    случая это ValueError, склеивать нечего.
    """
    if not levels:
        raise ValueError("deepstack needs at least one level")
    n_tokens = len(levels[0])
    for lvl in levels:
        if len(lvl) != n_tokens:
            raise ValueError("all levels must have the same number of tokens")
    # идём по токенам снаружи, по уровням внутри: порядок каналов должен быть
    # «весь уровень 1, затем весь уровень 2», а не чересполосица
    return [[value for lvl in levels for value in lvl[i]] for i in range(n_tokens)]


def cosine_similarity(a, b):
    """Косинус угла между векторами: мера согласованности картинки и текста.

    cosine_similarity([1.0, 0.0], [1.0, 0.0])   ->  1.0
    cosine_similarity([1.0, 0.0], [0.0, 1.0])   ->  0.0
    cosine_similarity([1.0, 0.0], [-1.0, 0.0])  ->  -1.0

    Длина векторов на ответ не влияет — только направление. Именно поэтому
    CLIP-подобные проверки нормализуют эмбеддинги перед скалярным произведением.

    Ловушки: разная длина векторов и нулевой вектор (у него нет направления) —
    оба случая ValueError.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def cross_modal_error_rate(image_embs, text_embs, confidences,
                           sim_threshold=0.25, conf_threshold=0.8):
    """CMER — доля ответов, где модель уверена, но картинку не подтверждает.

    Считаем ответ ошибочным, когда одновременно
      confidence > conf_threshold   (модель говорит уверенно)
      similarity < sim_threshold    (текст плохо согласован с изображением)

    cross_modal_error_rate([[1.0, 0.0], [1.0, 0.0]],
                           [[1.0, 0.0], [0.0, 1.0]],
                           [0.9, 0.9])              ->  0.5
    cross_modal_error_rate([], [], [])              ->  0.0

    Смысл метрики: это KPI галлюцинаций в проде. Высокий CMER не чинят
    «улучшением модели» — высокие CMER-ответы отправляют на ручную проверку.

    Ловушки: строгие неравенства (ровно на пороге — не ошибка), пустой вход
    даёт 0.0, а не деление на ноль. Разная длина трёх списков — ValueError.
    """
    if not (len(image_embs) == len(text_embs) == len(confidences)):
        raise ValueError("image_embs, text_embs and confidences must be the same length")
    if not confidences:
        return 0.0
    bad = 0
    for img, txt, conf in zip(image_embs, text_embs, confidences):
        if conf > conf_threshold and cosine_similarity(img, txt) < sim_threshold:
            bad += 1
    return bad / len(confidences)


def merge_image_tokens(text_embeds, vision_embeds, input_ids, image_token_id):
    """Подставить эмбеддинги картинки на места плейсхолдеров <image> в тексте.

    text_embeds    — список из M эмбеддингов текстовых токенов
    input_ids      — список из M id токенов, у части id == image_token_id
    vision_embeds  — список из N эмбеддингов, вышедших из projector-а

    merge_image_tokens([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
                       [[9.0, 9.0]], [5, 32000, 7], 32000)
        ->  [[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]]

    Плейсхолдеры могут стоять где угодно и вперемешку с текстом — заполняй их
    в порядке слева направо.

    Ловушки: число плейсхолдеров обязано совпадать с числом vision-токенов,
    иначе ValueError (в проде это самая частая ошибка при батчинге — сэмплы
    добиты до разного числа <image>). И не мутируй text_embeds: он ещё нужен
    вызывающему коду.
    """
    positions = [i for i, tid in enumerate(input_ids) if tid == image_token_id]
    if len(positions) != len(vision_embeds):
        raise ValueError(
            f"prompt has {len(positions)} image tokens but vision_embeds has "
            f"{len(vision_embeds)} patches")
    if len(input_ids) != len(text_embeds):
        raise ValueError("input_ids and text_embeds must be the same length")
    # копия верхнего уровня + копия каждой строки: возвращаем полностью
    # независимый результат, чтобы вызывающий мог свободно его править
    merged = [list(row) for row in text_embeds]
    for pos, vec in zip(positions, vision_embeds):
        merged[pos] = list(vec)
    return merged
