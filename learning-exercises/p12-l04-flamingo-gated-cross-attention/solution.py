"""
Flamingo и gated cross-attention — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

IMAGE = "image"
TEXT = "text"


def cross_attention(queries, keys, values):
    """Cross-attention: каждый запрос собирает взвешенную сумму values.

    Вернуть список выходов, по одному на запрос (веса наружу не отдаём —
    здесь они не нужны).

    scores[j] = dot(query, keys[j]) / sqrt(dim_key), дальше softmax, дальше
    взвешенная сумма values.

    cross_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[4.0], [0.0]])
        ->  [[2.679]]

    Ловушка: softmax наивным exp падает на больших логитах. Вычитай максимум.

    Пустые queries, keys или values, разное число keys и values — ValueError.
    """
    if not queries:
        raise ValueError("need at least one query")
    if not keys or not values:
        raise ValueError("keys and values must not be empty")
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")
    dim_key = len(keys[0])
    scale = math.sqrt(dim_key)
    dim_val = len(values[0])
    outputs = []
    for q in queries:
        if len(q) != dim_key:
            raise ValueError("query and keys must have the same dim")
        scores = [sum(a * b for a, b in zip(q, k)) / scale for k in keys]
        m = max(scores)  # сдвиг на максимум: тождество для softmax,
        e = [math.exp(s - m) for s in scores]  # спасение от OverflowError
        total = sum(e)
        out = [0.0] * dim_val
        for w, v in zip(e, values):
            for d in range(dim_val):
                out[d] += (w / total) * v[d]
        outputs.append(out)
    return outputs


def perceiver_resampler(patches, latents, blocks=1):
    """Сжать любое число патчей в фиксированное число латентов.

    Один блок = cross-attention латентов по патчам плюс residual:
        latents = latents + cross_attention(latents, patches, patches)
    Повторить blocks раз. Flamingo использует 6 блоков и 64 латента.

    perceiver_resampler(patches_900, latents_64)  ->  64 вектора
    perceiver_resampler(patches_196, latents_64)  ->  64 вектора

    В этом весь смысл: картинка 224x224 и картинка 480x480 выходят одной и
    той же длины, поэтому слой gated cross-attention в LLM всегда видит
    одну и ту же форму, сколько бы картинок ни было в промпте.

    В игрушечной версии латенты и патчи одной размерности — иначе residual
    не сложится. blocks=0 возвращает копию латентов, blocks<0 — ValueError.
    Входной список латентов не менять.
    """
    if blocks < 0:
        raise ValueError("blocks must be non-negative")
    current = [list(l) for l in latents]  # копия: параметры модели не портим
    for _ in range(blocks):
        attended = cross_attention(current, patches, patches)
        current = [[a + b for a, b in zip(c, at)] for c, at in zip(current, attended)]
    return current


def gated_residual(hidden, cross, alpha):
    """Затворённый residual Flamingo: out = hidden + tanh(alpha) * cross.

    gated_residual([[1.0, 2.0]], [[10.0, 10.0]], 0.0)  ->  [[1.0, 2.0]]

    Единственная по-настоящему важная строчка всей архитектуры. alpha
    инициализируется нулём, tanh(0) = 0, значит на шаге 0 новый слой —
    точный no-op, и замороженная LLM ведёт себя ровно как до вставки.
    Дальше alpha уезжает от нуля, и визуальный сигнал вливается плавно.

    Два следствия, которые стоит проверить:
      * при alpha = 0 равенство ТОЧНОЕ, а не приблизительное;
      * |tanh| <= 1, поэтому вклад никогда не превосходит сам cross —
        визуальная ветка не может затереть текстовое представление.

    Несовпадение форм hidden и cross — ValueError.
    """
    if len(hidden) != len(cross):
        raise ValueError("hidden and cross must have the same length")
    g = math.tanh(alpha)
    out = []
    for h, c in zip(hidden, cross):
        if len(h) != len(c):
            raise ValueError("hidden and cross vectors must have the same dim")
        out.append([a + g * b for a, b in zip(h, c)])
    return out


def gated_cross_attention_step(text_hidden, visual_tokens, alpha):
    """Полный вставной блок Flamingo между двумя слоями замороженной LLM.

    Текстовые скрытые состояния становятся запросами, визуальные токены —
    ключами и значениями, результат подмешивается через затвор.

    gated_cross_attention_step(hidden, visual, 0.0)   ->  hidden без изменений
    gated_cross_attention_step(hidden, visual, 2.0)   ->  hidden + 0.964*cross

    Обрати внимание, чего здесь НЕТ: входная последовательность LLM не
    меняется. Визуальные токены не подставляются в промпт, они живут сбоку
    и подмешиваются в скрытые состояния. Поэтому Flamingo умеет глотать
    сколько угодно картинок, не съедая контекст.
    """
    cross = cross_attention(text_hidden, visual_tokens, visual_tokens)
    return gated_residual(text_hidden, cross, alpha)


def most_recent_image(sequence):
    """Для каждой позиции — индекс ближайшей картинки слева (или None).

    sequence — список меток IMAGE и TEXT в порядке чтения.

    most_recent_image([TEXT, IMAGE, TEXT, IMAGE, TEXT])
        ->  [None, 0, 0, 1, 1]

    Индекс считается по картинкам, а не по позициям: первая картинка — 0,
    вторая — 1. Сама картинка «видит» себя.

    None у текста до первой картинки — не мелочь: это ровно те токены,
    которым визуальной информации ещё неоткуда взяться.

    Любая метка кроме IMAGE и TEXT — ValueError.
    """
    out = []
    seen = -1
    for kind in sequence:
        if kind == IMAGE:
            seen += 1
        elif kind != TEXT:
            raise ValueError(f"unknown token kind: {kind!r}")
        out.append(None if seen < 0 else seen)
    return out


def interleaved_cross_mask(sequence, tokens_per_image):
    """Маска cross-attention для перемешанной последовательности картинок и текста.

    Матрица len(sequence) x (число картинок * tokens_per_image) из True/False.
    True — этой позиции разрешено смотреть на этот визуальный токен.

    interleaved_cross_mask([IMAGE, TEXT, IMAGE, TEXT], 2)
        ->  [[T, T, F, F],
             [T, T, F, F],
             [F, F, T, T],
             [F, F, T, T]]

    Правило Flamingo: позиция видит ТОЛЬКО последнюю предшествующую
    картинку — не все предыдущие. Ограничение сознательное: так модель
    учится связывать подпись с ближайшей к ней картинкой, и few-shot
    примеры не сливаются в кашу.

    Позиции до первой картинки не видят ничего — целая строка False.
    tokens_per_image <= 0 — ValueError.
    """
    if tokens_per_image <= 0:
        raise ValueError("tokens_per_image must be positive")
    owners = most_recent_image(sequence)
    num_images = sum(1 for kind in sequence if kind == IMAGE)
    width = num_images * tokens_per_image
    mask = []
    for owner in owners:
        row = [False] * width
        if owner is not None:
            start = owner * tokens_per_image
            for j in range(start, start + tokens_per_image):
                row[j] = True
        mask.append(row)
    return mask


def build_few_shot_prompt(examples, query_image):
    """Собрать few-shot промпт Flamingo: пары (картинка, подпись) и запрос.

    examples — список пар (имя картинки, подпись). Вернуть список пар
    (метка, содержимое), где метка это IMAGE или TEXT.

    build_few_shot_prompt([("cat.jpg", "A photo of a cat.")], "bird.jpg")
        ->  [(IMAGE, "cat.jpg"), (TEXT, "A photo of a cat."), (IMAGE, "bird.jpg")]

    Хвост принципиален: последняя картинка идёт БЕЗ подписи. Именно
    незакрытый шаблон заставляет модель продолжить его — никаких
    градиентных шагов, чистое in-context обучение.

    Ноль примеров — это zero-shot, ровно одна картинка на выходе.
    Пустая подпись — ValueError: демонстрация без ответа ничему не учит.
    """
    prompt = []
    for name, caption in examples:
        if not caption or not caption.strip():
            raise ValueError("a few-shot example needs a non-empty caption")
        prompt.append((IMAGE, name))
        prompt.append((TEXT, caption))
    prompt.append((IMAGE, query_image))
    return prompt
