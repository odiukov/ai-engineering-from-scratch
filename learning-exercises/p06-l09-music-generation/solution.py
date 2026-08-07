"""
Генерация музыки — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def midi_to_hz(note):
    """MIDI-нота → частота в герцах. Символьная запись превращается в звук.

    midi_to_hz(69)  ->  440.0    (нота A4, опорная точка всего строя)
    midi_to_hz(81)  ->  880.0    (октавой выше — ровно вдвое)
    midi_to_hz(57)  ->  220.0    (октавой ниже — ровно вдвое меньше)

    Формула: 440 * 2 ** ((note - 69) / 12). Двенадцать полутонов в октаве,
    и каждый полутон — умножение на 2 ** (1/12), а не прибавление герц.
    Ловушка: слух логарифмичен, поэтому шкала здесь тоже логарифмическая.

    MusicGen и ACE-Step генерируют codec-токены, но условие по мелодии
    (melody conditioning) приходит символьно — вот отсюда и мост.
    """
    # 69 — это A4; всё остальное считается относительно неё
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def chroma_vector(notes):
    """Chromagram: 12-мерный вектор классов высоты, сумма компонент равна 1.

    chroma_vector([60])          ->  [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    chroma_vector([60, 72])      ->  то же самое: обе ноты — до, разные октавы
    chroma_vector([60, 64])      ->  0.5 в индексе 0 и 0.5 в индексе 4

    Класс высоты — это note % 12. Октава при этом СТИРАЕТСЯ: в том и смысл,
    что chromagram описывает мелодию, а не регистр, в котором её сыграли.

    Пустой список нот — ValueError: нормировать нечего.

    Это ровно то, что MusicGen-melody принимает на вход в
    generate_with_chroma: тембр он придумает сам, а мелодию возьмёт отсюда.
    """
    if not notes:
        raise ValueError("notes must not be empty")
    counts = [0.0] * 12
    for n in notes:
        # % 12 у Python корректен и для отрицательных нот, поэтому руками
        # выправлять знак не нужно
        counts[n % 12] += 1.0
    return [c / len(notes) for c in counts]


def sample_token(logits, rng, temperature=1.0, top_k=None):
    """Один шаг авторегрессивного декодера: логиты → индекс токена.

    sample_token([0.0, 5.0], rng, temperature=0)        ->  1   (argmax)
    sample_token([0.0, 5.0, 1.0], rng, top_k=1)         ->  1   (выбора нет)

    Шаги: оставить top_k самых больших логитов, поделить на temperature,
    посчитать softmax, разыграть по rng.random().

    Ловушки:
      * temperature == 0 — это НЕ деление на ноль, а жадный выбор argmax;
        при равенстве логитов берётся меньший индекс;
      * math.exp(1000) бросает OverflowError. Перед экспонентой вычитай
        максимум логита — сумма от этого не меняется, а переполнение уходит.

    Отрицательная temperature, top_k вне 1..len(logits), пустые логиты —
    ValueError. Случайность приходит параметром rng, чтобы генерация была
    воспроизводимой.
    """
    if not logits:
        raise ValueError("logits must not be empty")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    if top_k is not None and not (1 <= top_k <= len(logits)):
        raise ValueError("top_k must be between 1 and len(logits)")

    # ключ (-логит, индекс) — детерминированный разрыв ничьих в пользу
    # меньшего индекса
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    keep = sorted(order[: top_k if top_k is not None else len(logits)])

    if temperature == 0:
        return order[0]

    shift = max(logits[i] for i in keep)  # <- без этой строки exp переполнится
    weights = [math.exp((logits[i] - shift) / temperature) for i in keep]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in zip(keep, weights):
        acc += w
        if r < acc:
            return i
    return keep[-1]  # страховка от накопленной ошибки float


def generate_tokens(prompt, model, n, rng, temperature=1.0, top_k=None):
    """Авторегрессивная генерация n токенов: модель видит всё, что уже выдала.

    generate_tokens([0], model, 3, rng, temperature=0)  ->  [1, 2, 3]

    model — заглушка вместо MusicGen: принимает список токенов-контекста,
    возвращает список логитов. Каждый выданный токен дописывается в контекст
    и участвует в следующем шаге. Возвращаются ТОЛЬКО новые токены, без
    промпта.

    Ловушка: prompt менять нельзя — вызывающий передал свой список, и он
    должен остаться прежним. Делай копию.

    Это и есть «AR-модель над codec-токенами»: EnCodec-декодер потом
    превратит эту последовательность обратно в 32 кГц звук.
    """
    context = list(prompt)  # копия: чужой список не наш, чтобы его портить
    out = []
    for _ in range(n):
        token = sample_token(model(context), rng, temperature, top_k)
        context.append(token)
        out.append(token)
    return out


def repetition_rate(tokens, window=4):
    """Доля окон длины window, которые уже встречались раньше. Детектор зацикливания.

    repetition_rate([1, 2, 3, 4, 5], window=2)           ->  0.0
    repetition_rate([1, 2, 1, 2, 1, 2], window=2)        ->  0.6

    Берём все окна tokens[i:i+window], i от 0 до len(tokens) - window.
    Окно считается повтором, если точно такое уже попадалось левее.
    Ответ всегда лежит в [0, 1].

    window меньше 1 или больше длины последовательности — ValueError.

    Зачем: AR-модели после 30 секунд начинают гонять один и тот же мотив.
    Числовой порог по этой метрике — дешёвый регрессионный тест на «поплыло».
    """
    if window < 1:
        raise ValueError("window must be positive")
    if window > len(tokens):
        raise ValueError("window must not exceed the sequence length")
    seen = set()
    repeats = 0
    total = len(tokens) - window + 1
    for i in range(total):
        gram = tuple(tokens[i : i + window])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / total


def crossfade(a, b, n):
    """Склеить две генерации через линейный кроссфейд длиной n сэмплов.

    crossfade([1.0, 1.0], [1.0, 1.0], 1)  ->  [1.0, 1.0, 1.0]
    crossfade([0.0, 0.0], [0.0, 0.0], 0)  ->  [0.0, 0.0, 0.0, 0.0]

    Хвост a длиной n накладывается на голову b: вес i-го сэмпла окна равен
    w = (i + 1) / (n + 1), результат = a_хвост * (1 - w) + b_голова * w.
    Длина ответа — len(a) + len(b) - n.

    Веса в сумме дают 1, поэтому постоянный сигнал остаётся постоянным —
    именно это и убирает щелчок на стыке.

    n отрицательное или больше длины любого из кусков — ValueError.

    Зачем: MusicGen выдыхается на 30 секундах, длинный трек собирают из
    нескольких генераций, и стык нельзя делать встык.
    """
    if n < 0:
        raise ValueError("n must not be negative")
    if n > len(a) or n > len(b):
        raise ValueError("crossfade is longer than one of the clips")
    if n == 0:
        return list(a) + list(b)
    out = list(a[: len(a) - n])
    for i in range(n):
        w = (i + 1) / (n + 1)
        out.append(a[len(a) - n + i] * (1 - w) + b[i] * w)
    out.extend(b[n:])
    return out


def fad(real, fake):
    """Fréchet Audio Distance, диагональное приближение. Меньше — лучше.

    fad([[0.0], [2.0]], [[0.0], [2.0]])  ->  0.0   (одно и то же распределение)
    fad([[0.0], [0.0]], [[3.0], [3.0]])  ->  9.0   (сдвиг среднего на 3)

    Считаем по каждой размерности отдельно: (mu1 - mu2) ** 2 + v1 + v2 -
    2 * sqrt(v1 * v2), где v — дисперсия (делим на число примеров), и всё
    складываем.

    Настоящая FAD берёт ПОЛНУЮ ковариационную матрицу и её матричный корень
    (frechet_audio_distance поверх VGGish). Здесь ковариация считается
    диагональной — корреляции между размерностями теряются, зато формула
    влезает в стандартную библиотеку.

    Меньше двух примеров в любом наборе или разная размерность — ValueError:
    дисперсию не по чему считать.
    """
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("each set needs at least two embeddings")
    dim = len(real[0])
    if any(len(v) != dim for v in real) or any(len(v) != dim for v in fake):
        raise ValueError("all embeddings must have the same dimension")

    def stats(vectors):
        mu = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]
        var = [
            sum((v[d] - mu[d]) ** 2 for v in vectors) / len(vectors)
            for d in range(dim)
        ]
        return mu, var

    mu_r, var_r = stats(real)
    mu_f, var_f = stats(fake)
    return sum(
        (mu_r[d] - mu_f[d]) ** 2
        + var_r[d]
        + var_f[d]
        - 2 * math.sqrt(var_r[d] * var_f[d])
        for d in range(dim)
    )


def is_prompt_blocked(prompt, blocked):
    """Фильтр «в стиле такого-то»: есть ли в промпте имя из чёрного списка.

    is_prompt_blocked("song in the style of Taylor Swift", ["Taylor Swift"])  ->  True
    is_prompt_blocked("SONG LIKE taylor swift", ["Taylor Swift"])             ->  True
    is_prompt_blocked("queensland ambient loop", ["Queen"])                   ->  False

    Сравнение регистронезависимое и ПОСЛОВНОЕ: имя из списка должно совпасть
    с непрерывной цепочкой слов промпта целиком.

    Ловушка: наивная проверка `name.lower() in prompt.lower()` находит
    «Queen» внутри «queensland» и блокирует невиновный промпт. Режь текст на
    слова по не-буквенно-цифровым символам и сравнивай списки слов.

    Пустой чёрный список — ничего не блокируется, False.

    Suno и Udio такой фильтр держат сами (после иска Warner на $500M),
    открытые модели — нет, поэтому список ведёшь ты.
    """

    def words(text):
        out, cur = [], []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return out

    haystack = words(prompt)
    for name in blocked:
        needle = words(name)
        if not needle:
            continue
        # обычное подсписковое совпадение: длины небольшие, оптимизировать нечего
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return True
    return False
