"""
Show-o и masked discrete diffusion — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Модель урока: последовательность из SEQ_LEN дискретных VQ-токенов, где MASK
означает «ещё не раскрыт». Сэмплер MaskGIT/Show-o работает так: на каждом шаге
transformer предсказывает ВСЕ замаскированные позиции сразу, мы фиксируем
top-K самых уверенных, остальные оставляем под маской и повторяем. За T шагов
картинка собирается целиком.

Настоящего transformer здесь нет: везде, где он нужен, функция принимает
callable predict(tokens) -> logits. Так тест подставляет детерминированную
заглушку и считает, сколько раз её позвали — а это и есть главное число
урока: проходов ровно T, а не по одному на токен.

Только стандартная библиотека.
"""

import math

# Специальный id «позиция ещё замаскирована». В настоящем Show-o это
# отдельная запись словаря, тут хватит отрицательного числа.
MASK = -1


def cosine_schedule(T):
    """Косинусное расписание маски: T + 1 долей от 1.0 до 0.0.

    cosine_schedule(2)  ->  [1.0, 0.7071..., 0.0]

    Формула урока: mask_ratio(t) = cos(pi * t / (2 * T)) для t = 0..T.
    На нулевом шаге замаскировано всё, на T-м — ничего.

    Косинус держит долю маски высокой дольше линейного: основная масса
    раскрытий приходится на конец, когда контекста уже много.

    T <= 0 — ValueError: расписания из нуля шагов не бывает.
    """
    if T <= 0:
        raise ValueError(f"T должно быть положительным, получено {T}")
    return [math.cos(math.pi * t / (2 * T)) for t in range(T + 1)]


def linear_schedule(T):
    """Линейное расписание маски: T + 1 долей, равномерно от 1.0 до 0.0.

    linear_schedule(4)  ->  [1.0, 0.75, 0.5, 0.25, 0.0]

    Нужно для сравнения: линейное раскрывает поровну на каждом шаге,
    косинусное — по нарастающей.
    """
    if T <= 0:
        raise ValueError(f"T должно быть положительным, получено {T}")
    return [1.0 - t / T for t in range(T + 1)]


def unmask_counts(ratios, n_tokens):
    """Сколько токенов раскрывается на каждом шаге. Длина — len(ratios) - 1.

    unmask_counts([1.0, 0.5, 0.0], 8)  ->  [4, 4]
    unmask_counts([1.0, 0.0], 8)       ->  [8]

    Переводим доли в целые числа замаскированных (round от ratio * n_tokens)
    и берём разности соседних. Сумма разностей равна числу токенов, которые
    расписание обязано раскрыть.

    Ловушка: расписание обязано не возрастать. Если ratios где-то растёт,
    получится отрицательное число раскрытий — это не «замаскировать обратно»,
    а сломанное расписание. Бросай ValueError.
    """
    if len(ratios) < 2:
        raise ValueError("в расписании должно быть минимум два значения")
    if n_tokens < 0:
        raise ValueError(f"n_tokens не может быть отрицательным, получено {n_tokens}")
    # округляем именно число токенов, а не долю: иначе накопленная ошибка
    # округления уводит сумму от n_tokens
    masked = [int(round(r * n_tokens)) for r in ratios]
    counts = []
    for prev, cur in zip(masked, masked[1:]):
        if cur > prev:
            raise ValueError("расписание маски обязано не возрастать")
        counts.append(prev - cur)
    return counts


def softmax(logits):
    """Логиты -> распределение вероятностей.

    softmax([0.0, 0.0])  ->  [0.5, 0.5]
    softmax([1000.0, 0.0]) не должен падать с OverflowError

    Вычитай максимум перед exp: математически это ничего не меняет
    (множитель сокращается), численно спасает от переполнения.
    """
    if not logits:
        raise ValueError("пустые логиты")
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def top_k_confident(tokens, logits, k):
    """Индексы k самых уверенных ЗАМАСКИРОВАННЫХ позиций, по возрастанию.

    top_k_confident([MASK, 5, MASK], [[2.0, 0.0], [0.0, 0.0], [9.0, 0.0]], 1)
        ->  [2]

    Уверенность позиции — максимум её softmax-распределения. Уже раскрытые
    позиции не участвуют вовсе: они зафиксированы.

    При равной уверенности выигрывает меньший индекс — иначе сэмплер
    перестанет быть воспроизводимым.

    Если k больше числа замаскированных позиций, возвращаем их все.
    k <= 0 — ValueError: шаг, который ничего не раскрывает, зациклит сэмплер.
    """
    if k <= 0:
        raise ValueError(f"k должно быть положительным, получено {k}")
    if len(tokens) != len(logits):
        raise ValueError(f"разная длина: {len(tokens)} и {len(logits)}")
    masked = [i for i, t in enumerate(tokens) if t == MASK]
    # ключ (-уверенность, индекс): минус вместо reverse=True, чтобы при
    # равенстве индекс сортировался по возрастанию, а не по убыванию
    masked.sort(key=lambda i: (-max(softmax(logits[i])), i))
    return sorted(masked[:k])


def unmask_step(tokens, logits, k):
    """Один шаг сэмплера: раскрыть k самых уверенных позиций. Новый список.

    unmask_step([MASK, MASK], [[0.0, 9.0], [0.0, 0.0]], 1)  ->  [1, MASK]

    В раскрытые позиции кладём argmax логитов. Уже известные токены не
    трогаем — именно поэтому inpainting достаётся бесплатно: подай на вход
    частично заполненную картинку, и она останется на месте.

    Входной список не мутируем: трейс сэмплирования иначе схлопнется в
    N одинаковых ссылок на один и тот же список.
    """
    chosen = top_k_confident(tokens, logits, k)
    new_tokens = list(tokens)
    for i in chosen:
        row = logits[i]
        new_tokens[i] = max(range(len(row)), key=lambda v: (row[v], -v))
    return new_tokens


def sample_masked(predict, tokens, T):
    """Полный цикл Show-o. Вернуть трейс состояний, включая исходное.

    predict — callable: predict(tokens) -> список логитов на каждую позицию.
    tokens — стартовое состояние: MASK там, где надо сгенерировать.

    Количество раскрытий по шагам берётся из unmask_counts(cosine_schedule(T)),
    но не меньше одного за шаг: шаг, раскрывающий ноль токенов, тратит полный
    forward-проход впустую.

    Гарантии, ради которых всё и затевалось:
      * в последнем состоянии трейса не осталось ни одного MASK;
      * predict вызывается не больше T раз, сколько бы токенов ни было, —
        в этом вся разница с авторегрессией на 1024 прохода;
      * позиции, пришедшие уже заполненными, такими и остаются (inpainting).

    Если маскировать нечего, трейс состоит из одного состояния.
    """
    if T <= 0:
        raise ValueError(f"T должно быть положительным, получено {T}")
    state = list(tokens)
    traces = [list(state)]
    remaining = sum(1 for t in state if t == MASK)
    if remaining == 0:
        return traces

    for k in unmask_counts(cosine_schedule(T), remaining):
        if all(t != MASK for t in state):
            break
        state = unmask_step(state, predict(state), max(1, k))
        traces.append(list(state))

    # хвост: округление расписания может оставить пару позиций под маской
    while any(t == MASK for t in state):
        left = sum(1 for t in state if t == MASK)
        state = unmask_step(state, predict(state), left)
        traces.append(list(state))
    return traces


def compression_ratio(width, height, n_tokens, vocab, bits_per_pixel=24):
    """Во сколько раз VQ-код легче сырых пикселей.

    compression_ratio(512, 512, 1024, 16384)  ->  438.857...

    Разбор: сырые пиксели это 512 * 512 * 24 = 6 291 456 бит. Код это
    1024 токена по log2(16384) = 14 бит, итого 14 336 бит.

    Это упражнение 4 урока. Число показывает, чем платит дискретный
    токенизатор: 438-кратное сжатие не бывает бесплатным, и потерянная
    деталь — ровно та причина, по которой Transfusion держит патчи
    непрерывными.

    Ловушка: vocab < 2 даёт log2 <= 0 и деление на ноль. ValueError.
    """
    if width <= 0 or height <= 0:
        raise ValueError("размеры картинки должны быть положительными")
    if n_tokens <= 0:
        raise ValueError(f"n_tokens должно быть положительным, получено {n_tokens}")
    if vocab < 2:
        raise ValueError(f"vocab должен быть минимум 2, получено {vocab}")
    raw_bits = width * height * bits_per_pixel
    coded_bits = n_tokens * math.log2(vocab)
    return raw_bits / coded_bits
