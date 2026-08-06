"""
T5 и BART: encoder-decoder — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(scores):
    """Softmax по строке: набор чисел -> распределение вероятностей.

    softmax([0.0, 0.0])   ->  [0.5, 0.5]
    softmax([2.0, 0.0])   ->  [0.880..., 0.119...]

    Вычитай максимум перед экспонентой, иначе exp большого скора
    переполнится. Сумма результата от этого не меняется.
    """
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def cross_attention(Q_dec, K_enc, V_enc):
    """Cross-attention: запросы декодера смотрят на ключи и значения энкодера.

    Q_dec — матрица (N_target, d_k), K_enc — (N_source, d_k),
    V_enc — (N_source, d_v). Результат — (N_target, d_v).

    cross_attention([[1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[10.0], [0.0]])
        ->  [[8.044...]]   (софтмакс от [2/sqrt(2), 0] взвешивает 10 и 0)

    Три отличия от self-attention декодера:
      * Q берётся из декодера, а K и V из энкодера;
      * causal-маски здесь НЕТ: выходная позиция 0 имеет право видеть весь
        вход целиком, включая последний токен источника;
      * энкодер считается один раз, а декодер обращается к его выходу на
        каждом шаге генерации — поэтому выход энкодера кэшируют.

    Скоры делятся на sqrt(d_k): без деления дисперсия скоров растёт с
    размерностью, softmax насыщается и градиент умирает.

    Число строк K_enc и V_enc обязано совпадать — иначе ValueError.
    """
    if len(K_enc) != len(V_enc):
        raise ValueError("K_enc and V_enc must describe the same source tokens")
    d_k = len(Q_dec[0])
    scale = math.sqrt(d_k)
    out = []
    for q in Q_dec:
        scores = [sum(a * b for a, b in zip(q, k)) / scale for k in K_enc]
        weights = softmax(scores)
        # выход — выпуклая комбинация строк V: веса неотрицательны и дают 1
        out.append([sum(w * v[j] for w, v in zip(weights, V_enc)) for j in range(len(V_enc[0]))])
    return out


def shift_right(target_ids, start_id):
    """Вход декодера при teacher forcing: цели, сдвинутые вправо на один токен.

    shift_right([7, 8, 9], 0)  ->  [0, 7, 8]

    Длина не меняется: спереди добавляется start-токен, последняя цель
    выпадает — предсказывать после неё уже нечего.

    Смысл: на позиции i декодер получает ПРАВИЛЬНЫЙ предыдущий токен, а не
    свой собственный прошлый выход. Это teacher forcing, так учится и T5,
    и BART, и любой seq2seq.
    """
    return [start_id] + list(target_ids[:-1])


def pick_spans(n, rng, mask_rate=0.15, mean_span=3.0):
    """Выбрать непересекающиеся спаны для порчи: список (start, length).

    pick_spans(20, random.Random(0))               ->  [(6, 3)]
    pick_spans(30, random.Random(0), mask_rate=0.2) ->  [(6, 3), (20, 3)]

    Правила: спаны отсортированы по start, не пересекаются, между соседними
    остаётся хотя бы один живой токен, суммарная длина примерно
    round(n * mask_rate), число спанов примерно этой суммы делённой на
    mean_span. T5 использует mask_rate=0.15 и mean_span=3.

    Случайность идёт ТОЛЬКО через rng — тот же seed обязан давать те же
    спаны, иначе тест на round-trip невоспроизводим.

    mask_rate вне (0, 1) — ValueError. Если живых токенов не хватает на
    разделители между спанами — тоже ValueError.
    """
    if not 0 < mask_rate < 1:
        raise ValueError("mask_rate must be in (0, 1)")
    n_mask = max(1, round(n * mask_rate))
    n_spans = max(1, round(n_mask / mean_span))
    free = n - n_mask
    if free < n_spans - 1:
        raise ValueError("sequence too short for that many spans")

    # длины: делим бюджет как можно ровнее, остаток раздаём первым спанам
    base, rest = divmod(n_mask, n_spans)
    lengths = [base + (1 if i < rest else 0) for i in range(n_spans)]

    # раскладываем свободные токены по промежуткам: перед первым спаном,
    # между спанами и после последнего — итого n_spans + 1 промежутков
    extra = free - (n_spans - 1)
    gaps = [0] * (n_spans + 1)
    for _ in range(extra):
        gaps[rng.randrange(n_spans + 1)] += 1
    for i in range(1, n_spans):
        gaps[i] += 1  # обязательный разделитель между соседними спанами

    spans = []
    pos = 0
    for i, length in enumerate(lengths):
        pos += gaps[i]
        spans.append((pos, length))
        pos += length
    return spans


def corrupt_spans(tokens, spans):
    """T5 span corruption: (испорченный вход, цель декодера).

    corrupt_spans(["a", "b", "c", "d"], [(1, 2)])
        ->  (["a", "<extra_id_0>", "d"],
             ["<extra_id_0>", "b", "c", "<extra_id_1>"])

    Формат ровно такой, как в T5: в источнике каждый спан заменяется на
    свой sentinel `<extra_id_N>`, а в цели идут пары «sentinel + содержимое
    спана» и в самом конце закрывающий sentinel с номером len(spans).

    Экономия против BERT: декодер выписывает только испорченные куски, а не
    всю последовательность целиком.

    spans обязаны быть отсортированы, не пересекаться и лежать внутри
    tokens — иначе ValueError.
    """
    end = 0
    for start, length in spans:
        if start < end or length < 1 or start + length > len(tokens):
            raise ValueError("spans must be sorted, non-overlapping and in range")
        end = start + length

    source, target = [], []
    prev_end = 0
    for i, (start, length) in enumerate(spans):
        source.extend(tokens[prev_end:start])
        source.append(f"<extra_id_{i}>")
        target.append(f"<extra_id_{i}>")
        target.extend(tokens[start:start + length])
        prev_end = start + length
    source.extend(tokens[prev_end:])
    target.append(f"<extra_id_{len(spans)}>")  # закрывающий маркер
    return source, target


def round_trip(source, target):
    """Собрать исходную последовательность обратно из (source, target).

    round_trip(["a", "<extra_id_0>", "d"],
               ["<extra_id_0>", "b", "c", "<extra_id_1>"])
        ->  ["a", "b", "c", "d"]

    Разбери target на пары «sentinel -> его токены» и подставь их в source
    вместо соответствующих sentinel-ов.

    Настоящее обучение так не делает, но проверка стоит копейки и ловит
    любые ошибки на единицу в бухгалтерии спанов: если round_trip вернул
    исходный текст, значит порча обратима и обучающая пара корректна.

    Учти закрывающий sentinel в конце target: за ним токенов нет.
    """
    spans = {}
    key = None
    for token in target:
        if token.startswith("<extra_id_"):
            key = token
            spans[key] = []
        elif key is not None:
            spans[key].append(token)

    out = []
    for token in source:
        if token.startswith("<extra_id_"):
            out.extend(spans.get(token, []))
        else:
            out.append(token)
    return out


def text_infill(tokens, spans, mask_token="<mask>"):
    """Шум BART text infilling: каждый спан заменяется на ОДИН mask-токен.

    text_infill(["a", "b", "c", "d"], [(1, 2)])  ->  ["a", "<mask>", "d"]
    text_infill(["a", "b", "d"], [(1, 1)])       ->  ["a", "<mask>", "d"]

    Разница с T5 принципиальная: sentinel-ы пронумерованы и каждый спан
    попадает в цель, поэтому порча T5 обратима. Здесь все маски одинаковы и
    длина спана нигде не записана — по двум примерам выше видно, что из
    результата уже не восстановить, сколько токенов съели. Именно поэтому
    декодер BART учится восстанавливать ВСЮ последовательность.

    Требования к spans те же: отсортированы, не пересекаются, в границах.
    """
    end = 0
    for start, length in spans:
        if start < end or length < 1 or start + length > len(tokens):
            raise ValueError("spans must be sorted, non-overlapping and in range")
        end = start + length

    out = []
    prev_end = 0
    for start, length in spans:
        out.extend(tokens[prev_end:start])
        out.append(mask_token)
        prev_end = start + length
    out.extend(tokens[prev_end:])
    return out


def document_rotate(tokens, pivot):
    """Шум BART document rotation: последовательность прокручивается по кругу.

    document_rotate(["a", "b", "c", "d"], 1)  ->  ["b", "c", "d", "a"]
    document_rotate(["a", "b", "c", "d"], 0)  ->  ["a", "b", "c", "d"]

    Модель должна научиться находить настоящее начало документа. Порча
    обратима: прокрутка на pivot и следом на len(tokens) - pivot возвращает
    исходный порядок.

    pivot вне [0, len(tokens)) — ValueError. Пустой список тоже ValueError:
    крутить нечего.
    """
    if not tokens:
        raise ValueError("cannot rotate an empty sequence")
    if not 0 <= pivot < len(tokens):
        raise ValueError("pivot must be a valid index")
    return list(tokens[pivot:]) + list(tokens[:pivot])
