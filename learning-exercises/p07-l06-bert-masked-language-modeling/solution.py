"""
BERT и masked language modeling — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(logits):
    """Логиты -> распределение: всё положительное, сумма ровно 1.

    softmax([0.0, 0.0])  ->  [0.5, 0.5]
    softmax([2.0, 0.0])  ->  примерно [0.881, 0.119]

    Вычти максимум перед exp: ответ тот же (общий множитель сокращается),
    OverflowError не будет.

    Пустой список — ValueError.
    """
    if not logits:
        raise ValueError("softmax of an empty logit vector")
    shift = max(logits)
    exps = [math.exp(v - shift) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def build_bert_input(tokens_a, cls_id, sep_id, tokens_b=None):
    """Собрать вход BERT. Вернуть (token_ids, segment_ids).

    Одна последовательность:  [CLS] a... [SEP]
    Две последовательности:   [CLS] a... [SEP] b... [SEP]

    segment_ids: 0 у всего, что относится к первой части (включая [CLS] и
    первый [SEP]), 1 у второй части вместе с её [SEP].

    build_bert_input([7, 8], 1, 2)              ->  ([1, 7, 8, 2], [0, 0, 0, 0])
    build_bert_input([7], 1, 2, tokens_b=[9])   ->  ([1, 7, 2, 9, 2], [0, 0, 0, 1, 1])

    [CLS] всегда стоит нулевым: его финальный вектор и есть представление
    всей последовательности, за него потом цепляется голова классификации.
    [SEP] нужен, чтобы модель понимала, где кончается query и начинается
    документ — на этом живут cross-encoder реранкеры.
    """
    ids = [cls_id, *tokens_a, sep_id]
    segments = [0] * len(ids)
    if tokens_b is not None:
        ids += [*tokens_b, sep_id]
        segments += [1] * (len(tokens_b) + 1)
    return ids, segments


def create_mlm_batch(
    tokens,
    vocab_size,
    mask_id,
    rng,
    mask_prob=0.15,
    special_token_ids=(),
    decisions=None,
):
    """Разметка MLM по правилам BERT. Вернуть (input_ids, labels).

    Обычные токены независимо выбираются для предсказания с вероятностью
    mask_prob. Токены из special_token_ids (и сам mask_id) не выбираются.
    У выбранной позиции labels[i] = исходный токен, дальше ещё один бросок:
      < 0.8            -> input_ids[i] = mask_id
      от 0.8 до 0.9    -> input_ids[i] = случайный НЕспециальный токен
      >= 0.9           -> input_ids[i] остаётся исходным

    У невыбранных позиций labels[i] = -100 (соглашение ignore_index в
    PyTorch), а input_ids[i] обязан остаться нетронутым.

    rng — обязательный параметр (например random.Random(0)): без него
    прогон невоспроизводим. Глобальный random не использовать.

    Если передан список decisions, для каждой выбранной позиции добавь туда
    "mask", "random" или "unchanged". Ветку нельзя восстанавливать сравнением
    input_ids с labels: случайный токен может совпасть с исходным.

    create_mlm_batch([5, 5, 5], 1000, 1000, random.Random(0), mask_prob=0.0)
        ->  ([5, 5, 5], [-100, -100, -100])

    Зачем 10% случайных и 10% нетронутых: токена [MASK] на инференсе не
    существует. Если модель обучится ожидать его в 100% предсказываемых
    позиций, между предобучением и файнтюном возникнет сдвиг распределения.
    Эти 20% держат её честной.
    """
    special_ids = set(special_token_ids)
    special_ids.add(mask_id)
    random_token_ids = [token_id for token_id in range(vocab_size)
                        if token_id not in special_ids]
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, token in enumerate(tokens):
        if token in special_ids:
            continue
        if rng.random() >= mask_prob:
            continue
        labels[i] = token
        roll = rng.random()
        if roll < 0.8:
            input_ids[i] = mask_id
            branch = "mask"
        elif roll < 0.9:
            if not random_token_ids:
                raise ValueError("vocabulary has no non-special replacement tokens")
            input_ids[i] = random_token_ids[rng.randrange(len(random_token_ids))]
            branch = "random"
        else:
            # позиция остаётся как есть: предсказывать надо, подсказка на месте
            branch = "unchanged"
        if decisions is not None:
            decisions.append(branch)
    return input_ids, labels


def mlm_loss(logits, labels, ignore_index=-100):
    """Средняя кросс-энтропия ТОЛЬКО по предсказываемым позициям.

    logits — матрица (n, vocab): по строке логитов на позицию.
    labels — список длины n, где ignore_index означает «эту позицию не
    предсказываем».

    Возвращает -mean(log p[правильный класс]) по непроигнорированным
    позициям. Если предсказывать нечего — 0.0, а не деление на ноль.

    mlm_loss([[0.0, 0.0]], [0])       ->  log(2) = 0.6931...
    mlm_loss([[0.0, 0.0]], [-100])    ->  0.0

    Ловушка: делить надо на число ПРЕДСКАЗЫВАЕМЫХ позиций, а не на длину
    последовательности. При mask_prob=0.15 разница ровно в 6.7 раза, и
    ошибку легко не заметить — loss просто «подозрительно маленький».

    Ориентир для отладки: на равномерных логитах loss обязан быть равен
    log(vocab_size). Если на старте обучения он другой — сломана либо
    инициализация, либо сам подсчёт.
    """
    total = 0.0
    count = 0
    for row, label in zip(logits, labels):
        if label == ignore_index:
            continue
        probs = softmax(row)
        total -= math.log(probs[label])
        count += 1
    if count == 0:
        return 0.0
    return total / count


def mlm_loss_grad(logits, labels, ignore_index=-100):
    """Градиент mlm_loss по логитам. Форма совпадает с logits.

    Для предсказываемой позиции: (softmax(row) - one_hot(label)) / count,
    где count — число предсказываемых позиций. Для проигнорированной —
    строка нулей.

    mlm_loss_grad([[0.0, 0.0]], [0])     ->  [[-0.5, 0.5]]
    mlm_loss_grad([[0.0, 0.0]], [-100])  ->  [[0.0, 0.0]]

    Ради этой формулы softmax и кросс-энтропию всегда пишут одной
    функцией: по отдельности пришлось бы протаскивать якобиан softmax, а
    вместе он сокращается и остаётся «предсказание минус правда».

    Полезное свойство для отладки: сумма градиента по строке равна нулю.
    Softmax суммируется в 1, one-hot тоже, разность — в 0. Сумма не ноль —
    значит где-то потеряна нормировка.
    """
    count = sum(1 for label in labels if label != ignore_index)
    grad = []
    for row, label in zip(logits, labels):
        if label == ignore_index or count == 0:
            grad.append([0.0] * len(row))
            continue
        probs = softmax(row)
        # «предсказание минус правда»: якобиан softmax сократился с
        # производной log, поэтому здесь нет ни одного exp сверх softmax
        probs[label] -= 1.0
        grad.append([p / count for p in probs])
    return grad


def mlm_accuracy(logits, labels, ignore_index=-100):
    """Доля предсказываемых позиций, где argmax логитов совпал с меткой.

    mlm_accuracy([[9.0, 0.0], [0.0, 9.0]], [0, 1])     ->  1.0
    mlm_accuracy([[9.0, 0.0], [0.0, 9.0]], [0, 0])     ->  0.5
    mlm_accuracy([[9.0, 0.0]], [-100])                 ->  0.0

    Если предсказывать нечего — 0.0.

    Метрика смотрит только на argmax, поэтому loss может падать, а
    accuracy стоять на месте: модель становится увереннее там, где и так
    была права.
    """
    correct = 0
    count = 0
    for row, label in zip(logits, labels):
        if label == ignore_index:
            continue
        count += 1
        best = max(range(len(row)), key=lambda j: row[j])
        if best == label:
            correct += 1
    if count == 0:
        return 0.0
    return correct / count


def classify_from_cls(hidden, W, b):
    """Голова классификации поверх [CLS]: softmax(W @ hidden[0] + b).

    hidden — матрица (n, d) выходов энкодера, W — (n_classes, d),
    b — длины n_classes.

    classify_from_cls([[1.0, 0.0], [9.9, 9.9]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
        ->  примерно [0.731, 0.269]

    Обрати внимание на пример: вторая строка вообще не участвует. Голова
    берёт ТОЛЬКО нулевую позицию — это и есть весь «pooling» BERT.
    Меняй остальные токены сколько угодно, выход головы не дрогнет (при
    фиксированном hidden — то есть после того, как энкодер уже посчитан).

    Пустой hidden — ValueError: [CLS] обязан существовать.

    Это минимальная downstream-голова. При fine-tuning часто обучают её
    вместе со всем энкодером; заморозка энкодера — отдельный выбор для
    экономии памяти или очень маленького датасета, а не правило BERT.
    """
    if not hidden:
        raise ValueError("hidden states must contain at least the [CLS] position")
    cls = hidden[0]
    logits = [sum(w * h for w, h in zip(row, cls)) + bias for row, bias in zip(W, b)]
    return softmax(logits)
