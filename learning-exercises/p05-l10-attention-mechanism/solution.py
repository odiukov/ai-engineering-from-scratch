"""
Механизм внимания — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(scores):
    """Скоры -> веса внимания: всё положительное, сумма ровно 1.

    softmax([0.0, 0.0, 0.0])  ->  [1/3, 1/3, 1/3]
    softmax([2.0, 0.0])       ->  примерно [0.881, 0.119]

    Ловушка: math.exp(1000) кидает OverflowError. Вычти максимум перед exp —
    ответ тот же, переполнения нет.

    Пустой список — ValueError: распределения ни над чем не бывает.

    Именно softmax делает веса внимания «долями»: сумма 1 означает, что
    контекст — это средневзвешенное, а не просто сумма.
    """
    if not scores:
        raise ValueError("softmax of an empty score vector")
    shift = max(scores)
    exps = [math.exp(v - shift) for v in scores]
    total = sum(exps)
    return [e / total for e in exps]


def masked_softmax(scores, mask):
    """Softmax, который выкидывает паддинг: mask[i]=False -> вес РОВНО 0.0.

    mask — список тех же длин: True «позиция настоящая», False «паддинг».

    masked_softmax([1.0, 1.0, 5.0], [True, True, False])  ->  [0.5, 0.5, 0.0]

    Ловушка: подмена скора на -1e9 даёт не ноль, а крошечное ненулевое
    число, и паддинг всё равно подмешивается в контекст. Здесь нужен
    честный ноль — считай softmax только по оставленным позициям.

    Если mask прячет всё, это ValueError: смотреть не на что.
    """
    if len(mask) != len(scores):
        raise ValueError("mask and scores must have the same length")
    kept = [i for i, keep in enumerate(mask) if keep]
    if not kept:
        raise ValueError("mask hides every position")
    # softmax по подвыборке, потом раскладываем обратно по местам:
    # так нули получаются точными, а не «почти нулями»
    sub = softmax([scores[i] for i in kept])
    weights = [0.0] * len(scores)
    for position, weight in zip(kept, sub):
        weights[position] = weight
    return weights


def dot_score(query, keys):
    """Скор Luong dot: e_i = query · key_i. Один скор на позицию энкодера.

    dot_score([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])  ->  [1.0, 0.0]

    Ловушка: dot требует одинаковой размерности query и key. Если энкодер
    двунаправленный, а декодер нет — размерности разъедутся, и это ValueError,
    а не молчаливый обрез по короткому вектору (zip обрезал бы молча).
    """
    scores = []
    for key in keys:
        if len(key) != len(query):
            raise ValueError("dot score requires query and key of equal length")
        scores.append(sum(q * k for q, k in zip(query, key)))
    return scores


def general_score(query, keys, W):
    """Скор Luong general: e_i = query^T W key_i. W имеет форму (d_s, d_h).

    С единичной W это ровно dot_score. Смысл W в том, что она снимает
    жёсткое требование d_s == d_h: query может быть размерности 2,
    а ключи — 3.

    general_score([1.0, 0.0], [[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [1.0]

    Считать выгоднее так: сначала спроецировать query один раз
    (W^T @ query даёт вектор длины d_h), а потом скалярно умножать на
    каждый ключ. Иначе матрица прогоняется заново на каждую позицию.
    """
    if len(W) != len(query):
        raise ValueError("W must have one row per query dimension")
    d_h = len(W[0])
    # проекция query считается ОДИН раз, дальше только скалярные произведения:
    # T умножений матрицы превращаются в одно
    projected = [sum(query[j] * W[j][i] for j in range(len(query))) for i in range(d_h)]
    return dot_score(projected, keys)


def additive_score(query, keys, W_a, U_a, v_a):
    """Скор Bahdanau: e_i = v_a · tanh(W_a @ query + U_a @ key_i).

    W_a имеет форму (d_attn, d_s), U_a — (d_attn, d_h), v_a — длину d_attn.
    v_a не магия: это проекция, схлопывающая вектор размерности d_attn
    в один скаляр.

    Размерности query и ключей не обязаны совпадать — в этом и разница
    с dot.

    Полезное свойство для отладки: |tanh| < 1, поэтому |e_i| строго меньше
    суммы модулей v_a. Если скоры вылезли за эту границу — где-то потерян
    tanh.
    """
    d_attn = len(v_a)
    # проекция query от позиции энкодера не зависит — выносим из цикла
    proj_q = [sum(W_a[j][k] * query[k] for k in range(len(query))) for j in range(d_attn)]
    scores = []
    for key in keys:
        proj_k = [sum(U_a[j][k] * key[k] for k in range(len(key))) for j in range(d_attn)]
        total = 0.0
        for j in range(d_attn):
            total += v_a[j] * math.tanh(proj_q[j] + proj_k[j])
        scores.append(total)
    return scores


def attend(scores, values, mask=None):
    """Скоры + значения -> (context, weights). Сердце внимания.

    weights = softmax(scores) (с маской, если она дана),
    context = сумма weights[i] * values[i] — средневзвешенное значений.

    attend([0.0, 0.0], [[1.0], [3.0]])  ->  ([2.0], [0.5, 0.5])

    Обрати внимание: скоры считались против КЛЮЧЕЙ, а взвешивается
    VALUES. В классическом внимании keys и values — одно и то же, в
    self-attention это разные проекции.

    Ловушка: context — выпуклая комбинация, каждая его координата обязана
    лежать между минимумом и максимумом этой координаты по values. Вылезла
    за границы — значит веса не просуммировались в 1.
    """
    if len(scores) != len(values):
        raise ValueError("one score per value is required")
    weights = softmax(scores) if mask is None else masked_softmax(scores, mask)
    dim = len(values[0])
    context = [0.0] * dim
    for weight, value in zip(weights, values):
        for i in range(dim):
            context[i] += weight * value[i]
    return context, weights


def alignment_matrix(decoder_states, encoder_states, mask=None):
    """Матрица выравнивания (T_dec, T_enc): по строке весов на шаг декодера.

    Скоры считаются через dot_score, строки нормируются softmax, поэтому
    каждая строка суммируется в 1.

    Пустой список состояний декодера даёт пустую матрицу.

    Это та самая «красивая картинка» из статей: строка t показывает, на
    какие слова источника смотрел декодер, когда печатал слово t.
    Осторожно с интерпретацией — Jain & Wallace (2019) показали, что веса
    внимания можно подменить, не сломав предсказания.
    """
    rows = []
    for state in decoder_states:
        # values здесь не нужны, интересуют только веса — но переиспользуем
        # attend, чтобы правило нормировки жило в одном месте
        _, weights = attend(dot_score(state, encoder_states), encoder_states, mask)
        rows.append(weights)
    return rows


def multi_head_dot_attention(query, keys, values, n_heads):
    """Multi-head внимание: разрезать размерность на головы, каждую посчитать
    отдельно, контексты склеить обратно.

    Возвращает (context, head_weights): context длины d, head_weights —
    список из n_heads векторов весов.

    При n_heads=1 обязано совпасть с обычным dot-вниманием: одна голова —
    это и есть attend(dot_score(...)).

    Ловушка: d должно делиться на n_heads нацело, иначе головы получатся
    разной ширины — это ValueError.

    Смысл: одна голова усредняет всё в один вектор и вынуждена выбирать,
    на что смотреть. Несколько голов смотрят на разное одновременно.
    """
    dim = len(query)
    if n_heads <= 0 or dim % n_heads != 0:
        raise ValueError("query dimension must be divisible by n_heads")
    width = dim // n_heads
    context = []
    head_weights = []
    for head in range(n_heads):
        lo, hi = head * width, (head + 1) * width
        # срезы вместо копий всей матрицы: голова видит только свой кусок
        q = query[lo:hi]
        k = [key[lo:hi] for key in keys]
        v = [value[lo:hi] for value in values]
        head_context, weights = attend(dot_score(q, k), v)
        context.extend(head_context)
        head_weights.append(weights)
    return context, head_weights
