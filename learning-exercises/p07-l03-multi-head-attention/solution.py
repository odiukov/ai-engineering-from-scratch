"""
Multi-head attention — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def matmul(A, B):
    """Умножение матриц: (n, k) @ (k, m) -> (n, m).

    matmul([[1, 2]], [[3], [4]])                ->  [[11]]
    matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]])  ->  [[5, 6], [7, 8]]

    Число столбцов A обязано совпасть с числом строк B, иначе ValueError.
    Молчаливый обрез через zip прячет ошибку размерности — а именно в
    размерностях голов её проще всего сделать.
    """
    if not A:
        return []
    inner = len(A[0])
    if len(B) != inner:
        raise ValueError("matmul shape mismatch: A is (n, %d), B has %d rows" % (inner, len(B)))
    # транспонируем B один раз, а не на каждую пару индексов
    Bt = [list(column) for column in zip(*B)]
    return [[sum(a * b for a, b in zip(row, column)) for column in Bt] for row in A]


def softmax(scores):
    """Строка скоров -> строка весов: всё положительное, сумма ровно 1.

    softmax([0.0, 0.0, 0.0])  ->  [1/3, 1/3, 1/3]
    softmax([2.0, 0.0])       ->  примерно [0.881, 0.119]

    Вычти максимум перед exp: ответ тот же, OverflowError не будет.
    Пустой список — ValueError.
    """
    if not scores:
        raise ValueError("softmax of an empty score vector")
    shift = max(scores)
    exps = [math.exp(s - shift) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def split_heads(X, n_heads):
    """Разрезать (n, d) на n_heads кусков ширины d_head = d // n_heads.

    Вернуть список из n_heads матриц формы (n, d_head).

    split_heads([[1, 2, 3, 4]], 2)  ->  [[[1, 2]], [[3, 4]]]

    Голова h забирает столбцы [h * d_head : (h + 1) * d_head]. Строки не
    перемешиваются: токен остаётся токеном, режется только его вектор.

    Ловушка: d обязано делиться на n_heads нацело, иначе головы получатся
    разной ширины — это ValueError, а не молчаливый обрез.
    """
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if not X:
        return [[] for _ in range(n_heads)]
    d = len(X[0])
    if d % n_heads != 0:
        raise ValueError("width %d is not divisible by n_heads=%d" % (d, n_heads))
    width = d // n_heads
    return [[row[h * width:(h + 1) * width] for row in X] for h in range(n_heads)]


def combine_heads(heads):
    """Склеить головы обратно: список (n, d_head) -> одна (n, n_heads * d_head).

    combine_heads([[[1, 2]], [[3, 4]]])  ->  [[1, 2, 3, 4]]

    Это точная обратная операция к split_heads. Порядок голов сохраняется,
    иначе W_o будет применяться не к тем столбцам.
    """
    if not heads or not heads[0]:
        return []
    n = len(heads[0])
    return [[value for head in heads for value in head[i]] for i in range(n)]


def head_attention(Q, K, V, mask=None):
    """Внимание внутри ОДНОЙ головы: softmax(Q @ K^T / sqrt(d_head)) @ V.

    Вернуть (output, weights): output формы (n_q, d_v), weights — (n_q, n_k).

    head_attention([[0.0]], [[0.0], [0.0]], [[1.0], [3.0]])
        ->  ([[2.0]], [[0.5, 0.5]])

    mask — матрица (n_q, n_k) из bool (True = «смотреть можно») или None.
    Запрещённая позиция обязана получить вес РОВНО 0.0, а не -1e9 после
    экспоненты.

    Масштаб здесь sqrt(d_head), а НЕ sqrt(d_model). Голова живёт в своём
    подпространстве, и делить надо на его ширину — иначе при 8 головах
    скоры окажутся занижены в 2.8 раза.
    """
    if not Q:
        return [], []
    d_head = len(Q[0])
    scale = math.sqrt(d_head)
    Kt = [list(column) for column in zip(*K)]
    raw = matmul(Q, Kt)
    weights = []
    for i, row in enumerate(raw):
        scaled = [value / scale for value in row]
        if mask is None:
            weights.append(softmax(scaled))
            continue
        kept = [j for j, allowed in enumerate(mask[i]) if allowed]
        if not kept:
            raise ValueError("attention row %d has every position masked out" % i)
        sub = softmax([scaled[j] for j in kept])
        full = [0.0] * len(scaled)
        for j, weight in zip(kept, sub):
            full[j] = weight
        weights.append(full)
    return matmul(weights, V), weights


def repeat_kv_heads(kv_heads, n_heads):
    """GQA: размножить n_kv голов ключей/значений до n_heads запросов.

    Голова запроса i берёт kv-голову i // (n_heads // n_kv).

    repeat_kv_heads([A, B], 4)  ->  [A, A, B, B]
    repeat_kv_heads([A], 3)     ->  [A, A, A]        (это MQA)
    repeat_kv_heads([A, B], 2)  ->  [A, B]           (это обычный MHA)

    Ловушка: n_heads обязано делиться на число kv-голов, иначе группы
    получатся разного размера — ValueError.

    Смысл в KV-кэше: на инференсе хранятся только n_kv копий K и V.
    У Llama 3 70B это 64 головы запросов против 8 kv — кэш в 8 раз меньше.
    """
    n_kv = len(kv_heads)
    if n_kv == 0 or n_heads % n_kv != 0:
        raise ValueError("n_heads=%d is not divisible by n_kv_heads=%d" % (n_heads, n_kv))
    repeat = n_heads // n_kv
    return [kv_heads[i // repeat] for i in range(n_heads)]


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, n_kv_heads=None, mask=None):
    """Multi-head attention целиком. Вернуть (output, weights_per_head).

    Порядок действий:
      Q = X @ Wq, K = X @ Wk, V = X @ Wv;
      Q режется на n_heads голов, K и V — на n_kv_heads и размножаются;
      в каждой голове считается head_attention;
      головы склеиваются и умножаются на Wo.

    Формы: X это (n, d_model), Wq и Wo это (d_model, d_model),
    Wk и Wv это (d_model, n_kv_heads * d_head), где d_head = d_model // n_heads.
    n_kv_heads=None означает «столько же, сколько голов запросов» — обычный MHA.

    Выход — (n, d_model), weights_per_head — список из n_heads матриц (n, n).

    Проверь на себе: при n_heads=1 и единичной Wo это ровно head_attention
    от полных Q, K, V. Multi-head — не новая математика, а та же самая,
    применённая к кускам вектора.

    Wo — единственное место, где головы вообще узнают друг о друге. Убери
    его (поставь единичную матрицу), и каждый столбец выхода будет зависеть
    ровно от одной головы.
    """
    if n_kv_heads is None:
        n_kv_heads = n_heads
    Qh = split_heads(matmul(X, Wq), n_heads)
    # K и V режутся на СВОЁ число голов и потом размножаются — в этом вся GQA
    Kh = repeat_kv_heads(split_heads(matmul(X, Wk), n_kv_heads), n_heads)
    Vh = repeat_kv_heads(split_heads(matmul(X, Wv), n_kv_heads), n_heads)
    outputs, weights = [], []
    for q, k, v in zip(Qh, Kh, Vh):
        head_out, head_weights = head_attention(q, k, v, mask)
        outputs.append(head_out)
        weights.append(head_weights)
    return matmul(combine_heads(outputs), Wo), weights


def kv_cache_cells(seq_len, n_kv_heads, d_head, n_layers=1):
    """Сколько чисел лежит в KV-кэше: 2 * n_layers * n_kv_heads * seq_len * d_head.

    Двойка — потому что кэшируются и K, и V.

    kv_cache_cells(10, 8, 128)            ->  20480
    kv_cache_cells(10, 64, 128) / kv_cache_cells(10, 8, 128)  ->  8.0

    Это и есть вся арифметика GQA: кэш зависит от n_kv_heads, а качество —
    от n_heads. Уменьшая первое и не трогая второе, покупаешь память почти
    даром.
    """
    return 2 * n_layers * n_kv_heads * seq_len * d_head
