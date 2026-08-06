"""
Self-attention с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(scores):
    """Строка скоров -> строка весов: всё положительное, сумма ровно 1.

    softmax([0.0, 0.0, 0.0])  ->  [1/3, 1/3, 1/3]
    softmax([2.0, 0.0])       ->  примерно [0.881, 0.119]

    Ловушка: math.exp(1000) кидает OverflowError. Вычти максимум перед exp —
    математически ответ тот же (общий множитель сокращается), переполнения
    нет. Это же свойство означает, что softmax не меняется от сдвига всех
    скоров на константу.

    Пустой список — ValueError: распределения ни над чем не бывает.
    """
    if not scores:
        raise ValueError("softmax of an empty score vector")
    shift = max(scores)
    exps = [math.exp(s - shift) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def transpose(M):
    """Транспонирование матрицы: строки становятся столбцами.

    transpose([[1, 2, 3], [4, 5, 6]])  ->  [[1, 4], [2, 5], [3, 6]]
    transpose([])                      ->  []

    Нужно ровно затем, чтобы посчитать Q @ K^T: скор — это скалярное
    произведение строки Q на строку K, то есть на столбец K^T.
    """
    if not M:
        return []
    return [list(column) for column in zip(*M)]


def matmul(A, B):
    """Умножение матриц: (n, k) @ (k, m) -> (n, m).

    matmul([[1, 2]], [[3], [4]])  ->  [[11]]
    matmul([[1, 0], [0, 1]], [[5, 6], [7, 8]])  ->  [[5, 6], [7, 8]]

    Ловушка: число столбцов A обязано совпасть с числом строк B, иначе
    ValueError. Молчаливый обрез через zip прячет ошибку в размерностях —
    а в трансформере это самая частая поломка.
    """
    if not A:
        return []
    inner = len(A[0])
    if len(B) != inner:
        raise ValueError("matmul shape mismatch: A is (n, %d), B has %d rows" % (inner, len(B)))
    # B транспонируем один раз: иначе на каждую пару (i, j) пришлось бы
    # заново бегать по столбцу B, а это лишний проход по памяти
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, column)) for column in Bt] for row in A]


def attention_scores(Q, K):
    """Скоры внимания: (Q @ K^T) / sqrt(dk). Форма (n_q, n_k).

    attention_scores([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
        ->  [[0.7071..., 0.0]]        (1/sqrt(2) и 0)

    dk — ширина векторов Q и K. Деление на sqrt(dk) — не косметика:
    скалярное произведение двух случайных векторов длины dk растёт как
    sqrt(dk), и без деления softmax при dk=64 уезжает в почти-one-hot, где
    градиент нулевой. Урок называет это softmax saturation.

    Пустой Q даёт пустой список скоров.
    """
    if not Q:
        return []
    dk = len(Q[0])
    scale = math.sqrt(dk)
    raw = matmul(Q, transpose(K))
    return [[value / scale for value in row] for row in raw]


def causal_mask(n):
    """Причинная маска (n, n): True — «смотреть можно», False — «нельзя».

    Позиции i разрешено видеть только позиции j <= i.

    causal_mask(3)  ->  [[True,  False, False],
                         [True,  True,  False],
                         [True,  True,  True]]

    Это единственная разница между энкодером BERT и декодером GPT: у
    энкодера маски нет вовсе, у декодера — вот эта нижнетреугольная.
    """
    return [[j <= i for j in range(n)] for i in range(n)]


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Внимание целиком: вернуть (output, weights).

    weights = softmax по строкам от attention_scores(Q, K),
    output  = weights @ V.

    scaled_dot_product_attention([[0.0]], [[0.0], [0.0]], [[1.0], [3.0]])
        ->  ([[2.0]], [[0.5, 0.5]])

    mask — матрица (n_q, n_k) из bool или None. Запрещённая позиция обязана
    получить вес РОВНО 0.0.

    Ловушка: подмена скора на -1e9 даёт не ноль, а крошечное ненулевое
    число, и «будущее» всё равно подмешивается в выход. Считай softmax
    только по разрешённым позициям, потом разложи веса по местам.

    Строка, где маска запретила всё, — ValueError: смотреть не на что.
    """
    scores = attention_scores(Q, K)
    weights = []
    for i, row in enumerate(scores):
        if mask is None:
            weights.append(softmax(row))
            continue
        kept = [j for j, allowed in enumerate(mask[i]) if allowed]
        if not kept:
            raise ValueError("attention row %d has every position masked out" % i)
        # softmax по подвыборке и раскладка обратно: так нули получаются
        # точными, а не «почти нулями»
        sub = softmax([row[j] for j in kept])
        full = [0.0] * len(row)
        for j, weight in zip(kept, sub):
            full[j] = weight
        weights.append(full)
    return matmul(weights, V), weights


def self_attention(X, Wq, Wk, Wv, mask=None):
    """Self-attention: Q, K, V — три проекции ОДНОГО и того же X.

    Q = X @ Wq, K = X @ Wk, V = X @ Wv, дальше scaled dot-product.
    Вернуть (output, weights).

    Формы: X это (n, d_model), Wq и Wk это (d_model, dk), Wv это (d_model, dv).
    Выход — (n, dv), веса — (n, n).

    Слово «self» здесь означает ровно одно: источник у запросов, ключей и
    значений общий. В cross-attention декодера (урок 05) Q придёт из
    декодера, а K и V — из энкодера.

    Проверь на себе главное свойство: переставь строки X — строки выхода
    переставятся так же, и ничего больше не изменится. Внимание слепо к
    порядку, позицию придётся вносить отдельно (урок 04).
    """
    Q = matmul(X, Wq)
    K = matmul(X, Wk)
    V = matmul(X, Wv)
    return scaled_dot_product_attention(Q, K, V, mask)
