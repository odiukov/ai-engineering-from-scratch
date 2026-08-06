"""
Варианты внимания: sliding window, sparse, differential — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def causal_mask(n):
    """Обычная causal-маска n x n: 0.0 там, где смотреть можно, -inf где нельзя.

    causal_mask(3)  ->  [[0.0,  -inf, -inf],
                         [0.0,   0.0, -inf],
                         [0.0,   0.0,  0.0]]

    Запрос на позиции i видит все j <= i. Нижнетреугольная матрица.

    Ловушка: маска складывается с логитами ДО softmax, поэтому «нельзя» —
    это -inf, а не 0. Нулём глушат уже после softmax, и это другая
    операция: она ломает нормировку.
    """
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            M[i][j] = 0.0
    return M


def swa_mask(n, window):
    """Sliding-window causal-маска: запрос i видит только последние `window` позиций.

    swa_mask(4, 2)  ->  [[0.0,  -inf, -inf, -inf],
                         [0.0,   0.0, -inf, -inf],
                         [-inf,  0.0,  0.0, -inf],
                         [-inf, -inf,  0.0,  0.0]]

    swa_mask(n, window=1)  ->  только диагональ, каждый токен видит себя.
    swa_mask(n, window=n)  ->  ровно causal_mask(n).

    window считает позиции ВКЛЮЧАЯ саму себя: окно [i-window+1, i].
    Так же считает Gemma 3 (sliding_window=1024) и Mistral 7B.

    Зачем: KV-cache падает с O(N) до O(window) на слой. При N=128K и
    window=1024 это 128-кратная экономия памяти.
    """
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M


def strided_mask(n, window, stride):
    """Локальное окно плюс каждый `stride`-й токен назад до начала (sparse transformer).

    strided_mask(6, 2, 3)  ->  строка 5 разрешает j = 4, 5 (окно)
                                            и  j = 0, 3    (шаг 3)

    Ловушка: j = 0 попадает в шаг всегда (0 кратно любому stride), поэтому
    нулевой токен видят все. Это тот самый глобальный токен из Longformer
    и BigBird — и одновременно причина, по которой в обычном полном
    внимании на позиции 0 сам собой заводится attention sink.

    Сложность падает с O(N^2) до примерно O(N * sqrt(N)), если брать
    stride ~ sqrt(N).
    """
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M


def count_attended(M):
    """Сколько клеток маски разрешены (равны 0.0).

    count_attended(causal_mask(4))     ->  10   (это 4*5/2)
    count_attended(swa_mask(8, 4))     ->  26

    Прямой измеритель стоимости варианта: у полного внимания это N(N+1)/2,
    у SWA — примерно N*window. На этом числе и держится вся экономия.
    """
    return sum(1 for row in M for v in row if v == 0.0)


def effective_receptive_field(n_layers, window):
    """Насколько далеко назад видит стопка из n_layers слоёв SWA с окном window.

    effective_receptive_field(1, 1024)   ->  1024
    effective_receptive_field(4, 4)      ->  13
    effective_receptive_field(32, 1024)  ->  32737

    Формула: n_layers * (window - 1) + 1. Каждый слой отодвигает границу
    на window - 1 позиций, а единица — это сам токен.

    В уроке это округляют до «L * W», и на больших окнах разница
    незаметна (32*1024 = 32768 против 32737). Но на маленьком окне
    ошибка видна сразу, поэтому считаем честно.

    Смысл: у Mistral 7B информация «течёт вперёд» через перекрывающиеся
    окна, и стопка слоёв видит гораздо дальше одного окна.
    """
    return n_layers * (window - 1) + 1


def masked_attention_row(q, Ks, Vs, mask_row):
    """Внимание одного запроса с маской. Вернуть кортеж (out, weights).

    masked_attention_row([1.0], [[1.0], [1.0]], [[2.0], [6.0]], [0.0, 0.0])
        ->  ([4.0], [0.5, 0.5])
    masked_attention_row([1.0], [[1.0], [1.0]], [[2.0], [6.0]], [0.0, -inf])
        ->  ([2.0], [1.0, 0.0])

    weights обязаны суммироваться в 1.0 и быть ровно 0.0 там, где маска
    даёт -inf. Скалярные произведения делим на sqrt(len(q)).

    Ловушка: max() по всем логитам вернёт -inf, если разрешена ни одна
    позиция; и exp(-inf - (-inf)) — это nan, а не 0. Считай максимум
    только по разрешённым логитам.
    """
    scale = 1.0 / math.sqrt(len(q))
    scores = []
    for k, m in zip(Ks, mask_row):
        if m == float("-inf"):
            scores.append(None)  # маскированные позиции вообще не считаем
        else:
            scores.append(sum(qi * ki for qi, ki in zip(q, k)) * scale)
    allowed = [s for s in scores if s is not None]
    top = max(allowed)
    exps = [0.0 if s is None else math.exp(s - top) for s in scores]
    total = sum(exps)
    weights = [e / total for e in exps]
    out = [0.0] * len(Vs[0])
    for w, v in zip(weights, Vs):
        if w == 0.0:
            continue
        for j in range(len(out)):
            out[j] += w * v[j]
    return out, weights


def diff_attention_row(q1, q2, K1, K2, Vs, mask_row, lam):
    """Differential attention: две карты внимания, вторая вычитается с весом lam.

    Вернуть кортеж (out, weights), где weights = w1 - lam * w2, а
    out = сумма weights[j] * Vs[j].

    diff_attention_row(q, q, K, K, V, mask, lam=0.0)  ->  как обычное внимание
    diff_attention_row(q, q, K, K, V, mask, lam=1.0)  ->  weights все нули

    Сумма weights равна 1 - lam, а не 1: нормировка сознательно ломается.
    Отдельные веса могут быть отрицательными — softmax такого не умеет
    никогда, и именно поэтому обычное внимание не может «вычесть» лишнее.

    Зачем: softmax обязан отдать суммарный вес 1, поэтому неинформативный
    запрос сваливает его на позицию 0 (attention sink). Вторая карта
    ловит именно этот паразитный компонент, вычитание его снимает.
    DIFF Transformer (Microsoft, 2024): -5..-10% perplexity.
    """
    _, w1 = masked_attention_row(q1, K1, Vs, mask_row)
    _, w2 = masked_attention_row(q2, K2, Vs, mask_row)
    weights = [a - lam * b for a, b in zip(w1, w2)]
    out = [0.0] * len(Vs[0])
    for w, v in zip(weights, Vs):
        for j in range(len(out)):
            out[j] += w * v[j]
    return out, weights


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2, window=None):
    """KV-cache в байтах; при заданном window хранится только окно, а не весь префикс.

    kv_cache_bytes(80, 8, 128, 131072)                 ->  42949672960  (~43 GB)
    kv_cache_bytes(80, 8, 128, 131072, window=1024)    ->  335544320    (~0.34 GB)
    kv_cache_bytes(80, 8, 128, 131072, window=999999)  ->  как без окна

    window=None — полное внимание. Иначе хранится min(seq_len, window)
    токенов: окно не может быть больше самой последовательности.

    Ловушка: differential attention платит ДВА кэша (две пары Q/K), то
    есть удваивает эту цифру. Экономия SWA и удвоение DIFF — это два
    конца одной и той же линейки.
    """
    kept = seq_len if window is None else min(seq_len, window)
    return 2 * n_layers * n_kv_heads * d_head * kept * dtype_bytes
