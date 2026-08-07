"""
Native Sparse Attention (DeepSeek NSA) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

NSA прогоняет внимание трижды, по трём разным взглядам на KV-кэш:
  1. сжатая ветка   — блоки по l токенов, усреднённые в один «конспект»;
  2. выбранная ветка — top-k блоков по оценкам сжатой ветки, но токены
     берутся исходные, несжатые;
  3. оконная ветка  — последние w токенов, локальный контекст.
Три выхода складываются с обучаемыми гейтами.

Ключевая проверка каждой разреженной схемы: при полном окне она обязана
совпасть с плотным вниманием. Если не совпадает — сломана она, а не
плотное внимание.

Матрица — список строк. K[i] это ключ i-го токена.
"""

import math


def softmax(row):
    """Строка логитов -> распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([0.0, 1000.0])    ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум строки
    перед экспонентой.
    """
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def attention_weights(q, K):
    """Веса внимания запроса q по ключам K: softmax(q . k_i / sqrt(d)).

    attention_weights([1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])  ->  [0.5, 0.5]

    d берётся как len(q). Пустой список ключей -> ValueError: делить не на
    что, а вернуть [] значит уронить ветку тихо.
    """
    if not K:
        raise ValueError("список ключей пуст")
    scale = math.sqrt(len(q))
    scores = [sum(a * b for a, b in zip(q, k)) / scale for k in K]
    return softmax(scores)


def attend(weights, V):
    """Взвешенная сумма строк значений: out[j] = sum_i w_i * V[i][j].

    attend([0.5, 0.5], [[1.0, 0.0], [3.0, 4.0]])  ->  [2.0, 2.0]
    """
    d_v = len(V[0])
    return [sum(w * v[j] for w, v in zip(weights, V)) for j in range(d_v)]


def compress_blocks(K, l):
    """Сжатая ветка: блоки по l строк, каждый усредняется в одну строку.

    compress_blocks([[1.0], [3.0], [5.0]], 2)  ->  [[2.0], [5.0]]
    compress_blocks([[1.0], [3.0]], 1)         ->  [[1.0], [3.0]]

    Хвост короче l усредняется по тому, что в нём реально есть, — делить
    на l нельзя, иначе последний блок окажется заниженным.

    В настоящей NSA здесь обучаемый MLP; среднее — честная заглушка,
    которая не мешает увидеть остальную конструкцию.
    """
    if l < 1:
        raise ValueError(f"размер блока должен быть >= 1, получено {l}")
    d = len(K[0])
    out = []
    for start in range(0, len(K), l):
        block = K[start : start + l]
        out.append([sum(row[j] for row in block) / len(block) for j in range(d)])
    return out


def top_k_blocks(weights, k):
    """Индексы k блоков с наибольшими весами, по возрастанию.

    top_k_blocks([0.1, 0.5, 0.2, 0.4], 2)  ->  [1, 3]
    top_k_blocks([0.1, 0.5], 5)            ->  [0, 1]   (k больше числа блоков)

    При равных весах побеждает меньший индекс — выбор обязан быть
    воспроизводимым.

    Тонкость урока: это единственное недифференцируемое место NSA, и оно
    ни на что не влияет в графе — top_k только решает, какие блоки грузить
    из памяти. Градиент течёт через оценки сжатой ветки.
    """
    k = min(k, len(weights))
    ranked = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    return sorted(ranked[:k])


def selected_branch(q, K, V, l, k):
    """Выбранная ветка: top-k блоков по сжатым оценкам, токены — исходные.

    Порядок: сжали K -> посчитали веса по сжатым ключам -> взяли top-k
    блоков -> собрали ИСХОДНЫЕ токены этих блоков -> обычное внимание.

    Свойство для проверки: при l = 1 и k >= числа токенов ответ совпадает
    с плотным вниманием по всей последовательности. Так и должно быть —
    разреженность с полным окном обязана вырождаться в плотную.

    Веса пересчитываются по несжатым ключам: сжатые оценки нужны только
    чтобы выбрать блоки, а не чтобы взвешивать токены внутри них.
    """
    block_weights = attention_weights(q, compress_blocks(K, l))
    chosen = top_k_blocks(block_weights, k)

    ids = []
    for b in chosen:
        ids.extend(range(b * l, min((b + 1) * l, len(K))))

    sub_K = [K[i] for i in ids]
    sub_V = [V[i] for i in ids]
    return attend(attention_weights(q, sub_K), sub_V)


def nsa_attention(q, K, V, l, k, w, gates):
    """Три ветки NSA, сложенные с гейтами (g_cmp, g_sel, g_win).

    Модуль модели: NSA-блок целиком.

    out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win

    Гейты НЕ обязаны давать в сумме единицу: в статье это выход маленького
    MLP по запросу, ветки взвешиваются независимо.

    Три полезные проверки:
      gates = (0, 0, 1), w >= len(K)  ->  обычное плотное внимание;
      gates = (0, 1, 0), l = 1, k >= len(K)  ->  тоже плотное внимание;
      gates = (0, 0, 0)  ->  нулевой вектор.
    """
    g_cmp, g_sel, g_win = gates

    # сжатая ветка: и ключи, и значения усредняются по одним и тем же блокам
    out_cmp = attend(attention_weights(q, compress_blocks(K, l)), compress_blocks(V, l))
    out_sel = selected_branch(q, K, V, l, k)

    # оконная ветка: последние w токенов; w больше длины — берём всё
    window = max(1, min(w, len(K)))
    out_win = attend(attention_weights(q, K[-window:]), V[-window:])

    return [
        g_cmp * a + g_sel * b + g_win * c for a, b, c in zip(out_cmp, out_sel, out_win)
    ]


def keys_per_query(n, l, k, b, w):
    """Бюджет вычислений: сколько ключей видит один запрос в каждой ветке.

    Возвращает словарь с ключами compressed, selected, window, total,
    full, reduction (= full / total).

    keys_per_query(64000, 64, 16, 64, 512)["total"]  ->  2536
    keys_per_query(64000, 64, 16, 64, 512)["full"]   ->  64000

    compressed = ceil(n / l), selected = min(k * b, n), window = min(w, n).
    Ограничение по n обязательно: нельзя прочитать больше ключей, чем есть.

    Ради чего всё: на 64k выигрыш 25x, на 128k уже 36x. Экономия растёт
    вместе с длиной контекста — в этом и весь смысл.
    """
    compressed = -(-n // l)  # ceil без импорта
    selected = min(k * b, n)
    window = min(w, n)
    total = compressed + selected + window
    return {
        "compressed": compressed,
        "selected": selected,
        "window": window,
        "total": total,
        "full": n,
        "reduction": n / total,
    }
