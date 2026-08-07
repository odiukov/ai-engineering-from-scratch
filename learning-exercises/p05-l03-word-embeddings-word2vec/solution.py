"""
Эмбеддинги слов: Word2Vec с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def build_vocab(docs):
    """Словарь {слово: индекс} по списку токенизированных документов.

    build_vocab([["the", "cat"], ["the", "dog"]])  ->  {'the': 0, 'cat': 1, 'dog': 2}

    Индекс — порядок первого появления. Он же номер строки в матрице
    эмбеддингов: W[vocab["cat"]] — вектор слова "cat".
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def skipgram_pairs(docs, window=2):
    """Все пары (центральное слово, слово из окна) — обучающая выборка skip-gram.

    skipgram_pairs([["a", "b", "c"]], window=1)
        ->  [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b')]

    Идём по документу слева направо. Для позиции i берём все позиции j из
    [i - window, i + window], кроме самой i. Порядок пар — сначала по i,
    внутри — по возрастанию j.

    Ловушка: окно у краёв документа обрезается, а не заворачивается.
    Первое слово даёт меньше пар, чем слово в середине.

    Это и есть дистрибутивная гипотеза в коде: «слово определяется своим
    окружением». Больше ничего Word2Vec о языке не знает.
    """
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            # max/min обрезают окно по краям документа — так пары не
            # утекают в соседние документы и не выходят за границы списка
            lo = max(0, i - window)
            hi = min(len(doc), i + window + 1)
            for j in range(lo, hi):
                if i != j:
                    pairs.append((center, doc[j]))
    return pairs


def sigmoid(x):
    """Сигмоида с обрезкой аргумента до [-20, 20].

    sigmoid(0)      ->  0.5
    sigmoid(-1000)  ->  sigmoid(-20), то есть примерно 2.06e-9

    Обрезка — не косметика. math.exp(1000) роняет программу с
    OverflowError, а на расстоянии 20 от нуля сигмоида уже неотличима от
    своих пределов, так что смысл не теряется. Ровно это делает
    np.clip(x, -20, 20) в коде урока.

    В negative sampling сигмоида превращает скалярное произведение
    векторов в вероятность «эти слова встречаются рядом».
    """
    if x > 20:
        x = 20.0
    elif x < -20:
        x = -20.0
    return 1.0 / (1.0 + math.exp(-x))


def negative_samples(vocab_size, exclude, k, rng):
    """k случайных индексов слов, не попавших в exclude.

    rng — объект random.Random. Глобальный random использовать нельзя:
    обучение обязано воспроизводиться от seed.

    negative_samples(5, {0, 1}, 3, random.Random(0))  ->  список из 3 чисел,
        каждое в диапазоне 2..4, повторы разрешены

    Повторы разрешены сознательно: настоящий Word2Vec тянет негативы с
    возвращением, так проще и быстрее.

    Ловушка: если exclude накрывает весь словарь, цикл «тяни, пока не
    подойдёт» никогда не закончится. Такой вызов должен упасть с
    ValueError, а не зависнуть.

    Смысл: вместо softmax по 100 тысячам слов мы решаем бинарную задачу
    против горстки случайных слов. Отсюда вся скорость Word2Vec.
    """
    forbidden = set(exclude)
    if all(i in forbidden for i in range(vocab_size)):
        raise ValueError("negative sampling has no candidates left")
    out = []
    while len(out) < k:
        i = rng.randrange(vocab_size)
        if i not in forbidden:
            out.append(i)
    return out


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    """Один шаг negative sampling. Меняет W и W_prime НА МЕСТЕ, возвращает loss.

    W — таблица центральных векторов (её и оставляют как эмбеддинги),
    W_prime — таблица контекстных векторов (её обычно выбрасывают).
    Обе — списки списков.

    Функция потерь для одной пары:
        L = -log sigmoid(v_c . u_pos) - sum_neg log sigmoid(-v_c . u_neg)

    Градиенты (обозначим p = sigmoid(v_c . u_pos), n_i = sigmoid(v_c . u_i)):
        dL/dv_c   = (p - 1) * u_pos + sum_i n_i * u_i
        dL/du_pos = (p - 1) * v_c
        dL/du_i   = n_i * v_c

    Обновление: вычесть lr * градиент.

    Ловушка: все градиенты считаются от ЗНАЧЕНИЙ ДО шага. Если обновить
    W[center] раньше, чем посчитан градиент по W_prime, шаг перестанет
    совпадать с производной, и численная проверка это поймает.

    Смысл loss: положительную пару тянем к 1, негативные — к 0. Повторяя
    это миллиарды раз, получаем геометрию, где похожие слова рядом.
    """
    v_c = list(W[center_idx])  # копия: дальше W будет меняться
    u_pos = list(W_prime[context_idx])
    u_negs = [list(W_prime[n]) for n in negative_indices]
    dim = len(v_c)

    p = sigmoid(sum(a * b for a, b in zip(v_c, u_pos)))
    n_scores = [sigmoid(sum(a * b for a, b in zip(v_c, u))) for u in u_negs]

    grad_center = [(p - 1) * u_pos[k] for k in range(dim)]
    for score, u in zip(n_scores, u_negs):
        for k in range(dim):
            grad_center[k] += score * u[k]

    for k in range(dim):
        W_prime[context_idx][k] -= lr * (p - 1) * v_c[k]
    for neg_idx, score in zip(negative_indices, n_scores):
        for k in range(dim):
            W_prime[neg_idx][k] -= lr * score * v_c[k]
    for k in range(dim):
        W[center_idx][k] -= lr * grad_center[k]

    # 1e-12 под логарифмом: sigmoid может выдать ровно 0.0 на краю
    loss = -math.log(max(p, 1e-12))
    for score in n_scores:
        loss -= math.log(max(1.0 - score, 1e-12))
    return loss


def cosine_similarity(a, b):
    """Косинус угла между векторами.

    cosine_similarity([1, 0], [2, 0])  ->  1.0
    cosine_similarity([1, 0], [0, 1])  ->  0.0

    Длина вектора не влияет на ответ — поэтому эмбеддинги сравнивают
    косинусом, а не евклидовым расстоянием.

    Ловушка: нулевой вектор. Угла нет, возвращаем 0.0.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def nearest(vocab, W, target_vec, topk=5, exclude=None):
    """Топ-k слов, ближайших к target_vec по косинусу.

    Вернуть список пар (слово, косинус), отсортированный по убыванию косинуса.
    exclude — множество ИНДЕКСОВ, которые надо пропустить (не слов).

    vocab = {'a': 0, 'b': 1}, W = [[1, 0], [0, 1]]
    nearest(vocab, W, [1, 0], topk=1)  ->  [('a', 1.0)]

    Ловушка: при равных косинусах порядок должен быть детерминированным.
    Сортируй по ключу (-косинус, индекс слова), иначе один и тот же вызов
    будет давать разные ответы на разных запусках.

    Это ядро любого векторного поиска: тот же перебор, только по миллиарду
    строк и с ANN-индексом вместо полного прохода.
    """
    skip = set(exclude or ())
    scored = [
        (-cosine_similarity(W[idx], target_vec), idx, word)
        for word, idx in vocab.items()
        if idx not in skip
    ]
    scored.sort()  # ключ (-косинус, индекс) делает порядок однозначным
    return [(word, -neg_sim) for neg_sim, _, word in scored[:topk]]


def analogy(vocab, W, a, b, c, topk=5):
    """Аналогия «a относится к b, как c относится к ?».

    Вектор запроса: W[b] - W[a] + W[c]. Сами a, b, c из ответа исключаются,
    иначе первым же местом вернётся c — он ближе всех к самому себе.

    analogy(vocab, W, "man", "king", "woman")  ->  [('queen', 0.71), ...]

    king - man ловит что-то вроде «королевскости»; прибавив это к woman,
    попадаем в область королев. Модель при этом не знает ни что такое
    король, ни что такое пол.

    Ловушка: без исключения a, b, c аналогия почти всегда возвращает c.
    """
    ia, ib, ic = vocab[a], vocab[b], vocab[c]
    target = [W[ib][k] - W[ia][k] + W[ic][k] for k in range(len(W[ia]))]
    return nearest(vocab, W, target, topk=topk, exclude={ia, ib, ic})
