"""
GloVe, FastText и subword-эмбеддинги — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def build_cooccurrence(docs, window=5):
    """Матрица совстречаемости GloVe: словарь и веса пар с затуханием 1/расстояние.

    Вернуть кортеж (vocab, counts):
      vocab  — {слово: индекс}, индекс по первому появлению;
      counts — {(i, j): вес}, где вес — сумма 1/|i - j| по всем вхождениям.

    build_cooccurrence([["a", "b", "c"]], window=2)
        ->  ({'a': 0, 'b': 1, 'c': 2},
             {(0, 1): 1.0, (0, 2): 0.5, (1, 0): 1.0, (1, 2): 1.0,
              (2, 0): 0.5, (2, 1): 1.0})

    Соседнее слово даёт 1.0, слово через одно — 0.5, через два — 1/3.
    В этом вся разница с bag-of-words: GloVe помнит, насколько близко.

    Ловушка: окно обрезается по границе документа, пары через границу
    двух документов не считаются. Пары (i, i) не бывает.
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    counts = {}
    for doc in docs:
        # индексы считаем один раз на документ: поиск по dict дешевле,
        # чем повторять его во вложенном цикле по окну
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            lo = max(0, i - window)
            hi = min(len(indexed), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                key = (center, indexed[j])
                counts[key] = counts.get(key, 0.0) + 1.0 / abs(i - j)
    return vocab, counts


def glove_weight(x, x_max=100.0, alpha=0.75):
    """Весовая функция GloVe: f(x) = (x/x_max)^alpha при x < x_max, иначе 1.0.

    glove_weight(0.0)    ->  0.0
    glove_weight(100.0)  ->  1.0
    glove_weight(1000.0) ->  1.0   (выше x_max вес не растёт)

    Смысл: пара (the, and) встречается в миллион раз чаще пары (deep,
    learning). Без обрезки сверху первая задавила бы функцию потерь целиком,
    а без степени alpha < 1 редкие пары не получили бы вообще никакого веса.
    """
    if x >= x_max:
        return 1.0
    return (x / x_max) ** alpha


def glove_step(W, W_tilde, b, b_tilde, i, j, x_ij, lr, x_max=100.0, alpha=0.75):
    """Один шаг SGD для GloVe по одной паре (i, j). Меняет таблицы НА МЕСТЕ.

    W, W_tilde — списки списков (центральные и контекстные векторы),
    b, b_tilde — списки чисел (сдвиги). Вернуть значение потерь ДО шага.

    Потери на одной паре:
        diff = W[i] . W_tilde[j] + b[i] + b_tilde[j] - log(x_ij)
        L    = f(x_ij) * diff^2

    Градиенты:
        dL/dW[i]       = 2 * f * diff * W_tilde[j]
        dL/dW_tilde[j] = 2 * f * diff * W[i]
        dL/db[i]       = dL/db_tilde[j] = 2 * f * diff

    Ловушка: оба вектора обновляются от значений ДО шага. Если сначала
    записать новый W[i], а потом считать градиент по W_tilde[j], шаг
    перестанет совпадать с производной — численная проверка это поймает.

    Ловушка: в коде урока множитель 2 опущен (его прячут в lr). Здесь он
    оставлен, чтобы шаг был честной производной написанной функции потерь.
    """
    dim = len(W[i])
    v_i = list(W[i])  # копии: дальше обе таблицы будут меняться
    u_j = list(W_tilde[j])

    diff = sum(a * c for a, c in zip(v_i, u_j)) + b[i] + b_tilde[j] - math.log(x_ij)
    f = glove_weight(x_ij, x_max, alpha)
    coef = 2.0 * f * diff

    for k in range(dim):
        W[i][k] -= lr * coef * u_j[k]
        W_tilde[j][k] -= lr * coef * v_i[k]
    b[i] -= lr * coef
    b_tilde[j] -= lr * coef

    return f * diff * diff


def char_ngrams(word, n_min=3, n_max=6):
    """Множество символьных n-грамм слова плюс само слово в угловых скобках.

    Слово сначала оборачивается: "where" -> "<where>". Скобки — это маркеры
    начала и конца, они отличают приставку от такого же куска в середине.

    char_ngrams("cat", n_min=3, n_max=3)  ->  {'<ca', 'cat', 'at>', '<cat>'}
    len(char_ngrams("where"))             ->  13

    Ловушка: "<where>" попадает в результат ВСЕГДА, даже когда его длина
    больше n_max. Это отдельный элемент, а не n-грамм.

    Это ядро FastText: слово — сумма своих кусков, поэтому у никогда не
    виденного "whereupon" всё равно найдётся вектор.
    """
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i : i + n])
    return grams


def fasttext_vector(word, ngram_table, n_min=3, n_max=6):
    """Вектор слова по FastText: сумма векторов его известных n-грамм.

    ngram_table — {n-грамм: вектор-список}. Неизвестные n-граммы просто
    пропускаются. Если не нашлось ни одной — вернуть None.

    table = {"<ca": [1, 0], "cat": [0, 1]}
    fasttext_vector("cat", table, 3, 3)  ->  [1, 1]
    fasttext_vector("zzz", table, 3, 3)  ->  None

    Ровно это делает ft.get_word_vector(word) в библиотеке fasttext.

    Смысл: OOV-слово перестаёт быть проблемой. Word2Vec и GloVe на
    "dogecoin" разводят руками, FastText собирает вектор из кусков.
    """
    vecs = [ngram_table[g] for g in char_ngrams(word, n_min, n_max) if g in ngram_table]
    if not vecs:
        return None
    # суммируем поэлементно; порядок слагаемых не важен, сложение коммутативно,
    # поэтому множество n-грамм даёт однозначный ответ
    out = [0.0] * len(vecs[0])
    for v in vecs:
        for k in range(len(out)):
            out[k] += v[k]
    return out


def merge_pair(tokens, pair):
    """Склеить все вхождения соседней пары в списке токенов, слева направо.

    merge_pair(["l", "o", "w"], ("l", "o"))       ->  ['lo', 'w']
    merge_pair(["a", "a", "a"], ("a", "a"))       ->  ['aa', 'a']
    merge_pair(["c", "a", "t"], ("x", "y"))       ->  ['c', 'a', 't']

    Ловушка: после склейки индекс сдвигается на ДВА. Иначе на входе
    ["a","a","a"] средний токен успеет склеиться дважды.

    Это единственная операция, из которой собраны и learn_bpe, и apply_bpe.
    """
    a, b = pair
    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def learn_bpe(corpus, k_merges):
    """Обучение byte-pair encoding: список слияний в порядке их появления.

    corpus — {слово: частота}. Каждое слово разбивается на символы, в конец
    дописывается маркер конца слова "</w>". Дальше k_merges раз: посчитать
    частоты соседних пар по всему корпусу, слить самую частую в один токен.

    learn_bpe({"low": 5, "lower": 2}, 2)  ->  [('l', 'o'), ('lo', 'w')]

    Ловушка: при равной частоте нужен явный порядок, иначе один и тот же
    корпус даст разные словари на разных запусках. Берём лексикографически
    меньшую пару.

    Ловушка: пары кончились (все слова из одного токена) — цикл надо
    прервать, а не слить пустоту.

    Так устроен токенизатор любого современного LLM, только там 30-100 тысяч
    слияний вместо двух.
    """
    words = {tuple(word) + ("</w>",): freq for word, freq in corpus.items()}

    merges = []
    for _ in range(k_merges):
        pair_freq = {}
        for tokens, freq in words.items():
            for pair in zip(tokens, tokens[1:]):
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
        if not pair_freq:
            break
        # ключ (-частота, пара): максимум по частоте, при равенстве —
        # лексикографический минимум, и результат перестаёт зависеть
        # от порядка обхода словаря
        best = min(pair_freq, key=lambda p: (-pair_freq[p], p))
        merges.append(best)

        merged = {}
        for tokens, freq in words.items():
            key = tuple(merge_pair(list(tokens), best))
            merged[key] = merged.get(key, 0) + freq
        words = merged
    return merges


def apply_bpe(word, merges):
    """Разбить слово на subword-токены выученными слияниями.

    Слово разбивается на символы, в конец дописывается "</w>", затем
    слияния применяются В ТОМ ЖЕ ПОРЯДКЕ, в каком их выучил learn_bpe.

    merges = [('l', 'o'), ('lo', 'w')]
    apply_bpe("low", merges)  ->  ['low', '</w>']
    apply_bpe("cat", merges)  ->  ['c', 'a', 't', '</w>']

    Ловушка: порядок слияний принципиален. ('lo','w') не сработает, пока не
    применено ('l','o') — токена "lo" ещё не существует.

    Ловушка: слово с незнакомыми символами не падает, а остаётся набором
    отдельных символов. Именно поэтому у BPE не бывает OOV.
    """
    tokens = list(word) + ["</w>"]
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens
