"""
Признаки: конструирование и отбор — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def min_max_scale(values):
    """Линейно сжать список в отрезок [0, 1]: минимум в 0, максимум в 1.

    min_max_scale([10, 20, 30])   ->  [0.0, 0.5, 1.0]
    min_max_scale([5, 5, 5])      ->  [0.0, 0.0, 0.0]

    Ловушка: когда все значения одинаковы, знаменатель max - min равен нулю.
    Делить нельзя — договорились возвращать нули.

    Зачем в AI: KNN, k-means и SVM считают расстояния. Признак «зарплата» в
    рублях без масштабирования просто задавит признак «возраст» в годах.
    """
    low, high = min(values), max(values)
    span = high - low
    if span == 0:
        return [0.0] * len(values)
    return [(v - low) / span for v in values]


def standardize(values):
    """Z-оценка: вычесть среднее, поделить на стандартное отклонение.

    standardize([1, 2, 3])        ->  [-1.2247..., 0.0, 1.2247...]
    standardize([5, 5, 5])        ->  [0.0, 0.0, 0.0]

    В отличие от min_max_scale не загоняет в отрезок: выбросы остаются далеко,
    и это плюс, когда «далеко» — полезный сигнал.

    Ловушка: стандартное отклонение считается по всей выборке (делим на n, а не
    на n - 1) — так метрика согласована с тем, что делают sklearn-скейлеры.
    """
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance == 0:
        return [0.0] * n
    std = math.sqrt(variance)
    return [(v - mean) / std for v in values]


def bin_values(values, n_bins=5):
    """Разложить числа по n_bins равным по ширине корзинам. Возвращает номера.

    bin_values([0, 5, 10], n_bins=2)      ->  [0, 1, 1]   (граница ровно на 5)
    bin_values([1, 2, 3, 4], n_bins=4)    ->  [0, 1, 2, 3]

    Ловушка: максимальное значение после масштабирования равно ровно 1.0 и даёт
    номер n_bins — на единицу больше последней корзины. Его надо прижать.

    Зачем в AI: линейная модель не умеет «до 30 лет одно, после 30 другое».
    Корзины превращают ступенчатую зависимость в набор категорий.
    """
    # переиспользуем масштабирование вместо повторного min/max: та же формула,
    # написанная дважды, — гарантированный источник расхождений
    scaled = min_max_scale(values)
    return [min(int(s * n_bins), n_bins - 1) for s in scaled]


def impute_median(values):
    """Заполнить None медианой. Возвращает (заполненный список, медиана).

    impute_median([1.0, None, 3.0])        ->  ([1.0, 2.0, 3.0], 2.0)
    impute_median([1.0, 2.0, 3.0, 4.0])    ->  ([1.0, 2.0, 3.0, 4.0], 2.5)

    Медиана, а не среднее: одна опечатка «зарплата 10 000 000» сдвинет среднее
    и испортит все пропуски, а медиану не тронет.

    Ловушка про утечку: медиана обязана считаться ТОЛЬКО по обучающей выборке.
    Поэтому функция и возвращает её вторым значением — то же число потом
    применяют к тесту, а не пересчитывают заново.
    """
    present = sorted(v for v in values if v is not None)
    if not present:
        # ни одного наблюдения — считать нечего, отдаём нейтральный ноль
        return [0.0] * len(values), 0.0

    mid = len(present) // 2
    if len(present) % 2 == 0:
        median = (present[mid - 1] + present[mid]) / 2
    else:
        median = present[mid]
    return [median if v is None else v for v in values], median


def one_hot_encode(values):
    """Категории -> двоичные столбцы. Возвращает (матрица, список категорий).

    one_hot_encode(["a", "b", "a"])  ->  ([[1, 0], [0, 1], [1, 0]], ["a", "b"])
    one_hot_encode(["z"])            ->  ([[1]], ["z"])

    Категории сортируются: порядок столбцов не должен зависеть от того, какая
    строка попалась в данных первой, иначе обученная модель развалится на
    новом файле с теми же категориями в другом порядке.

    Ловушка: столбцов ровно столько, сколько категорий в ЭТОЙ выборке. Категория,
    впервые встреченная на тесте, кодироваться уже нечем.
    """
    categories = sorted(set(values))
    index = {cat: i for i, cat in enumerate(categories)}

    rows = []
    for value in values:
        row = [0] * len(categories)
        row[index[value]] = 1
        rows.append(row)
    return rows, categories


def target_encode(feature_values, target_values, smoothing=10):
    """Категория -> сглаженное среднее целевой переменной. (Закодировано, словарь).

    target_encode(["a", "a", "b"], [10, 20, 100], smoothing=0)  ->
        ([15.0, 15.0, 100.0], {"a": 15.0, "b": 100.0})
    target_encode(["a", "a", "b"], [10, 20, 100], smoothing=1)  ->
        примерно ([24.44, 24.44, 71.67], {...})

    Формула: weight = count / (count + smoothing), результат =
    weight * среднее_категории + (1 - weight) * общее_среднее. Чем реже
    категория, тем сильнее её тянет к общему среднему.

    Ловушка (главная в уроке): при smoothing=0 категория, встреченная один раз,
    кодируется собственным ответом — это прямая утечка. Модель «запоминает»
    целевую переменную и на тесте разваливается. Словарь возвращается для того,
    чтобы применить к тесту кодировку, посчитанную на train.
    """
    global_mean = sum(target_values) / len(target_values)

    sums, counts = {}, {}
    for feature, target in zip(feature_values, target_values):
        sums[feature] = sums.get(feature, 0.0) + target
        counts[feature] = counts.get(feature, 0) + 1

    mapping = {}
    for category, total in sums.items():
        count = counts[category]
        weight = count / (count + smoothing)
        mapping[category] = weight * (total / count) + (1 - weight) * global_mean

    return [mapping[v] for v in feature_values], mapping


def tfidf(documents):
    """TF-IDF матрица корпуса. Возвращает (векторы, словарь слово -> столбец).

    tfidf(["a b", "a c"])  ->  у слова "a" вес 0.0 в обоих документах
    tfidf(["cat", "dog"])  ->  каждое слово встречается в 1 из 2 документов

    TF = доля слова в документе, IDF = log(всего документов / документов со
    словом), вес = TF * IDF.

    Ловушка: слово, которое есть во ВСЕХ документах, получает IDF = log(1) = 0
    и вылетает из представления. Это не баг, а весь смысл IDF: «the» ничего не
    отличает.

    Зачем в AI: до эмбеддингов это был стандартный вход для классификаторов
    текста, и до сих пор — крепкий бейзлайн, который лень бить нейросетью.
    """
    tokenised = [doc.lower().split() for doc in documents]

    vocab = {}
    for words in tokenised:
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    # в скольких документах слово встретилось хотя бы раз — считаем по set,
    # иначе документ с десятью «the» накрутит счётчик в десять раз
    doc_freq = {}
    for words in tokenised:
        for word in set(words):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    n_docs = len(documents)
    vectors = []
    for words in tokenised:
        vector = [0.0] * len(vocab)
        for word in set(words):
            tf = words.count(word) / len(words)
            idf = math.log(n_docs / doc_freq[word])
            vector[vocab[word]] = tf * idf
        vectors.append(vector)
    return vectors, vocab


def mutual_information(feature, target, n_bins=10):
    """Взаимная информация числового признака и меток класса, в натах. >= 0.

    mutual_information([0, 0, 1, 1], [0, 0, 1, 1], n_bins=2)  ->  0.6931... (ln 2)
    mutual_information([5, 5, 5, 5], [0, 0, 1, 1], n_bins=2)  ->  0.0

    Сумма по парам (корзина, класс) от p(x, y) * log(p(x, y) / (p(x) * p(y))).
    Ноль означает независимость: признак ничего не говорит о цели.

    Ловушка: MI не бывает отрицательной, и пары с нулевой вероятностью в сумму
    не входят вовсе — log(0) взорвётся.

    Зачем в AI: фильтр отбора признаков, который, в отличие от корреляции,
    ловит и нелинейные связи.
    """
    n = len(feature)
    binned = bin_values(feature, n_bins)

    p_feature = {}
    for b in binned:
        p_feature[b] = p_feature.get(b, 0) + 1 / n
    p_target = {}
    for t in target:
        p_target[t] = p_target.get(t, 0) + 1 / n

    joint = {}
    for b, t in zip(binned, target):
        joint[(b, t)] = joint.get((b, t), 0) + 1 / n

    return sum(
        p * math.log(p / (p_feature[b] * p_target[t]))
        for (b, t), p in joint.items()
    )
