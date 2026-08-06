"""
Метод k ближайших соседей и расстояния — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def l2_distance(a, b):
    """Евклидово расстояние: корень из суммы квадратов разностей.

    l2_distance([0.0, 0.0], [3.0, 4.0])  ->  5.0
    l2_distance([1.0], [1.0])            ->  0.0

    Метрика по умолчанию. Чувствительна к масштабу признаков: столбец в
    тысячах затопчет столбец в единицах, поэтому перед KNN данные почти
    всегда стандартизуют.
    """
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def l1_distance(a, b):
    """Манхэттенское расстояние: сумма модулей разностей.

    l1_distance([0.0, 0.0], [3.0, 4.0])  ->  7.0
    l1_distance([1.0], [-2.0])           ->  3.0

    Разности не возводятся в квадрат, поэтому один выброс по одному
    признаку не перекрывает собой все остальные. На данных с выбросами L1
    устойчивее L2.
    """
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def cosine_distance(a, b):
    """Косинусное расстояние: 1 - (a·b) / (|a| * |b|).

    cosine_distance([1.0, 0.0], [5.0, 0.0])   ->  0.0  (одно направление)
    cosine_distance([1.0, 0.0], [0.0, 1.0])   ->  1.0  (перпендикуляры)
    cosine_distance([1.0, 0.0], [-1.0, 0.0])  ->  2.0  (противоположны)

    Длина векторов не влияет вообще — важен только угол. Ловушка: нулевой
    вектор даёт нулевую норму, делить нельзя. Верни 1.0.

    Это метрика номер один для эмбеддингов: у текста длина вектора несёт
    больше шума, чем смысла, а направление и есть смысл. Векторные базы и
    RAG ищут ближайших соседей именно по косинусу.
    """
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def minkowski_distance(a, b, p=2):
    """Обобщение L1 и L2: (сумма |a_i - b_i|^p)^(1/p).

    minkowski_distance([0.0, 0.0], [3.0, 4.0], p=1)  ->  7.0   (это L1)
    minkowski_distance([0.0, 0.0], [3.0, 4.0], p=2)  ->  5.0   (это L2)
    minkowski_distance([0.0, 0.0], [3.0, 4.0], p=float("inf"))  ->  4.0

    Ловушка: при p = inf формула не работает — возводить в бесконечную
    степень нельзя. Это предельный случай, расстояние Чебышёва:
    максимум модуля разности по координатам. Обработай его отдельно.
    """
    if p == float("inf"):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)


def k_nearest(X_train, query, k, distance_fn=l2_distance):
    """k ближайших к query объектов как список пар (расстояние, индекс).

    Список отсортирован по возрастанию расстояния. Если k больше размера
    выборки, вернуть все объекты.

    k_nearest([[0.0], [10.0], [1.0]], [0.0], 2)  ->  [(0.0, 0), (1.0, 2)]

    Индекс нужен, чтобы потом достать метку из y_train, а расстояние —
    чтобы взвесить голос соседа.

    Ловушка: сортировать надо пары, а не расстояния отдельно от индексов.

    Обучения здесь нет вообще: вся работа происходит в момент запроса,
    O(n*d) на каждый. Это ленивое обучение — и ровно так же устроен поиск
    в векторной базе, только с индексом поверх.
    """
    pairs = [(distance_fn(x, query), i) for i, x in enumerate(X_train)]
    pairs.sort()
    return pairs[:k]


def knn_classify(X_train, y_train, query, k=3, distance_fn=l2_distance, weighted=False):
    """Классификация голосованием k соседей.

    weighted=False — каждый сосед даёт один голос.
    weighted=True  — вес соседа равен 1 / (расстояние + 1e-10), близкие
    соседи весят больше.

    knn_classify([[0.0], [1.0], [10.0]], [0, 0, 1], [0.5], k=3)  ->  0

    При равенстве голосов побеждает меньшая метка — иначе ответ зависит от
    порядка словаря и тесты начинают мигать.

    Ловушки: epsilon в знаменателе обязателен, иначе запрос, совпавший с
    обучающим объектом, роняет функцию делением на ноль. И k стоит брать
    нечётным для двух классов, чтобы ничьих было меньше.
    """
    votes = {}
    for distance, index in k_nearest(X_train, query, k, distance_fn):
        weight = 1.0 / (distance + 1e-10) if weighted else 1.0
        label = y_train[index]
        votes[label] = votes.get(label, 0.0) + weight
    return max(sorted(votes), key=votes.get)


def knn_regress(X_train, y_train, query, k=3, distance_fn=l2_distance, weighted=False):
    """Регрессия усреднением значений k соседей.

    weighted=False — обычное среднее.
    weighted=True  — сумма(w_i * y_i) / сумма(w_i), где w_i = 1/(d_i + 1e-10).

    knn_regress([[0.0], [2.0]], [10.0, 20.0], [1.0], k=2)  ->  15.0

    Важное свойство: ответ всегда лежит между минимумом и максимумом
    обучающих значений. KNN не умеет экстраполировать — обучали на
    значениях от 0 до 100, значит 200 он не предскажет никогда.
    """
    neighbours = k_nearest(X_train, query, k, distance_fn)
    if not weighted:
        return sum(y_train[i] for _, i in neighbours) / len(neighbours)
    weights = [1.0 / (d + 1e-10) for d, _ in neighbours]
    total = sum(weights)
    return sum(w * y_train[i] for w, (_, i) in zip(weights, neighbours)) / total


def standardize(X):
    """Привести каждый признак к нулевому среднему и единичному разбросу.

    Вернуть тройку (X_scaled, means, stds). Разброс считается по формуле
    популяции (делим на n, не на n-1).

    standardize([[0.0], [2.0]])  ->  ([[-1.0], [1.0]], [1.0], [1.0])

    Ловушка: постоянный признак даёт std = 0. Такой столбец должен стать
    нулями, а не NaN и не исключением.

    Для KNN это не украшение, а условие работоспособности: расстояние —
    сумма по всем признакам, и признак с размахом в тысячи единиц один
    определяет, кто чей сосед. Остальные признаки просто перестают
    существовать.
    """
    n, d = len(X), len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = [
        (sum((row[j] - means[j]) ** 2 for row in X) / n) ** 0.5 for j in range(d)
    ]
    scaled = [
        [(row[j] - means[j]) / stds[j] if stds[j] else 0.0 for j in range(d)]
        for row in X
    ]
    return scaled, means, stds
