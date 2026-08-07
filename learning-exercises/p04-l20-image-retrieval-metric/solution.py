"""
Поиск по картинкам и metric learning — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def l2_normalize(vec):
    """Привести вектор к единичной длине: поделить на его L2-норму.

    l2_normalize([3.0, 4.0])  ->  [0.6, 0.8]
    l2_normalize([0.0, 0.0])  ->  [0.0, 0.0]

    Ловушка: нулевой вектор. Делить на ноль нельзя, а падать посреди
    индексации миллиона картинок — тем более. Верни его как есть.

    Зачем: на нормированных эмбеддингах скалярное произведение равно
    косинусу, а FAISS IndexFlatIP умеет только скалярное произведение.
    Нормировка — это и есть переход "IP-индекс считает косинус".
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def cosine_similarity(a, b):
    """Косинус угла между векторами: от -1 (противоположны) до 1 (сонаправлены).

    cosine_similarity([1.0, 0.0], [2.0, 0.0])  ->  1.0
    cosine_similarity([1.0, 0.0], [0.0, 1.0])  ->  0.0
    cosine_similarity([1.0, 0.0], [-1.0, 0.0]) ->  -1.0

    Косинус не зависит от длины векторов — только от направления. Именно
    поэтому "яркая копия картинки" и "тусклая копия" оказываются рядом.

    Нулевой вектор: угла нет, возвращай 0.0 вместо деления на ноль.
    """
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def euclidean_distance(a, b):
    """Обычное евклидово расстояние между двумя векторами.

    euclidean_distance([0.0, 0.0], [3.0, 4.0])  ->  5.0
    euclidean_distance([1.0], [1.0])            ->  0.0

    Полезное тождество: для НОРМИРОВАННЫХ векторов
    ||a - b||^2 = 2 - 2 * cos(a, b). То есть на единичной сфере косинус и
    евклид упорядочивают соседей одинаково — можно выбирать любой, но
    смешивать их в одной системе нельзя.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def triplet_loss(anchor, positive, negative, margin=0.2):
    """Triplet loss: притянуть positive, оттолкнуть negative с зазором margin.

    L = max(0, d(a, p) - d(a, n) + margin)

    triplet_loss([0.0, 0.0], [1.0, 0.0], [0.0, 3.0])  ->  0.0
    triplet_loss([0.0, 0.0], [0.0, 2.0], [1.0, 0.0])  ->  1.2

    Ноль означает "этот триплет уже разведён, учить нечему". Отсюда и всё
    значение майнинга: батч из лёгких триплетов даёт нулевой градиент, и
    сеть просто стоит на месте.
    """
    d_ap = euclidean_distance(anchor, positive)
    d_an = euclidean_distance(anchor, negative)
    return max(0.0, d_ap - d_an + margin)


def semi_hard_negative(anchor, positive, negatives, margin=0.2):
    """Индекс semi-hard negative: дальше positive, но ближе, чем positive + margin.

    Из подходящих берём САМЫЙ БЛИЗКИЙ к anchor — он самый информативный.
    Если ни один негатив не попал в зону, возвращаем самый близкий из всех
    (hardest negative) — иначе шаг обучения пропадёт впустую.

    semi_hard_negative([0.0], [1.0], [[5.0], [1.1], [0.5]], margin=0.5)  ->  1
    semi_hard_negative([0.0], [1.0], [[9.0], [8.0]], margin=0.5)         ->  1

    Разбор первого примера: d_ap = 1. Кандидаты должны лежать в (1, 1.5).
    [5.0] слишком далеко, [0.5] ближе positive (это hard negative, на нём
    обучение разваливается), остаётся [1.1] — индекс 1.

    Ловушка: hard negative (ближе positive) сознательно исключается. Рецепт
    FaceNet 2015 года: на них лосс взрывается и коллапсирует эмбеддинг.
    """
    d_ap = euclidean_distance(anchor, positive)
    dists = [euclidean_distance(anchor, n) for n in negatives]

    band = [i for i, d in enumerate(dists) if d_ap < d < d_ap + margin]
    pool = band if band else list(range(len(dists)))
    # min по расстоянию: самый трудный из допустимых. Ключ с индексом делает
    # выбор детерминированным при совпадающих расстояниях
    return min(pool, key=lambda i: (dists[i], i))


def rank_gallery(query, gallery):
    """Индексы галереи, отсортированные по убыванию косинуса с запросом.

    rank_gallery([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])  ->  [1, 0, 2]

    При равных косинусах порядок задаём индексом — ранжирование обязано
    быть детерминированным, иначе метрики поедут от запуска к запуску.

    Это ровно то, что делает FAISS IndexFlatIP: полный перебор, точный
    ответ. Приближённые индексы (IVF, HNSW) отвечают быстрее, но не всегда
    тем же списком.
    """
    sims = [cosine_similarity(query, g) for g in gallery]
    return sorted(range(len(gallery)), key=lambda i: (-sims[i], i))


def recall_at_k(queries, query_labels, gallery, gallery_labels, k=1):
    """Recall@K: доля запросов, у которых ХОТЯ БЫ один верный сосед в топ-K.

    Верный сосед — тот, у кого метка совпала с меткой запроса.

    recall_at_k([[1.0, 0.0]], [0], [[1.0, 0.0], [0.0, 1.0]], [0, 1], k=1)  ->  1.0

    Ловушка: это ХОТЯ БЫ один, а не "сколько всего". Recall@K не убывает
    при росте K — это свойство надо помнить, оно ловит половину ошибок в
    реализации.
    """
    hits = 0
    for q, label in zip(queries, query_labels):
        top = rank_gallery(q, gallery)[:k]
        if any(gallery_labels[i] == label for i in top):
            hits += 1
    return hits / len(queries) if queries else 0.0


def precision_at_k(queries, query_labels, gallery, gallery_labels, k=1):
    """Precision@K: средняя доля верных среди топ-K по каждому запросу.

    precision_at_k([[1.0, 0.0]], [0],
                   [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], [0, 0, 1], k=3)
        ->  примерно 0.667

    Отличие от recall@K: один верный сосед из десяти даёт recall 1.0 и
    precision 0.1. Для поиска дублей важна precision (каждый ложный
    результат видит пользователь), для визуального поиска — recall.

    Полезная проверка: precision@K никогда не больше recall@K.
    """
    total = 0.0
    for q, label in zip(queries, query_labels):
        top = rank_gallery(q, gallery)[:k]
        total += sum(1 for i in top if gallery_labels[i] == label) / k
    return total / len(queries) if queries else 0.0
