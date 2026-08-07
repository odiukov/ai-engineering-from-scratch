"""
Эмбеддинги: похожесть, индекс, сжатие — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Метрики, которые понимает search. Держим списком, чтобы тест мог пройтись
# по всем и не переписывать перечень в двух местах.
METRICS = ("cosine", "dot", "euclidean", "hamming")


def dot(a, b):
    """Скалярное произведение двух векторов одинаковой длины.

    dot([1, 2, 3], [4, 5, 6])  ->  32.0
    dot([1, 0], [0, 1])        ->  0.0   (перпендикулярны)

    Разная длина — ValueError: молча обрезать по zip нельзя, иначе ошибка
    в размерности эмбеддинга превратится в тихо неверный поиск.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return float(sum(x * y for x, y in zip(a, b)))


def norm(v):
    """Евклидова длина вектора.

    norm([3, 4])  ->  5.0
    norm([0, 0])  ->  0.0

    Это корень из скалярного произведения вектора на себя — считай через
    dot, а не переписывай сумму квадратов заново.
    """
    return math.sqrt(dot(v, v))


def cosine_similarity(a, b):
    """Косинус угла между векторами: от -1 до 1.

    cosine_similarity([1, 0], [1, 0])   ->  1.0
    cosine_similarity([1, 0], [0, 1])   ->  0.0
    cosine_similarity([1, 0], [-1, 0])  ->  -1.0

    Формула: dot(a, b) / (norm(a) * norm(b)).

    Нулевой вектор: длины нет, угла нет — возвращай 0.0, а не деление на ноль.
    Пустой запрос в поиске это обычное дело.

    Главное свойство: косинус не зависит от длины векторов. Документ на 500
    слов и запрос на 3 слова сравнимы напрямую — поэтому косинус и стал
    метрикой по умолчанию в retrieval.
    """
    denom = norm(a) * norm(b)
    return dot(a, b) / denom if denom else 0.0


def euclidean_distance(a, b):
    """Евклидово расстояние между векторами. Чем меньше, тем ближе.

    euclidean_distance([0, 0], [3, 4])  ->  5.0
    euclidean_distance([1, 2], [1, 2])  ->  0.0

    В отличие от косинуса, расстояние чувствительно к длине вектора:
    удвоенный документ уедет далеко от оригинала, хотя смысл тот же.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def normalize(v):
    """Вектор той же длины и того же направления, но с нормой 1.

    normalize([3, 4])  ->  [0.6, 0.8]
    normalize([0, 0])  ->  [0.0, 0.0]

    Нулевой вектор нормировать некуда — возвращай нули, не деление на ноль.

    Зачем: на нормированных векторах скалярное произведение РАВНО косинусу.
    Именно поэтому провайдеры (OpenAI) отдают нормированные эмбеддинги —
    в проде тогда достаточно быстрого dot.
    """
    length = norm(v)
    return [x / length for x in v] if length else [0.0] * len(v)


def truncate_embedding(v, dims):
    """Matryoshka-усечение: первые dims координат, снова нормированные.

    truncate_embedding([0.6, 0.8, 0.0], 2)  ->  [0.6, 0.8]
    truncate_embedding([3.0, 4.0], 5)       ->  [0.6, 0.8]  (короче — берём всё)

    Модели с Matryoshka Representation Learning обучены так, что первые
    координаты несут больше всего информации. Обрезал 1536 до 256 — хранилище
    ужалось в шесть раз, точность просела на 3-5%.

    Нормировка после обрезки обязательна: иначе длина усечённого вектора
    зависит от того, сколько координат ты отрезал, и косинус поплывёт.

    Соответствует параметру dimensions в OpenAI embeddings API.
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")
    return normalize(v[:dims])


def binary_quantize(v):
    """Бинарная квантизация: знак каждой координаты в бит.

    binary_quantize([0.4, -0.1, 0.0, -2.0])  ->  [1, 0, 0, 0]

    Положительное — 1, ноль и отрицательное — 0. float32 (4 байта на
    координату) превращается в один бит: 32-кратная экономия памяти.

    Похожесть после этого считают расстоянием Хэмминга — числом
    несовпавших битов. Точность retrieval падает на 5-10%, поэтому в проде
    так делают только первый проход, а топ-1000 пересчитывают в полной
    точности.
    """
    return [1 if x > 0 else 0 for x in v]


def search(query, vectors, top_k=5, metric="cosine"):
    """Перебором найти top_k ближайших векторов. Возвращает [(индекс, score)].

    Больший score — ближе. Для euclidean score = -расстояние, для hamming
    score = -число несовпавших битов, чтобы сортировка везде была одна.

    search([1, 0], [[1, 0], [0, 1], [-1, 0]], top_k=2)
        ->  [(0, 1.0), (1, 0.0)]

    metric — одно из METRICS, иначе ValueError.
    При равных score первым идёт меньший индекс: без этого два запуска
    поиска дадут разный порядок и тесты начнут мигать.

    Это FlatIndex из FAISS: честный перебор, O(n) на запрос. Настоящая база
    строит HNSW и отвечает за O(log n), но с приближением.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}. available: {list(METRICS)}")

    if metric == "hamming":
        # квантуем запрос один раз, а не внутри цикла сравнения
        q_bits = binary_quantize(query)
        scored = []
        for i, v in enumerate(vectors):
            if len(v) != len(q_bits):
                raise ValueError(f"dimension mismatch: {len(q_bits)} vs {len(v)}")
            bits = binary_quantize(v)
            scored.append((i, -float(sum(x != y for x, y in zip(q_bits, bits)))))
    elif metric == "cosine":
        scored = [(i, cosine_similarity(query, v)) for i, v in enumerate(vectors)]
    elif metric == "dot":
        scored = [(i, dot(query, v)) for i, v in enumerate(vectors)]
    else:
        scored = [(i, -euclidean_distance(query, v)) for i, v in enumerate(vectors)]

    # сортировка стабильна, а исходный порядок — по индексу: равные score
    # автоматически идут по возрастанию индекса
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
