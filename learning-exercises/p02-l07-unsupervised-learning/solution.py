"""
Обучение без учителя — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def euclidean_distance(a, b):
    """Евклидово расстояние между двумя точками одинаковой размерности.

    euclidean_distance([0, 0], [3, 4])   ->  5.0
    euclidean_distance([1, 1], [1, 1])   ->  0.0

    Это единственная мера близости во всём уроке: и k-means, и силуэт, и
    DBSCAN считают «похожесть» именно через неё. Поменяешь метрику —
    поменяются все кластеры.
    """
    # zip обрывается по короткой последовательности: если размерности разные,
    # тихо посчитается ерунда. Здесь это допустимо — данные приходят из одного
    # источника, а лишняя проверка на каждый вызов внутри O(n^2) циклов дорога.
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def assign_clusters(data, centroids):
    """Каждой точке — индекс ближайшего центроида. Список той же длины, что data.

    assign_clusters([[0, 0], [9, 9]], [[0, 0], [10, 10]])  ->  [0, 1]
    assign_clusters([[4, 0], [6, 0]], [[0, 0], [10, 0]])   ->  [0, 1]

    Это шаг E алгоритма Ллойда: центры зафиксированы, двигаются только метки.
    """
    assignments = []
    for point in data:
        # min по ключу вместо построения полного списка расстояний и index(min):
        # один проход вместо трёх и никакого хранения промежуточного списка
        distances = [euclidean_distance(point, c) for c in centroids]
        assignments.append(distances.index(min(distances)))
    return assignments


def update_centroids(data, assignments, centroids):
    """Новые центры — средние точек своего кластера. Пустой кластер не двигается.

    update_centroids([[0, 0], [2, 2]], [0, 0], [[9, 9]])  ->  [[1.0, 1.0]]
    update_centroids([[0, 0]], [0], [[0, 0], [5, 5]])     ->  [[0.0, 0.0], [5, 5]]

    Ловушка: кластер может остаться без точек. Делить на ноль нельзя, поэтому
    старый центр возвращается как есть — тогда следующая итерация даст ему
    ещё один шанс подобрать точки.
    """
    n_features = len(data[0])
    members = [[] for _ in centroids]
    for point, cluster_id in zip(data, assignments):
        members[cluster_id].append(point)

    new_centroids = []
    for old, group in zip(centroids, members):
        if not group:
            new_centroids.append(list(old))
            continue
        new_centroids.append(
            [sum(p[j] for p in group) / len(group) for j in range(n_features)]
        )
    return new_centroids


def kmeans(data, k, max_iterations=100, seed=42):
    """Алгоритм Ллойда. Возвращает (assignments, centroids).

    kmeans([[0, 0], [0, 1], [9, 9], [9, 8]], 2)  ->  метки вида [0, 0, 1, 1]
    kmeans(data, 3, seed=1) == kmeans(data, 3, seed=1)   ->  True

    Цикл: назначить точки центрам, пересчитать центры, повторить. Останавливаемся,
    когда метки перестали меняться.

    Ловушка: k-means находит только ЛОКАЛЬНЫЙ минимум, ответ зависит от старта.
    Поэтому seed — обязательный параметр, а не глобальный random.seed().
    """
    # свой генератор вместо random.seed(): глобальное состояние процесса не
    # трогается, два вызова с одним seed дают один результат независимо от того,
    # что крутилось между ними
    rng = random.Random(seed)
    # копии, а не ссылки: иначе update_centroids вернёт списки, которые
    # указывают внутрь data, и один шаг испортит исходные точки
    centroids = [list(p) for p in rng.sample(data, k)]

    assignments = assign_clusters(data, centroids)
    for _ in range(max_iterations):
        centroids = update_centroids(data, assignments, centroids)
        new_assignments = assign_clusters(data, centroids)
        # сравниваем метки, а не сдвиг центров: метки — это целые числа,
        # сравнение точное, порога сходимости подбирать не нужно
        if new_assignments == assignments:
            break
        assignments = new_assignments
    return assignments, centroids


def inertia(data, assignments, centroids):
    """Сумма квадратов расстояний от точек до их центров. Меньше — плотнее.

    inertia([[0, 0], [2, 0]], [0, 0], [[1, 0]])  ->  2.0
    inertia([[5, 5]], [0], [[5, 5]])             ->  0.0

    Это ровно та величина, которую k-means минимизирует. Растёт число
    кластеров — инерция падает всегда, вплоть до нуля при k = числу точек.
    Поэтому «выбрать k по минимуму инерции» бессмысленно, ищут излом (локоть).
    """
    return sum(
        euclidean_distance(point, centroids[cluster_id]) ** 2
        for point, cluster_id in zip(data, assignments)
    )


def silhouette_score(data, assignments):
    """Средний силуэт разбиения: от -1 (метки перепутаны) до +1 (кластеры чёткие).

    silhouette_score([[0, 0], [0, 1], [9, 9], [9, 8]], [0, 0, 1, 1])  ->  ~0.917
    silhouette_score([[0, 0], [1, 1]], [0, 0])                        ->  0.0

    Для точки: a — среднее расстояние до своих, b — среднее до ближайшего
    чужого кластера, силуэт = (b - a) / max(a, b).

    Ловушки: при одном кластере b не существует — договорились возвращать 0.0;
    точка-одиночка в своём кластере тоже даёт 0.0, а не единицу.
    """
    clusters = {}
    for i, cluster_id in enumerate(assignments):
        clusters.setdefault(cluster_id, []).append(i)

    # один кластер (или ноль точек) — сравнивать не с чем, метрика не определена
    if len(clusters) < 2:
        return 0.0

    scores = []
    for i, own in enumerate(assignments):
        own_members = [j for j in clusters[own] if j != i]
        if not own_members:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)
        b = min(
            sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            for cluster_id, members in clusters.items()
            if cluster_id != own
        )
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return sum(scores) / len(scores)


def best_k_by_silhouette(data, k_values, seed=42):
    """Из списка кандидатов вернуть k с наибольшим средним силуэтом.

    best_k_by_silhouette(три_плотных_облака, [2, 3, 4])  ->  3
    best_k_by_silhouette(data, [2])                      ->  2

    Инерция для этого не годится: она монотонно падает с ростом k. Силуэт же
    штрафует и за слипшиеся, и за разорванные кластеры, поэтому у него есть
    настоящий максимум.
    """
    best_k, best_score = None, -2.0  # -2 заведомо меньше любого силуэта (>= -1)
    for k in k_values:
        assignments, _ = kmeans(data, k, seed=seed)
        score = silhouette_score(data, assignments)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def dbscan(data, eps, min_samples):
    """Плотностная кластеризация. Возвращает метки, -1 — шум (выброс).

    dbscan([[0, 0], [0, 1], [1, 0], [50, 50]], 2.0, 3)  ->  [0, 0, 0, -1]
    dbscan([[0, 0], [9, 9]], 1.0, 2)                    ->  [-1, -1]

    Точка — ядровая, если в радиусе eps (включая её саму) не меньше min_samples
    соседей. Ядровые точки, дотягивающиеся друг до друга, склеиваются в один
    кластер; неядровые соседи ядровых становятся граничными.

    В отличие от k-means, число кластеров не задаётся, а форма может быть любой
    (две дуги, кольцо). Зато результат детерминирован — случайности здесь нет.
    """
    n = len(data)
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def neighbours(i):
        return [j for j in range(n) if euclidean_distance(data[i], data[j]) <= eps]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        seeds = neighbours(i)
        if len(seeds) < min_samples:
            # не ядровая: пока шум. Метку могут перебить позже, если точка
            # окажется в eps-окрестности чьего-то ядра (станет граничной)
            continue

        labels[i] = cluster_id
        # очередь растёт прямо во время обхода: список, а не set, чтобы порядок
        # обхода был воспроизводим и метки кластеров не прыгали от запуска к запуску
        queue = [j for j in seeds if j != i]
        pos = 0
        while pos < len(queue):
            q = queue[pos]
            pos += 1
            if not visited[q]:
                visited[q] = True
                q_neighbours = neighbours(q)
                if len(q_neighbours) >= min_samples:
                    queue.extend(nb for nb in q_neighbours if nb not in queue)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels
