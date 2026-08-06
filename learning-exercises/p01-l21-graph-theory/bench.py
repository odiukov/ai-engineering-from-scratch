"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _random_undirected(n, edges_per_node):
    """Случайный неориентированный граф без петель, рёбра записаны с обоих концов."""
    adjacency = {node: set() for node in range(n)}
    for node in range(n):
        for _ in range(edges_per_node):
            other = random.randrange(n)
            if other != node:
                adjacency[node].add(other)
                adjacency[other].add(node)
    # sorted, а не list(set): порядок соседей влияет на порядок обхода,
    # а замер должен быть воспроизводим от запуска к запуску
    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


# крупный разреженный граф: обходы и агрегация — O(V + E)
_BIG = _random_undirected(3000, 3)

# отдельный граф поменьше для матриц: они O(n^2) и по памяти, и по времени,
# на 3000 узлов одна матрица — это девять миллионов ячеек
_SMALL = _random_undirected(700, 3)

# несколько несвязных кусков, чтобы connected_components не отработал за один BFS
_PIECES = {}
for _offset in range(0, 3000, 100):
    for _node, _neighbors in _random_undirected(100, 2).items():
        _PIECES[_offset + _node] = [_offset + _n for _n in _neighbors]

# фичи узлов: 16 признаков на узел, как маленький скрытый слой GNN
_FEATURES = {node: [random.random() for _ in range(16)] for node in _BIG}

BENCH = {
    "degrees": (_BIG,),
    "adjacency_matrix": (_SMALL,),
    "laplacian_matrix": (_SMALL,),
    "bfs_layers": (_BIG, 0),
    "bfs_distances": (_BIG, 0),
    "dfs_order": (_BIG, 0),
    "connected_components": (_PIECES,),
    "message_passing": (_BIG, _FEATURES),
}
