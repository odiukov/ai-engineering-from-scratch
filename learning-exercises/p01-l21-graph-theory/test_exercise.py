"""Тесты к уроку «Теория графов». Правь exercise.py."""

import math

import pytest

from exercise import (
    adjacency_matrix,
    bfs_distances,
    bfs_layers,
    connected_components,
    degrees,
    dfs_order,
    laplacian_matrix,
    message_passing,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [value for row in matrix for value in row]


TRIANGLE = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
PATH = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
TREE = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}
RING = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
TWO_PIECES = {0: [1], 1: [0], 2: [3], 3: [2]}


# ----------------------------------------------------------------- degrees
def test_degrees_counts_neighbours_of_each_node():
    assert degrees(TRIANGLE) == {0: 2, 1: 2, 2: 2}


def test_degrees_of_a_node_without_edges_is_zero():
    """Ловушка: узел без исходящих рёбер обязан попасть в ответ с нулём."""
    assert degrees({0: [1], 1: [], 2: []}) == {0: 1, 1: 0, 2: 0}


def test_degrees_sum_to_twice_the_edge_count():
    """В неориентированном графе каждое ребро посчитано с двух концов."""
    assert sum(degrees(PATH).values()) == 2 * 3


# -------------------------------------------------------- adjacency_matrix
def test_adjacency_matrix_of_a_triangle():
    assert flat(adjacency_matrix(TRIANGLE)) == flat([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_adjacency_matrix_is_symmetric_for_undirected_graph():
    A = adjacency_matrix(PATH)
    n = len(A)
    assert all(A[i][j] == A[j][i] for i in range(n) for j in range(n))


def test_adjacency_matrix_is_asymmetric_for_directed_graph():
    """Ребро 0 -> 1 записано один раз: A[0][1] = 1, а A[1][0] = 0."""
    A = adjacency_matrix({0: [1], 1: []})
    assert A[0][1] == 1 and A[1][0] == 0


def test_adjacency_matrix_row_order_does_not_depend_on_dict_order():
    """Ловушка: строки нумеруются по sorted(graph), а не по порядку ключей."""
    shuffled = {2: [0], 0: [2], 1: []}
    assert flat(adjacency_matrix(shuffled)) == flat([[0, 0, 1], [0, 0, 0], [1, 0, 0]])


# -------------------------------------------------------- laplacian_matrix
def test_laplacian_of_a_triangle():
    expected = [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]
    assert flat(laplacian_matrix(TRIANGLE)) == flat(expected)


def test_laplacian_rows_sum_to_zero():
    """Смысловое свойство L: степень на диагонали ровно гасит минус-единицы."""
    assert [sum(row) for row in laplacian_matrix(PATH)] == [0, 0, 0, 0]


def test_laplacian_diagonal_holds_the_degrees():
    L = laplacian_matrix(TREE)
    deg = degrees(TREE)
    assert [L[i][i] for i in range(len(L))] == [deg[node] for node in sorted(TREE)]


def test_laplacian_equals_degree_matrix_minus_adjacency():
    """L = D - A: вне диагонали остаётся ровно -A."""
    L = laplacian_matrix(RING)
    A = adjacency_matrix(RING)
    off_diagonal = [
        L[i][j] == -A[i][j] for i in range(len(L)) for j in range(len(L)) if i != j
    ]
    assert all(off_diagonal)


# -------------------------------------------------------------- bfs_layers
def test_bfs_layers_group_nodes_by_hop_distance():
    assert bfs_layers(TREE, 0) == [[0], [1, 2], [3]]


def test_bfs_visits_nearest_nodes_first():
    """Все соседи старта идут раньше любого узла со второго слоя."""
    layers = bfs_layers(PATH, 0)
    assert layers[0] == [0] and layers[1] == [1] and layers[2] == [2]


def test_bfs_terminates_on_a_cycle():
    """Ловушка: в кольце обход без множества посещённых крутился бы вечно."""
    assert bfs_layers(RING, 0) == [[0], [1, 3], [2]]


def test_bfs_layers_skip_unreachable_nodes():
    assert bfs_layers({0: [1], 1: [0], 2: []}, 0) == [[0], [1]]


# ----------------------------------------------------------- bfs_distances
def test_bfs_distances_count_hops_from_the_start():
    assert bfs_distances(TREE, 0) == {0: 0, 1: 1, 2: 1, 3: 2}


def test_bfs_distance_to_the_start_is_zero():
    assert bfs_distances(PATH, 2)[2] == 0


def test_bfs_distance_is_the_shortest_not_the_first_found():
    """У узла 3 есть длинный путь 0-1-2-3 и короткий 0-3: ответ 1, а не 3."""
    graph = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    assert bfs_distances(graph, 0)[3] == 1


def test_bfs_distance_to_unreachable_node_is_infinite():
    """Ловушка: недостижимый узел присутствует в ответе со значением inf."""
    distances = bfs_distances({0: [1], 1: [0], 2: []}, 0)
    assert math.isinf(distances[2])


# --------------------------------------------------------------- dfs_order
def test_dfs_goes_deep_before_wide():
    assert dfs_order(TREE, 0) == [0, 1, 3, 2]


def test_dfs_follows_the_first_neighbour_first():
    """Ловушка стека: LIFO переворачивает порядок, если не класть соседей задом наперёд."""
    assert dfs_order({0: [1, 2], 1: [], 2: []}, 0) == [0, 1, 2]


def test_dfs_terminates_on_a_cycle():
    assert sorted(dfs_order(RING, 0)) == [0, 1, 2, 3]


def test_dfs_visits_the_same_nodes_as_bfs():
    """Обходы разные, множество достижимых узлов — одно и то же."""
    bfs = {node for layer in bfs_layers(TREE, 0) for node in layer}
    assert set(dfs_order(TREE, 0)) == bfs


# ------------------------------------------------------ connected_components
def test_connected_components_split_two_pieces():
    assert connected_components(TWO_PIECES) == [[0, 1], [2, 3]]


def test_connected_graph_has_exactly_one_component():
    assert connected_components(TRIANGLE) == [[0, 1, 2]]


def test_isolated_node_is_its_own_component():
    assert connected_components({0: [1], 1: [0], 2: []}) == [[0, 1], [2]]


def test_components_cover_every_node_exactly_once():
    """Компоненты не пересекаются и вместе дают весь граф."""
    graph = {0: [1], 1: [0], 2: [3, 4], 3: [2], 4: [2], 5: []}
    nodes = [node for component in connected_components(graph) for node in component]
    assert sorted(nodes) == [0, 1, 2, 3, 4, 5] and len(nodes) == len(set(nodes))


def test_component_order_does_not_depend_on_dict_order():
    """Ловушка недетерминизма: ответ — списки, а не set-ы, и он отсортирован."""
    shuffled = {3: [2], 1: [0], 2: [3], 0: [1]}
    assert connected_components(shuffled) == [[0, 1], [2, 3]]


# --------------------------------------------------------- message_passing
def test_message_passing_averages_neighbour_features():
    graph = {0: [1, 2], 1: [0], 2: [0]}
    features = {0: [0.0], 1: [1.0], 2: [3.0]}
    assert message_passing(graph, features)[0] == APPROX([2.0])


def test_message_passing_ignores_own_features():
    """Это только D^-1 A H; полный GCN ещё нормирует, учит W и применяет активацию."""
    graph = {0: [1], 1: [0]}
    features = {0: [1.0, 0.0], 1: [0.0, 1.0]}
    assert message_passing(graph, features)[0] == APPROX([0.0, 1.0])


def test_node_without_neighbours_gets_zeros():
    """Ловушка: деления на нулевую степень быть не должно."""
    graph = {0: [1], 1: [0], 2: []}
    features = {0: [1.0, 2.0], 1: [3.0, 4.0], 2: [5.0, 6.0]}
    assert message_passing(graph, features)[2] == APPROX([0.0, 0.0])


def test_message_passing_keeps_the_feature_width():
    graph = {0: [1], 1: [0]}
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    result = message_passing(graph, features)
    assert all(len(vector) == 3 for vector in result.values())


def test_one_round_does_not_reach_two_hop_neighbours():
    """Сигнал из узла 2 за один раунд доходит только до соседа 1."""
    graph = {0: [1], 1: [0, 2], 2: [1]}
    features = {0: [0.0], 1: [0.0], 2: [1.0]}
    once = message_passing(graph, features)
    assert once[0] == APPROX([0.0]) and once[1] == APPROX([0.5])


def test_two_rounds_reach_two_hop_neighbours():
    """Второй раунд приносит узлу 0 информацию из узла 2 — K раундов = K хопов."""
    graph = {0: [1], 1: [0, 2], 2: [1]}
    features = {0: [0.0], 1: [0.0], 2: [1.0]}
    twice = message_passing(graph, message_passing(graph, features))
    assert twice[0][0] > 0.0
