"""Тесты к уроку «3D-зрение: облака точек и NeRF». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    alpha_from_density,
    pointnet_global_feature,
    positional_encoding,
    sample_ray,
    segment_deltas,
    shared_mlp,
    transmittance,
    volumetric_render,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    return [x for row in M for x in row]


# веса общего MLP: 2 входа -> 2 выхода, тождественное отображение
EYE2 = [[1.0, 0.0], [0.0, 1.0]]
ZERO2 = [0.0, 0.0]


# --------------------------------------------------------------- shared_mlp
def test_shared_mlp_computes_linear_layer_before_relu():
    """W @ p + b, пока всё положительное — ReLU ничего не трогает."""
    got = shared_mlp([1.0, 2.0], [[1.0, 1.0], [2.0, 0.5]], [0.5, 1.0])
    assert got == APPROX([3.5, 4.0])


def test_shared_mlp_relu_zeroes_negative_outputs():
    """Отрицательный нейрон обнуляется, положительный проходит как есть."""
    got = shared_mlp([1.0, 2.0], [[1.0, 0.0], [0.0, -1.0]], ZERO2)
    assert got == APPROX([1.0, 0.0])


def test_shared_mlp_output_length_equals_number_of_rows():
    """Ловушка с порядком индексов: строка weights — это выходной нейрон."""
    got = shared_mlp([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, 0.0])
    assert len(got) == 3
    assert got == APPROX([1.0, 1.0, 2.0])


# --------------------------------------------------- pointnet_global_feature
def test_pointnet_is_the_max_over_point_features():
    cloud = [[1.0, 2.0], [3.0, 0.0]]
    assert pointnet_global_feature(cloud, EYE2, ZERO2) == APPROX([3.0, 2.0])


def test_pointnet_is_permutation_invariant():
    """Весь PointNet ради этого свойства: порядок точек не влияет на ответ."""
    rng = random.Random(0)
    cloud = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(30)]
    weights = [[0.7, -0.3], [-1.2, 0.4], [0.1, 0.9]]
    biases = [0.0, 0.2, -0.1]

    straight = pointnet_global_feature(cloud, weights, biases)
    shuffled = list(cloud)
    rng.shuffle(shuffled)
    assert shuffled != cloud  # перестановка действительно другая
    assert pointnet_global_feature(shuffled, weights, biases) == APPROX(straight)


def test_pointnet_ignores_duplicated_points():
    """Max-пул идемпотентен: копия точки ничего не добавляет."""
    cloud = [[1.0, 2.0], [3.0, 0.0]]
    once = pointnet_global_feature(cloud, EYE2, ZERO2)
    twice = pointnet_global_feature(cloud + [cloud[0]] * 5, EYE2, ZERO2)
    assert twice == APPROX(once)


def test_pointnet_handles_clouds_of_any_size():
    """Одна и та же модель принимает облака разной длины N."""
    rng = random.Random(1)
    weights = [[0.5, 0.5], [-1.0, 2.0], [1.0, 0.0]]
    biases = [0.0, 0.0, 0.0]
    for n in (1, 7, 100):
        cloud = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(n)]
        assert len(pointnet_global_feature(cloud, weights, biases)) == 3


def test_pointnet_rejects_an_empty_cloud():
    with pytest.raises(ValueError):
        pointnet_global_feature([], EYE2, ZERO2)


# ------------------------------------------------------- positional_encoding
def test_positional_encoding_length_is_dim_times_two_levels():
    assert len(positional_encoding([0.1, 0.2, 0.3], 10)) == 3 * 2 * 10
    assert len(positional_encoding([0.1], 4)) == 8


def test_positional_encoding_at_zero_is_sin_zero_cos_one():
    assert positional_encoding([0.0], 3) == APPROX([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_positional_encoding_norm_is_sqrt_dim_times_levels():
    """Каждая пара даёт sin^2 + cos^2 = 1, значит длина вектора известна заранее."""
    p = [0.37, -1.4, 2.9]
    L = 4
    enc = positional_encoding(p, L)
    norm = math.sqrt(sum(v * v for v in enc))
    assert norm == pytest.approx(math.sqrt(len(p) * L), abs=1e-12)


def test_positional_encoding_frequencies_double_each_level():
    """Уровень l+1 — это уровень l, посчитанный в удвоенной точке."""
    enc = positional_encoding([0.1], 3)
    assert enc[2:4] == pytest.approx(positional_encoding([0.2], 1), abs=1e-12)
    assert enc[4:6] == pytest.approx(positional_encoding([0.4], 1), abs=1e-12)
    # то же самое через формулу двойного угла: sin(2a) = 2 sin(a) cos(a)
    assert enc[2] == pytest.approx(2 * enc[0] * enc[1], abs=1e-12)


def test_positional_encoding_with_zero_levels_is_empty():
    assert positional_encoding([0.3, 0.4], 0) == []


# ---------------------------------------------------------------- sample_ray
def test_sample_ray_walks_along_the_direction():
    got = sample_ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.5])
    assert flat(got) == APPROX(flat([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.5]]))


def test_sample_ray_at_t_zero_returns_the_origin():
    origin = [1.0, -2.0, 3.0]
    assert sample_ray(origin, [4.0, 5.0, 6.0], [0.0])[0] == APPROX(origin)


def test_sample_ray_does_not_mutate_the_origin():
    origin = [1.0, 2.0, 3.0]
    sample_ray(origin, [1.0, 1.0, 1.0], [0.5, 1.0])
    assert origin == [1.0, 2.0, 3.0]


# ------------------------------------------------------------ segment_deltas
def test_segment_deltas_keeps_the_length_of_t_vals():
    """Ловушка: не на единицу короче, иначе sigma и delta разъедутся."""
    t_vals = [0.5, 1.0, 1.75, 4.0]
    assert len(segment_deltas(t_vals)) == len(t_vals)


def test_segment_deltas_are_gaps_between_neighbours():
    got = segment_deltas([1.0, 2.0, 4.0], last=0.0)
    assert got == APPROX([1.0, 2.0, 0.0])


def test_segment_deltas_last_entry_reaches_to_infinity():
    assert segment_deltas([2.0]) == APPROX([1e10])
    assert segment_deltas([0.0, 0.5], last=1.0) == APPROX([0.5, 1.0])


# --------------------------------------------------------- alpha_from_density
def test_alpha_is_zero_in_empty_space():
    """sigma = 0 — пустота, отсчёт полностью прозрачен."""
    assert alpha_from_density(0.0, 3.0) == APPROX(0.0)
    assert alpha_from_density(0.0, 1e10) == APPROX(0.0)


def test_alpha_matches_the_exponential_formula():
    assert alpha_from_density(1.0, 1.0) == APPROX(1.0 - math.exp(-1.0))
    assert alpha_from_density(2.0, 0.25) == APPROX(1.0 - math.exp(-0.5))


def test_alpha_stays_between_zero_and_one():
    rng = random.Random(2)
    for _ in range(50):
        a = alpha_from_density(rng.uniform(0.0, 5.0), rng.uniform(0.0, 3.0))
        assert 0.0 <= a < 1.0


def test_alpha_saturates_to_one_on_huge_optical_depth():
    """exp(-огромное) уходит в машинный ноль — так стенка закрывает луч."""
    assert alpha_from_density(1.0, 1e10) == 1.0
    assert alpha_from_density(1e6, 1.0) == 1.0


def test_alpha_rejects_negative_density():
    with pytest.raises(ValueError):
        alpha_from_density(-0.5, 1.0)


# --------------------------------------------------------------- transmittance
def test_transmittance_starts_at_one():
    assert transmittance([0.3, 0.7])[0] == 1.0


def test_transmittance_multiplies_the_survivors():
    assert transmittance([0.5, 0.5, 0.5]) == APPROX([1.0, 0.5, 0.25])


def test_transmittance_is_non_increasing():
    """Свет по дороге только теряется."""
    rng = random.Random(3)
    alphas = [rng.uniform(0.0, 1.0) for _ in range(40)]
    T = transmittance(alphas)
    assert all(T[i + 1] <= T[i] + 1e-12 for i in range(len(T) - 1))


def test_transmittance_is_zero_behind_a_fully_opaque_sample():
    """Перекрытие: за стенкой не видно ничего."""
    T = transmittance([0.2, 1.0, 0.9, 0.4])
    assert T[2] == APPROX(0.0)
    assert T[3] == APPROX(0.0)


# ------------------------------------------------------------ volumetric_render
def test_volumetric_renders_a_two_sample_ray():
    """Пустота, потом стенка: виден цвет стенки на её расстоянии."""
    color, depth, weights = volumetric_render(
        [0.0, 100.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1.0, 2.0]
    )
    assert color == APPROX([0.0, 1.0, 0.0])
    assert depth == APPROX(2.0)
    assert weights == APPROX([0.0, 1.0])


def test_volumetric_weights_equal_transmittance_times_alpha():
    """Рендер обязан складываться из уже написанных кусков, а не из своих формул."""
    sigmas = [0.3, 1.2, 0.0, 2.5]
    t_vals = [2.0, 2.5, 3.0, 4.0]
    colors = [[0.1, 0.2, 0.3]] * 4

    deltas = segment_deltas(t_vals)
    alphas = [alpha_from_density(s, d) for s, d in zip(sigmas, deltas)]
    want = [T * a for T, a in zip(transmittance(alphas), alphas)]

    _, _, weights = volumetric_render(sigmas, colors, t_vals)
    assert weights == APPROX(want)


def test_volumetric_weights_sum_at_most_one():
    """Вес — доля поглощённого света, больше единицы её не бывает."""
    rng = random.Random(4)
    n = 32
    t_vals = [2.0 + 4.0 * i / (n - 1) for i in range(n)]
    for _ in range(20):
        sigmas = [rng.uniform(0.0, 3.0) for _ in range(n)]
        colors = [[rng.random(), rng.random(), rng.random()] for _ in range(n)]
        _, _, weights = volumetric_render(sigmas, colors, t_vals)
        assert sum(weights) <= 1.0 + 1e-12


def test_volumetric_color_is_the_first_opaque_sample():
    """Occlusion: непрозрачный первый отсчёт прячет всё, что за ним."""
    colors = [[0.25, 0.5, 0.75], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    color, depth, weights = volumetric_render([1e3, 1e3, 1e3], colors, [1.0, 2.0, 3.0])
    assert color == APPROX(colors[0])
    assert depth == APPROX(1.0)
    assert weights[1] == APPROX(0.0)
    assert weights[2] == APPROX(0.0)


def test_volumetric_color_is_black_when_density_is_zero():
    """Луч ушёл в пустоту — пиксель чёрный, каким бы ни был цвет отсчётов."""
    colors = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.2, 0.9, 0.4]]
    color, _, weights = volumetric_render([0.0, 0.0, 0.0], colors, [1.0, 2.0, 3.0])
    assert color == APPROX([0.0, 0.0, 0.0])
    assert weights == APPROX([0.0, 0.0, 0.0])


def test_volumetric_depth_lies_between_the_first_and_last_sample():
    rng = random.Random(5)
    n = 16
    t_vals = [1.0 + 0.5 * i for i in range(n)]
    for _ in range(10):
        sigmas = [rng.uniform(0.05, 2.0) for _ in range(n)]
        colors = [[rng.random(), rng.random(), rng.random()] for _ in range(n)]
        _, depth, _ = volumetric_render(sigmas, colors, t_vals)
        assert min(t_vals) - 1e-9 <= depth <= max(t_vals) + 1e-9
