"""Тесты к уроку «3D Gaussian Splatting своими руками». Правь exercise.py."""

import math

import pytest

from exercise import (
    alpha_composite,
    covariance_2d,
    densify_decision,
    gaussian_density,
    gaussian_float_count,
    inverse_2x2,
    render_pixel,
    eval_sh_degree_1,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Развернуть матрицу в плоский список: pytest.approx не умеет вложенность."""
    return [v for row in M for v in row]


def splat(mean, colour, opacity, depth, cov=None):
    """Один сплат со сферической ковариацией по умолчанию."""
    return {
        "mean": mean,
        "cov": cov if cov is not None else [[1.0, 0.0], [0.0, 1.0]],
        "colour": colour,
        "opacity": opacity,
        "depth": depth,
    }


# ----------------------------------------------------------- covariance_2d
def test_covariance_holds_squared_scales():
    assert flat(covariance_2d(2.0, 1.0, 0.0)) == APPROX([4.0, 0.0, 0.0, 1.0])


def test_quarter_turn_swaps_the_two_axes():
    """Проверка порядка R и S: при 'S R R^T S^T' оси не поменяются местами."""
    assert flat(covariance_2d(2.0, 1.0, math.pi / 2)) == pytest.approx(
        [1.0, 0.0, 0.0, 4.0], abs=1e-12
    )


def test_isotropic_splat_does_not_care_about_rotation():
    assert flat(covariance_2d(3.0, 3.0, 0.7)) == pytest.approx([9.0, 0.0, 0.0, 9.0], abs=1e-12)


def test_rotation_preserves_the_area_of_the_ellipse():
    """det Sigma = (sx * sy)^2 при любом угле: поворот не меняет площадь."""
    for angle in (0.0, 0.3, 1.1, 2.9):
        (a, b), (c, d) = covariance_2d(2.0, 0.5, angle)
        assert a * d - b * c == pytest.approx(1.0, abs=1e-12)


def test_covariance_rejects_a_collapsed_scale():
    with pytest.raises(ValueError):
        covariance_2d(0.0, 1.0, 0.0)


# ------------------------------------------------------------- inverse_2x2
def test_inverse_times_the_matrix_is_the_identity():
    M = [[2.0, 1.0], [1.0, 3.0]]
    inv = inverse_2x2(M)
    product = [[sum(M[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert flat(product) == APPROX([1.0, 0.0, 0.0, 1.0])


def test_inverse_of_a_diagonal_matrix_inverts_each_axis():
    assert flat(inverse_2x2([[2.0, 0.0], [0.0, 4.0]])) == APPROX([0.5, 0.0, 0.0, 0.25])


def test_inverting_twice_gives_the_original_matrix():
    M = [[2.0, 1.0], [1.0, 3.0]]
    assert flat(inverse_2x2(inverse_2x2(M))) == APPROX(flat(M))


def test_a_splat_collapsed_to_a_line_is_rejected():
    with pytest.raises(ValueError):
        inverse_2x2([[1.0, 2.0], [2.0, 4.0]])


# --------------------------------------------------------- gaussian_density
def test_density_at_the_centre_is_exactly_one():
    assert gaussian_density((3.0, 5.0), [[1.0, 0.0], [0.0, 1.0]], (3.0, 5.0)) == APPROX(1.0)


def test_density_at_one_sigma_is_exp_minus_half():
    cov = [[4.0, 0.0], [0.0, 4.0]]
    assert gaussian_density((0.0, 0.0), cov, (2.0, 0.0)) == APPROX(math.exp(-0.5))


def test_stretched_splat_reaches_further_along_its_long_axis():
    """Расстояние Махаланобиса, а не евклидово: анизотропия и держит всю сцену."""
    cov = covariance_2d(4.0, 1.0, 0.0)
    along = gaussian_density((0.0, 0.0), cov, (1.0, 0.0))
    across = gaussian_density((0.0, 0.0), cov, (0.0, 1.0))
    assert along > across


def test_density_survives_a_joint_rotation_of_splat_and_pixel():
    angle = 0.9
    base = gaussian_density((0.0, 0.0), covariance_2d(3.0, 1.0, 0.0), (2.0, 1.0))
    c, s = math.cos(angle), math.sin(angle)
    turned_point = (2.0 * c - 1.0 * s, 2.0 * s + 1.0 * c)
    turned = gaussian_density((0.0, 0.0), covariance_2d(3.0, 1.0, angle), turned_point)
    assert turned == pytest.approx(base, abs=1e-12)


# -------------------------------------------------------- alpha_composite
def test_an_opaque_front_layer_hides_everything_behind_it():
    colour, T = alpha_composite([(1.0, (1.0, 0.0, 0.0)), (1.0, (0.0, 0.0, 1.0))])
    assert colour == APPROX([1.0, 0.0, 0.0])
    assert T == APPROX(0.0)


def test_layer_weights_and_transmittance_always_sum_to_one():
    """Единица непрозрачности делится между слоями и фоном — ничего не теряется."""
    white = [(0.3, (1.0, 1.0, 1.0)), (0.5, (1.0, 1.0, 1.0)), (0.25, (1.0, 1.0, 1.0))]
    colour, T = alpha_composite(white)
    assert colour[0] + T == APPROX(1.0)


def test_swapping_two_overlapping_layers_changes_the_pixel():
    """Поэтому в 3DGS есть отдельная стадия depth-sort на каждый тайл."""
    front = (0.5, (1.0, 0.0, 0.0))
    back = (0.5, (0.0, 0.0, 1.0))
    assert alpha_composite([front, back])[0] != APPROX(alpha_composite([back, front])[0])


def test_an_empty_stack_is_a_black_fully_transparent_pixel():
    assert alpha_composite([]) == ([0.0, 0.0, 0.0], 1.0)


def test_alpha_outside_the_unit_range_is_rejected():
    with pytest.raises(ValueError):
        alpha_composite([(1.2, (1.0, 1.0, 1.0))])


# ------------------------------------------------------------- render_pixel
def test_one_splat_at_its_own_centre_paints_almost_its_colour():
    colour, T = render_pixel([splat((0.0, 0.0), (1.0, 1.0, 1.0), 1.0, 0.0)], (0.0, 0.0))
    assert colour == APPROX([0.99, 0.99, 0.99])
    assert T == APPROX(0.01)


def test_the_nearer_splat_wins_the_pixel():
    red = splat((0.0, 0.0), (1.0, 0.0, 0.0), 0.6, depth=0.0)
    blue = splat((0.0, 0.0), (0.0, 0.0, 1.0), 0.6, depth=1.0)
    front_red = render_pixel([red, blue], (0.0, 0.0))[0]
    swapped = render_pixel([dict(red, depth=1.0), dict(blue, depth=0.0)], (0.0, 0.0))[0]
    assert front_red[0] > front_red[2]
    assert swapped[2] > swapped[0]


def test_depth_order_stops_mattering_when_footprints_do_not_overlap():
    """Композитинг переставим ровно тогда, когда сплаты не спорят за пиксель."""
    here = splat((0.0, 0.0), (1.0, 0.0, 0.0), 0.6, depth=0.0)
    far = splat((100.0, 0.0), (0.0, 0.0, 1.0), 0.6, depth=1.0)
    a = render_pixel([here, far], (0.0, 0.0))[0]
    b = render_pixel([dict(here, depth=1.0), dict(far, depth=0.0)], (0.0, 0.0))[0]
    assert a == pytest.approx(b, abs=1e-12)


def test_alpha_is_capped_so_hidden_splats_keep_a_gradient():
    """При alpha ровно 1.0 множитель (1 - alpha) навсегда обнулил бы задним градиент."""
    front = splat((0.0, 0.0), (1.0, 0.0, 0.0), 1.0, depth=0.0)
    back = splat((0.0, 0.0), (0.0, 0.0, 1.0), 1.0, depth=1.0)
    colour, T = render_pixel([front, back], (0.0, 0.0))
    assert colour[2] > 0.0
    assert T > 0.0


def test_an_empty_scene_renders_a_black_pixel():
    assert render_pixel([], (0.0, 0.0)) == ([0.0, 0.0, 0.0], 1.0)


# -------------------------------------------------------- eval_sh_degree_1
def test_degree_zero_colour_is_the_same_from_every_direction():
    """Ламбертова поверхность: одна константа на канал, вид не важен."""
    coeffs = [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    front = eval_sh_degree_1(coeffs, (1.0, 0.0, 0.0))
    side = eval_sh_degree_1(coeffs, (0.0, 0.0, 1.0))
    assert front == APPROX(side)
    assert front == APPROX([0.2820947917738781] * 3)


def test_the_degree_one_term_flips_with_the_viewing_direction():
    """Это и есть блик: виден с одной стороны, погашен с противоположной."""
    coeffs = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]
    up = eval_sh_degree_1(coeffs, (0.0, 0.0, 1.0))
    down = eval_sh_degree_1(coeffs, (0.0, 0.0, -1.0))
    assert up[0] > 0.0
    assert up[0] == APPROX(-down[0])


def test_a_non_unit_viewing_direction_is_rejected():
    coeffs = [(1.0, 1.0, 1.0)] * 4
    with pytest.raises(ValueError):
        eval_sh_degree_1(coeffs, (2.0, 0.0, 0.0))


def test_a_wrong_number_of_sh_coefficients_is_rejected():
    with pytest.raises(ValueError):
        eval_sh_degree_1([(1.0, 1.0, 1.0)] * 3, (1.0, 0.0, 0.0))


# ------------------------------------------------------- densify_decision
def test_a_transparent_splat_is_pruned_before_anything_else():
    """Клонировать невидимый сплат — значит вырастить два невидимых."""
    assert densify_decision(grad_norm=1.0, scale=0.5, opacity=0.0001) == "prune"


def test_a_large_under_fit_splat_is_split():
    assert densify_decision(grad_norm=0.001, scale=0.5, opacity=0.9) == "split"


def test_a_small_under_fit_splat_is_cloned():
    assert densify_decision(grad_norm=0.001, scale=0.001, opacity=0.9) == "clone"


def test_a_converged_splat_is_left_alone():
    assert densify_decision(grad_norm=0.0, scale=0.5, opacity=0.9) == "keep"


# ----------------------------------------------------- gaussian_float_count
def test_a_degree_three_splat_costs_fifty_nine_floats():
    assert gaussian_float_count(3) == 59


def test_colour_dominates_the_budget():
    """48 из 59 float уходят на SH — поэтому экспорт начинают с квантования цвета."""
    geometry = gaussian_float_count(0) - 3
    assert gaussian_float_count(3) - geometry == 48


def test_a_negative_sh_degree_is_rejected():
    with pytest.raises(ValueError):
        gaussian_float_count(-1)
