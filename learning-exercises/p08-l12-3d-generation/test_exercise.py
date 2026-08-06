"""Тесты к уроку «Генерация 3D: гауссовы сплаты и подгонка градиентом». Правь exercise.py."""

import math

import pytest

from exercise import (
    alpha_composite,
    color_gradients,
    fit_colors,
    gaussian_value,
    image_mse,
    prune_gaussians,
    render,
    split_gaussian,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SIZE = 8


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def splat(px, py, sigma, color):
    return {"pos": [px, py], "sigma": sigma, "color": color}


def two_splats(c1=0.8, c2=0.3):
    return [splat(2.0, 2.0, 1.2, c1), splat(6.0, 6.0, 1.2, c2)]


# ----------------------------------------------------------- gaussian_value
def test_gaussian_value_at_the_centre_is_the_colour():
    assert gaussian_value(1, 1, splat(1.0, 1.0, 1.0, 0.8)) == APPROX(0.8)


def test_gaussian_value_falls_off_with_distance():
    g = splat(1.0, 1.0, 1.0, 0.8)
    assert gaussian_value(2, 1, g) == pytest.approx(0.8 * math.exp(-0.5), abs=1e-12)


def test_gaussian_value_is_monotone_in_distance():
    g = splat(4.0, 4.0, 1.5, 1.0)
    values = [gaussian_value(4 + d, 4, g) for d in (0, 1, 2, 3)]
    assert values == sorted(values, reverse=True)


def test_gaussian_pos_is_x_then_y_not_row_then_column():
    """Асимметричная позиция ловит перепутанные оси."""
    g = splat(0.0, 3.0, 1.0, 1.0)
    assert gaussian_value(0, 3, g) > gaussian_value(3, 0, g)


def test_wider_sigma_spreads_the_same_colour_further():
    narrow = splat(0.0, 0.0, 0.5, 1.0)
    wide = splat(0.0, 0.0, 3.0, 1.0)
    assert gaussian_value(2, 0, wide) > gaussian_value(2, 0, narrow)


# ------------------------------------------------------------------- render
def test_render_without_gaussians_is_black():
    assert flat(render(3, [])) == APPROX([0.0] * 9)


def test_render_matches_the_kernel_pixel_by_pixel():
    g = splat(0.0, 0.0, 1.0, 1.0)
    assert flat(render(2, [g])) == pytest.approx(
        [1.0, math.exp(-0.5), math.exp(-0.5), math.exp(-1.0)], abs=1e-12)


def test_render_adds_gaussians_together():
    """2D-игрушка складывает сплаты — значит рендер линеен по набору."""
    a, b = two_splats()
    both = flat(render(SIZE, [a, b]))
    apart = [x + y for x, y in zip(flat(render(SIZE, [a])), flat(render(SIZE, [b])))]
    assert both == pytest.approx(apart, abs=1e-12)


def test_render_is_brightest_at_the_gaussian_centre():
    img = render(SIZE, [splat(5.0, 2.0, 1.0, 1.0)])
    peak = max((v, y, x) for y, row in enumerate(img) for x, v in enumerate(row))
    assert (peak[1], peak[2]) == (2, 5)


# ---------------------------------------------------------------- image_mse
def test_image_mse_of_identical_images_is_zero():
    img = render(SIZE, two_splats())
    assert image_mse(img, img) == APPROX(0.0)


def test_image_mse_averages_the_squared_errors():
    assert image_mse([[0.0, 0.0]], [[1.0, 3.0]]) == APPROX(5.0)


def test_image_mse_is_symmetric():
    a, b = render(4, two_splats()), render(4, [splat(1.0, 1.0, 2.0, 0.4)])
    assert image_mse(a, b) == APPROX(image_mse(b, a))


# ---------------------------------------------------------- color_gradients
def test_color_gradients_match_central_differences():
    """Аналитика обязана совпасть с численной производной, иначе она неверна."""
    gaussians = two_splats()
    target = render(SIZE, two_splats(0.2, 0.9))
    analytic = color_gradients(SIZE, gaussians, target)
    h = 1e-6
    for i in range(len(gaussians)):
        up = [dict(g) for g in gaussians]
        down = [dict(g) for g in gaussians]
        up[i]["color"] += h
        down[i]["color"] -= h
        numeric = (image_mse(render(SIZE, up), target)
                   - image_mse(render(SIZE, down), target)) / (2 * h)
        assert analytic[i] == pytest.approx(numeric, abs=1e-6)


def test_color_gradients_vanish_at_the_exact_solution():
    gaussians = two_splats()
    target = render(SIZE, two_splats())
    assert color_gradients(SIZE, gaussians, target) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_color_gradient_is_negative_when_the_splat_is_too_dim():
    """Слишком тускло — производная отрицательна, шаг против неё поднимет яркость."""
    gaussians = two_splats(0.0, 0.0)
    target = render(SIZE, two_splats(0.8, 0.8))
    assert all(g < 0 for g in color_gradients(SIZE, gaussians, target))


def test_color_gradients_return_one_number_per_gaussian():
    gaussians = [splat(1.0, 1.0, 1.0, 0.5), splat(3.0, 3.0, 1.0, 0.5),
                 splat(5.0, 5.0, 1.0, 0.5)]
    assert len(color_gradients(SIZE, gaussians, render(SIZE, gaussians))) == 3


# --------------------------------------------------------------- fit_colors
def test_fit_colors_recovers_the_colours_of_the_target():
    """Цель отрисована из тех же позиций — значит яркости восстановимы точно."""
    target = render(SIZE, two_splats(0.8, 0.3))
    fitted = fit_colors(SIZE, two_splats(0.0, 0.0), target)
    assert [g["color"] for g in fitted] == pytest.approx([0.8, 0.3], abs=1e-3)


def test_fit_colors_lowers_the_loss():
    target = render(SIZE, two_splats(0.8, 0.3))
    start = two_splats(0.0, 0.0)
    before = image_mse(render(SIZE, start), target)
    after = image_mse(render(SIZE, fit_colors(SIZE, start, target, steps=20)), target)
    assert after < before


def test_fit_colors_does_not_mutate_the_input():
    """Иначе второй запуск с теми же аргументами даст другой ответ."""
    target = render(SIZE, two_splats(0.8, 0.3))
    start = two_splats(0.0, 0.0)
    fit_colors(SIZE, start, target, steps=10)
    assert [g["color"] for g in start] == APPROX([0.0, 0.0])


def test_fit_colors_is_deterministic():
    target = render(SIZE, two_splats(0.8, 0.3))
    a = fit_colors(SIZE, two_splats(0.1, 0.1), target, steps=25)
    b = fit_colors(SIZE, two_splats(0.1, 0.1), target, steps=25)
    assert [g["color"] for g in a] == APPROX([g["color"] for g in b])


def test_more_gaussians_fit_the_target_better():
    """Тот же вывод, что в упражнении урока: 4 сплата точнее двух."""
    truth = [splat(2.0, 2.0, 1.2, 0.8), splat(6.0, 6.0, 1.2, 0.3),
             splat(2.0, 6.0, 1.2, 0.5), splat(6.0, 2.0, 1.2, 0.6)]
    target = render(SIZE, truth)
    few = fit_colors(SIZE, [dict(g, color=0.0) for g in truth[:2]], target)
    many = fit_colors(SIZE, [dict(g, color=0.0) for g in truth], target)
    assert image_mse(render(SIZE, many), target) < image_mse(render(SIZE, few), target)


# ---------------------------------------------------------- alpha_composite
def test_opaque_front_layer_hides_everything_behind_it():
    assert alpha_composite([(1.0, 1.0), (0.0, 1.0)]) == APPROX(1.0)


def test_alpha_composite_is_not_commutative():
    """Поэтому 3D-GS обязан сортировать гауссианы по глубине перед рендером."""
    front_first = alpha_composite([(0.0, 1.0), (1.0, 1.0)])
    back_first = alpha_composite([(1.0, 1.0), (0.0, 1.0)])
    assert front_first != back_first


def test_semi_transparent_layers_accumulate():
    assert alpha_composite([(1.0, 0.5), (1.0, 1.0)]) == APPROX(1.0)


def test_fully_transparent_layers_contribute_nothing():
    assert alpha_composite([(1.0, 0.0), (0.5, 0.0)]) == APPROX(0.0)


def test_alpha_composite_of_no_layers_is_zero():
    assert alpha_composite([]) == APPROX(0.0)


# --------------------------------------------------------- prune_gaussians
def test_prune_drops_invisible_splats():
    kept = prune_gaussians([splat(0.0, 0.0, 1.0, 0.001), splat(1.0, 1.0, 1.0, 0.5)])
    assert [g["color"] for g in kept] == APPROX([0.5])


def test_prune_judges_by_absolute_value():
    """Сплат с цветом -0.5 виден не меньше, чем с +0.5."""
    assert len(prune_gaussians([splat(0.0, 0.0, 1.0, -0.5)])) == 1


def test_prune_barely_changes_the_render():
    visible = two_splats()
    noisy = visible + [splat(4.0, 4.0, 1.0, 1e-4)]
    assert image_mse(render(SIZE, prune_gaussians(noisy)), render(SIZE, noisy)) < 1e-7


# ---------------------------------------------------------- split_gaussian
def test_split_returns_two_children():
    assert len(split_gaussian(splat(4.0, 4.0, 2.0, 0.6), [1.0, 0.0])) == 2


def test_split_children_are_narrower_than_the_parent():
    parent = splat(4.0, 4.0, 2.0, 0.6)
    children = split_gaussian(parent, [1.0, 0.0])
    assert all(c["sigma"] < parent["sigma"] for c in children)
    assert children[0]["sigma"] == APPROX(2.0 / 1.6)


def test_split_keeps_the_centre_of_the_parent():
    """Середина между детьми — позиция родителя, иначе densification уводит сцену."""
    a, b = split_gaussian(splat(4.0, 5.0, 2.0, 0.6), [1.0, -0.5])
    assert [(x + y) / 2 for x, y in zip(a["pos"], b["pos"])] == APPROX([4.0, 5.0])


def test_split_inherits_the_colour_and_leaves_the_parent_alone():
    parent = splat(4.0, 4.0, 2.0, 0.6)
    children = split_gaussian(parent, [1.0, 0.0])
    assert all(c["color"] == APPROX(0.6) for c in children)
    assert parent == {"pos": [4.0, 4.0], "sigma": 2.0, "color": 0.6}
