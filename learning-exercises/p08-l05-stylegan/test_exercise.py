"""Тесты к уроку «StyleGAN». Правь exercise.py."""

import math
import random
import statistics

import pytest

from exercise import (
    adain,
    average_w,
    leaky_relu,
    mapping_network,
    modulate,
    style_mixing,
    synthesis,
    truncate_w,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Плоский список из матрицы: pytest.approx не умеет вложенные списки."""
    return [v for row in M for v in row]


def mapping_layers():
    """Двухслойная mapping-сеть 2 -> 2 -> 2, веса руками."""
    return [
        ([[1.0, 0.5], [-0.5, 1.0]], [0.0, 0.0]),
        ([[0.8, 0.2], [0.1, -0.9]], [0.1, -0.1]),
    ]


def blocks(num=3, dim=4):
    """num блоков синтеза с нулевыми bias: так проверяется независимость от const."""
    out = []
    for i in range(num):
        W = [
            [0.4 + 0.1 * ((r + c + i) % 3) for c in range(dim)]
            for r in range(dim)
        ]
        out.append(
            {
                "W": W,
                "b": [0.0] * dim,
                "scale_w": [1.0, 0.0],
                "bias_w": [0.0, 1.0],
            }
        )
    return out


# ------------------------------------------------------------ leaky_relu
def test_leaky_relu_passes_the_positive_side_unchanged():
    assert leaky_relu(3.0) == APPROX(3.0)


def test_leaky_relu_shrinks_instead_of_killing_the_negative_side():
    """Главное отличие от ReLU: на отрицательной стороне ноль не выдаётся."""
    assert leaky_relu(-3.0) == APPROX(-0.6)
    assert leaky_relu(-3.0) != 0.0


def test_leaky_relu_at_zero_is_zero():
    assert leaky_relu(0.0) == APPROX(0.0)


def test_leaky_relu_slope_is_tunable():
    assert leaky_relu(-10.0, slope=0.5) == APPROX(-5.0)


# -------------------------------------------------------- mapping_network
def test_mapping_network_applies_one_layer():
    assert mapping_network([1.0], [([[2.0]], [0.0])]) == APPROX([2.0])


def test_mapping_network_leaks_the_negative_side():
    assert mapping_network([-1.0], [([[2.0]], [0.0])]) == APPROX([-0.4])


def test_mapping_network_composes_layers_in_order():
    """Два слоя подряд: сначала *2, потом *3 — итого *6, а не *5."""
    layers = [([[2.0]], [0.0]), ([[3.0]], [0.0])]
    assert mapping_network([1.0], layers) == APPROX([6.0])


def test_mapping_network_adds_the_bias():
    assert mapping_network([0.0], [([[5.0]], [1.0])]) == APPROX([1.0])


def test_mapping_network_without_layers_returns_z_itself():
    assert mapping_network([1.0, 2.0], []) == APPROX([1.0, 2.0])


def test_mapping_network_does_not_mutate_z():
    z = [1.0, 2.0]
    mapping_network(z, mapping_layers())
    assert z == [1.0, 2.0]


def test_mapping_network_gives_different_w_for_different_z():
    a = mapping_network([1.0, 0.0], mapping_layers())
    b = mapping_network([0.0, 1.0], mapping_layers())
    assert a != b


# ------------------------------------------------------------------ adain
def test_adain_worked_example():
    assert adain([1.0, 3.0], 1.0, 0.0) == pytest.approx([-1.0, 1.0], abs=1e-7)


def test_adain_sets_the_mean_to_the_bias():
    """«Стиль» — это первые два момента. Первый задаётся bias'ом."""
    out = adain([1.0, 5.0, -2.0, 9.0], 3.0, 7.0)
    assert statistics.fmean(out) == pytest.approx(7.0, abs=1e-6)


def test_adain_sets_the_std_to_the_scale():
    """Второй момент задаётся scale'ом, ровно и с точностью до знака."""
    out = adain([1.0, 5.0, -2.0, 9.0], 3.0, 7.0)
    m = statistics.fmean(out)
    sd = math.sqrt(sum((v - m) ** 2 for v in out) / len(out))
    assert sd == pytest.approx(3.0, abs=1e-6)


def test_adain_erases_the_input_mean_and_scale():
    """Растянули и сдвинули вход — выход не изменился. Слою нечего унаследовать."""
    x = [1.0, 5.0, -2.0, 9.0]
    stretched = [2.0 * v + 5.0 for v in x]
    assert adain(x, 1.5, -0.5) == pytest.approx(adain(stretched, 1.5, -0.5), abs=1e-6)


def test_adain_survives_a_constant_feature_map():
    """Постоянный вход даёт std = 0 — без добавки 1e-8 здесь было бы деление на ноль."""
    assert adain([0.0, 0.0], 2.0, 5.0) == pytest.approx([5.0, 5.0], abs=1e-6)


def test_adain_with_a_negative_scale_flips_the_order():
    out = adain([1.0, 3.0], -1.0, 0.0)
    assert out[0] > out[1]


def test_adain_on_an_empty_vector_raises_value_error():
    with pytest.raises(ValueError):
        adain([], 1.0, 0.0)


# --------------------------------------------------------------- modulate
def test_modulate_projects_w_into_scale_and_bias():
    assert modulate([1.0, 2.0], [1.0, 0.0], [0.0, 3.0]) == pytest.approx((1.0, 6.0))


def test_modulate_of_the_zero_w_is_a_neutral_style():
    assert modulate([0.0, 0.0], [1.0, 2.0], [3.0, 4.0]) == pytest.approx((0.0, 0.0))


def test_modulate_is_linear_in_w():
    """Линейная проекция: удвоили w — удвоились и scale, и bias."""
    one = modulate([1.0, 2.0], [0.3, -0.7], [0.5, 0.1])
    two = modulate([2.0, 4.0], [0.3, -0.7], [0.5, 0.1])
    assert two == pytest.approx((2 * one[0], 2 * one[1]))


def test_modulate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        modulate([1.0, 2.0], [1.0], [1.0, 2.0])


# -------------------------------------------------------------- average_w
def test_average_w_averages_coordinatewise():
    assert average_w([[0.0, 2.0], [2.0, 0.0]]) == APPROX([1.0, 1.0])


def test_average_of_identical_vectors_is_that_vector():
    assert average_w([[1.0, -2.0]] * 7) == APPROX([1.0, -2.0])


def test_average_w_on_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        average_w([])


def test_average_w_rejects_ragged_input():
    with pytest.raises(ValueError):
        average_w([[1.0, 2.0], [1.0]])


# ------------------------------------------------------------- truncate_w
def test_psi_one_leaves_w_untouched():
    assert truncate_w([3.0], [1.0], 1.0) == APPROX([3.0])


def test_psi_zero_collapses_everything_into_the_mean():
    """psi = 0 — один и тот же выход на любой z. Качество максимально, разнообразия нет."""
    assert truncate_w([3.0], [1.0], 0.0) == APPROX([1.0])
    assert truncate_w([-9.0], [1.0], 0.0) == APPROX([1.0])


def test_psi_half_lands_midway():
    assert truncate_w([3.0], [1.0], 0.5) == APPROX([2.0])


def test_truncation_shrinks_the_spread_by_exactly_psi():
    """Вот и обмен разнообразия на качество, в числах."""
    ws = [[v] for v in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    bar = average_w(ws)
    before = statistics.stdev(w[0] for w in ws)
    after = statistics.stdev(truncate_w(w, bar, 0.7)[0] for w in ws)
    assert after == pytest.approx(0.7 * before, abs=1e-9)


def test_truncate_w_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        truncate_w([1.0, 2.0], [0.0], 0.7)


# ------------------------------------------------------------ style_mixing
def test_crossover_splits_the_layers():
    assert flat(style_mixing([1.0], [2.0], 3, 1)) == APPROX([1.0, 2.0, 2.0])


def test_full_crossover_uses_only_the_first_style():
    assert flat(style_mixing([1.0], [2.0], 3, 3)) == APPROX([1.0, 1.0, 1.0])


def test_zero_crossover_uses_only_the_second_style():
    assert flat(style_mixing([1.0], [2.0], 3, 0)) == APPROX([2.0, 2.0, 2.0])


def test_style_mixing_returns_one_w_per_layer():
    assert len(style_mixing([1.0, 2.0], [3.0, 4.0], 5, 2)) == 5


def test_style_mixing_copies_instead_of_aliasing():
    """Один и тот же объект на всех слоях — и правка одного слоя поехала бы по всем."""
    mixed = style_mixing([1.0], [2.0], 3, 3)
    mixed[0][0] = 99.0
    assert mixed[1][0] == APPROX(1.0)


def test_crossover_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        style_mixing([1.0], [2.0], 3, 4)


# --------------------------------------------------------------- synthesis
def test_synthesis_is_deterministic_without_noise():
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    assert synthesis([0.3, -0.2, 0.5, 0.1], b, w) == synthesis(
        [0.3, -0.2, 0.5, 0.1], b, w
    )


def test_adain_makes_the_output_independent_of_the_constant_scale():
    """z в сеть не подаётся, а масштаб const первый же AdaIN нормализует начисто."""
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    const = [0.3, -0.2, 0.5, 0.1]
    big = [10.0 * v for v in const]
    assert synthesis(const, b, w, adain_on=True) == pytest.approx(
        synthesis(big, b, w, adain_on=True), abs=1e-6
    )


def test_without_adain_the_constant_scale_leaks_straight_through():
    """Контраст к предыдущему тесту: выключили AdaIN — и вход снова всё решает."""
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    const = [0.3, -0.2, 0.5, 0.1]
    big = [10.0 * v for v in const]
    plain = synthesis(const, b, w, adain_on=False)
    scaled = synthesis(big, b, w, adain_on=False)
    assert plain != pytest.approx(scaled, abs=1e-6)


def test_changing_w_changes_the_output():
    b = blocks()
    const = [0.3, -0.2, 0.5, 0.1]
    a = synthesis(const, b, style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3))
    c = synthesis(const, b, style_mixing([-2.0, 3.0], [-2.0, 3.0], 3, 3))
    assert a != pytest.approx(c, abs=1e-6)


def test_style_mixing_result_feeds_straight_into_synthesis():
    """Разные w на разных слоях дают не то же, что один w везде."""
    b = blocks()
    const = [0.3, -0.2, 0.5, 0.1]
    mixed = synthesis(const, b, style_mixing([1.0, 0.5], [-2.0, 3.0], 3, 1))
    pure = synthesis(const, b, style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3))
    assert mixed != pytest.approx(pure, abs=1e-6)


def test_noise_changes_the_stochastic_detail():
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    const = [0.3, -0.2, 0.5, 0.1]
    a = synthesis(const, b, w, noise_sigma=0.01, rng=random.Random(0))
    c = synthesis(const, b, w, noise_sigma=0.01, rng=random.Random(1))
    assert a != c


def test_noise_leaves_the_global_style_in_place():
    """Разный сид шума — те же общие статистики выхода. Поры меняются, лицо нет."""
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    const = [0.3, -0.2, 0.5, 0.1]
    a = synthesis(const, b, w, noise_sigma=0.01, rng=random.Random(0))
    c = synthesis(const, b, w, noise_sigma=0.01, rng=random.Random(1))
    assert statistics.fmean(a) == pytest.approx(statistics.fmean(c), abs=0.05)


def test_noise_without_an_rng_raises_value_error():
    """Молчаливый глобальный random сделал бы прогон невоспроизводимым."""
    b = blocks()
    w = style_mixing([1.0, 0.5], [1.0, 0.5], 3, 3)
    with pytest.raises(ValueError):
        synthesis([0.3, -0.2, 0.5, 0.1], b, w, noise_sigma=0.1)


def test_wrong_number_of_w_vectors_raises_value_error():
    b = blocks()
    with pytest.raises(ValueError):
        synthesis([0.3, -0.2, 0.5, 0.1], b, style_mixing([1.0, 0.5], [1.0, 0.5], 2, 2))


def test_truncated_w_produces_a_narrower_range_of_outputs():
    """Полная цепочка урока: z -> w -> truncation -> synthesis."""
    b = blocks()
    const = [0.3, -0.2, 0.5, 0.1]
    layers = mapping_layers()
    rng = random.Random(0)
    zs = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(60)]
    ws = [mapping_network(z, layers) for z in zs]
    bar = average_w(ws)

    def spread(vectors):
        firsts = [
            synthesis(const, b, style_mixing(w, w, 3, 3))[0] for w in vectors
        ]
        return statistics.stdev(firsts)

    full = spread(ws)
    narrow = spread([truncate_w(w, bar, 0.3) for w in ws])
    assert narrow < full
