"""Тесты к уроку «Воплощённые VLA: RT-2, OpenVLA, pi0, GR00T».

Правь exercise.py.
"""

import math
import random

import pytest

from exercise import (
    FORMATS,
    cofinetune_mix,
    dct,
    discretize,
    fast_reconstruct,
    fast_tokens,
    format_token_budget,
    idct,
    undiscretize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [v for row in M for v in row]


def mean_abs_error(a, b):
    fa, fb = flat(a), flat(b)
    return sum(abs(x - y) for x, y in zip(fa, fb)) / len(fa)


# --------------------------------------------------------------- discretize
def test_discretize_splits_the_range_into_equal_bins():
    assert discretize([-1.0, 0.0, 1.0], bins=4) == [0, 2, 3]


def test_discretize_puts_the_upper_bound_in_the_last_bin():
    """int((v - lo) / span * bins) на v == hi даёт номер bins — несуществующий бин."""
    assert discretize([1.0], bins=256) == [255]


def test_discretize_clips_out_of_range_values():
    assert discretize([5.0, -5.0], bins=256) == [255, 0]


def test_discretize_is_monotone():
    """Больший угол сустава не может попасть в меньший бин."""
    tokens = discretize([-0.9, -0.3, 0.0, 0.4, 0.95], bins=256)
    assert tokens == sorted(tokens)


# ------------------------------------------------------------- undiscretize
def test_undiscretize_returns_bin_centers():
    assert undiscretize([0, 2, 3], bins=4) == pytest.approx([-0.75, 0.25, 0.75])


def test_undiscretize_stays_inside_the_range():
    values = undiscretize(list(range(256)), bins=256)
    assert all(-1.0 < v < 1.0 for v in values)


def test_round_trip_error_never_exceeds_half_a_bin():
    """Максимальная ошибка квантования — (hi - lo) / (2 * bins), для 256 бинов 0.0039."""
    half_bin = 2.0 / (2 * 256)
    originals = [-0.999, -0.5, -0.001, 0.0, 0.333, 0.87, 0.999]
    restored = undiscretize(discretize(originals, bins=256), bins=256)
    assert all(abs(a - b) <= half_bin + 1e-12 for a, b in zip(originals, restored))


def test_more_bins_means_smaller_quantization_error():
    """Это и есть цена дискретизации, которую pi0 убирает непрерывной головой."""
    xs = [-0.777, 0.123, 0.456]
    coarse = undiscretize(discretize(xs, bins=8), bins=8)
    fine = undiscretize(discretize(xs, bins=256), bins=256)
    assert mean_abs_error([xs], [fine]) < mean_abs_error([xs], [coarse])


# ---------------------------------------------------------------------- dct
def test_dct_of_a_constant_keeps_only_the_dc_coefficient():
    assert dct([1.0, 1.0, 1.0, 1.0]) == pytest.approx([4.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_dct_is_linear():
    a = [0.3, -0.7, 0.1, 0.9]
    b = [1.0, 0.2, -0.4, 0.5]
    combined = dct([x + y for x, y in zip(a, b)])
    separate = [x + y for x, y in zip(dct(a), dct(b))]
    assert combined == pytest.approx(separate, abs=1e-12)


def test_dct_concentrates_a_smooth_signal_in_the_low_coefficients():
    """Ради этого FAST и существует: гладкая траектория живёт в первых коэффициентах."""
    smooth = [0.8 * math.sin(2 * math.pi * i / 128) for i in range(32)]
    coeffs = dct(smooth)
    energy = sum(c * c for c in coeffs)
    low = sum(c * c for c in coeffs[:4])
    assert low / energy > 0.99


def test_dct_pushes_an_alternating_signal_into_the_top_coefficient():
    """Дрожь на частоте Найквиста — вся энергия в САМОМ высоком коэффициенте."""
    drum = [0.8 if i % 2 == 0 else -0.8 for i in range(32)]
    coeffs = dct(drum)
    assert max(range(32), key=lambda k: abs(coeffs[k])) == 31


# --------------------------------------------------------------------- idct
def test_idct_inverts_dct():
    x = [0.3, -0.7, 0.1, 0.9, -0.2, 0.55]
    assert idct(dct(x)) == pytest.approx(x, abs=1e-12)


def test_dct_inverts_idct():
    y = [4.0, -1.0, 0.5, 0.25]
    assert dct(idct(y)) == pytest.approx(y, abs=1e-12)


def test_idct_of_a_lone_dc_coefficient_is_a_constant():
    """Вес нулевого коэффициента 1/N, а не 2/N — на этом спотыкаются все."""
    assert idct([4.0, 0.0, 0.0, 0.0]) == pytest.approx([1.0] * 4, abs=1e-12)


# -------------------------------------------------------------- fast_tokens
def test_fast_tokens_count_is_dof_times_kept_coefficients():
    assert len(fast_tokens([[0.0, 0.0]] * 30, keep_coeff=4)) == 8


def test_fast_tokens_do_not_grow_with_the_horizon():
    """В этом весь смысл: 30 шагов и 60 шагов стоят одинаково."""
    short = fast_tokens([[0.1, -0.2]] * 15, keep_coeff=4)
    long = fast_tokens([[0.1, -0.2]] * 60, keep_coeff=4)
    assert len(short) == len(long) == 8


def test_fast_tokens_are_valid_bin_indices():
    trajectory = [[math.sin(i / 5.0) * 0.9, math.cos(i / 7.0) * 0.9] for i in range(24)]
    tokens = fast_tokens(trajectory, keep_coeff=5, bins=256)
    assert all(isinstance(t, int) and 0 <= t < 256 for t in tokens)


def test_fast_tokens_group_coefficients_by_axis():
    """Ось 0 постоянная, ось 1 нулевая: их блоки токенов обязаны отличаться."""
    trajectory = [[0.6, 0.0] for _ in range(16)]
    tokens = fast_tokens(trajectory, keep_coeff=3, bins=256)
    assert tokens[0] != tokens[3]
    assert tokens[3:] == discretize([0.0, 0.0, 0.0], bins=256)


# --------------------------------------------------------- fast_reconstruct
def test_fast_reconstruct_shape():
    tokens = fast_tokens([[0.1, -0.2]] * 12, keep_coeff=4)
    restored = fast_reconstruct(tokens, 12, 2, keep_coeff=4)
    assert len(restored) == 12
    assert all(len(step) == 2 for step in restored)


def test_fast_round_trip_on_a_constant_trajectory():
    trajectory = [[0.5]] * 8
    restored = fast_reconstruct(fast_tokens(trajectory, keep_coeff=4), 8, 1, keep_coeff=4)
    assert mean_abs_error(trajectory, restored) < 0.05


def test_fast_reconstructs_smooth_motion_far_better_than_drumming():
    """Ответ на вопрос урока: FAST теряет именно высокочастотную дрожь."""
    smooth = [[0.8 * math.sin(2 * math.pi * i / 128)] for i in range(32)]
    drum = [[0.8 if i % 2 == 0 else -0.8] for i in range(32)]
    err_smooth = mean_abs_error(
        smooth, fast_reconstruct(fast_tokens(smooth, keep_coeff=4), 32, 1, keep_coeff=4)
    )
    err_drum = mean_abs_error(
        drum, fast_reconstruct(fast_tokens(drum, keep_coeff=4), 32, 1, keep_coeff=4)
    )
    assert err_smooth < 0.05
    assert err_drum > 10 * err_smooth


def test_keeping_more_coefficients_reduces_the_reconstruction_error():
    trajectory = [[math.sin(i / 3.0) * 0.9] for i in range(32)]
    err = {}
    for keep in (2, 8, 24):
        tokens = fast_tokens(trajectory, keep_coeff=keep)
        err[keep] = mean_abs_error(
            trajectory, fast_reconstruct(tokens, 32, 1, keep_coeff=keep)
        )
    assert err[24] < err[8] < err[2]


# ----------------------------------------------------- format_token_budget
def test_format_token_budget_covers_every_known_format():
    assert set(format_token_budget(10, 30)) == set(FORMATS)


def test_format_token_budget_worked_example():
    assert format_token_budget(10, 30) == {"discrete_bin": 300, "fast": 40, "flow": 0}


def test_flow_matching_spends_no_vocabulary_tokens():
    """Голова pi0 выдаёт вектор чисел, а не токены словаря."""
    assert format_token_budget(30, 50)["flow"] == 0


def test_fast_beats_discrete_bins_only_on_long_horizons():
    """При горизонте, равном keep_coeff, сжимать уже нечего."""
    long_horizon = format_token_budget(10, 30, keep_coeff=4)
    assert long_horizon["fast"] < long_horizon["discrete_bin"]
    even = format_token_budget(10, 4, keep_coeff=4)
    assert even["fast"] == even["discrete_bin"]


# ------------------------------------------------------------ cofinetune_mix
def test_cofinetune_mix_respects_the_batch_size():
    batch = cofinetune_mix(["w"], ["r"], 1.0, 17, random.Random(1))
    assert len(batch) == 17


def test_zero_ratio_gives_a_robot_only_batch():
    assert cofinetune_mix(["w"], ["r"], 0.0, 3, random.Random(0)) == ["r", "r", "r"]


def test_the_same_seed_gives_the_same_batch():
    """Без воспроизводимости состав батча не отладить."""
    a = cofinetune_mix(list("abcd"), list("XYZ"), 1.0, 50, random.Random(7))
    b = cofinetune_mix(list("abcd"), list("XYZ"), 1.0, 50, random.Random(7))
    assert a == b
    assert a != cofinetune_mix(list("abcd"), list("XYZ"), 1.0, 50, random.Random(8))


def test_the_ratio_controls_the_web_share():
    """1:1 у RT-2 — примерно половина батча; 0.5:1 у OpenVLA — примерно треть."""
    web, robot = ["w"], ["r"]
    even = cofinetune_mix(web, robot, 1.0, 4000, random.Random(3)).count("w") / 4000
    openvla = cofinetune_mix(web, robot, 0.5, 4000, random.Random(3)).count("w") / 4000
    assert 0.47 < even < 0.53
    assert 0.30 < openvla < 0.37
