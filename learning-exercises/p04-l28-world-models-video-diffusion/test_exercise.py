"""Тесты к уроку «Модели мира и видео-диффузия». Правь exercise.py."""

import pytest

from exercise import (
    attention_pairs,
    axis_position_encoding,
    divided_attention_groups,
    flat_index,
    imagine_rollout,
    inverse_dynamics,
    token_count,
    token_grid,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """Развернуть матрицу в плоский список — pytest.approx не умеет вложенность."""
    return [value for row in matrix for value in row]


# -------------------------------------------------------------- token_grid
def test_token_grid_halves_every_axis_with_patch_two():
    assert token_grid((8, 16, 16), (2, 2, 2)) == (4, 8, 8)


def test_token_grid_of_five_seconds_of_360p():
    """120 кадров 640x360 при патче (2, 8, 8)."""
    assert token_grid((120, 360, 640), (2, 8, 8)) == (60, 45, 80)


def test_token_grid_rejects_a_side_that_does_not_divide():
    """Тихое округление вниз отрезало бы последний кадр видео."""
    with pytest.raises(ValueError):
        token_grid((9, 16, 16), (2, 2, 2))


# ------------------------------------------------------------- token_count
def test_token_count_multiplies_the_grid():
    assert token_count((8, 16, 16), (2, 2, 2)) == 256


def test_token_count_of_five_seconds_of_360p_is_huge():
    """216 тысяч токенов — вот почему полный joint attention по видео не считают."""
    assert token_count((120, 360, 640), (2, 8, 8)) == 216000


def test_token_count_agrees_with_the_grid_it_comes_from():
    shape, patch = (12, 24, 36), (3, 4, 6)
    grid = token_grid(shape, patch)
    assert token_count(shape, patch) == grid[0] * grid[1] * grid[2]


# -------------------------------------------------------------- flat_index
def test_flat_index_of_the_first_token_is_zero():
    assert flat_index((4, 8, 8), 0, 0, 0) == 0


def test_flat_index_moves_by_a_row_when_h_grows():
    assert flat_index((4, 8, 8), 0, 1, 0) == 8


def test_flat_index_moves_by_a_whole_frame_when_t_grows():
    assert flat_index((4, 8, 8), 1, 0, 0) == 64


def test_flat_index_covers_every_position_exactly_once():
    grid = (2, 3, 4)
    seen = [
        flat_index(grid, t, h, w)
        for t in range(2)
        for h in range(3)
        for w in range(4)
    ]
    assert sorted(seen) == list(range(24))


def test_flat_index_rejects_coordinates_outside_the_grid():
    """Отрицательный индекс в Python валиден — ошибку надо ловить самим."""
    with pytest.raises(ValueError):
        flat_index((4, 8, 8), 0, 8, 0)


# ------------------------------------------------ divided_attention_groups
def test_time_groups_hold_the_same_spatial_point_across_frames():
    assert divided_attention_groups((2, 1, 2), "time") == [[0, 2], [1, 3]]


def test_space_groups_hold_a_whole_frame():
    assert divided_attention_groups((2, 1, 2), "space") == [[0, 1], [2, 3]]


def test_time_groups_are_as_many_as_spatial_positions():
    groups = divided_attention_groups((6, 3, 4), "time")
    assert len(groups) == 12
    assert all(len(g) == 6 for g in groups)


def test_space_groups_are_as_many_as_frames():
    groups = divided_attention_groups((6, 3, 4), "space")
    assert len(groups) == 6
    assert all(len(g) == 12 for g in groups)


def test_both_groupings_partition_every_token_exactly_once():
    grid = (3, 2, 5)
    total = list(range(30))
    for axis in ("time", "space"):
        indices = sorted(i for group in divided_attention_groups(grid, axis) for i in group)
        assert indices == total


def test_divided_attention_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        divided_attention_groups((2, 2, 2), "depth")


# ---------------------------------------------------------- attention_pairs
def test_joint_attention_is_the_square_of_the_token_count():
    assert attention_pairs((4, 2, 2), "joint") == 256


def test_divided_attention_splits_the_cost_in_two_terms():
    assert attention_pairs((4, 2, 2), "divided") == 128


def test_divided_attention_is_an_order_of_magnitude_cheaper_on_real_video():
    grid = (60, 45, 80)
    assert attention_pairs(grid, "divided") * 10 < attention_pairs(grid, "joint")


def test_divided_attention_does_not_pay_off_on_a_single_frame():
    """При T = 1 лишний проход по времени делает divided дороже joint."""
    grid = (1, 8, 8)
    assert attention_pairs(grid, "divided") > attention_pairs(grid, "joint")


def test_attention_pairs_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        attention_pairs((2, 2, 2), "window")


# ---------------------------------------------------- axis_position_encoding
def test_position_zero_is_zeros_in_sines_and_ones_in_cosines():
    assert flat(axis_position_encoding((1, 1, 1), 2, 2, 2)) == APPROX(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    )


def test_position_encoding_gives_one_vector_per_token():
    encoding = axis_position_encoding((2, 3, 4), 4, 4, 2)
    assert len(encoding) == 24
    assert all(len(vector) == 10 for vector in encoding)


def test_tokens_of_the_same_frame_share_the_time_part():
    """Первые t_dim чисел зависят только от t — и ни от чего больше."""
    encoding = axis_position_encoding((2, 2, 2), 4, 2, 2)
    same_frame = [vector[:4] for vector in encoding[:4]]
    for vector in same_frame[1:]:
        assert vector == APPROX(same_frame[0])
    assert encoding[4][:4] != APPROX(encoding[0][:4])


def test_tokens_differing_only_in_w_share_the_time_and_height_parts():
    encoding = axis_position_encoding((1, 2, 2), 2, 2, 2)
    assert encoding[0][:4] == APPROX(encoding[1][:4])
    assert encoding[0][4:] != APPROX(encoding[1][4:])


def test_neighbouring_positions_stay_distinguishable():
    """Ловушка: частоты от dim вместо dim // 2 склеили бы соседние кадры."""
    encoding = axis_position_encoding((4, 1, 1), 8, 2, 2)
    for i in range(3):
        gap = max(abs(a - b) for a, b in zip(encoding[i][:8], encoding[i + 1][:8]))
        assert gap > 1e-3


def test_position_encoding_rejects_an_odd_axis_dimension():
    with pytest.raises(ValueError):
        axis_position_encoding((1, 1, 1), 3, 2, 2)


# ------------------------------------------------------- inverse_dynamics
def test_inverse_dynamics_returns_the_difference_of_states():
    assert inverse_dynamics([0.0, 0.0], [1.0, -2.0]) == APPROX([1.0, -2.0])


def test_inverse_dynamics_points_from_now_to_next():
    """Знак: next - state. Наоборот — робот поедет от цели."""
    assert inverse_dynamics([5.0], [1.0]) == APPROX([-4.0])


def test_inverse_dynamics_rejects_states_of_different_length():
    with pytest.raises(ValueError):
        inverse_dynamics([1.0], [1.0, 2.0])


# --------------------------------------------------------- imagine_rollout
def test_rollout_applies_the_actions_in_order():
    step = lambda s, a: [s[0] + a[0]]
    assert flat(imagine_rollout([0.0], [[1.0], [1.0]], step)) == APPROX([1.0, 2.0])


def test_rollout_without_actions_imagines_nothing():
    assert imagine_rollout([0.0], [], lambda s, a: s) == []


def test_rollout_does_not_mutate_the_starting_state():
    start = [0.0, 0.0]
    imagine_rollout(start, [[1.0, 1.0]], lambda s, a: [x + y for x, y in zip(s, a)])
    assert start == [0.0, 0.0]


def test_inverse_dynamics_recovers_the_actions_of_a_rollout():
    """Полный круг: воображаем прогон, потом восстанавливаем действия из кадров."""
    start = [0.0, 0.0]
    actions = [[1.0, 0.0], [0.0, 2.0], [-3.0, 1.0]]
    step = lambda s, a: [x + y for x, y in zip(s, a)]
    states = imagine_rollout(start, actions, step)
    recovered = [
        inverse_dynamics(prev, nxt)
        for prev, nxt in zip([start] + states[:-1], states)
    ]
    assert flat(recovered) == APPROX(flat(actions))
