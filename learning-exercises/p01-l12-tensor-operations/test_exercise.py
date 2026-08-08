"""Тесты к уроку «Операции с тензорами». Правь exercise.py."""

import pytest

from exercise import (
    add,
    broadcast_shapes,
    broadcast_to,
    flat_index,
    permute,
    reduce_sum,
    reshape,
    strides,
)


# ----------------------------------------------------------------- strides
def test_strides_of_a_matrix():
    assert strides((3, 4)) == (4, 1)


def test_strides_of_a_four_dimensional_tensor():
    assert strides((2, 3, 4, 5)) == (60, 20, 5, 1)


def test_last_stride_is_always_one():
    """Последняя ось идёт подряд в памяти — шаг по ней всегда единица."""
    assert strides((7, 2, 9))[-1] == 1


def test_strides_of_a_scalar_are_empty():
    assert strides(()) == ()


def test_first_stride_equals_the_size_of_one_slice():
    """Шаг по первой оси — это сколько элементов в одном срезе."""
    assert strides((5, 3, 2))[0] == 3 * 2


# -------------------------------------------------------------- flat_index
def test_flat_index_of_the_origin_is_zero():
    assert flat_index((3, 4), (0, 0)) == 0


def test_flat_index_of_a_matrix_cell():
    assert flat_index((3, 4), (1, 2)) == 6


def test_flat_index_of_the_last_cell_is_size_minus_one():
    assert flat_index((2, 3, 4), (1, 2, 3)) == 23


def test_moving_one_step_along_an_axis_costs_exactly_its_stride():
    """Определение шага: сдвиг индекса на 1 сдвигает позицию на stride."""
    shape = (4, 5, 6)
    for axis in range(3):
        base = [1, 1, 1]
        moved = [1, 1, 1]
        moved[axis] += 1
        assert flat_index(shape, moved) - flat_index(shape, base) == strides(shape)[axis]


# ----------------------------------------------------------------- reshape
def test_reshape_to_an_explicit_shape():
    assert reshape((2, 6), (3, 4)) == (3, 4)


def test_reshape_infers_the_minus_one_axis():
    assert reshape((2, 6), (-1, 3)) == (4, 3)


def test_reshape_can_flatten_to_one_dimension():
    assert reshape((2, 3, 4), (-1,)) == (24,)


def test_reshape_rejects_a_shape_with_the_wrong_element_count():
    with pytest.raises(ValueError):
        reshape((2, 6), (5, 5))


def test_reshape_rejects_two_inferred_axes():
    """Две минус-единицы нечем развести: система недоопределена."""
    with pytest.raises(ValueError):
        reshape((2, 6), (-1, -1))


def test_reshape_rejects_a_non_divisible_inference():
    with pytest.raises(ValueError):
        reshape((2, 6), (-1, 5))


def test_reshape_rejects_negative_dimensions_other_than_minus_one():
    """Равное произведение не делает форму (-2, -6) допустимой."""
    with pytest.raises(ValueError):
        reshape((2, 6), (-2, -6))


# ----------------------------------------------------------------- permute
def test_permute_transposes_a_matrix():
    assert permute([0, 1, 2, 3, 4, 5], (2, 3), (1, 0)) == ([0, 3, 1, 4, 2, 5], (3, 2))


def test_permute_reorders_the_shape_tuple():
    _, shape = permute(list(range(24)), (1, 2, 3, 4), (0, 2, 3, 1))
    assert shape == (1, 3, 4, 2)


def test_permuting_twice_returns_the_original():
    """(1, 0) — сама себе обратная: два транспонирования дают исходное."""
    data, shape = list(range(12)), (3, 4)
    once = permute(data, shape, (1, 0))
    assert permute(once[0], once[1], (1, 0)) == (data, shape)


def test_identity_permutation_changes_nothing():
    assert permute([9, 8, 7, 6], (2, 2), (0, 1)) == ([9, 8, 7, 6], (2, 2))


def test_permute_actually_moves_data_not_only_the_shape():
    """Ловушка: поменять форму мало — порядок элементов тоже другой."""
    data, _ = permute([0, 1, 2, 3, 4, 5], (2, 3), (1, 0))
    assert data != [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------- broadcast_shapes
def test_broadcast_shapes_expands_both_sides():
    assert broadcast_shapes((3, 1), (1, 4)) == (3, 4)


def test_broadcast_shapes_pads_the_shorter_shape_on_the_left():
    assert broadcast_shapes((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)


def test_broadcast_shapes_is_symmetric():
    assert broadcast_shapes((3, 1), (1, 4)) == broadcast_shapes((1, 4), (3, 1))


def test_bias_of_shape_d_fits_a_batch_but_bias_of_shape_b_does_not():
    """Выравнивание справа: (B, T, D) дружит с (D,) и ссорится с (B,)."""
    assert broadcast_shapes((16, 128, 768), (768,)) == (16, 128, 768)
    with pytest.raises(ValueError):
        broadcast_shapes((16, 128, 768), (16,))


def test_broadcast_shapes_rejects_incompatible_axes():
    with pytest.raises(ValueError):
        broadcast_shapes((3,), (4,))


# ------------------------------------------------------------- broadcast_to
def test_broadcast_to_repeats_a_column():
    assert broadcast_to([1, 2, 3], (3, 1), (3, 2)) == [1, 1, 2, 2, 3, 3]


def test_broadcast_to_repeats_a_row_by_padding_on_the_left():
    assert broadcast_to([1, 2], (2,), (3, 2)) == [1, 2, 1, 2, 1, 2]


def test_broadcast_to_the_same_shape_is_a_copy():
    assert broadcast_to([5, 6, 7, 8], (2, 2), (2, 2)) == [5, 6, 7, 8]


def test_broadcast_to_spreads_a_scalar_everywhere():
    assert broadcast_to([7], (), (2, 2)) == [7, 7, 7, 7]


def test_broadcast_to_refuses_to_stretch_a_non_unit_axis():
    """Ось размера 3 растянуть до 5 нечем — данных для этого нет."""
    with pytest.raises(ValueError):
        broadcast_to([1, 2, 3], (3,), (5,))


# --------------------------------------------------------------------- add
def test_add_of_equal_shapes():
    assert add([1, 2], (2,), [10, 20], (2,)) == ([11, 22], (2,))


def test_add_builds_an_outer_style_grid_from_a_column_and_a_row():
    assert add([1, 2, 3], (3, 1), [10, 20], (1, 2)) == (
        [11, 21, 12, 22, 13, 23],
        (3, 2),
    )


def test_add_of_a_bias_vector_to_every_row_of_a_batch():
    data, shape = add([0, 0, 0, 0, 0, 0], (2, 3), [1, 2, 3], (3,))
    assert (data, shape) == ([1, 2, 3, 1, 2, 3], (2, 3))


def test_add_is_commutative_including_the_resulting_shape():
    left = add([1, 2, 3], (3, 1), [10, 20], (1, 2))
    right = add([10, 20], (1, 2), [1, 2, 3], (3, 1))
    assert left == right


def test_add_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        add([1, 2, 3], (3,), [1, 2, 3, 4], (4,))


# -------------------------------------------------------------- reduce_sum
def test_reduce_sum_over_the_first_axis_sums_columns():
    assert reduce_sum([1, 2, 3, 4, 5, 6], (2, 3), 0) == ([5, 7, 9], (3,))


def test_reduce_sum_over_the_last_axis_sums_rows():
    assert reduce_sum([1, 2, 3, 4, 5, 6], (2, 3), 1) == ([6, 15], (2,))


def test_reduce_sum_of_a_vector_produces_a_scalar_shape():
    assert reduce_sum([1, 2, 3], (3,), 0) == ([6], ())


def test_reduce_sum_over_a_middle_axis_of_a_three_dimensional_tensor():
    assert reduce_sum(list(range(24)), (2, 3, 4), 1) == (
        [12, 15, 18, 21, 48, 51, 54, 57],
        (2, 4),
    )


def test_the_axis_number_changes_the_answer_not_just_the_shape():
    """Перепутанный axis не роняет программу — он молча считает не то."""
    assert reduce_sum([1, 2, 3, 4, 5, 6], (2, 3), 0)[0] != reduce_sum(
        [1, 2, 3, 4, 5, 6], (3, 2), 0
    )[0]


def test_summing_over_every_axis_in_turn_gives_the_total():
    partial, shape = reduce_sum(list(range(24)), (2, 3, 4), 2)
    assert reduce_sum(*reduce_sum(partial, shape, 1), 0)[0] == [sum(range(24))]
