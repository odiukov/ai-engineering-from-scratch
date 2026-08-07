"""Тесты к уроку «Vision-Language Models: паттерн ViT-MLP-LLM». Правь exercise.py."""

import pytest

from exercise import (
    cosine_similarity,
    count_projector_params,
    cross_modal_error_rate,
    deepstack_concat,
    gelu,
    linear,
    merge_image_tokens,
    projector_forward,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


# --------------------------------------------------------------------- gelu
def test_gelu_at_zero_is_zero():
    assert gelu(0.0) == APPROX(0.0)


def test_gelu_matches_the_erf_formula_at_one():
    assert gelu(1.0) == APPROX(0.8413447460685429)


def test_gelu_leaks_negative_values_unlike_relu():
    """Главное отличие от ReLU: слегка отрицательный вход не обнуляется."""
    assert gelu(-1.0) == APPROX(-0.15865525393145707)
    assert gelu(-1.0) < 0.0


def test_gelu_saturates_to_identity_and_to_zero():
    assert gelu(10.0) == pytest.approx(10.0, abs=1e-6)
    assert gelu(-10.0) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------- linear
def test_linear_applies_rows_as_output_neurons():
    out = linear([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, 0.0])
    assert flat(out) == APPROX([1.0, 2.0, 3.0])


def test_linear_changes_the_token_dimension_not_the_token_count():
    out = linear([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 1.0]], [0.0])
    assert len(out) == 3
    assert all(len(row) == 1 for row in out)


def test_linear_adds_the_bias():
    out = linear([[1.0], [2.0]], [[3.0]], [1.0])
    assert flat(out) == APPROX([4.0, 7.0])


def test_linear_rejects_a_vector_of_the_wrong_width():
    with pytest.raises(ValueError):
        linear([[1.0, 2.0, 3.0]], [[1.0, 0.0]], [0.0])


# --------------------------------------------------------- projector_forward
def test_projector_forward_worked_example():
    out = projector_forward([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0],
                            [[1.0, 1.0]], [0.0])
    assert flat(out) == APPROX([0.8413447460685429])


def test_projector_maps_vit_dim_to_llm_dim_keeping_token_count():
    tokens = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    w1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    b1 = [0.0] * 4
    w2 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    b2 = [0.0, 0.0]
    out = projector_forward(tokens, w1, b1, w2, b2)
    assert len(out) == 2                      # токенов столько же
    assert all(len(row) == 2 for row in out)  # размерность стала d_llm


def test_projector_is_not_linear_because_of_the_activation():
    """Если бы GELU не было, projector(2x) равнялся бы 2*projector(x)."""
    w1, b1 = [[1.0]], [0.0]
    w2, b2 = [[1.0]], [0.0]
    one = projector_forward([[1.0]], w1, b1, w2, b2)[0][0]
    two = projector_forward([[2.0]], w1, b1, w2, b2)[0][0]
    assert two != pytest.approx(2 * one, abs=1e-3)


# --------------------------------------------------- count_projector_params
def test_count_projector_params_on_a_tiny_shape():
    assert count_projector_params(2, 2, 1) == 9


def test_count_projector_params_on_a_production_shape():
    assert count_projector_params(768, 4096, 4096) == 19931136


def test_count_projector_params_grows_with_hidden_width():
    assert count_projector_params(768, 8192, 4096) > count_projector_params(768, 4096, 4096)


# ---------------------------------------------------------- deepstack_concat
def test_deepstack_concat_glues_channels_of_two_levels():
    assert flat(deepstack_concat([[[1.0, 2.0]], [[3.0]]])) == APPROX([1.0, 2.0, 3.0])


def test_deepstack_keeps_token_count_and_sums_dimensions():
    levels = [[[1.0, 1.0], [2.0, 2.0]], [[3.0], [4.0]], [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]]
    out = deepstack_concat(levels)
    assert len(out) == 2
    assert all(len(row) == 2 + 1 + 3 for row in out)


def test_deepstack_rejects_levels_with_different_token_counts():
    with pytest.raises(ValueError):
        deepstack_concat([[[1.0], [2.0]], [[3.0]]])


def test_deepstack_rejects_an_empty_level_list():
    with pytest.raises(ValueError):
        deepstack_concat([])


# --------------------------------------------------------- cosine_similarity
def test_cosine_of_identical_directions_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == APPROX(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_of_opposite_directions_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_cosine_ignores_vector_length():
    """Именно поэтому эмбеддинги можно не нормализовать заранее."""
    assert cosine_similarity([3.0, 4.0], [30.0, 40.0]) == APPROX(1.0)
    assert cosine_similarity([1.0, 2.0], [5.0, 10.0]) == APPROX(1.0)


def test_cosine_rejects_a_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


# --------------------------------------------------- cross_modal_error_rate
def test_cmer_flags_only_confident_and_unaligned_answers():
    rate = cross_modal_error_rate([[1.0, 0.0], [1.0, 0.0]],
                                  [[1.0, 0.0], [0.0, 1.0]],
                                  [0.9, 0.9])
    assert rate == APPROX(0.5)


def test_cmer_of_an_empty_batch_is_zero_not_a_division_error():
    assert cross_modal_error_rate([], [], []) == APPROX(0.0)


def test_cmer_ignores_unaligned_answers_the_model_is_unsure_about():
    """Низкая уверенность — не галлюцинация, а честное «не знаю»."""
    rate = cross_modal_error_rate([[1.0, 0.0]], [[0.0, 1.0]], [0.1])
    assert rate == APPROX(0.0)


def test_cmer_ignores_confident_answers_that_match_the_image():
    rate = cross_modal_error_rate([[1.0, 0.0]], [[1.0, 0.0]], [0.99])
    assert rate == APPROX(0.0)


def test_cmer_thresholds_are_strict_inequalities():
    """Ровно на пороге ответ ошибкой не считается."""
    rate = cross_modal_error_rate([[1.0, 0.0]], [[0.0, 1.0]], [0.8], conf_threshold=0.8)
    assert rate == APPROX(0.0)


def test_cmer_rejects_mismatched_batch_lengths():
    with pytest.raises(ValueError):
        cross_modal_error_rate([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [0.9])


# -------------------------------------------------------- merge_image_tokens
def test_merge_replaces_the_placeholder_position():
    out = merge_image_tokens([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
                             [[9.0, 9.0]], [5, 32000, 7], 32000)
    assert flat(out) == APPROX([1.0, 1.0, 9.0, 9.0, 2.0, 2.0])


def test_merge_fills_interleaved_placeholders_left_to_right():
    out = merge_image_tokens([[0.0], [0.0], [0.0], [0.0], [0.0]],
                             [[1.0], [2.0]], [7, 99, 7, 99, 7], 99)
    assert flat(out) == APPROX([0.0, 1.0, 0.0, 2.0, 0.0])


def test_merge_rejects_a_placeholder_count_mismatch():
    """В батче все сэмплы обязаны иметь одинаковое число <image>."""
    with pytest.raises(ValueError):
        merge_image_tokens([[0.0], [0.0]], [[1.0], [2.0]], [99, 7], 99)


def test_merge_does_not_mutate_the_text_embeddings():
    text = [[1.0, 1.0], [0.0, 0.0]]
    merge_image_tokens(text, [[9.0, 9.0]], [7, 99], 99)
    assert flat(text) == APPROX([1.0, 1.0, 0.0, 0.0])


def test_merge_without_placeholders_returns_the_text_unchanged():
    out = merge_image_tokens([[1.0], [2.0]], [], [7, 8], 99)
    assert flat(out) == APPROX([1.0, 2.0])


def test_merged_sequence_length_equals_the_prompt_length():
    """LLM видит один поток токенов: картинка не удлиняет последовательность,
    она занимает уже зарезервированные под неё места."""
    ids = [7, 99, 99, 8]
    out = merge_image_tokens([[0.0]] * 4, [[1.0], [2.0]], ids, 99)
    assert len(out) == len(ids)
