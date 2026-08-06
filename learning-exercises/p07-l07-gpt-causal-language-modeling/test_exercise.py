"""Тесты к уроку «GPT и causal language modeling». Правь exercise.py."""

import math

import pytest

from exercise import (
    causal_attention_weights,
    causal_mask,
    cross_entropy_shifted,
    min_p_filter,
    prefix_average_matrix,
    softmax,
    top_k_filter,
    top_p_filter,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
NEG = float("-inf")


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


# ---------------------------------------------------------------- softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0, 0.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_sums_to_one():
    assert sum(softmax([2.5, -1.0, 0.3, 7.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    """Наивный exp(1000) это OverflowError. Вычитание максимума спасает."""
    assert softmax([1000.0, 1000.0]) == APPROX([0.5, 0.5])


def test_low_temperature_sharpens_the_distribution():
    peak_cold = max(softmax([1.0, 0.0, 0.0], temperature=0.25))
    peak_warm = max(softmax([1.0, 0.0, 0.0], temperature=1.0))
    assert peak_cold > peak_warm


def test_high_temperature_flattens_towards_uniform():
    probs = softmax([5.0, 0.0, -5.0], temperature=1000.0)
    assert probs == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-2)


def test_softmax_gives_minus_inf_exactly_zero_probability():
    """На этом держится вся causal-маска: exp(-inf) = 0."""
    assert softmax([0.0, NEG, NEG]) == APPROX([1.0, 0.0, 0.0])


def test_softmax_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature=0.0)


# ------------------------------------------------------------ causal_mask
def test_causal_mask_blocks_only_the_future():
    assert causal_mask(3) == [
        [0.0, NEG, NEG],
        [0.0, 0.0, NEG],
        [0.0, 0.0, 0.0],
    ]


def test_causal_mask_keeps_the_diagonal_open():
    """Позиция обязана видеть саму себя, иначе первая строка пуста."""
    mask = causal_mask(5)
    assert all(mask[i][i] == 0.0 for i in range(5))


# -------------------------------------------------- prefix_average_matrix
def test_prefix_average_rows_sum_to_one():
    assert [sum(row) for row in prefix_average_matrix(5)] == APPROX([1.0] * 5)


def test_prefix_average_first_row_is_only_itself():
    assert prefix_average_matrix(3)[0] == APPROX([1.0, 0.0, 0.0])


def test_prefix_average_equals_attention_on_flat_scores():
    """Ключевая связка урока: attention с одинаковыми скорами — это то же
    самое префиксное среднее. Треугольник не добавили к attention, он
    остался от обычного среднего."""
    n = 4
    flat_scores = [[0.0] * n for _ in range(n)]
    assert flat(causal_attention_weights(flat_scores)) == APPROX(
        flat(prefix_average_matrix(n))
    )


# ------------------------------------------------ causal_attention_weights
def test_attention_gives_no_weight_to_the_future():
    scores = [[1.0, 9.0, 9.0], [2.0, 3.0, 9.0], [0.5, 0.5, 0.5]]
    weights = causal_attention_weights(scores)
    assert all(weights[i][j] == 0.0 for i in range(3) for j in range(3) if j > i)


def test_attention_rows_are_probability_distributions():
    scores = [[1.0, 9.0, 9.0], [2.0, 3.0, 9.0], [0.5, 0.5, 0.5]]
    assert [sum(row) for row in causal_attention_weights(scores)] == APPROX([1.0] * 3)


def test_first_position_attends_only_to_itself():
    scores = [[0.0, 100.0], [0.0, 0.0]]
    assert causal_attention_weights(scores)[0] == APPROX([1.0, 0.0])


def test_future_scores_never_change_earlier_rows():
    """Смысл causal LM: правка последней колонки не может испортить
    предсказание первых позиций."""
    base = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    changed = [row[:2] + [row[2] + 50.0] for row in base]
    before = causal_attention_weights(base)
    after = causal_attention_weights(changed)
    assert flat(before[:2]) == APPROX(flat(after[:2]))


def test_attention_temperature_reaches_the_masked_row_too():
    """Температура заостряет и причинные строки, а маска остаётся маской."""
    scores = [[0.0, 0.0], [1.0, 0.0]]
    cold = causal_attention_weights(scores, temperature=0.2)
    assert cold[1][0] > 0.9 and cold[0][1] == 0.0


# -------------------------------------------------- cross_entropy_shifted
def test_uniform_logits_give_the_log_of_vocab_size():
    """Необученная модель на словаре V стоит ln V — базовый ориентир."""
    logits = [[0.0] * 8 for _ in range(4)]
    assert cross_entropy_shifted(logits, [1, 3, 5, 7]) == APPROX(math.log(8))


def test_confident_correct_prediction_has_near_zero_loss():
    tokens = [0, 1, 2]
    logits = [[0.0] * 3 for _ in tokens]
    for i in range(len(tokens) - 1):
        logits[i][tokens[i + 1]] = 50.0
    assert cross_entropy_shifted(logits, tokens) < 1e-6


def test_loss_is_shifted_by_one_not_aligned():
    """Модель, предсказывающая ТЕКУЩИЙ токен вместо следующего, получает
    большой loss — иначе сдвиг не реализован."""
    tokens = [0, 1, 2, 3]
    aligned = [[0.0] * 4 for _ in tokens]
    shifted = [[0.0] * 4 for _ in tokens]
    for i in range(len(tokens)):
        aligned[i][tokens[i]] = 10.0
        if i + 1 < len(tokens):
            shifted[i][tokens[i + 1]] = 10.0
    assert cross_entropy_shifted(aligned, tokens) > cross_entropy_shifted(shifted, tokens)


def test_boosting_the_correct_logit_lowers_the_loss():
    tokens = [2, 0, 1]
    base = [[0.0] * 3 for _ in tokens]
    better = [list(row) for row in base]
    for i in range(len(tokens) - 1):
        better[i][tokens[i + 1]] += 1.0
    assert cross_entropy_shifted(better, tokens) < cross_entropy_shifted(base, tokens)


def test_last_position_has_no_target_and_is_ignored():
    """У последней позиции нет следующего токена: её логиты в loss не входят."""
    tokens = [0, 1]
    quiet = [[0.0, 5.0], [0.0, 0.0]]
    loud = [[0.0, 5.0], [99.0, -99.0]]
    assert cross_entropy_shifted(quiet, tokens) == APPROX(
        cross_entropy_shifted(loud, tokens)
    )


def test_single_token_sequence_is_rejected():
    with pytest.raises(ValueError):
        cross_entropy_shifted([[0.0, 0.0]], [1])


# ----------------------------------------------------------- top_k_filter
def test_top_k_keeps_exactly_k_tokens():
    assert sum(1 for p in top_k_filter([0.4, 0.3, 0.2, 0.1], 2) if p > 0) == 2


def test_top_k_renormalizes_the_survivors():
    assert top_k_filter([0.5, 0.3, 0.2], 2) == APPROX([0.625, 0.375, 0.0])


def test_top_k_larger_than_the_vocab_changes_nothing():
    assert top_k_filter([0.5, 0.3, 0.2], 9) == APPROX([0.5, 0.3, 0.2])


def test_top_k_below_one_is_rejected():
    with pytest.raises(ValueError):
        top_k_filter([0.5, 0.5], 0)


# ----------------------------------------------------------- top_p_filter
def test_top_p_keeps_the_smallest_set_reaching_the_mass():
    assert top_p_filter([0.6, 0.3, 0.1], 0.9) == APPROX([2 / 3, 1 / 3, 0.0])


def test_top_p_always_keeps_at_least_one_token():
    assert top_p_filter([0.7, 0.2, 0.1], 0.01) == APPROX([1.0, 0.0, 0.0])


def test_top_p_adapts_to_the_shape_of_the_distribution():
    """Тот же p режет острое распределение до одного токена, а плоское
    не режет почти совсем — этим top-p и отличается от top-k."""
    peaked = sum(1 for x in top_p_filter([0.95, 0.03, 0.01, 0.01], 0.9) if x > 0)
    flat_d = sum(1 for x in top_p_filter([0.25, 0.25, 0.25, 0.25], 0.9) if x > 0)
    assert peaked == 1 and flat_d == 4


def test_top_p_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        top_p_filter([0.5, 0.5], 0.0)


# ---------------------------------------------------------- min_p_filter
def test_min_p_cuts_the_tail_of_a_peaked_distribution():
    assert min_p_filter([0.9, 0.05, 0.05], 0.1) == APPROX([1.0, 0.0, 0.0])


def test_min_p_leaves_a_flat_distribution_intact():
    """Порог относительный, поэтому на плоском распределении он не срабатывает."""
    assert min_p_filter([0.34, 0.33, 0.33], 0.1) == APPROX([0.34, 0.33, 0.33])


def test_min_p_zero_keeps_everything():
    assert min_p_filter([0.7, 0.2, 0.1], 0.0) == APPROX([0.7, 0.2, 0.1])


def test_min_p_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        min_p_filter([0.5, 0.5], -0.1)
