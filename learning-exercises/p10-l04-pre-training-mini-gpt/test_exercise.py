"""Тесты к уроку «Предобучение mini-GPT». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    GPT2_SMALL,
    causal_attention,
    count_parameters,
    cross_entropy,
    d_cross_entropy,
    layer_norm,
    sample_next_token,
    softmax,
    top_k_top_p,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [x for row in M for x in row]


# ------------------------------------------------------------------- softmax
def test_softmax_of_equal_scores_is_uniform():
    assert softmax([0.0, 0.0]) == APPROX([0.5, 0.5])
    assert softmax([]) == []


def test_softmax_sums_to_one_and_keeps_the_order_of_scores():
    probs = softmax([0.0, 5.0, 2.0])
    assert sum(probs) == APPROX(1.0)
    assert probs[1] > probs[2] > probs[0]


def test_softmax_ignores_a_constant_shift():
    """softmax(x - c) == softmax(x) — на этом и держится вычитание максимума."""
    assert softmax([1.0, 2.0, 3.0]) == pytest.approx(softmax([101.0, 102.0, 103.0]))


def test_softmax_survives_huge_logits():
    """Наивный exp(1000) — это OverflowError, а не большое число."""
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0], abs=1e-9)


# ---------------------------------------------------------------- layer_norm
def test_layer_norm_centres_and_scales():
    assert layer_norm([1.0, 3.0], [1.0, 1.0], [0.0, 0.0]) == pytest.approx(
        [-1.0, 1.0], abs=1e-4
    )


def test_layer_norm_output_has_zero_mean_and_unit_variance():
    out = layer_norm([4.0, -2.0, 7.0, 1.0], [1.0] * 4, [0.0] * 4)
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert var == pytest.approx(1.0, abs=1e-4)


def test_layer_norm_ignores_shift_and_scale_of_the_input():
    """Активации разъезжаются от слоя к слою — LayerNorm возвращает их к одному виду."""
    base = layer_norm([1.0, 2.0, 3.0], [1.0] * 3, [0.0] * 3)
    shifted = layer_norm([101.0, 102.0, 103.0], [1.0] * 3, [0.0] * 3)
    scaled = layer_norm([10.0, 20.0, 30.0], [1.0] * 3, [0.0] * 3)
    assert shifted == pytest.approx(base, abs=1e-4)
    assert scaled == pytest.approx(base, abs=1e-4)


def test_layer_norm_applies_gamma_and_beta():
    out = layer_norm([1.0, 3.0], [2.0, 2.0], [5.0, 5.0])
    assert out == pytest.approx([3.0, 7.0], abs=1e-4)


def test_layer_norm_does_not_explode_on_a_constant_vector():
    """Все компоненты равны — дисперсия ноль, спасает только eps."""
    assert layer_norm([5.0, 5.0], [1.0, 1.0], [0.0, 0.0]) == APPROX([0.0, 0.0])


# ----------------------------------------------------------- causal_attention
def test_first_position_can_only_attend_to_itself():
    Q = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0], [30.0, 40.0]]
    out = causal_attention(Q, Q, V)
    assert out[0] == APPROX([1.0, 2.0])
    assert causal_attention([[1.0]], [[1.0]], [[7.0, 8.0]]) == [[7.0, 8.0]]


def test_the_future_never_leaks_into_the_past():
    """Причинная маска: подмена последней строки V не трогает предыдущие выходы."""
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
    V_changed = [[1.0, 0.0], [0.0, 1.0], [99.0, -99.0]]
    out = causal_attention(Q, Q, V)
    changed = causal_attention(Q, Q, V_changed)
    assert flat(out[:2]) == pytest.approx(flat(changed[:2]))
    assert out[2] != changed[2]


def test_zero_queries_average_everything_seen_so_far():
    """Все скоры равны — веса равномерные, выход это среднее прошлого."""
    Q = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    V = [[1.0, 1.0], [3.0, 5.0], [8.0, 0.0]]
    out = causal_attention(Q, Q, V)
    assert flat(out) == pytest.approx([1.0, 1.0, 2.0, 3.0, 4.0, 2.0])


def test_attention_scores_are_divided_by_sqrt_of_head_dim():
    """Без деления на sqrt(d_k) softmax насыщается и градиент исчезает."""
    Q = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0], [3.0, 4.0]]
    out = causal_attention(Q, Q, V)
    w = math.exp(1.0 / math.sqrt(2))
    expected = [(1.0 + 3.0 * w) / (1.0 + w), (2.0 + 4.0 * w) / (1.0 + w)]
    assert out[1] == pytest.approx(expected)


# ------------------------------------------------------------- cross_entropy
def test_cross_entropy_of_a_coin_flip_is_ln_two():
    assert cross_entropy([[0.0, 0.0]], [0]) == pytest.approx(math.log(2))
    assert cross_entropy([[100.0, 0.0]], [0]) == pytest.approx(0.0, abs=1e-9)


def test_untrained_model_starts_at_ln_of_vocabulary_size():
    """Байтовый словарь: первый шаг обязан показать 5.545, иначе ищи ошибку."""
    assert cross_entropy([[0.0] * 256], [7]) == pytest.approx(math.log(256), abs=1e-9)


def test_cross_entropy_survives_huge_logits():
    assert cross_entropy([[1000.0, 999.0]], [0]) == pytest.approx(math.log(1 + math.e ** -1))


def test_cross_entropy_grows_when_the_prediction_gets_worse():
    good = cross_entropy([[3.0, 0.0]], [0])
    bad = cross_entropy([[3.0, 0.0]], [1])
    assert bad > good


# ----------------------------------------------------------- d_cross_entropy
def test_d_cross_entropy_of_a_coin_flip():
    assert flat(d_cross_entropy([[0.0, 0.0]], [0])) == APPROX([-0.5, 0.5])


def test_d_cross_entropy_rows_sum_to_zero():
    """Вероятности остаются вероятностями: поднять один логит можно только за счёт других."""
    grad = d_cross_entropy([[0.5, -1.2, 2.0], [0.1, 0.3, -0.7]], [2, 0])
    for row in grad:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)


def test_d_cross_entropy_pushes_the_correct_token_up():
    grad = d_cross_entropy([[0.5, -1.2, 2.0]], [1])
    assert grad[0][1] < 0
    assert grad[0][0] > 0 and grad[0][2] > 0


def test_d_cross_entropy_matches_the_numeric_gradient():
    """Зелёная формула без сверки с численной производной ничего не доказывает."""
    logits = [[0.5, -1.2, 2.0], [0.1, 0.3, -0.7]]
    targets = [2, 0]
    analytic = d_cross_entropy(logits, targets)
    h = 1e-5
    for i in range(len(logits)):
        for j in range(len(logits[0])):
            up = [row[:] for row in logits]
            down = [row[:] for row in logits]
            up[i][j] += h
            down[i][j] -= h
            numeric = (cross_entropy(up, targets) - cross_entropy(down, targets)) / (2 * h)
            assert analytic[i][j] == pytest.approx(numeric, abs=1e-6)


# ---------------------------------------------------------- count_parameters
def test_gpt2_small_has_124_million_parameters():
    """Итог из таблицы урока (124 438 272) забывает финальный LayerNorm."""
    assert count_parameters(**GPT2_SMALL) == 124_439_808


def test_weight_tying_saves_a_whole_embedding_matrix():
    tied = count_parameters(**GPT2_SMALL)
    untied = count_parameters(**GPT2_SMALL, tie_weights=False)
    assert untied - tied == 50257 * 768


def test_doubling_width_costs_far_more_than_doubling_depth():
    deeper = count_parameters(**{**GPT2_SMALL, "num_layers": 24})
    wider = count_parameters(**{**GPT2_SMALL, "embed_dim": 1536, "ff_dim": 6144})
    assert wider > deeper


def test_embeddings_dominate_a_small_vocabulary_model():
    """У крошечной модели словарь съедает почти всё — отсюда и weight tying."""
    tiny = count_parameters(50257, 128, 2, 64, 512)
    embeddings = 50257 * 128
    assert embeddings / tiny > 0.9


# ---------------------------------------------------------------- top_k_top_p
def test_top_k_one_leaves_only_the_best_token():
    assert top_k_top_p([0.6, 0.3, 0.1], top_k=1) == APPROX([1.0, 0.0, 0.0])


def test_top_p_keeps_the_shortest_prefix_that_reaches_p():
    assert top_k_top_p([0.6, 0.3, 0.1], top_p=0.85) == APPROX([2 / 3, 1 / 3, 0.0])


def test_filtered_distribution_still_sums_to_one():
    out = top_k_top_p([0.4, 0.3, 0.2, 0.1], top_k=2)
    assert sum(out) == APPROX(1.0)
    assert out[2] == 0.0 and out[3] == 0.0


def test_a_tiny_p_still_keeps_one_token_so_generation_can_continue():
    assert top_k_top_p([0.6, 0.3, 0.1], top_p=0.01) == APPROX([1.0, 0.0, 0.0])


def test_top_p_of_one_changes_nothing():
    probs = [0.4, 0.35, 0.25]
    assert top_k_top_p(probs, top_p=1.0) == pytest.approx(probs)


# ------------------------------------------------------- sample_next_token
def test_zero_temperature_is_greedy_and_needs_no_randomness():
    """Урок делит логиты на температуру — на нуле это ZeroDivisionError."""
    assert sample_next_token([0.0, 100.0], None, temperature=0.0) == 1


def test_same_seed_gives_the_same_sequence():
    logits = [1.0, 0.5, 0.2, -1.0]
    first = [sample_next_token(logits, random.Random(3)) for _ in range(5)]
    second = [sample_next_token(logits, random.Random(3)) for _ in range(5)]
    assert first == second
    assert all(0 <= i < len(logits) for i in first)


def test_top_k_one_makes_sampling_deterministic():
    logits = [1.0, 0.5, 0.2, -1.0]
    rng = random.Random(11)
    assert {sample_next_token(logits, rng, top_k=1) for _ in range(50)} == {0}


def test_low_temperature_concentrates_on_the_top_token():
    """Низкая температура заостряет распределение, высокая размывает."""
    logits = [2.0, 1.0, 0.0]
    rng = random.Random(0)
    cold = [sample_next_token(logits, rng, temperature=0.2) for _ in range(200)]
    rng = random.Random(0)
    hot = [sample_next_token(logits, rng, temperature=2.0) for _ in range(200)]
    assert cold.count(0) > hot.count(0)
    assert len(set(hot)) > len(set(cold))
