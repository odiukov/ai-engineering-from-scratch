"""Тесты к уроку «Differential Attention (V2)». Правь exercise.py."""

import math

import pytest

from exercise import (
    attend,
    attention_param_count,
    attention_weights,
    best_lambda,
    diff_attention,
    diff_weights,
    signal_to_noise,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def keys_from_scores(scores):
    """q = [2, 0, 0, 0] и k = [s, 0, 0, 0] дают ровно score = s (делитель sqrt(4) = 2)."""
    return [[s, 0.0, 0.0, 0.0] for s in scores]


QUERY = [2.0, 0.0, 0.0, 0.0]
SIGNAL = 3  # позиция «нужного» токена в синтетической последовательности

# Общий шум обеих веток: exp(score) равны 1, 2, 1, _, 3, 1, 2, 1 -> сумма 11.
# Ветка 1 видит сигнал (exp = 13, полная сумма 24), ветка 2 не видит
# (exp = 1, полная сумма 12). Значит lambda* = 12 / 24 = 0.5 гасит шум ТОЧНО.
NOISE_SCORES = [0.0, math.log(2), 0.0, None, math.log(3), 0.0, math.log(2), 0.0]
K_BRANCH1 = keys_from_scores([math.log(13) if s is None else s for s in NOISE_SCORES])
K_BRANCH2 = keys_from_scores([0.0 if s is None else s for s in NOISE_SCORES])
LAMBDA_STAR = 0.5


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, 2.0, 3.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0])


def test_softmax_never_produces_an_exact_zero():
    """Это и есть шумовой пол: даже у безнадёжного токена вес положителен."""
    weights = softmax([0.0, 50.0])
    assert weights[0] > 0.0


# -------------------------------------------------------- attention_weights
def test_identical_keys_give_uniform_attention():
    assert attention_weights([1.0, 0.0], [[0.0, 0.0]] * 4) == pytest.approx([0.25] * 4)


def test_attention_weights_sum_to_one():
    K = [[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]]
    assert sum(attention_weights([1.0, 1.0], K)) == APPROX(1.0)


def test_attention_prefers_the_matching_key():
    K = [[0.0, 0.0], [5.0, 0.0]]
    w = attention_weights([1.0, 0.0], K)
    assert w[1] > w[0]


def test_attention_divides_scores_by_sqrt_of_the_dimension():
    """Без делителя sqrt(d) веса были бы другими — проверяем число."""
    w = attention_weights([2.0, 0.0, 0.0, 0.0], keys_from_scores([0.0, math.log(9)]))
    assert w == pytest.approx([0.1, 0.9])


def test_attention_weights_are_never_negative():
    K = [[-5.0, 3.0], [2.0, -7.0]]
    assert all(x >= 0 for x in attention_weights([1.0, 1.0], K))


# ------------------------------------------------------------------- attend
def test_attend_averages_the_value_rows():
    assert attend([0.5, 0.5], [[1.0, 0.0], [3.0, 4.0]]) == APPROX([2.0, 2.0])


def test_attend_accepts_negative_weights():
    """После вычитания двух softmax веса становятся знаковыми — это норма."""
    assert attend([1.0, -1.0], [[1.0, 0.0], [3.0, 4.0]]) == APPROX([-2.0, -4.0])


def test_attend_with_zero_weights_gives_a_zero_vector():
    assert attend([0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]]) == APPROX([0.0, 0.0])


# ------------------------------------------------------------- diff_weights
def test_diff_weights_subtract_elementwise():
    assert diff_weights([0.6, 0.4], [0.5, 0.5], 1.0) == APPROX([0.1, -0.1])


def test_lambda_zero_leaves_the_first_branch_alone():
    assert diff_weights([0.6, 0.4], [0.5, 0.5], 0.0) == APPROX([0.6, 0.4])


def test_diff_weights_sum_to_one_minus_lambda():
    """Каждая карта суммируется в 1, значит разность — в 1 - lambda."""
    w1 = attention_weights([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    w2 = attention_weights([0.5, -1.0], [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    assert sum(diff_weights(w1, w2, 0.8)) == APPROX(0.2)


def test_identical_branches_cancel_completely_at_lambda_one():
    w = attention_weights([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]])
    assert diff_weights(w, w, 1.0) == APPROX([0.0, 0.0])


# ---------------------------------------------------- noise cancellation ---
def test_both_branches_carry_the_same_noise_before_subtraction():
    """Предпосылка: шум присутствует в ОБЕИХ ветках, и он положительный."""
    w1 = attention_weights(QUERY, K_BRANCH1)
    w2 = attention_weights(QUERY, K_BRANCH2)
    for i in range(len(w1)):
        if i != SIGNAL:
            assert w1[i] > 0.0 and w2[i] > 0.0


def test_the_shared_noise_cancels_exactly_at_the_right_lambda():
    """Ядро урока: вычитание гасит общий шум в обеих ветках до нуля."""
    w1 = attention_weights(QUERY, K_BRANCH1)
    w2 = attention_weights(QUERY, K_BRANCH2)
    d = diff_weights(w1, w2, LAMBDA_STAR)
    noise = [x for i, x in enumerate(d) if i != SIGNAL]
    assert noise == pytest.approx([0.0] * len(noise), abs=1e-12)


def test_the_signal_survives_the_subtraction():
    w1 = attention_weights(QUERY, K_BRANCH1)
    w2 = attention_weights(QUERY, K_BRANCH2)
    assert diff_weights(w1, w2, LAMBDA_STAR)[SIGNAL] == pytest.approx(0.5, abs=1e-12)


def test_differential_beats_plain_attention_on_signal_to_noise():
    w1 = attention_weights(QUERY, K_BRANCH1)
    w2 = attention_weights(QUERY, K_BRANCH2)
    plain = signal_to_noise(w1, SIGNAL)
    diff = signal_to_noise(diff_weights(w1, w2, LAMBDA_STAR), SIGNAL)
    assert plain < 20  # обычное внимание застревает на 8.3
    assert diff > 1000 * plain


def test_cancellation_works_on_arbitrary_shared_noise():
    """Не подгонка под красивые числа: lambda* = Z2 / Z1 гасит любой общий шум."""
    scores = [0.3, -1.2, 0.7, None, 2.1, -0.5, 1.4, 0.2]
    s1 = [4.0 if s is None else s for s in scores]
    s2 = [-0.9 if s is None else s for s in scores]
    lam_star = sum(math.exp(s) for s in s2) / sum(math.exp(s) for s in s1)
    w1 = attention_weights(QUERY, keys_from_scores(s1))
    w2 = attention_weights(QUERY, keys_from_scores(s2))
    assert signal_to_noise(diff_weights(w1, w2, lam_star), SIGNAL) > 1e6


# ----------------------------------------------------------- diff_attention
def test_lambda_zero_reproduces_plain_attention():
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    Q = [[1.0, 0.0], [0.0, 1.0]]
    plain = [attend(attention_weights(q, K), V) for q in Q]
    assert flat(diff_attention(Q, K, Q, K, V, 0.0)) == pytest.approx(flat(plain))


def test_identical_branches_at_lambda_one_output_nothing():
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0], [3.0, 4.0]]
    Q = [[1.0, 1.0]]
    assert flat(diff_attention(Q, K, Q, K, V, 1.0)) == pytest.approx([0.0, 0.0])


def test_diff_attention_returns_one_row_per_query():
    K = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    out = diff_attention(Q, K, Q, K, V, 0.3)
    assert len(out) == 3 and all(len(row) == 3 for row in out)


def test_diff_attention_output_can_be_negative():
    """Отрицательные веса дают знаковый выход — проекция O_W это переварит."""
    V = [[1.0]] * 8
    out = diff_attention([QUERY], K_BRANCH1, [QUERY], K_BRANCH2, V, 2.0)
    assert out[0][0] < 0


# ---------------------------------------------------------- signal_to_noise
def test_signal_to_noise_on_a_hand_made_row():
    assert signal_to_noise([0.7, 0.1, 0.1, 0.1], 0) == pytest.approx(7.0)


def test_zero_noise_reads_as_infinite_ratio():
    """Идеальное сокращение — не аварийный случай, а цель."""
    assert signal_to_noise([0.5, 0.0, 0.0, 0.0], 0) == math.inf


def test_signal_to_noise_uses_absolute_values():
    assert signal_to_noise([0.5, -0.1, 0.1], 0) == pytest.approx(5.0)


# --------------------------------------------------------------- best_lambda
def test_best_lambda_finds_the_cancelling_value():
    grid = [i / 100 for i in range(201)]
    assert best_lambda(QUERY, K_BRANCH1, QUERY, K_BRANCH2, SIGNAL, grid) == pytest.approx(0.5)


def test_best_lambda_beats_the_no_subtraction_baseline():
    grid = [i / 100 for i in range(201)]
    chosen = best_lambda(QUERY, K_BRANCH1, QUERY, K_BRANCH2, SIGNAL, grid)
    assert chosen != 0.0


# -------------------------------------------------- attention_param_count
def test_v2_adds_exactly_one_projection_worth_of_parameters():
    base = attention_param_count(4096, 32, 128, "baseline")
    v2 = attention_param_count(4096, 32, 128, "v2")
    extra = 4096 * 32 * 128 + 4 * 32 * 128
    assert v2["total"] - base["total"] == extra


def test_v1_costs_only_the_lambda_parameters():
    base = attention_param_count(4096, 32, 128, "baseline")
    v1 = attention_param_count(4096, 32, 128, "v1")
    assert v1["total"] - base["total"] == v1["lam"]


def test_kv_projections_are_identical_across_all_variants():
    """Смысл V2: KV-кэш не растёт, значит decode остаётся на скорости базы."""
    variants = [attention_param_count(4096, 32, 128, v) for v in ("baseline", "v1", "v2")]
    assert len({(c["k"], c["v"]) for c in variants}) == 1


def test_v2_doubles_only_the_query_projection():
    base = attention_param_count(4096, 32, 128, "baseline")
    v2 = attention_param_count(4096, 32, 128, "v2")
    assert v2["q"] == 2 * base["q"]
    assert (v2["k"], v2["v"], v2["o"]) == (base["k"], base["v"], base["o"])


def test_baseline_has_no_lambda_parameters():
    assert attention_param_count(4096, 32, 128, "baseline")["lam"] == 0


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        attention_param_count(4096, 32, 128, "v3")
