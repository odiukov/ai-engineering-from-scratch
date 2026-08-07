"""Тесты к уроку «DPO: Direct Preference Optimization». Правь exercise.py."""

import math

import pytest

from exercise import (
    dpo_grad,
    dpo_logit,
    dpo_loss,
    implicit_rewards,
    log_softmax,
    preference_accuracy,
    sequence_logprob,
    sigmoid,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

LOG2 = math.log(2.0)


def numeric_partial(f, args, i, h=1e-6):
    """Численная производная f по i-му аргументу — центральная разность."""
    up, down = list(args), list(args)
    up[i] += h
    down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_survives_large_negative_input():
    """Наивная 1/(1+exp(-x)) падает с OverflowError на x = -1000."""
    assert sigmoid(-1000.0) == APPROX(0.0)


# ------------------------------------------------------------ log_softmax
def test_log_softmax_of_equal_logits():
    assert log_softmax([0.0, 0.0]) == pytest.approx([-LOG2, -LOG2])


def test_log_softmax_values_are_never_positive():
    """Это логарифмы вероятностей, значит все они <= 0."""
    assert all(v <= 1e-12 for v in log_softmax([5.0, -3.0, 0.1, 2.2]))


def test_log_softmax_exponentiates_back_to_one():
    assert sum(math.exp(v) for v in log_softmax([3.0, -1.0, 0.5])) == APPROX(1.0)


def test_log_softmax_survives_huge_logits():
    """log(softmax(x)) в лоб даёт OverflowError на exp(1000)."""
    assert log_softmax([1000.0, 0.0]) == pytest.approx([0.0, -1000.0], abs=1e-6)


def test_log_softmax_is_shift_invariant():
    assert log_softmax([1.0, 2.0]) == pytest.approx(log_softmax([501.0, 502.0]))


# ------------------------------------------------------- sequence_logprob
def test_sequence_logprob_sums_over_steps():
    assert sequence_logprob([[0.0, 0.0], [0.0, 0.0]], [0, 1]) == APPROX(-2 * LOG2)


def test_sequence_logprob_of_empty_sequence_is_zero():
    """Пустая сумма — ноль, то есть вероятность 1, а не ошибка."""
    assert sequence_logprob([], []) == APPROX(0.0)


def test_sequence_logprob_is_higher_for_the_token_the_model_expects():
    logits = [[3.0, 0.0]]
    assert sequence_logprob(logits, [0]) > sequence_logprob(logits, [1])


def test_sequence_logprob_does_not_underflow_on_a_long_sequence():
    """Перемножение 500 вероятностей дало бы ровно 0.0 и log(0) следом."""
    value = sequence_logprob([[0.0, 0.0]] * 500, [0] * 500)
    assert value == pytest.approx(-500 * LOG2, abs=1e-9)


# ------------------------------------------------------------- dpo_logit
def test_dpo_logit_is_zero_when_policy_equals_reference():
    """Старт обучения: политика — копия reference, преимущества нет ни у кого."""
    assert dpo_logit(-2.0, -3.0, -2.0, -3.0) == APPROX(0.0)


def test_dpo_logit_is_positive_when_policy_prefers_chosen():
    assert dpo_logit(-1.0, -4.0, -2.0, -3.0, 0.1) == APPROX(0.2)


def test_dpo_logit_flips_sign_when_the_pair_is_swapped():
    """Ловушка порядка аргументов: перепутал chosen и rejected — обучение поедет назад."""
    right = dpo_logit(-1.0, -4.0, -2.0, -3.0)
    swapped = dpo_logit(-4.0, -1.0, -3.0, -2.0)
    assert swapped == APPROX(-right)


def test_dpo_logit_scales_linearly_with_beta():
    assert dpo_logit(-1.0, -4.0, -2.0, -3.0, 0.4) == APPROX(
        4 * dpo_logit(-1.0, -4.0, -2.0, -3.0, 0.1)
    )


# --------------------------------------------------------------- dpo_loss
def test_dpo_loss_starts_at_log_two_for_any_pair():
    """Пока политика == reference, лосс равен log 2 независимо от данных."""
    assert dpo_loss(-2.0, -3.0, -2.0, -3.0) == APPROX(LOG2)
    assert dpo_loss(-40.0, -1.0, -40.0, -1.0) == APPROX(LOG2)


def test_dpo_loss_falls_when_chosen_overtakes_rejected():
    """Главное смысловое свойство урока."""
    behind = dpo_loss(-4.0, -1.0, -2.0, -3.0)
    same = dpo_loss(-2.0, -3.0, -2.0, -3.0)
    ahead = dpo_loss(-1.0, -4.0, -2.0, -3.0)
    assert behind > same > ahead


def test_dpo_loss_is_always_positive():
    assert all(
        dpo_loss(w, l, -2.0, -3.0) > 0
        for w, l in [(-1.0, -4.0), (-2.0, -3.0), (-40.0, -1.0)]
    )


def test_dpo_loss_survives_a_confidently_wrong_pair():
    """z = -80: наивный -log(sigmoid(z)) даёт log(0.0) -> ValueError."""
    assert dpo_loss(-802.0, -3.0, -2.0, -3.0, beta=0.1) == pytest.approx(80.0, abs=1e-6)


def test_bigger_beta_reacts_harder_to_the_same_drift():
    small = dpo_loss(-1.0, -4.0, -2.0, -3.0, beta=0.05)
    large = dpo_loss(-1.0, -4.0, -2.0, -3.0, beta=0.5)
    assert large < small


# --------------------------------------------------------------- dpo_grad
def test_dpo_grad_matches_numeric_derivative_for_chosen():
    args = (-1.0, -4.0, -2.0, -3.0, 0.1)
    numeric = numeric_partial(dpo_loss, args, 0)
    assert dpo_grad(*args)[0] == pytest.approx(numeric, abs=1e-8)


def test_dpo_grad_matches_numeric_derivative_for_rejected():
    args = (-1.0, -4.0, -2.0, -3.0, 0.1)
    numeric = numeric_partial(dpo_loss, args, 1)
    assert dpo_grad(*args)[1] == pytest.approx(numeric, abs=1e-8)


def test_dpo_grad_matches_numeric_derivative_at_a_large_beta():
    args = (-0.5, -6.0, -2.0, -3.0, 0.7)
    numeric = numeric_partial(dpo_loss, args, 0)
    assert dpo_grad(*args)[0] == pytest.approx(numeric, abs=1e-8)


def test_dpo_grad_raises_chosen_and_lowers_rejected():
    g_chosen, g_rejected = dpo_grad(-2.0, -3.0, -2.0, -3.0)
    assert g_chosen < 0 < g_rejected


def test_dpo_grad_vanishes_on_an_already_solved_pair():
    """Множитель (1 - s) сам гасит сигнал — DPO не тратит шаги впустую."""
    assert abs(dpo_grad(-1.0, -401.0, -2.0, -3.0)[0]) < 1e-12


# -------------------------------------------------------- implicit_rewards
def test_implicit_rewards_are_zero_at_the_reference():
    out = implicit_rewards(-2.0, -3.0, -2.0, -3.0)
    assert (out["chosen"], out["rejected"], out["margin"]) == pytest.approx((0.0, 0.0, 0.0))


def test_implicit_reward_margin_grows_as_the_policy_learns():
    early = implicit_rewards(-1.8, -3.2, -2.0, -3.0)["margin"]
    late = implicit_rewards(-1.0, -6.0, -2.0, -3.0)["margin"]
    assert late > early > 0


def test_implicit_reward_margin_equals_the_dpo_logit():
    """Тот же множитель beta, та же разность — logit и есть margin."""
    args = (-1.0, -4.0, -2.0, -3.0, 0.1)
    assert implicit_rewards(*args)["margin"] == APPROX(dpo_logit(*args))


def test_implicit_reward_is_negative_when_the_policy_abandons_a_response():
    assert implicit_rewards(-5.0, -3.0, -2.0, -3.0)["chosen"] < 0


# ----------------------------------------------------- preference_accuracy
def test_preference_accuracy_counts_positive_margins():
    rows = [(-1.0, -4.0, -2.0, -3.0), (-4.0, -1.0, -3.0, -2.0)]
    assert preference_accuracy(rows) == APPROX(0.5)


def test_preference_accuracy_of_an_untrained_policy_is_zero():
    """Политика == reference, все margin ровно нулевые — это ноль, не половина."""
    assert preference_accuracy([(-2.0, -3.0, -2.0, -3.0)]) == APPROX(0.0)


def test_preference_accuracy_of_empty_data_is_zero():
    assert preference_accuracy([]) == APPROX(0.0)


def test_preference_accuracy_does_not_depend_on_beta():
    """beta — положительный множитель, знак margin он изменить не может."""
    rows = [(-1.0, -4.0, -2.0, -3.0), (-4.0, -1.0, -3.0, -2.0), (-0.5, -9.0, -2.0, -3.0)]
    assert preference_accuracy(rows, beta=0.01) == APPROX(preference_accuracy(rows, beta=5.0))


# ---------------------------------------------------------- всё вместе
def test_dpo_loss_falls_after_a_step_that_raises_chosen_logits():
    """Сквозной проход: логиты -> log-вероятности -> лосс.

    Reference одинаково относится к обоим ответам. Политика подняла логит
    токена chosen-ответа. Лосс обязан упасть ниже стартового log 2.
    """
    ref_logits = [[0.0, 0.0]]
    policy_logits = [[1.0, 0.0]]

    ref_w = sequence_logprob(ref_logits, [0])
    ref_l = sequence_logprob(ref_logits, [1])
    pi_w = sequence_logprob(policy_logits, [0])
    pi_l = sequence_logprob(policy_logits, [1])

    assert dpo_loss(pi_w, pi_l, ref_w, ref_l) < LOG2
    assert implicit_rewards(pi_w, pi_l, ref_w, ref_l)["margin"] > 0
