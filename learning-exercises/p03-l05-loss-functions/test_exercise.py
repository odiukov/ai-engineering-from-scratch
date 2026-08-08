"""Тесты к уроку «Функции потерь». Правь exercise.py."""

import math

import pytest

from exercise import (
    bce_gradient,
    binary_cross_entropy,
    categorical_cross_entropy,
    cce_gradient,
    label_smoothed_cce,
    mse,
    mse_gradient,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def numeric_gradient(loss_fn, values, h=1e-6):
    """Центральная разность по каждому элементу — независимая проверка градиентов."""
    grads = []
    for i in range(len(values)):
        up, down = list(values), list(values)
        up[i] += h
        down[i] -= h
        grads.append((loss_fn(up) - loss_fn(down)) / (2.0 * h))
    return grads


# -------------------------------------------------------------------- mse
def test_mse_of_a_perfect_prediction_is_zero():
    assert mse([1.0, 2.0], [1.0, 2.0]) == APPROX(0.0)


def test_mse_worked_example():
    assert mse([0.0, 0.0], [1.0, 3.0]) == APPROX(5.0)


def test_mse_is_averaged_not_summed():
    """Тот же промах на вдвое большем наборе даёт тот же loss."""
    assert mse([0.0] * 4, [1.0] * 4) == APPROX(mse([0.0] * 8, [1.0] * 8))


def test_mse_punishes_big_errors_quadratically():
    """Ошибка втрое больше стоит в девять раз дороже — отсюда чувствительность к выбросам."""
    assert mse([0.0], [3.0]) == APPROX(9 * mse([0.0], [1.0]))


# ----------------------------------------------------------- mse_gradient
def test_mse_gradient_worked_example():
    assert mse_gradient([0.0, 0.0], [1.0, 3.0]) == pytest.approx([-1.0, -3.0])


def test_mse_gradient_is_zero_at_the_optimum():
    assert mse_gradient([2.0, 5.0], [2.0, 5.0]) == pytest.approx([0.0, 0.0])


def test_mse_gradient_matches_the_numeric_derivative():
    preds, targets = [0.3, -1.2, 4.0], [1.0, 0.0, 2.5]
    assert mse_gradient(preds, targets) == pytest.approx(
        numeric_gradient(lambda v: mse(v, targets), preds), abs=1e-6
    )


def test_mse_gradient_is_divided_by_the_batch_size():
    """Без деления на n градиент растёт вместе с батчем и уносит learning rate."""
    small = mse_gradient([0.0] * 2, [1.0] * 2)[0]
    big = mse_gradient([0.0] * 8, [1.0] * 8)[0]
    assert small == APPROX(4 * big)


# --------------------------------------------------- binary_cross_entropy
def test_bce_rewards_a_confident_correct_answer():
    assert binary_cross_entropy([0.9], [1.0]) == pytest.approx(0.105361, abs=1e-6)


def test_bce_of_a_coin_flip_is_log_two():
    assert binary_cross_entropy([0.5], [1.0]) == pytest.approx(math.log(2), abs=1e-9)


def test_bce_survives_log_of_zero():
    """Модель вправе выдать ровно 0.0 — loss обязан остаться конечным числом."""
    loss = binary_cross_entropy([0.0], [1.0])
    assert math.isfinite(loss)
    assert loss > 30.0


def test_bce_survives_log_of_one_for_a_zero_target():
    loss = binary_cross_entropy([1.0], [0.0])
    assert math.isfinite(loss)
    assert loss > 30.0


def test_bce_is_symmetric_between_the_classes():
    assert binary_cross_entropy([0.3], [1.0]) == APPROX(binary_cross_entropy([0.7], [0.0]))


def test_bce_is_averaged_not_summed():
    assert binary_cross_entropy([0.3] * 4, [1.0] * 4) == APPROX(
        binary_cross_entropy([0.3] * 9, [1.0] * 9)
    )


# ----------------------------------------------------------- bce_gradient
def test_bce_gradient_worked_example():
    assert bce_gradient([0.5], [1.0]) == pytest.approx([-2.0])


def test_bce_gradient_matches_the_numeric_derivative():
    preds, targets = [0.2, 0.75, 0.5], [1.0, 0.0, 1.0]
    assert bce_gradient(preds, targets) == pytest.approx(
        numeric_gradient(lambda v: binary_cross_entropy(v, targets), preds, h=1e-6),
        rel=1e-4,
    )


def test_bce_gradient_explodes_on_a_confident_mistake():
    """t = 1, p почти ноль: сигнал «немедленно чини» вместо затухающего градиента."""
    assert bce_gradient([1e-6], [1.0])[0] < -100000.0


def test_bce_gradient_is_zero_on_the_clipped_probability_plateau():
    assert bce_gradient([0.0, 1.0], [1.0, 0.0]) == APPROX([0.0, 0.0])


def test_bce_gradient_matches_finite_difference_at_probability_boundaries():
    """Loss постоянен около 0 и 1 из-за clip, значит и производная нулевая."""
    preds, targets = [0.0, 1.0], [1.0, 0.0]
    assert bce_gradient(preds, targets, eps=1e-3) == pytest.approx(
        numeric_gradient(
            lambda v: binary_cross_entropy(v, targets, eps=1e-3),
            preds,
            h=1e-6,
        ),
        abs=1e-9,
    )


def test_bce_gradient_is_small_when_already_right():
    assert abs(bce_gradient([0.99], [1.0])[0]) < abs(bce_gradient([0.6], [1.0])[0])


def test_bce_gradient_beats_mse_gradient_on_a_confident_mistake():
    """Ровно тот случай, ради которого классификацию не учат на MSE:
    у MSE градиент упирается в -2, у кросс-энтропии — нет потолка."""
    wrong_and_sure = ([0.01], [1.0])
    assert abs(bce_gradient(*wrong_and_sure)[0]) > 20 * abs(mse_gradient(*wrong_and_sure)[0])


def test_bce_gradient_is_averaged_over_the_batch():
    """Loss усредняется по батчу — градиент обязан делиться на то же n."""
    assert bce_gradient([0.5] * 2, [1.0] * 2)[0] == APPROX(
        4 * bce_gradient([0.5] * 8, [1.0] * 8)[0]
    )


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([2.0, 1.0, 0.0, -1.0])) == APPROX(1.0)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])


def test_softmax_survives_huge_logits():
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


def test_softmax_is_shift_invariant():
    assert softmax([2.0, 1.0, 0.0]) == pytest.approx(softmax([-98.0, -99.0, -100.0]))


# ------------------------------------------- categorical_cross_entropy
def test_cce_of_a_uniform_guess_is_log_of_class_count():
    assert categorical_cross_entropy([0.0] * 10, 3) == pytest.approx(math.log(10), abs=1e-9)


def test_cce_of_a_certain_correct_answer_is_zero():
    assert categorical_cross_entropy([1000.0, 0.0], 0) == pytest.approx(0.0, abs=1e-9)


def test_cce_survives_a_confidently_wrong_answer():
    """Наивный -log(softmax(...)[target]) здесь даёт log(0.0) и падает."""
    loss = categorical_cross_entropy([0.0, 1000.0], 0)
    assert math.isfinite(loss)
    assert loss == pytest.approx(1000.0, abs=1e-6)


def test_cce_is_shift_invariant():
    assert categorical_cross_entropy([2.0, 1.0, 0.0], 0) == pytest.approx(
        categorical_cross_entropy([502.0, 501.0, 500.0], 0), abs=1e-9
    )


def test_cce_grows_as_the_true_class_loses_probability():
    assert categorical_cross_entropy([3.0, 0.0], 0) < categorical_cross_entropy([1.0, 0.0], 0)


# ----------------------------------------------------------- cce_gradient
def test_cce_gradient_worked_example():
    assert cce_gradient([0.0, 0.0], 0) == pytest.approx([-0.5, 0.5])


def test_cce_gradient_sums_to_zero():
    """Softmax перекладывает вероятность, а не создаёт её: сумма сдвигов нулевая."""
    assert sum(cce_gradient([2.0, 1.0, -3.0, 0.5], 2)) == pytest.approx(0.0, abs=1e-12)


def test_cce_gradient_is_negative_only_for_the_true_class():
    grads = cce_gradient([1.0, 2.0, 3.0], 1)
    assert grads[1] < 0
    assert grads[0] > 0 and grads[2] > 0


def test_cce_gradient_matches_the_numeric_derivative():
    logits = [1.3, -0.7, 2.2, 0.4]
    assert cce_gradient(logits, 2) == pytest.approx(
        numeric_gradient(lambda v: categorical_cross_entropy(v, 2), logits), abs=1e-6
    )


def test_cce_gradient_vanishes_when_the_answer_is_certain():
    assert cce_gradient([50.0, 0.0], 0) == pytest.approx([0.0, 0.0], abs=1e-15)


# ------------------------------------------------------ label_smoothed_cce
def test_label_smoothing_with_zero_alpha_is_plain_cce():
    assert label_smoothed_cce([2.0, 0.0], 0, alpha=0.0) == pytest.approx(
        categorical_cross_entropy([2.0, 0.0], 0), abs=1e-9
    )


def test_label_smoothing_worked_example():
    assert label_smoothed_cce([2.0, 0.0], 0, alpha=0.2) == pytest.approx(0.326928, abs=1e-6)


def test_label_smoothing_makes_overconfidence_expensive():
    """Логиты уехали в бесконечность — обычная CCE довольна, сглаженная нет."""
    plain = categorical_cross_entropy([50.0, 0.0, 0.0], 0)
    smoothed = label_smoothed_cce([50.0, 0.0, 0.0], 0, alpha=0.1)
    assert plain == pytest.approx(0.0, abs=1e-9)
    assert smoothed > 3.0


def test_label_smoothing_has_a_minimum_at_the_smoothed_target():
    """Оптимум сместился с 1.0 на 1 - alpha + alpha/K — в этом весь смысл приёма."""
    k, alpha = 3, 0.3
    target_p = 1.0 - alpha + alpha / k
    best_logit = math.log(target_p / ((1.0 - target_p) / (k - 1)))
    at_optimum = label_smoothed_cce([best_logit, 0.0, 0.0], 0, alpha=alpha)
    assert at_optimum < label_smoothed_cce([best_logit + 1.0, 0.0, 0.0], 0, alpha=alpha)
    assert at_optimum < label_smoothed_cce([best_logit - 1.0, 0.0, 0.0], 0, alpha=alpha)


def test_label_smoothing_stays_finite_on_extreme_logits():
    assert math.isfinite(label_smoothed_cce([0.0, 1000.0], 0, alpha=0.1))
