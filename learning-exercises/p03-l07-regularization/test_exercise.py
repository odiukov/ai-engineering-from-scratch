"""Тесты к уроку «Регуляризация». Правь exercise.py."""

import math

import pytest

from exercise import (
    apply_dropout,
    batch_norm,
    dropout_mask,
    early_stop_epoch,
    generalization_gap,
    l2_gradient,
    l2_penalty,
    layer_norm,
    rms_norm,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in matrix for x in row]


def mean(values):
    return sum(values) / len(values)


def variance(values):
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def numeric_gradient(f, values, h=1e-6):
    grads = []
    for i in range(len(values)):
        up, down = list(values), list(values)
        up[i] += h
        down[i] -= h
        grads.append((f(up) - f(down)) / (2.0 * h))
    return grads


# ------------------------------------------------------------ dropout_mask
def test_dropout_mask_with_zero_probability_keeps_everything():
    assert dropout_mask(4, 0.0, seed=0) == [1, 1, 1, 1]


def test_dropout_mask_with_probability_one_drops_everything():
    assert dropout_mask(4, 1.0, seed=0) == [0, 0, 0, 0]


def test_dropout_mask_is_reproducible_for_one_seed():
    assert dropout_mask(64, 0.5, seed=1) == dropout_mask(64, 0.5, seed=1)


def test_dropout_mask_differs_between_seeds():
    assert dropout_mask(64, 0.5, seed=1) != dropout_mask(64, 0.5, seed=2)


def test_dropout_mask_drops_about_p_of_the_neurons():
    mask = dropout_mask(4000, 0.3, seed=7)
    assert mask.count(0) / len(mask) == pytest.approx(0.3, abs=0.03)


def test_dropout_mask_has_the_requested_length():
    assert len(dropout_mask(17, 0.5, seed=0)) == 17


# ----------------------------------------------------------- apply_dropout
def test_apply_dropout_zeroes_the_masked_values():
    assert apply_dropout([2.0, 2.0], [1, 0], 0.5)[1] == APPROX(0.0)


def test_apply_dropout_scales_the_survivors():
    """Inverted dropout: выжившие делятся на (1 - p), иначе масштаб уедет."""
    assert apply_dropout([2.0, 2.0], [1, 0], 0.5)[0] == APPROX(4.0)


def test_apply_dropout_with_zero_probability_changes_nothing():
    assert apply_dropout([2.0, -3.0], [1, 1], 0.0) == pytest.approx([2.0, -3.0])


def test_apply_dropout_preserves_the_expected_sum():
    """Ради этого равенства и нужно деление: инференс видит тот же масштаб."""
    values = [1.0] * 2000
    total = 0.0
    for seed in range(20):
        mask = dropout_mask(len(values), 0.4, seed=seed)
        total += sum(apply_dropout(values, mask, 0.4))
    assert total / 20 == pytest.approx(sum(values), rel=0.02)


def test_apply_dropout_survives_probability_one():
    """p = 1.0 выключает всё — делить не на что, ZeroDivisionError недопустим."""
    assert apply_dropout([2.0, 2.0], [0, 0], 1.0) == pytest.approx([0.0, 0.0])


# --------------------------------------------------- l2_penalty / l2_gradient
def test_l2_penalty_worked_example():
    assert l2_penalty([3.0, 4.0], 1.0) == APPROX(12.5)


def test_l2_penalty_is_zero_without_lambda():
    assert l2_penalty([3.0, 4.0], 0.0) == APPROX(0.0)


def test_l2_penalty_punishes_one_big_weight_more_than_many_small():
    """Штраф толкает к решениям, где ни один вес не доминирует."""
    assert l2_penalty([4.0, 0.0, 0.0, 0.0], 1.0) > l2_penalty([1.0, 1.0, 1.0, 1.0], 1.0)


def test_l2_gradient_worked_example():
    assert l2_gradient([3.0, 4.0], 1.0) == pytest.approx([3.0, 4.0])


def test_l2_gradient_matches_the_numeric_derivative():
    """Половинка в формуле штрафа существует ровно ради того, чтобы здесь не было двойки."""
    weights = [1.5, -2.5, 0.3]
    assert l2_gradient(weights, 0.7) == pytest.approx(
        numeric_gradient(lambda w: l2_penalty(w, 0.7), weights), abs=1e-6
    )


def test_l2_gradient_always_points_back_to_zero():
    assert all(w * g > 0 for w, g in zip([3.0, -3.0], l2_gradient([3.0, -3.0], 0.1)))


# -------------------------------------------------------------- batch_norm
def test_batch_norm_zeroes_the_mean_of_each_feature():
    batch = [[1.0, 10.0], [3.0, 20.0], [5.0, 60.0], [7.0, 10.0]]
    out = batch_norm(batch, [1.0, 1.0], [0.0, 0.0])
    for j in range(2):
        assert mean([sample[j] for sample in out]) == pytest.approx(0.0, abs=1e-9)


def test_batch_norm_makes_variance_one():
    batch = [[1.0], [3.0], [5.0], [7.0]]
    out = batch_norm(batch, [1.0], [0.0])
    assert variance([sample[0] for sample in out]) == pytest.approx(1.0, abs=1e-4)


def test_batch_norm_applies_gamma_and_beta():
    out = batch_norm([[1.0], [3.0]], [2.0], [5.0])
    assert flat(out) == pytest.approx([3.0, 7.0], abs=1e-4)


def test_batch_norm_normalizes_features_independently():
    """Второй признак в сто раз крупнее — после нормировки разницы нет."""
    batch = [[1.0, 100.0], [3.0, 300.0]]
    out = batch_norm(batch, [1.0, 1.0], [0.0, 0.0])
    assert out[0][0] == pytest.approx(out[0][1], abs=1e-4)


def test_batch_norm_output_depends_on_the_neighbours():
    """Тот же пример в другом батче нормируется иначе — вот слабость BatchNorm."""
    x = [1.0, 3.0, 8.0]
    quiet = batch_norm([x, [0.0, 0.0, 0.0]], [1.0] * 3, [0.0] * 3)[0]
    loud = batch_norm([x, [50.0, 50.0, 50.0]], [1.0] * 3, [0.0] * 3)[0]
    assert quiet != pytest.approx(loud)


def test_batch_norm_degenerates_on_a_single_sample():
    """Дисперсия батча из одного примера равна нулю — вот зачем нужен LayerNorm."""
    assert flat(batch_norm([[5.0]], [1.0], [0.0])) == pytest.approx([0.0], abs=1e-9)


# -------------------------------------------------------------- layer_norm
def test_layer_norm_zeroes_the_mean_of_the_sample():
    out = layer_norm([1.0, 3.0, 8.0, -2.0], [1.0] * 4, [0.0] * 4)
    assert mean(out) == pytest.approx(0.0, abs=1e-9)


def test_layer_norm_makes_variance_one():
    out = layer_norm([1.0, 3.0, 8.0, -2.0], [1.0] * 4, [0.0] * 4)
    assert variance(out) == pytest.approx(1.0, abs=1e-5)


def test_layer_norm_applies_gamma_and_beta():
    assert layer_norm([1.0, 3.0], [2.0, 2.0], [1.0, 1.0]) == pytest.approx(
        [-1.0, 3.0], abs=1e-4
    )


def test_layer_norm_ignores_a_constant_shift():
    """Прибавили ко всем признакам одно и то же — центрирование это съело."""
    assert layer_norm([1.0, 3.0, 8.0], [1.0] * 3, [0.0] * 3) == pytest.approx(
        layer_norm([101.0, 103.0, 108.0], [1.0] * 3, [0.0] * 3), abs=1e-6
    )


def test_layer_norm_ignores_a_positive_rescale():
    """Умножили все признаки на 10 — среднее и разброс выросли одинаково."""
    x = [1.0, 3.0, 8.0]
    assert layer_norm(x, [1.0] * 3, [0.0] * 3, eps=0.0) == pytest.approx(
        layer_norm([10 * v for v in x], [1.0] * 3, [0.0] * 3, eps=0.0)
    )


# ---------------------------------------------------------------- rms_norm
def test_rms_norm_worked_example():
    assert rms_norm([3.0, 4.0], [1.0, 1.0]) == pytest.approx([0.848528, 1.131371], abs=1e-6)


def test_rms_norm_survives_the_zero_vector():
    """Без eps под корнем оказался бы ноль — и деление на него."""
    assert rms_norm([0.0, 0.0], [1.0, 1.0]) == pytest.approx([0.0, 0.0])


def test_rms_norm_makes_the_rms_one():
    x = [1.0, -3.0, 8.0, 2.0]
    out = rms_norm(x, [1.0] * 4, eps=0.0)
    assert math.sqrt(mean([v * v for v in out])) == pytest.approx(1.0, abs=1e-9)


def test_rms_norm_is_scale_invariant():
    """Умножили вектор на 100 — результат тот же, нормировка съела масштаб."""
    x = [1.0, -3.0, 8.0]
    assert rms_norm(x, [1.0] * 3, eps=0.0) == pytest.approx(
        rms_norm([100 * v for v in x], [1.0] * 3, eps=0.0)
    )


def test_rms_norm_keeps_the_shift_that_layer_norm_removes():
    """Вот и вся разница: RMSNorm не вычитает среднее, поэтому сдвиг для неё виден."""
    x = [1.0, 3.0, 8.0]
    shifted = [v + 100.0 for v in x]
    assert rms_norm(x, [1.0] * 3) != pytest.approx(rms_norm(shifted, [1.0] * 3))


def test_rms_norm_equals_layer_norm_on_a_centered_vector():
    """Когда среднее и так ноль, вычитать нечего — формулы совпадают."""
    x = [-3.0, -1.0, 1.0, 3.0]
    assert rms_norm(x, [1.0] * 4, eps=1e-9) == pytest.approx(
        layer_norm(x, [1.0] * 4, [0.0] * 4, eps=1e-9), abs=1e-6
    )


def test_rms_norm_applies_gamma():
    assert rms_norm([3.0, 4.0], [2.0, 0.0]) == pytest.approx([1.697056, 0.0], abs=1e-6)


# ------------------------------------------------------- generalization_gap
def test_generalization_gap_flags_overfitting():
    assert generalization_gap(0.999, 0.65) == pytest.approx(0.349, abs=1e-9)


def test_generalization_gap_is_small_when_underfitting():
    """Одинаково слабо везде — регуляризация здесь не поможет, нужна ёмкость."""
    assert generalization_gap(0.60, 0.58) < 0.05


# --------------------------------------------------------- early_stop_epoch
def test_early_stop_triggers_after_patience_bad_epochs():
    assert early_stop_epoch([1.0, 0.9, 0.8, 0.85, 0.9, 0.95], patience=2) == 4


def test_early_stop_runs_to_the_end_while_it_improves():
    assert early_stop_epoch([1.0, 0.9, 0.8, 0.7], patience=2) == 3


def test_early_stop_patience_resets_on_improvement():
    """Одно ухудшение между улучшениями терпение не тратит насовсем."""
    assert early_stop_epoch([1.0, 1.1, 0.9, 1.0, 0.8, 0.7], patience=2) == 5


def test_bigger_patience_stops_later():
    losses = [1.0, 0.8, 0.9, 0.95, 1.0, 1.1]
    assert early_stop_epoch(losses, patience=1) < early_stop_epoch(losses, patience=3)


def test_early_stop_saves_epochs_on_an_overfitting_curve():
    """Кривая ушла вверх на третьей эпохе — досчитывать сотню незачем."""
    losses = [1.0, 0.5, 0.4] + [0.4 + 0.01 * i for i in range(100)]
    assert early_stop_epoch(losses, patience=5) < 15
