"""Тесты к уроку «Multi-Token Prediction (MTP)». Правь exercise.py."""

import math

import pytest

from exercise import (
    cross_entropy,
    depth_hidden,
    joint_loss,
    matvec,
    mtp_depth_losses,
    mtp_extra_params,
    rms_norm,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Игрушечная модель: словарь из 4 токенов, hidden = 2.
EMBEDDING = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
W_OUT = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
M1 = [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]]
M2 = [[0.5, 0.25, 1.0, 0.0], [0.25, 0.5, 0.0, 1.0]]
H0 = [0.5, -0.25]


# ------------------------------------------------------------------- matvec
def test_matvec_multiplies_rows_by_the_vector():
    assert matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]) == APPROX([3.0, 7.0])


def test_matvec_of_a_zero_row_is_zero():
    assert matvec([[0.0, 0.0]], [5.0, 5.0]) == APPROX([0.0])


def test_matvec_rejects_a_shape_mismatch():
    """zip обрежет молча, и ошибка вылезет через два слоя — ловим здесь."""
    with pytest.raises(ValueError):
        matvec([[1.0, 2.0]], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, 2.0, 3.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0])


# ------------------------------------------------------------ cross_entropy
def test_uniform_logits_cost_log_of_the_vocabulary():
    """Потеря необученной модели: log(V), с этого начинается любая кривая."""
    assert cross_entropy([0.0] * 32, 7) == pytest.approx(math.log(32))


def test_confident_and_correct_costs_almost_nothing():
    assert cross_entropy([100.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)


def test_confident_and_wrong_does_not_crash_on_log_of_zero():
    """softmax загоняет вероятность в 0.0, и наивный -log(p) падает."""
    assert cross_entropy([1000.0, 0.0], 1) == pytest.approx(1000.0)


def test_cross_entropy_agrees_with_minus_log_softmax():
    logits = [0.5, -1.2, 2.0, 0.1]
    assert cross_entropy(logits, 2) == pytest.approx(-math.log(softmax(logits)[2]))


def test_a_better_prediction_costs_less():
    assert cross_entropy([3.0, 0.0], 0) < cross_entropy([1.0, 0.0], 0)


# ----------------------------------------------------------------- rms_norm
def test_rms_norm_gives_unit_root_mean_square():
    out = rms_norm([3.0, 4.0], 0.0)
    assert math.sqrt(sum(v * v for v in out) / 2) == pytest.approx(1.0)


def test_rms_norm_is_invariant_to_input_scale():
    assert rms_norm([1.0, 2.0], 0.0) == pytest.approx(rms_norm([10.0, 20.0], 0.0))


def test_rms_norm_does_not_center_the_vector():
    """Это RMSNorm, а не LayerNorm: среднее выхода не обязано быть нулём."""
    assert sum(rms_norm([1.0, 1.0], 0.0)) == pytest.approx(2.0)


def test_rms_norm_survives_an_all_zero_vector():
    assert rms_norm([0.0, 0.0]) == pytest.approx([0.0, 0.0])


# -------------------------------------------------------------- depth_hidden
def test_depth_hidden_normalizes_each_half_separately():
    """Склейка ПОСЛЕ нормировки: масштаб эмбеддинга не должен ни на что влиять."""
    a = depth_hidden([1.0, 1.0], [2.0, 2.0], M1)
    b = depth_hidden([1.0, 1.0], [1.0, 1.0], M1)
    assert a == pytest.approx(b)


def test_normalizing_the_concatenation_instead_gives_a_different_answer():
    """Контроль к предыдущему тесту: разные по форме входы дают разный выход."""
    a = depth_hidden([1.0, 1.0], [1.0, 0.0], M1)
    b = depth_hidden([1.0, 1.0], [0.0, 1.0], M1)
    assert a != pytest.approx(b)


def test_depth_hidden_depends_on_the_next_token():
    """Тот самый causal chain: h^(1) обязан видеть t_{i+1}."""
    a = depth_hidden(H0, EMBEDDING[0], M1)
    b = depth_hidden(H0, EMBEDDING[3], M1)
    assert a != pytest.approx(b)


def test_zero_projection_kills_the_depth():
    assert depth_hidden(H0, EMBEDDING[1], [[0.0] * 4, [0.0] * 4]) == APPROX([0.0, 0.0])


def test_depth_hidden_returns_a_hidden_sized_vector():
    assert len(depth_hidden(H0, EMBEDDING[1], M1)) == len(H0)


# --------------------------------------------------------- mtp_depth_losses
def test_one_loss_per_depth():
    losses = mtp_depth_losses(H0, [0, 2, 1], [M1, M2], EMBEDDING, W_OUT)
    assert len(losses) == 3


def test_depth_one_is_the_main_model_prediction():
    """logits для k=1 берутся прямо из h^(0) — это предсказание основной модели."""
    losses = mtp_depth_losses(H0, [2, 0], [M1], EMBEDDING, W_OUT)
    assert losses[0] == pytest.approx(cross_entropy(matvec(W_OUT, H0), 2))


def test_a_change_at_depth_one_moves_the_depth_two_loss():
    """Последовательный MTP: глубина 2 обусловлена токеном глубины 1.

    У параллельного MTP (Gloeckle) обе головы смотрят на один и тот же
    h^(0), и этот тест бы не сработал. В этом вся разница конструкций.
    """
    a = mtp_depth_losses(H0, [0, 2], [M1], EMBEDDING, W_OUT)
    b = mtp_depth_losses(H0, [1, 2], [M1], EMBEDDING, W_OUT)
    assert a[1] != pytest.approx(b[1])


def test_a_change_at_depth_two_leaves_depth_one_alone():
    """Причинность в другую сторону: глубина 1 не подглядывает в будущее."""
    a = mtp_depth_losses(H0, [0, 2], [M1], EMBEDDING, W_OUT)
    b = mtp_depth_losses(H0, [0, 3], [M1], EMBEDDING, W_OUT)
    assert a[0] == pytest.approx(b[0])


def test_a_single_depth_needs_no_projection():
    losses = mtp_depth_losses(H0, [1], [], EMBEDDING, W_OUT)
    assert losses == pytest.approx([cross_entropy(matvec(W_OUT, H0), 1)])


def test_too_few_projections_is_rejected():
    with pytest.raises(ValueError):
        mtp_depth_losses(H0, [0, 1, 2], [M1], EMBEDDING, W_OUT)


def test_losses_are_never_negative():
    losses = mtp_depth_losses(H0, [3, 1, 0], [M1, M2], EMBEDDING, W_OUT)
    assert all(x >= 0 for x in losses)


def test_mtp_does_not_mutate_the_backbone_hidden_state():
    h = list(H0)
    mtp_depth_losses(h, [0, 2], [M1], EMBEDDING, W_OUT)
    assert h == H0


# --------------------------------------------------------------- joint_loss
def test_joint_loss_averages_the_depths_and_scales_by_lambda():
    assert joint_loss(2.0, [1.0, 3.0], 0.3) == APPROX(2.6)


def test_lambda_zero_leaves_only_the_main_loss():
    assert joint_loss(2.0, [10.0, 10.0], 0.0) == APPROX(2.0)


def test_no_mtp_modules_means_no_extra_loss():
    """Пустой список глубин не должен делить на ноль."""
    assert joint_loss(2.0, [], 0.3) == APPROX(2.0)


def test_joint_loss_averages_rather_than_sums_over_depth():
    """D в знаменателе: добавление глубин не раздувает вклад MTP."""
    two = joint_loss(0.0, [4.0, 4.0], 0.3)
    four = joint_loss(0.0, [4.0, 4.0, 4.0, 4.0], 0.3)
    assert two == APPROX(four)


def test_the_schedule_lowers_the_mtp_contribution():
    """0.3 на старте, 0.1 дальше — вклад MTP падает втрое."""
    early = joint_loss(1.0, [3.0], 0.3)
    late = joint_loss(1.0, [3.0], 0.1)
    assert early - 1.0 == pytest.approx(3 * (late - 1.0))


# --------------------------------------------------------- mtp_extra_params
def test_one_module_costs_about_fourteen_h_squared():
    hidden = 7168
    total = mtp_extra_params(hidden)["total"]
    assert total == pytest.approx(14 * hidden * hidden, rel=0.01)


def test_deepseek_v3_module_is_about_720m_parameters():
    assert mtp_extra_params(7168)["total"] == pytest.approx(720e6, rel=0.01)


def test_shared_embedding_and_head_cost_nothing():
    """Общие таблица и голова — буквально те же тензоры, не копии."""
    assert mtp_extra_params(4096, depths=3)["shared"] == 0


def test_components_add_up_to_one_module():
    counts = mtp_extra_params(4096)
    assert counts["projection"] + counts["attention"] + counts["mlp"] == counts["per_module"]


def test_the_projection_is_h_by_two_h():
    """M_k складывает склейку длиной 2h обратно в h."""
    assert mtp_extra_params(4096)["projection"] == 2 * 4096 * 4096


def test_extra_params_scale_with_the_number_of_depths():
    one = mtp_extra_params(4096, depths=1)["total"]
    assert mtp_extra_params(4096, depths=3)["total"] == 3 * one


def test_mtp_overhead_is_a_small_fraction_of_a_70b_model():
    """Порядок величины из урока: несколько процентов, а не «ещё одна модель»."""
    overhead = mtp_extra_params(8192)["total"]
    assert overhead / 70e9 < 0.05
