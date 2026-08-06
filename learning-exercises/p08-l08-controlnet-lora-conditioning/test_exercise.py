"""Тесты к уроку «ControlNet, LoRA и обусловливание». Правь exercise.py."""

import random

import pytest

from exercise import (
    apply_controls,
    lora_delta,
    lora_forward,
    lora_grads,
    lora_param_count,
    matrix_rank,
    matvec,
    merge_lora,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [v for row in M for v in row]


def random_matrix(rows, cols, rng, scale=1.0):
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def sq_loss(W, A, B, x, target, alpha=1.0):
    pred = lora_forward(W, A, B, x, alpha)
    return sum((p - t) ** 2 for p, t in zip(pred, target))


# ------------------------------------------------------------------- matvec
def test_matvec_multiplies_rows_by_the_vector():
    assert matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]) == pytest.approx([3.0, 7.0])


def test_identity_matrix_leaves_the_vector_alone():
    assert matvec([[1.0, 0.0], [0.0, 1.0]], [5.0, -2.0]) == pytest.approx([5.0, -2.0])


def test_matvec_result_length_follows_the_row_count():
    """Прямоугольная матрица 3x2 превращает 2-мерный вход в 3-мерный выход."""
    assert len(matvec([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])) == 3


# --------------------------------------------------------------- lora_delta
def test_lora_delta_of_rank_one_is_an_outer_product():
    got = lora_delta([[1.0, 2.0]], [[3.0], [4.0]])
    assert flat(got) == pytest.approx(flat([[3.0, 6.0], [4.0, 8.0]]))


def test_zero_initialised_B_gives_exactly_a_zero_delta():
    """Главная гарантия LoRA: до обучения адаптер — тождественный ноль."""
    got = lora_delta([[1.0, 2.0, -3.0]], [[0.0], [0.0]])
    assert flat(got) == pytest.approx([0.0] * 6)


def test_alpha_scales_the_delta_linearly():
    one = lora_delta([[1.0, 2.0]], [[3.0], [4.0]], 1.0)
    half = lora_delta([[1.0, 2.0]], [[3.0], [4.0]], 0.5)
    assert flat(half) == pytest.approx([0.5 * v for v in flat(one)])


def test_delta_has_the_shape_of_the_frozen_weight():
    """B: d_out x r, A: r x d_in — значит дельта d_out x d_in, как у W."""
    rng = random.Random(0)
    A = random_matrix(3, 5, rng)
    B = random_matrix(7, 3, rng)
    delta = lora_delta(A, B)
    assert len(delta) == 7
    assert all(len(row) == 5 for row in delta)


# ------------------------------------------------------------- lora_forward
def test_lora_forward_with_zero_B_equals_the_frozen_layer():
    """Инициализация B нулями обязана давать РОВНО исходный слой."""
    rng = random.Random(1)
    W = random_matrix(4, 4, rng)
    A = random_matrix(2, 4, rng)
    B = [[0.0, 0.0] for _ in range(4)]
    x = [rng.gauss(0, 1) for _ in range(4)]
    assert lora_forward(W, A, B, x) == pytest.approx(matvec(W, x))


def test_alpha_zero_disables_the_adapter():
    rng = random.Random(2)
    W = random_matrix(4, 4, rng)
    A = random_matrix(2, 4, rng)
    B = random_matrix(4, 2, rng)
    x = [rng.gauss(0, 1) for _ in range(4)]
    assert lora_forward(W, A, B, x, alpha=0.0) == pytest.approx(matvec(W, x))


def test_lora_forward_adds_the_low_rank_correction():
    W = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0, 0.0]]
    B = [[1.0], [0.0]]
    # поправка: B @ A @ [2, 3] = [2, 0]
    assert lora_forward(W, A, B, [2.0, 3.0]) == pytest.approx([4.0, 3.0])


def test_a_trained_adapter_actually_changes_the_output():
    """Обратная сторона проверки с нулевым B: ненулевой адаптер обязан влиять."""
    rng = random.Random(3)
    W = random_matrix(4, 4, rng)
    A = random_matrix(1, 4, rng)
    B = random_matrix(4, 1, rng)
    x = [rng.gauss(0, 1) for _ in range(4)]
    assert lora_forward(W, A, B, x) != pytest.approx(matvec(W, x))


# --------------------------------------------------------------- merge_lora
def test_merged_weight_reproduces_the_adapter_forward_pass():
    """Вплавленный вариант и рантайм-вариант обязаны считать одно и то же."""
    rng = random.Random(4)
    W = random_matrix(5, 5, rng)
    A = random_matrix(2, 5, rng)
    B = random_matrix(5, 2, rng)
    x = [rng.gauss(0, 1) for _ in range(5)]
    merged = merge_lora(W, A, B, 0.8)
    assert matvec(merged, x) == pytest.approx(lora_forward(W, A, B, x, 0.8))


def test_merging_a_zero_adapter_returns_the_frozen_weight():
    W = [[1.0, 2.0], [3.0, 4.0]]
    merged = merge_lora(W, [[1.0, 1.0]], [[0.0], [0.0]])
    assert flat(merged) == pytest.approx(flat(W))


def test_merging_does_not_mutate_the_frozen_weight():
    """База заморожена — испортить её на месте нельзя."""
    W = [[1.0, 2.0], [3.0, 4.0]]
    merge_lora(W, [[1.0, 1.0]], [[1.0], [1.0]])
    assert flat(W) == pytest.approx([1.0, 2.0, 3.0, 4.0])


# -------------------------------------------------------------- matrix_rank
def test_rank_of_the_identity_is_full():
    assert matrix_rank([[1.0, 0.0], [0.0, 1.0]]) == 2


def test_rank_of_a_duplicated_row_is_one():
    assert matrix_rank([[1.0, 2.0], [2.0, 4.0]]) == 1


def test_rank_of_a_zero_matrix_is_zero():
    assert matrix_rank([[0.0, 0.0], [0.0, 0.0]]) == 0


def test_lora_delta_rank_never_exceeds_r():
    """То самое «low-rank» из названия: дельта в 8x8 живёт в 2 измерениях."""
    rng = random.Random(5)
    A = random_matrix(2, 8, rng)
    B = random_matrix(8, 2, rng)
    assert matrix_rank(lora_delta(A, B), tol=1e-8) == 2


def test_a_full_rank_weight_is_not_reachable_by_a_rank_one_adapter():
    """Ранг 1 не дотянется до полноценной матрицы — отсюда и выбор r в 4-16."""
    rng = random.Random(6)
    A = random_matrix(1, 4, rng)
    B = random_matrix(4, 1, rng)
    assert matrix_rank(lora_delta(A, B), tol=1e-8) == 1


# --------------------------------------------------------- lora_param_count
def test_sdxl_attention_adapter_is_twenty_times_smaller():
    full, lora = lora_param_count(640, 640, 16)
    assert (full, lora) == (409600, 20480)
    assert full / lora == pytest.approx(20.0)


def test_param_count_grows_linearly_in_rank():
    _, r4 = lora_param_count(640, 640, 4)
    _, r8 = lora_param_count(640, 640, 8)
    assert r8 == 2 * r4


def test_a_full_rank_adapter_costs_more_than_the_layer_itself():
    """При r, сравнимом с d, LoRA теряет смысл — это надо увидеть числом."""
    full, lora = lora_param_count(4, 4, 4)
    assert lora > full


# --------------------------------------------------------------- lora_grads
def test_gradient_for_B_matches_the_numeric_gradient():
    rng = random.Random(7)
    W = random_matrix(3, 3, rng)
    A = random_matrix(2, 3, rng)
    B = random_matrix(3, 2, rng)
    x = [rng.gauss(0, 1) for _ in range(3)]
    target = [rng.gauss(0, 1) for _ in range(3)]
    _, grad_B = lora_grads(W, A, B, x, target)

    h = 1e-6
    for i in range(3):
        for k in range(2):
            up = [row[:] for row in B]
            down = [row[:] for row in B]
            up[i][k] += h
            down[i][k] -= h
            numeric = (sq_loss(W, A, up, x, target) - sq_loss(W, A, down, x, target)) / (2 * h)
            assert grad_B[i][k] == pytest.approx(numeric, abs=1e-4)


def test_gradient_for_A_matches_the_numeric_gradient():
    rng = random.Random(8)
    W = random_matrix(3, 3, rng)
    A = random_matrix(2, 3, rng)
    B = random_matrix(3, 2, rng)
    x = [rng.gauss(0, 1) for _ in range(3)]
    target = [rng.gauss(0, 1) for _ in range(3)]
    grad_A, _ = lora_grads(W, A, B, x, target)

    h = 1e-6
    for k in range(2):
        for j in range(3):
            up = [row[:] for row in A]
            down = [row[:] for row in A]
            up[k][j] += h
            down[k][j] -= h
            numeric = (sq_loss(W, up, B, x, target) - sq_loss(W, down, B, x, target)) / (2 * h)
            assert grad_A[k][j] == pytest.approx(numeric, abs=1e-4)


def test_gradient_is_zero_at_a_perfect_fit():
    W = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0, 1.0]]
    B = [[0.0], [0.0]]
    x = [2.0, 3.0]
    grad_A, grad_B = lora_grads(W, A, B, x, matvec(W, x))
    assert flat(grad_A) + flat(grad_B) == pytest.approx([0.0] * 4)


def test_gradient_for_A_vanishes_while_B_is_still_zero():
    """Поэтому нулями инициализируют только B: если занулить и A, и B,
    адаптер навсегда останется нулевым."""
    rng = random.Random(9)
    W = random_matrix(3, 3, rng)
    A = random_matrix(2, 3, rng)
    B = [[0.0, 0.0] for _ in range(3)]
    x = [rng.gauss(0, 1) for _ in range(3)]
    target = [rng.gauss(0, 1) for _ in range(3)]
    grad_A, grad_B = lora_grads(W, A, B, x, target)
    assert flat(grad_A) == pytest.approx([0.0] * 6)
    assert flat(grad_B) != pytest.approx([0.0] * 6)


def test_a_gradient_step_reduces_the_loss():
    rng = random.Random(10)
    W = random_matrix(3, 3, rng)
    A = random_matrix(2, 3, rng)
    B = [[0.0, 0.0] for _ in range(3)]
    x = [rng.gauss(0, 1) for _ in range(3)]
    target = [rng.gauss(0, 1) for _ in range(3)]
    before = sq_loss(W, A, B, x, target)
    grad_A, grad_B = lora_grads(W, A, B, x, target)
    lr = 0.01
    A2 = [[a - lr * g for a, g in zip(ra, rg)] for ra, rg in zip(A, grad_A)]
    B2 = [[b - lr * g for b, g in zip(rb, rg)] for rb, rg in zip(B, grad_B)]
    assert sq_loss(W, A2, B2, x, target) < before


# ------------------------------------------------------------ apply_controls
def test_zero_gates_leave_the_base_untouched():
    """Наш zero-convolution: до обучения ControlNet — ровно no-op."""
    assert apply_controls([1.0, 2.0], [[10.0, 10.0], [-9.0, 7.0]], [0.0, 0.0]) == pytest.approx(
        [1.0, 2.0]
    )


def test_gate_scales_the_side_signal():
    assert apply_controls([1.0, 2.0], [[10.0, 10.0]], [0.5]) == pytest.approx([6.0, 7.0])


def test_two_controls_add_up():
    """Pose 1.0 + Depth 1.0 складываются буквально — вот откуда перебор."""
    both = apply_controls([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    assert both == pytest.approx([1.0, 1.0])


def test_control_order_does_not_change_the_result():
    a = apply_controls([1.0], [[2.0], [3.0]], [0.5, 0.25])
    b = apply_controls([1.0], [[3.0], [2.0]], [0.25, 0.5])
    assert a == pytest.approx(b)


def test_apply_controls_does_not_mutate_the_base():
    base = [1.0, 2.0]
    apply_controls(base, [[10.0, 10.0]], [1.0])
    assert base == pytest.approx([1.0, 2.0])
