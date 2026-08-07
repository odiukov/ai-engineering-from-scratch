"""Тесты к уроку «Fine-tuning с LoRA и QLoRA». Правь exercise.py."""

import pytest

from exercise import (
    count_trainable,
    init_lora,
    linear,
    lora_forward,
    lora_grads,
    merge_lora,
    quantize_dequantize,
    train_lora,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [v for row in M for v in row]


W = [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]      # d_out=2, d_in=3
A = [[0.3, -0.2, 0.1], [0.4, 0.5, -0.3]]      # rank=2, d_in=3
B = [[0.2, -0.4], [0.6, 0.1]]                 # d_out=2, rank=2
ALPHA = 4.0
X = [1.0, -2.0, 0.5]
TARGET = [0.7, -0.3]


def bumped(M, i, j, delta):
    """Копия матрицы с одним сдвинутым элементом — для численной производной."""
    out = [row[:] for row in M]
    out[i][j] += delta
    return out


# ----------------------------------------------------------------- linear
def test_linear_multiplies_rows_by_the_vector():
    assert linear([[1, 2], [3, 4]], [1, 1]) == APPROX([3.0, 7.0])


def test_linear_with_identity_returns_the_input():
    assert linear([[1, 0], [0, 1]], [5, 9]) == APPROX([5.0, 9.0])


def test_linear_output_length_is_the_number_of_rows():
    assert len(linear([[1, 1, 1]], [2, 3, 4])) == 1


# -------------------------------------------------------------- init_lora
def test_init_lora_shapes_follow_rank():
    a, b = init_lora(4, 3, 2, seed=0)
    assert (len(a), len(a[0])) == (2, 4)
    assert (len(b), len(b[0])) == (3, 2)


def test_init_lora_starts_b_at_zero():
    """B = 0 — не мелочь: с неё адаптер стартует, не трогая базовую модель."""
    _, b = init_lora(4, 3, 2, seed=7)
    assert flat(b) == APPROX([0.0] * 6)


def test_init_lora_a_is_not_all_zeros():
    a, _ = init_lora(8, 8, 4, seed=1)
    assert any(v != 0.0 for v in flat(a))


def test_init_lora_is_reproducible_for_the_same_seed():
    assert flat(init_lora(6, 6, 3, seed=42)[0]) == APPROX(
        flat(init_lora(6, 6, 3, seed=42)[0])
    )


def test_init_lora_differs_for_a_different_seed():
    assert flat(init_lora(6, 6, 3, seed=1)[0]) != flat(init_lora(6, 6, 3, seed=2)[0])


def test_fresh_adapter_does_not_change_the_base_model():
    """Главное свойство инициализации: сразу после init выход тот же самый."""
    a, b = init_lora(3, 2, 2, seed=3)
    assert lora_forward(W, a, b, ALPHA, X) == APPROX(linear(W, X))


# ----------------------------------------------------------- lora_forward
def test_lora_forward_worked_example():
    assert lora_forward([[1, 0], [0, 1]], [[1, 1]], [[2], [0]], 1, [1, 2]) == APPROX(
        [7.0, 2.0]
    )


def test_lora_forward_scales_the_delta_by_alpha_over_rank():
    """Удвоение alpha удваивает добавку адаптера, база остаётся на месте."""
    base = linear(W, X)
    one = lora_forward(W, A, B, 2.0, X)
    two = lora_forward(W, A, B, 4.0, X)
    for b, o, t in zip(base, one, two):
        assert (t - b) == pytest.approx(2 * (o - b), abs=1e-12)


def test_lora_forward_with_alpha_equal_to_rank_scales_by_one():
    rank = len(A)
    delta = [y - b for y, b in zip(lora_forward(W, A, B, rank, X), linear(W, X))]
    expected = linear(B, linear(A, X))
    assert delta == APPROX(expected)


# ------------------------------------------------------------- merge_lora
def test_merged_weights_give_the_same_output_as_the_adapter():
    """Слияние обязано быть точным: иначе после merge_and_unload модель другая."""
    merged = merge_lora(W, A, B, ALPHA)
    assert linear(merged, X) == pytest.approx(lora_forward(W, A, B, ALPHA, X), abs=1e-12)


def test_merge_lora_does_not_mutate_the_base_weights():
    """W держат другие адаптеры — портить её на месте нельзя."""
    original = [row[:] for row in W]
    merge_lora(W, A, B, ALPHA)
    assert flat(W) == APPROX(flat(original))


def test_merging_a_zero_adapter_changes_nothing():
    zeros_b = [[0.0, 0.0], [0.0, 0.0]]
    assert flat(merge_lora(W, A, zeros_b, ALPHA)) == APPROX(flat(W))


def test_merge_delta_of_rank_one_adapter_has_rank_one():
    """B A при rank=1 — матрица ранга 1: все её 2x2 миноры нулевые."""
    zero_w = [[0.0] * 3 for _ in range(3)]
    a1 = [[0.5, -1.0, 2.0]]
    b1 = [[1.0], [-3.0], [0.25]]
    delta = merge_lora(zero_w, a1, b1, 1.0)
    for i in range(3):
        for j in range(i + 1, 3):
            for k in range(3):
                for m in range(k + 1, 3):
                    minor = delta[i][k] * delta[j][m] - delta[i][m] * delta[j][k]
                    assert minor == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------- count_trainable
def test_count_trainable_reproduces_the_lesson_numbers():
    assert count_trainable(4096, 4096, 16) == {
        "full": 16777216,
        "lora": 131072,
        "ratio": pytest.approx(0.0078125),
    }


def test_trainable_parameters_shrink_with_the_rank():
    big = count_trainable(4096, 4096, 32)["lora"]
    small = count_trainable(4096, 4096, 8)["lora"]
    assert small < big < count_trainable(4096, 4096, 4096)["lora"]


def test_lora_stops_saving_at_half_the_dimension():
    """При rank = d/2 адаптер ровно такого же размера, что полная матрица."""
    assert count_trainable(4096, 4096, 2048)["ratio"] == pytest.approx(1.0)


def test_trainable_ratio_is_under_one_percent_at_rank_16():
    assert count_trainable(4096, 4096, 16)["ratio"] < 0.01


# ---------------------------------------------------- quantize_dequantize
def test_quantization_is_lossless_on_exact_multiples_of_the_step():
    row = [[-3.5, -1.5, 0.0, 2.0, 3.5]]
    assert flat(quantize_dequantize(row, block_size=5)) == APPROX(flat(row))


def test_quantization_keeps_the_shape():
    m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    out = quantize_dequantize(m, block_size=4)
    assert [len(r) for r in out] == [3, 3]


def test_quantization_survives_an_all_zero_block():
    """scale = 0 — деление на ноль, если не разобрать этот случай отдельно."""
    assert flat(quantize_dequantize([[0.0, 0.0, 0.0]], block_size=3)) == APPROX(
        [0.0, 0.0, 0.0]
    )


def test_largest_weight_in_a_block_survives_exactly():
    m = [[0.31, -2.5, 1.7, 0.04]]
    out = quantize_dequantize(m, block_size=4)
    assert out[0][1] == pytest.approx(-2.5, abs=1e-12)


def test_quantization_error_is_bounded_by_half_a_step():
    m = [[0.31, -2.5, 1.7, 0.04, -0.9, 2.2, -1.1, 0.5]]
    out = quantize_dequantize(m, block_size=8)
    step = 2.5 / 7
    assert max(abs(a - b) for a, b in zip(flat(m), flat(out))) <= step / 2 + 1e-12


def test_an_outlier_eats_the_precision_of_its_neighbours():
    """Почему в QLoRA блоки короткие: один большой вес губит соседей."""
    m = [[1000.0, 1.0, 1.0, 1.0]]
    coarse = quantize_dequantize(m, block_size=4)
    fine = quantize_dequantize(m, block_size=2)
    coarse_err = sum(abs(a - b) for a, b in zip(flat(m), flat(coarse)))
    fine_err = sum(abs(a - b) for a, b in zip(flat(m), flat(fine)))
    assert fine_err < coarse_err


# ------------------------------------------------------------- lora_grads
def test_lora_grads_worked_example():
    loss, ga, gb = lora_grads([[0.0]], [[1.0]], [[1.0]], 1.0, [2.0], [0.0])
    assert loss == APPROX(4.0)
    assert flat(ga) == APPROX([8.0])
    assert flat(gb) == APPROX([8.0])


def test_grad_a_matches_the_numeric_derivative():
    h = 1e-5
    _, grad_a, _ = lora_grads(W, A, B, ALPHA, X, TARGET)
    for r in range(len(A)):
        for j in range(len(A[0])):
            up = lora_grads(W, bumped(A, r, j, +h), B, ALPHA, X, TARGET)[0]
            down = lora_grads(W, bumped(A, r, j, -h), B, ALPHA, X, TARGET)[0]
            assert grad_a[r][j] == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_grad_b_matches_the_numeric_derivative():
    h = 1e-5
    _, _, grad_b = lora_grads(W, A, B, ALPHA, X, TARGET)
    for i in range(len(B)):
        for r in range(len(B[0])):
            up = lora_grads(W, A, bumped(B, i, r, +h), ALPHA, X, TARGET)[0]
            down = lora_grads(W, A, bumped(B, i, r, -h), ALPHA, X, TARGET)[0]
            assert grad_b[i][r] == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_first_step_moves_only_b_because_a_gets_no_gradient():
    """При B = 0 градиент по A нулевой: A оживает лишь после того, как B ушёл."""
    a0, b0 = init_lora(3, 2, 2, seed=5)
    _, grad_a, grad_b = lora_grads(W, a0, b0, ALPHA, X, TARGET)
    assert flat(grad_a) == APPROX([0.0] * 6)
    assert any(abs(v) > 1e-9 for v in flat(grad_b))


def test_lora_grads_does_not_touch_the_frozen_weights():
    snapshot = flat(W) + flat(A) + flat(B)
    lora_grads(W, A, B, ALPHA, X, TARGET)
    assert flat(W) + flat(A) + flat(B) == APPROX(snapshot)


def test_zero_loss_gives_zero_gradients():
    exact = lora_forward(W, A, B, ALPHA, X)
    loss, grad_a, grad_b = lora_grads(W, A, B, ALPHA, X, exact)
    assert loss == pytest.approx(0.0, abs=1e-18)
    assert flat(grad_a) + flat(grad_b) == pytest.approx([0.0] * 10, abs=1e-12)


# ------------------------------------------------------------- train_lora
SWAP_W = [[1.0, 0.0], [0.0, 1.0]]
SWAP_DATA = [([1.0, 0.0], [0.0, 1.0]), ([0.0, 1.0], [1.0, 0.0])]


def test_training_reduces_the_loss():
    _, _, losses = train_lora(SWAP_W, SWAP_DATA, rank=2, alpha=4.0, lr=0.1, epochs=200)
    assert losses[-1] < losses[0] / 10


def test_training_records_one_loss_per_epoch():
    _, _, losses = train_lora(SWAP_W, SWAP_DATA, rank=1, epochs=7)
    assert len(losses) == 7


def test_training_leaves_the_base_weights_frozen():
    """Всё выученное лежит в A и B — потому адаптер и весит 10-100 МБ."""
    weights = [[0.3, -0.7], [1.2, 0.4]]
    snapshot = flat(weights)
    train_lora(weights, SWAP_DATA, rank=2, epochs=20)
    assert flat(weights) == APPROX(snapshot)


def test_trained_adapter_fits_the_targets():
    a, b, _ = train_lora(SWAP_W, SWAP_DATA, rank=2, alpha=4.0, lr=0.1, epochs=200)
    for x, target in SWAP_DATA:
        assert lora_forward(SWAP_W, a, b, 4.0, x) == pytest.approx(target, abs=0.1)


def test_merging_the_trained_adapter_keeps_the_answers():
    a, b, _ = train_lora(SWAP_W, SWAP_DATA, rank=2, alpha=4.0, lr=0.1, epochs=100)
    merged = merge_lora(SWAP_W, a, b, 4.0)
    for x, _ in SWAP_DATA:
        assert linear(merged, x) == pytest.approx(
            lora_forward(SWAP_W, a, b, 4.0, x), abs=1e-12
        )


def test_training_is_reproducible_for_the_same_seed():
    first = train_lora(SWAP_W, SWAP_DATA, rank=2, epochs=15, seed=11)[2]
    second = train_lora(SWAP_W, SWAP_DATA, rank=2, epochs=15, seed=11)[2]
    assert first == APPROX(second)


def test_higher_rank_is_not_worse_on_the_same_budget():
    """r=1 не хватает выразительности на эту задачу, r=2 хватает."""
    low = train_lora(SWAP_W, SWAP_DATA, rank=1, alpha=2.0, lr=0.1, epochs=200)[2]
    high = train_lora(SWAP_W, SWAP_DATA, rank=2, alpha=4.0, lr=0.1, epochs=200)[2]
    assert high[-1] <= low[-1] + 1e-9
