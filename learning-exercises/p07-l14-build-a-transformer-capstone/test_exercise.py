"""Тесты к уроку «Собираем трансформер с нуля — капстоун». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    cross_entropy_next_token,
    gpt_forward,
    init_params,
    linear,
    multi_head_attention,
    rms_norm,
    softmax,
    swiglu_ffn,
    transformer_block,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не сравнивает вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def toy_sequence(n, d, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]


def toy_block(d, n_heads, seed):
    """Полный набор весов одного блока, детерминированный по seed."""
    rng = random.Random(seed)

    def mat(rows, cols):
        return [[rng.gauss(0, 0.3) for _ in range(cols)] for _ in range(rows)]

    return {
        "n_heads": n_heads,
        "norm1": [1.0] * d,
        "wq": mat(d, d), "wk": mat(d, d), "wv": mat(d, d), "wo": mat(d, d),
        "norm2": [1.0] * d,
        "w1": mat(2 * d, d), "w3": mat(2 * d, d), "w2": mat(d, 2 * d),
    }


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([3.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_softmax_is_shift_invariant():
    assert softmax([1.0, 2.0, 3.0]) == pytest.approx(softmax([101.0, 102.0, 103.0]))


def test_softmax_survives_huge_logits():
    """math.exp(1000) — это OverflowError. Сдвиг на максимум обязателен."""
    assert sum(softmax([1000.0, 999.0, 1.0])) == APPROX(1.0)


# ----------------------------------------------------------------- linear
def test_linear_with_identity_returns_the_input():
    assert linear([1.0, 2.0, 3.0], eye(3)) == pytest.approx([1.0, 2.0, 3.0])


def test_linear_output_length_is_the_number_of_rows():
    """Строка W — один выходной нейрон. Матрица может быть неквадратной."""
    assert len(linear([1.0, 2.0], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])) == 3


def test_linear_reduces_a_vector_to_its_sum():
    assert linear([1.0, 2.0, 4.0], [[1.0, 1.0, 1.0]]) == pytest.approx([7.0])


def test_linear_is_additive():
    """f(a + b) == f(a) + f(b): без bias слой строго линеен."""
    W = [[0.5, -1.0], [2.0, 0.25]]
    a, b = [1.0, 2.0], [-3.0, 0.5]
    both = linear([x + y for x, y in zip(a, b)], W)
    apart = [x + y for x, y in zip(linear(a, W), linear(b, W))]
    assert both == pytest.approx(apart)


# --------------------------------------------------------------- rms_norm
def test_rms_norm_makes_the_root_mean_square_one():
    out = rms_norm([3.0, 4.0, 12.0, 0.0], [1.0] * 4)
    assert math.sqrt(sum(x * x for x in out) / 4) == pytest.approx(1.0, abs=1e-6)


def test_rms_norm_does_not_subtract_the_mean():
    """Это не LayerNorm: [1, 1, 1] остаётся единицами, а не превращается в нули."""
    assert rms_norm([1.0, 1.0, 1.0], [1.0] * 3) == pytest.approx([1.0] * 3, abs=1e-6)


def test_rms_norm_is_scale_invariant():
    """Умножили вход на 100 — выход тот же. Норма и существует ради этого."""
    x = [0.3, -1.2, 5.0]
    a = rms_norm(x, [1.0] * 3)
    b = rms_norm([100 * xi for xi in x], [1.0] * 3)
    assert a == pytest.approx(b, abs=1e-6)


def test_rms_norm_weight_rescales_each_channel():
    assert rms_norm([2.0, 2.0], [3.0, 5.0]) == pytest.approx([3.0, 5.0], abs=1e-6)


def test_rms_norm_survives_the_zero_vector():
    """Без eps здесь деление на нуль, а нулевые активации реально бывают."""
    assert rms_norm([0.0, 0.0], [1.0, 1.0]) == pytest.approx([0.0, 0.0])


# ------------------------------------------------------------- swiglu_ffn
def test_swiglu_with_zero_value_branch_outputs_zero():
    """silu(0) = 0, значит всё произведение — ноль, каким бы ни был gate."""
    assert swiglu_ffn([1.0, 2.0], zeros(4, 2), [[1.0, 1.0]] * 4, zeros(2, 4)) == pytest.approx(
        [0.0, 0.0]
    )


def test_swiglu_output_length_matches_the_last_matrix():
    out = swiglu_ffn([1.0, 2.0, 3.0], [[0.1] * 3] * 6, [[0.2] * 3] * 6, [[0.3] * 6] * 3)
    assert len(out) == 3


def test_swiglu_is_not_linear():
    """Гейт умножает два линейных выхода — значит функция квадратичная, не линейная."""
    W1, W3, W2 = [[1.0]], [[1.0]], [[1.0]]
    single = swiglu_ffn([1.0], W1, W3, W2)[0]
    double = swiglu_ffn([2.0], W1, W3, W2)[0]
    assert double != pytest.approx(2 * single, abs=1e-6)


def test_swiglu_gate_can_shut_the_block_down():
    """Нулевые ворота (W3 = 0) глушат блок целиком, каким бы ни был W1."""
    assert swiglu_ffn([1.0, 2.0], [[1.0, 1.0]] * 4, zeros(4, 2), [[1.0] * 4] * 2) == pytest.approx(
        [0.0, 0.0]
    )


def test_swiglu_survives_large_negative_preactivations():
    """silu через exp(-t) при t = -1000 даёт OverflowError. Нужна устойчивая сигмоида."""
    out = swiglu_ffn([-1000.0], [[1.0]], [[1.0]], [[1.0]])
    assert out[0] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------- multi_head_attention
def test_attention_keeps_the_sequence_shape():
    X = toy_sequence(5, 4, seed=1)
    out = multi_head_attention(X, eye(4), eye(4), eye(4), eye(4), 2)
    assert len(out) == 5
    assert all(len(row) == 4 for row in out)


def test_first_position_can_only_attend_to_itself():
    """Один доступный ключ -> softmax даёт вес 1.0 -> выход это чистые Wo @ Wv @ x."""
    X = toy_sequence(6, 4, seed=2)
    Wv = [[0.5, 0.1, -0.2, 0.3], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.5, 0.0], [0.2, 0.2, 0.2, 0.2]]
    Wo = [[0.3, 0.0, 0.0, 1.0], [1.0, -0.5, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.1, 0.1, 0.1, 0.1]]
    out = multi_head_attention(X, eye(4), eye(4), Wv, Wo, 2)
    assert out[0] == pytest.approx(linear(linear(X[0], Wv), Wo))


def test_attention_is_causal():
    """Главный тест урока: подмена токена не трогает выходы предыдущих позиций."""
    X = toy_sequence(6, 4, seed=3)
    Y = [row[:] for row in X]
    Y[4] = [9.0, -9.0, 3.0, 1.0]
    a = multi_head_attention(X, eye(4), eye(4), eye(4), eye(4), 2)
    b = multi_head_attention(Y, eye(4), eye(4), eye(4), eye(4), 2)
    assert flat(a[:4]) == pytest.approx(flat(b[:4]), abs=1e-12)


def test_attention_does_react_to_the_past():
    """Обратная половина: прошлое влиять обязано, иначе тест причинности пустой."""
    X = toy_sequence(6, 4, seed=4)
    Y = [row[:] for row in X]
    Y[1] = [9.0, -9.0, 3.0, 1.0]
    a = multi_head_attention(X, eye(4), eye(4), eye(4), eye(4), 2)
    b = multi_head_attention(Y, eye(4), eye(4), eye(4), eye(4), 2)
    assert max(abs(x - y) for x, y in zip(a[5], b[5])) > 1e-3


def test_zero_output_projection_kills_attention_entirely():
    X = toy_sequence(4, 4, seed=5)
    out = multi_head_attention(X, eye(4), eye(4), eye(4), zeros(4, 4), 2)
    assert flat(out) == pytest.approx([0.0] * 16)


def test_head_count_changes_the_result():
    """Головы режут пространство на подпространства — одна голова это другая модель."""
    X = toy_sequence(6, 4, seed=6)
    W = [[0.7, -0.3, 0.2, 0.1], [0.1, 0.9, -0.4, 0.0], [0.3, 0.2, 0.8, -0.1], [-0.2, 0.4, 0.1, 0.6]]
    one = multi_head_attention(X, W, W, eye(4), eye(4), 1)
    two = multi_head_attention(X, W, W, eye(4), eye(4), 2)
    assert max(abs(a - b) for a, b in zip(flat(one), flat(two))) > 1e-6


def test_attention_with_identical_keys_averages_the_values():
    """Все ключи одинаковы -> веса равномерные -> последняя позиция видит среднее V."""
    X = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    Wv = [[1.0, 0.0], [0.0, 1.0]]
    out = multi_head_attention(X, zeros(2, 2), zeros(2, 2), Wv, eye(2), 1)
    assert out[2] == pytest.approx([1.0, 0.0])


# --------------------------------------------------------- transformer_block
def test_block_with_zero_residual_branches_is_the_identity():
    """wo = 0 и w2 = 0 -> оба residual-а прозрачны. Так проверяются сложения."""
    d = 4
    block = toy_block(d, 2, seed=7)
    block["wo"] = zeros(d, d)
    block["w2"] = zeros(d, 2 * d)
    X = toy_sequence(5, d, seed=8)
    assert flat(transformer_block(X, block)) == pytest.approx(flat(X), abs=1e-12)


def test_block_keeps_the_sequence_shape():
    X = toy_sequence(5, 4, seed=9)
    out = transformer_block(X, toy_block(4, 2, seed=10))
    assert len(out) == 5
    assert all(len(row) == 4 for row in out)


def test_block_actually_changes_the_input():
    X = toy_sequence(5, 4, seed=11)
    out = transformer_block(X, toy_block(4, 2, seed=12))
    assert max(abs(a - b) for a, b in zip(flat(X), flat(out))) > 1e-3


def test_block_is_causal():
    X = toy_sequence(6, 4, seed=13)
    Y = [row[:] for row in X]
    Y[5] = [4.0, 4.0, -4.0, 0.5]
    block = toy_block(4, 2, seed=14)
    a = transformer_block(X, block)
    b = transformer_block(Y, block)
    assert flat(a[:5]) == pytest.approx(flat(b[:5]), abs=1e-12)


def test_block_survives_huge_activations_thanks_to_the_norm():
    """Без нормы SwiGLU на входе 1e6 выдал бы 1e12 и утащил бы residual за собой."""
    X = [[1e6, -1e6, 1e6, -1e6] for _ in range(3)]
    out = transformer_block(X, toy_block(4, 2, seed=15))
    assert all(abs(v) < 1e7 for v in flat(out))


# ------------------------------------------------------------- init_params
def test_init_params_shapes_match_the_config():
    p = init_params(65, 8, 2, 3, 16, random.Random(0))
    assert len(p["tok_emb"]) == 65 and len(p["tok_emb"][0]) == 8
    assert len(p["pos_emb"]) == 16 and len(p["pos_emb"][0]) == 8
    assert len(p["blocks"]) == 3
    assert len(p["norm_f"]) == 8


def test_init_params_block_shapes_match_the_config():
    b = init_params(65, 8, 2, 1, 16, random.Random(0))["blocks"][0]
    assert b["n_heads"] == 2
    assert len(b["wq"]) == len(b["wq"][0]) == 8
    assert len(b["w1"]) == 16 and len(b["w1"][0]) == 8
    assert len(b["w2"]) == 8 and len(b["w2"][0]) == 16


def test_norm_weights_start_at_one():
    """Норма на старте — тождество; иначе первый forward уже искажает сигнал."""
    p = init_params(20, 8, 2, 2, 16, random.Random(0))
    assert p["norm_f"] == [1.0] * 8
    assert p["blocks"][0]["norm1"] == [1.0] * 8
    assert p["blocks"][0]["norm2"] == [1.0] * 8


def test_init_params_is_reproducible_for_the_same_seed():
    a = init_params(20, 8, 2, 2, 16, random.Random(3))
    b = init_params(20, 8, 2, 2, 16, random.Random(3))
    assert flat(a["tok_emb"]) == pytest.approx(flat(b["tok_emb"]))
    assert flat(a["blocks"][1]["wq"]) == pytest.approx(flat(b["blocks"][1]["wq"]))


def test_init_scale_controls_the_weight_magnitude():
    small = init_params(50, 16, 2, 1, 16, random.Random(4), scale=0.01)
    big = init_params(50, 16, 2, 1, 16, random.Random(4), scale=1.0)
    mean_small = sum(abs(v) for v in flat(small["tok_emb"])) / (50 * 16)
    mean_big = sum(abs(v) for v in flat(big["tok_emb"])) / (50 * 16)
    assert mean_big / mean_small == pytest.approx(100.0, rel=1e-6)


# ------------------------------------------------------------- gpt_forward
def test_forward_returns_one_logit_row_per_token():
    p = init_params(65, 16, 4, 2, 32, random.Random(5))
    logits = gpt_forward([1, 2, 3, 4, 5], p)
    assert len(logits) == 5
    assert all(len(row) == 65 for row in logits)


def test_lm_head_is_tied_to_the_token_embedding():
    """Отдельной матрицы lm_head нет: тронул строку tok_emb — поехал ровно её логит."""
    p = init_params(30, 8, 2, 1, 16, random.Random(6))
    before = gpt_forward([3, 7], p)
    assert len(before[0]) == len(p["tok_emb"])
    # токен 29 во входе не встречается, значит меняется только выходная проекция
    p["tok_emb"][29] = [v + 1.0 for v in p["tok_emb"][29]]
    after = gpt_forward([3, 7], p)
    assert abs(after[0][29] - before[0][29]) > 1e-6
    assert [after[0][j] for j in range(29)] == pytest.approx(
        [before[0][j] for j in range(29)], abs=1e-12
    )


def test_forward_is_causal_to_the_logits():
    """Подмена последнего токена не двигает логиты предыдущих позиций."""
    p = init_params(30, 8, 2, 2, 16, random.Random(7))
    a = gpt_forward([1, 2, 3, 4], p)
    b = gpt_forward([1, 2, 3, 29], p)
    assert flat(a[:3]) == pytest.approx(flat(b[:3]), abs=1e-12)


def test_forward_uses_position_not_just_token_identity():
    """Один и тот же токен на разных позициях даёт разные логиты — иначе это bag-of-words."""
    p = init_params(30, 8, 2, 1, 16, random.Random(8))
    logits = gpt_forward([5, 5, 5], p)
    assert max(abs(a - b) for a, b in zip(logits[0], logits[2])) > 1e-6


def test_forward_reacts_to_the_prefix():
    """Логиты последней позиции обязаны зависеть от того, что было раньше."""
    p = init_params(30, 8, 2, 2, 16, random.Random(9))
    a = gpt_forward([1, 2, 3], p)
    b = gpt_forward([1, 20, 3], p)
    assert max(abs(x - y) for x, y in zip(a[2], b[2])) > 1e-6


def test_every_block_in_the_stack_is_applied():
    """Те же эмбеддинги, второй блок отрезан — ответ обязан измениться."""
    p = init_params(30, 8, 2, 2, 16, random.Random(10))
    shallow = dict(p, blocks=p["blocks"][:1])
    one = gpt_forward([1, 2, 3], shallow)
    two = gpt_forward([1, 2, 3], p)
    assert max(abs(a - b) for a, b in zip(one[2], two[2])) > 1e-6


# ------------------------------------------------- cross_entropy_next_token
def test_cross_entropy_of_uniform_logits_is_log_vocab():
    logits = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
    assert cross_entropy_next_token(logits, [0, 1, 2]) == pytest.approx(math.log(4))


def test_cross_entropy_of_a_confident_correct_prediction_is_zero():
    assert cross_entropy_next_token([[0.0, 50.0], [0.0, 0.0]], [0, 1]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_cross_entropy_uses_the_shift_by_one_target():
    """Логиты позиции i отвечают за токен i+1. Сдвинешь не туда — числа поедут."""
    logits = [[0.0, 10.0], [10.0, 0.0]]
    # позиция 0 предсказывает tokens[1] = 1, и делает это уверенно верно
    assert cross_entropy_next_token(logits, [0, 1]) == pytest.approx(
        math.log(1 + math.exp(-10.0))
    )


def test_last_position_has_no_target_and_is_excluded():
    """Средним делим на len(tokens) - 1: у последней позиции цели нет."""
    good = [[0.0, 50.0], [0.0, 0.0], [999.0, -999.0]]
    assert cross_entropy_next_token(good, [0, 1, 0]) == pytest.approx(
        math.log(2) / 2, abs=1e-6
    )


def test_cross_entropy_survives_huge_logits():
    """Через softmax потом log здесь получился бы log(0). Нужен log-sum-exp."""
    value = cross_entropy_next_token([[1000.0, 0.0], [0.0, 0.0]], [0, 1])
    assert value == pytest.approx(1000.0, abs=1e-6)


def test_fresh_model_loss_is_log_of_vocab_size():
    """Классическая проверка nanoGPT: до обучения loss ровно ln(vocab_size)."""
    vocab = 65
    p = init_params(vocab, 32, 4, 3, 32, random.Random(11))
    tokens = [i % vocab for i in range(16)]
    loss = cross_entropy_next_token(gpt_forward(tokens, p), tokens)
    assert loss == pytest.approx(math.log(vocab), abs=0.1)
