"""Тесты к уроку «Полный трансформер». Правь exercise.py."""

import math

import pytest

from exercise import (
    block_params,
    ffn_swiglu,
    layer_norm,
    post_norm_sublayer,
    pre_norm_sublayer,
    rms_norm,
    silu,
    transformer_block,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


def norm_of(rows):
    return math.sqrt(sum(v * v for v in flat(rows)))


def zero_sublayer(rows):
    """Подслой, который ничего не добавляет к резидуальному потоку."""
    return [[0.0] * len(row) for row in rows]


def triple_sublayer(rows):
    """Подслой, усиливающий вход втрое — чтобы поймать рост активаций."""
    return [[3.0 * v for v in row] for row in rows]


# -------------------------------------------------------------- layer_norm
def test_layer_norm_centres_and_scales():
    assert layer_norm([1.0, 2.0, 3.0]) == ROUGH([-1.2247449, 0.0, 1.2247449])


def test_layer_norm_output_has_zero_mean_and_unit_deviation():
    out = layer_norm([4.0, -1.0, 7.0, 0.5])
    assert sum(out) / len(out) == pytest.approx(0.0, abs=1e-9)
    variance = sum(v * v for v in out) / len(out)
    assert variance == pytest.approx(1.0, abs=1e-4)


def test_layer_norm_ignores_a_constant_shift():
    """LayerNorm стирает сдвиг — вычитание среднего для этого и стоит."""
    assert layer_norm([1.0, 2.0, 3.0]) == ROUGH(layer_norm([101.0, 102.0, 103.0]))


def test_layer_norm_ignores_a_positive_rescale():
    assert layer_norm([1.0, 2.0, 3.0]) == ROUGH(layer_norm([10.0, 20.0, 30.0]))


def test_layer_norm_of_a_constant_vector_is_zeros():
    """Дисперсия нулевая, спасает только eps под корнем."""
    assert layer_norm([5.0, 5.0, 5.0]) == ROUGH([0.0, 0.0, 0.0])


def test_layer_norm_rejects_an_empty_vector():
    with pytest.raises(ValueError):
        layer_norm([])


# ---------------------------------------------------------------- rms_norm
def test_rms_norm_divides_by_the_root_mean_square():
    assert rms_norm([3.0, 4.0]) == ROUGH([3.0 / 3.5355339, 4.0 / 3.5355339])


def test_rms_norm_output_has_unit_root_mean_square():
    out = rms_norm([4.0, -1.0, 7.0, 0.5])
    assert math.sqrt(sum(v * v for v in out) / len(out)) == pytest.approx(1.0, abs=1e-6)


def test_rms_norm_equals_layer_norm_on_a_zero_mean_vector():
    """Вычитать было нечего — значит и разницы нет."""
    x = [-3.0, 1.0, 2.0]
    assert rms_norm(x) == ROUGH(layer_norm(x))


def test_rms_norm_does_not_ignore_a_constant_shift():
    """Вот и вся разница с LayerNorm: среднее не вычитается."""
    x = [1.0, 2.0, 3.0]
    shifted = [v + 100.0 for v in x]
    assert rms_norm(shifted) != ROUGH(rms_norm(x))
    assert layer_norm(shifted) == ROUGH(layer_norm(x))


def test_rms_norm_rejects_an_empty_vector():
    with pytest.raises(ValueError):
        rms_norm([])


# -------------------------------------------------------------------- silu
def test_silu_at_zero_is_zero():
    assert silu(0.0) == APPROX(0.0)


def test_silu_approaches_identity_for_large_positive_input():
    assert silu(10.0) == pytest.approx(10.0, abs=1e-3)


def test_silu_dips_below_zero_where_relu_is_flat():
    """Отрицательный провал — причина, по которой SiLU обошла ReLU."""
    assert -0.3 < silu(-2.0) < 0.0


def test_silu_survives_a_large_negative_input():
    """Наивная 1/(1+exp(-x)) на x = -1000 кидает OverflowError."""
    assert silu(-1000.0) == APPROX(0.0)


def test_silu_is_monotone_on_the_positive_side():
    values = [silu(x) for x in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


# -------------------------------------------------------------- ffn_swiglu
def test_ffn_swiglu_returns_a_vector_of_model_width():
    x = [1.0, 2.0]
    W1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    W3 = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
    W2 = [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    assert len(ffn_swiglu(x, W1, W2, W3)) == 2


def test_closing_the_gate_zeroes_the_output():
    """W3 = 0 — «пропускать нечего», при любых W1 и W2 выход нулевой."""
    x = [1.0, 2.0]
    W1 = [[3.0, 3.0], [3.0, 3.0]]
    W3 = [[0.0, 0.0], [0.0, 0.0]]
    W2 = [[1.0, 1.0], [1.0, 1.0]]
    assert ffn_swiglu(x, W1, W2, W3) == APPROX([0.0, 0.0])


def test_a_zero_gate_projection_also_zeroes_the_output():
    """silu(0) = 0, значит W1 = 0 закрывает гейт так же наглухо."""
    x = [1.0, 2.0]
    W1 = [[0.0, 0.0], [0.0, 0.0]]
    W3 = [[3.0, 3.0], [3.0, 3.0]]
    W2 = [[1.0, 1.0], [1.0, 1.0]]
    assert ffn_swiglu(x, W1, W2, W3) == APPROX([0.0, 0.0])


def test_ffn_swiglu_multiplies_the_two_branches_elementwise():
    """Один скрытый нейрон, ручной счёт: silu(2) * 3, потом на 1."""
    x = [1.0]
    W1 = [[2.0]]
    W3 = [[3.0]]
    W2 = [[1.0]]
    assert ffn_swiglu(x, W1, W2, W3) == APPROX([silu(2.0) * 3.0])


def test_the_two_branches_are_not_interchangeable():
    """W1 идёт через silu, W3 — нет. Поменяй местами, и ответ другой."""
    x = [1.0]
    W1, W3, W2 = [[2.0]], [[5.0]], [[1.0]]
    assert ffn_swiglu(x, W1, W2, W3) != APPROX(ffn_swiglu(x, W3, W2, W1))


# ------------------------------------------------------- pre_norm_sublayer
def test_pre_norm_with_a_silent_sublayer_passes_the_input_through():
    """Резидуальный поток не тронут — на этом держится обучение глубоких стеков."""
    rows = [[1.0, 2.0], [3.0, 4.0]]
    assert flat(pre_norm_sublayer(rows, zero_sublayer)) == APPROX(flat(rows))


def test_pre_norm_normalizes_what_the_sublayer_sees():
    seen = []

    def spy(rows):
        seen.append([row[:] for row in rows])
        return zero_sublayer(rows)

    pre_norm_sublayer([[10.0, 20.0, 30.0]], spy)
    assert sum(seen[0][0]) == pytest.approx(0.0, abs=1e-9)


def test_pre_norm_adds_the_sublayer_output_to_the_raw_input():
    rows = [[1.0, 1.0]]
    out = pre_norm_sublayer(rows, lambda m: [[7.0, 7.0]])
    assert flat(out) == APPROX([8.0, 8.0])


def test_pre_norm_accepts_rms_norm_instead_of_layer_norm():
    """Подмена нормы — вся «модернизация 2026» в одну строку."""
    rows = [[1.0, 2.0, 3.0]]
    with_ln = pre_norm_sublayer(rows, triple_sublayer, norm=layer_norm)
    with_rms = pre_norm_sublayer(rows, triple_sublayer, norm=rms_norm)
    assert flat(with_ln) != APPROX(flat(with_rms))


# ------------------------------------------------------ post_norm_sublayer
def test_post_norm_with_a_silent_sublayer_still_rewrites_the_input():
    """Норма стоит на выходе, поэтому «ничего не делать» уже невозможно."""
    rows = [[1.0, 2.0]]
    out = post_norm_sublayer(rows, zero_sublayer)
    assert flat(out) != APPROX(flat(rows))
    assert flat(out) == ROUGH(layer_norm(rows[0]))


def test_post_norm_gives_the_sublayer_the_raw_input():
    """Контраст с pre-norm: здесь подслой видит ненормализованный вход."""
    seen = []

    def spy(rows):
        seen.append([row[:] for row in rows])
        return zero_sublayer(rows)

    rows = [[10.0, 20.0, 30.0]]
    post_norm_sublayer(rows, spy)
    assert flat(seen[0]) == APPROX(flat(rows))


def test_post_norm_output_is_always_normalized():
    rows = [[100.0, -100.0, 5.0]]
    out = post_norm_sublayer(rows, triple_sublayer)
    assert sum(out[0]) == pytest.approx(0.0, abs=1e-9)


def test_post_norm_crushes_the_residual_stream_that_pre_norm_lets_grow():
    """Двенадцать слоёв: pre-norm накапливает сигнал, post-norm обнуляет память."""
    rows = [[1.0, 2.0, 3.0, 4.0]]
    pre, post = rows, rows
    for _ in range(12):
        pre = pre_norm_sublayer(pre, triple_sublayer)
        post = post_norm_sublayer(post, triple_sublayer)
    assert norm_of(pre) > 5 * norm_of(post)


# ------------------------------------------------------- transformer_block
def test_a_block_without_sublayers_is_the_identity():
    rows = [[1.0, 2.0]]
    assert flat(transformer_block(rows, [])) == APPROX(flat(rows))


def test_encoder_block_runs_two_sublayers_and_decoder_block_three():
    """Вся разница энкодера и декодера — количество подслоёв."""
    calls = []

    def named(tag):
        def sublayer(rows):
            calls.append(tag)
            return zero_sublayer(rows)

        return sublayer

    rows = [[1.0, 2.0]]
    transformer_block(rows, [named("self"), named("ffn")])
    assert calls == ["self", "ffn"]

    calls.clear()
    transformer_block(rows, [named("self"), named("cross"), named("ffn")])
    assert calls == ["self", "cross", "ffn"]


def test_a_block_of_silent_sublayers_leaves_the_stream_untouched():
    rows = [[1.0, 2.0], [3.0, 4.0]]
    out = transformer_block(rows, [zero_sublayer, zero_sublayer, zero_sublayer])
    assert flat(out) == APPROX(flat(rows))


def test_a_block_composes_its_sublayers_in_order():
    rows = [[1.0, 1.0]]
    once = pre_norm_sublayer(rows, lambda m: [[1.0, 2.0]])
    twice = pre_norm_sublayer(once, lambda m: [[10.0, 20.0]])
    out = transformer_block(rows, [lambda m: [[1.0, 2.0]], lambda m: [[10.0, 20.0]]])
    assert flat(out) == APPROX(flat(twice))


def test_a_block_keeps_the_shape():
    rows = [[1.0, 2.0, 3.0] for _ in range(5)]
    out = transformer_block(rows, [triple_sublayer, triple_sublayer])
    assert len(out) == 5 and all(len(row) == 3 for row in out)


# ------------------------------------------------------------ block_params
def test_block_params_counts_attention_plus_feed_forward():
    assert block_params(512) == 4 * 512 ** 2 + 2 * 4 * 512 ** 2


def test_cross_attention_adds_exactly_one_more_attention_block():
    assert block_params(512, cross_attention=True) - block_params(512) == 4 * 512 ** 2


def test_swiglu_needs_three_matrices_instead_of_two():
    assert block_params(512, swiglu=True) - block_params(512) == 4 * 512 ** 2


def test_swiglu_at_ratio_2_6_matches_relu_at_ratio_4():
    """Табличка урока: расширение снижают именно затем, чтобы бюджет сошёлся."""
    relu = block_params(512, ffn_ratio=4.0)
    swiglu = block_params(512, ffn_ratio=2.6, swiglu=True)
    assert abs(swiglu - relu) < 0.03 * relu


def test_block_params_grows_with_the_square_of_the_width():
    assert block_params(1024) == 4 * block_params(512)
