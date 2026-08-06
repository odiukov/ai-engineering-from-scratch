"""Тесты к уроку «Латентная диффузия и Stable Diffusion». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    classifier_free_guidance,
    cross_attention,
    decode,
    drop_label_for_cfg,
    encode,
    latent_compression_ratio,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def std(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ------------------------------------------------------------ encode/decode
def test_encode_applies_the_scaling_factor():
    assert encode(10.0) == APPROX(1.8215)
    assert encode(4.0, 0.5) == APPROX(2.0)


def test_decode_undoes_encode():
    assert decode(encode(7.3)) == pytest.approx(7.3, abs=1e-12)


def test_scaling_factor_brings_latents_to_unit_variance():
    """Зачем вообще магическое 0.18215: сырые латенты SD имеют разброс ~5.5,
    после множителя — примерно единицу, как и ждёт расписание DDPM."""
    rng = random.Random(0)
    raw = [rng.gauss(0.0, 1.0 / 0.18215) for _ in range(4000)]
    assert std(raw) > 4.0
    assert std([encode(v) for v in raw]) == pytest.approx(1.0, abs=0.05)


def test_mismatched_scaling_factor_does_not_round_trip():
    """Латенты SDXL, SD3 и Flux несовместимы: чужой множитель ломает картинку."""
    assert decode(encode(7.3, 0.18215), 0.13025) != pytest.approx(7.3, abs=1e-3)


# ------------------------------------------------- latent_compression_ratio
def test_stable_diffusion_compresses_pixels_forty_eight_times():
    assert latent_compression_ratio(512, 512, 3, 8, 4) == APPROX(48.0)


def test_no_downsample_and_matching_channels_gives_ratio_one():
    assert latent_compression_ratio(64, 64, 4, 1, 4) == APPROX(1.0)


def test_ratio_grows_with_the_square_of_the_downsample():
    """downsample режет обе пространственные оси, поэтому вклад квадратичный."""
    eight = latent_compression_ratio(512, 512, 3, 8, 4)
    sixteen = latent_compression_ratio(512, 512, 3, 16, 4)
    assert sixteen == APPROX(4.0 * eight)


def test_more_latent_channels_mean_less_compression():
    """SD3 перешёл с 4 каналов на 16 ради запаса на детали — и заплатил сжатием."""
    sd15 = latent_compression_ratio(512, 512, 3, 8, 4)
    sd3 = latent_compression_ratio(512, 512, 3, 8, 16)
    assert sd3 < sd15


# ----------------------------------------------------------------- softmax
def test_softmax_weights_sum_to_one():
    assert sum(softmax([0.3, -1.2, 4.0])) == pytest.approx(1.0, abs=1e-12)


def test_uniform_scores_give_uniform_weights():
    assert softmax([2.0, 2.0, 2.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_preserves_the_order_of_scores():
    w = softmax([0.0, 1.0, 2.0])
    assert w[0] < w[1] < w[2]


def test_softmax_survives_huge_scores():
    """Наивный math.exp(1000) — OverflowError. Вычитание максимума спасает."""
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


def test_softmax_is_invariant_to_a_constant_shift():
    assert softmax([1.0, 2.0]) == pytest.approx(softmax([101.0, 102.0]))


# --------------------------------------------------------- cross_attention
def test_identical_keys_average_the_values():
    out = cross_attention([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 2.0], [4.0, 6.0]])
    assert out == pytest.approx([2.0, 4.0])


def test_a_dominant_key_selects_its_value():
    """Так текст и «выбирает» нужный токен: один скор побеждает остальные."""
    out = cross_attention([10.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[5.0], [-5.0]])
    assert out[0] > 4.9


def test_attention_output_stays_inside_the_range_of_values():
    """Выход — выпуклая комбинация values, вылезти за их пределы он не может."""
    values = [[1.0], [7.0], [3.0]]
    keys = [[0.4, -1.0], [2.0, 0.5], [-0.3, 0.9]]
    out = cross_attention([0.7, 1.3], keys, values)
    assert 1.0 <= out[0] <= 7.0


def test_attention_ignores_the_order_of_text_tokens():
    """Без позиционного кодирования attention — это множество, а не список."""
    keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    values = [[1.0], [2.0], [3.0]]
    q = [0.6, -0.4]
    direct = cross_attention(q, keys, values)
    shuffled = cross_attention(q, keys[::-1], values[::-1])
    assert direct == pytest.approx(shuffled)


def test_attention_divides_scores_by_sqrt_of_dimension():
    """Без деления на sqrt(d) softmax насыщается и градиенты умирают."""
    out = cross_attention([1.0, 0.0, 0.0, 0.0],
                          [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                          [[1.0], [0.0]])
    scaled = math.exp(0.5) / (math.exp(0.5) + 1.0)
    assert out[0] == pytest.approx(scaled, abs=1e-9)


# ------------------------------------------------------ drop_label_for_cfg
def test_zero_dropout_always_keeps_the_label():
    rng = random.Random(0)
    assert all(drop_label_for_cfg(1, 2, 0.0, rng) == 1 for _ in range(200))


def test_full_dropout_always_returns_the_null_label():
    rng = random.Random(0)
    assert all(drop_label_for_cfg(1, 2, 1.0, rng) == 2 for _ in range(200))


def test_dropout_frequency_matches_the_requested_probability():
    """Стандартные 10%: без них не появится eps_uncond, и CFG собрать не из чего."""
    rng = random.Random(4)
    n = 20000
    dropped = sum(1 for _ in range(n) if drop_label_for_cfg(1, 2, 0.1, rng) == 2)
    assert dropped / n == pytest.approx(0.1, abs=0.01)


def test_dropout_sequence_is_reproducible_and_actually_mixed():
    """Одинаковый seed — одинаковая последовательность; при p = 0.5 в ней обязаны
    встретиться и метка, и null, иначе тест ничего не проверяет."""
    make = lambda: [drop_label_for_cfg(1, 2, 0.5, rng) for _ in range(20)]
    rng = random.Random(7)
    first = make()
    rng = random.Random(7)
    assert make() == first
    assert set(first) == {1, 2}


# ------------------------------------------------ classifier_free_guidance
def test_guidance_scale_one_reduces_to_the_conditional_prediction():
    """guidance_scale = 1 + w, значит w = 0 — это чистое условное предсказание."""
    assert classifier_free_guidance([1.0, -2.0], [0.3, 5.0], 0.0) == pytest.approx([1.0, -2.0])


def test_guidance_does_nothing_when_conditioning_changes_nothing():
    """Если условное и безусловное совпали, усиливать нечего при любом w."""
    same = [0.4, -1.1]
    assert classifier_free_guidance(same, same, 7.0) == pytest.approx(same)


def test_guidance_extrapolates_past_the_conditional_prediction():
    """CFG не интерполирует между двумя предсказаниями, а уезжает за условное."""
    out = classifier_free_guidance([1.0], [0.0], 3.0)
    assert out[0] > 1.0
    assert out == pytest.approx([4.0])


def test_guidance_is_linear_in_the_guidance_weight():
    cond, unc = [2.0], [0.5]
    w1 = classifier_free_guidance(cond, unc, 1.0)[0]
    w2 = classifier_free_guidance(cond, unc, 2.0)[0]
    w3 = classifier_free_guidance(cond, unc, 3.0)[0]
    assert w2 - w1 == pytest.approx(w3 - w2)


def test_guidance_amplifies_the_conditional_direction():
    """Формулу можно прочитать как eps_uncond + (1 + w) * (eps_cond - eps_uncond)."""
    cond, unc, w = [1.5, -0.5], [0.5, 0.5], 4.0
    expected = [u + (1.0 + w) * (c - u) for c, u in zip(cond, unc)]
    assert classifier_free_guidance(cond, unc, w) == pytest.approx(expected)
