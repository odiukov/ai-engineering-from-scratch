"""Тесты к уроку «Stable Diffusion — архитектура и дообучение». Правь exercise.py."""

import math

import pytest

from exercise import (
    classifier_free_guidance,
    cross_attention,
    img2img_timesteps,
    inpaint_blend,
    latent_compression_factor,
    lora_update,
    scale_latents,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [v for row in M for v in row]


# ------------------------------------------------ latent_compression_factor
def test_stable_diffusion_latent_is_forty_eight_times_smaller():
    """Та самая цифра 48x из статьи про latent diffusion."""
    assert latent_compression_factor((3, 512, 512), (4, 64, 64)) == APPROX(48.0)


def test_channels_count_too_not_just_the_sides():
    """Ловушка: у латента 4 канала против 3 у картинки, и это меняет ответ."""
    by_sides_only = (512 * 512) / (64 * 64)
    assert latent_compression_factor((3, 512, 512), (4, 64, 64)) != APPROX(
        by_sides_only
    )


def test_halving_every_side_of_the_latent_quadruples_the_factor():
    big = latent_compression_factor((3, 512, 512), (4, 64, 64))
    small = latent_compression_factor((3, 512, 512), (4, 32, 32))
    assert small == APPROX(4.0 * big)


# ------------------------------------------------------------ scale_latents
def test_scaling_multiplies_every_value():
    assert scale_latents([10.0, -2.0], 0.5) == APPROX([5.0, -1.0])


def test_scale_then_unscale_returns_the_original_latents():
    """Забыл поделить перед декодером — получил выцветшую картинку."""
    latents = [4.7, -3.1, 0.0, 12.5]
    there = scale_latents(latents)
    back = scale_latents(there, inverse=True)
    assert back == pytest.approx(latents, abs=1e-9)
    assert there != pytest.approx(latents, abs=1e-3)


def test_default_factor_brings_raw_vae_latents_to_roughly_unit_spread():
    """Сырой выход VAE гуляет в районе +-5.5; после 0.18215 — около +-1."""
    raw = [5.5, -5.5, 5.5, -5.5]
    scaled = scale_latents(raw)
    assert max(abs(v) for v in scaled) == pytest.approx(1.0, abs=0.05)


# ------------------------------------------------------------------ softmax
def test_softmax_of_equal_scores_is_uniform():
    assert softmax([0.0, 0.0]) == APPROX([0.5, 0.5])
    assert softmax([7.0, 7.0, 7.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_sums_to_one_and_stays_positive():
    weights = softmax([-3.0, 0.5, 2.0, 9.0])
    assert sum(weights) == APPROX(1.0)
    assert all(w > 0.0 for w in weights)


def test_softmax_is_invariant_to_a_constant_shift():
    """softmax(s) = softmax(s + c): на этом и держится вычитание максимума."""
    base = softmax([1.0, 2.0, 3.0])
    shifted = softmax([101.0, 102.0, 103.0])
    assert shifted == pytest.approx(base, abs=1e-12)


def test_softmax_does_not_overflow_on_huge_scores():
    """Ловушка: math.exp(1000) падает с OverflowError."""
    weights = softmax([1000.0, 1000.0, 900.0])
    assert sum(weights) == APPROX(1.0)
    assert weights[2] < 1e-30


# ---------------------------------------------------------- cross_attention
def test_equal_keys_average_all_the_values():
    """Промпт ничего не выделяет — берём среднее по токенам."""
    assert cross_attention([0.0], [[0.0], [0.0]], [[1.0], [3.0]]) == APPROX([2.0])


def test_attention_picks_the_value_of_the_closest_key():
    out = cross_attention([10.0], [[1.0], [-1.0]], [[1.0], [3.0]])
    assert out[0] == pytest.approx(1.0, abs=1e-6)


def test_attention_output_is_a_convex_combination_of_the_values():
    """Ответ обязан лежать внутри выпуклой оболочки values, а не снаружи."""
    values = [[1.0, 0.0], [5.0, -2.0], [3.0, 4.0]]
    out = cross_attention([0.3, -0.7], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], values)
    for i in range(2):
        column = [v[i] for v in values]
        assert min(column) <= out[i] <= max(column)


def test_scores_are_divided_by_sqrt_of_the_dimension():
    """Без деления на sqrt(d) softmax насыщается и градиент умирает."""
    query = [2.0, 2.0, 2.0, 2.0]
    keys = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    out = cross_attention(query, keys, [[1.0], [0.0]])
    # score_0 = 8 / sqrt(4) = 4, score_1 = 0
    expected = math.exp(4.0) / (math.exp(4.0) + 1.0)
    assert out[0] == APPROX(expected)


# -------------------------------------------------- classifier_free_guidance
def test_guidance_scale_one_is_exactly_the_conditional_prediction():
    """scale = 1 — это обычное условное предсказание, без всякой добавки."""
    cond = [1.0, -0.5, 2.0]
    assert classifier_free_guidance([0.0, 0.3, -1.0], cond, 1.0) == APPROX(cond)


def test_guidance_scale_zero_is_exactly_the_unconditional_prediction():
    uncond = [0.0, 0.3, -1.0]
    assert classifier_free_guidance(uncond, [1.0, -0.5, 2.0], 0.0) == APPROX(uncond)


def test_default_scale_extrapolates_past_the_conditional_prediction():
    """7.5 — не интерполяция: результат уезжает дальше условного."""
    out = classifier_free_guidance([0.0], [1.0], 7.5)
    assert out == APPROX([7.5])
    assert out[0] > 1.0


def test_guidance_moves_along_the_line_between_the_two_predictions():
    uncond, cond = [1.0, 2.0], [3.0, -2.0]
    for scale in (0.5, 2.0, 7.5):
        out = classifier_free_guidance(uncond, cond, scale)
        for u, c, o in zip(uncond, cond, out):
            assert o - u == APPROX(scale * (c - u))


def test_identical_predictions_make_guidance_a_no_op():
    """Промпт ни на что не повлиял — крутить scale бессмысленно."""
    same = [0.4, -0.4]
    assert classifier_free_guidance(same, same, 15.0) == APPROX(same)


# --------------------------------------------------------- img2img_timesteps
def test_full_strength_runs_the_whole_schedule():
    """strength = 1.0 — входная картинка полностью игнорируется."""
    assert img2img_timesteps(50, 1.0) == list(range(49, -1, -1))


def test_zero_strength_runs_nothing_at_all():
    assert img2img_timesteps(50, 0.0) == []


def test_strength_sets_how_many_steps_are_run():
    assert len(img2img_timesteps(50, 0.6)) == 30
    assert img2img_timesteps(50, 0.6)[0] == 29


def test_timesteps_go_from_noisy_to_clean():
    steps = img2img_timesteps(20, 0.5)
    assert steps == sorted(steps, reverse=True)
    assert steps[-1] == 0


def test_more_strength_never_means_fewer_steps():
    lengths = [len(img2img_timesteps(40, s)) for s in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert lengths == sorted(lengths)
    assert lengths[-1] == 40


# ------------------------------------------------------------ inpaint_blend
def test_empty_mask_keeps_the_original_untouched():
    assert inpaint_blend([9.0, 9.0], [1.0, 2.0], [0.0, 0.0]) == APPROX([1.0, 2.0])


def test_full_mask_takes_the_whole_denoised_latent():
    assert inpaint_blend([9.0, 9.0], [1.0, 2.0], [1.0, 1.0]) == APPROX([9.0, 9.0])


def test_soft_mask_interpolates_at_the_edge():
    assert inpaint_blend([9.0], [1.0], [0.5]) == APPROX([5.0])


def test_blend_works_per_pixel_not_all_or_nothing():
    out = inpaint_blend([9.0, 9.0, 9.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.25])
    assert out == APPROX([9.0, 1.0, 3.0])


def test_blend_does_not_mutate_its_inputs():
    """Оригинал понадобится ещё раз на следующем шаге сэмплирования."""
    denoised, original, mask = [9.0], [1.0], [0.5]
    inpaint_blend(denoised, original, mask)
    assert (denoised, original, mask) == ([9.0], [1.0], [0.5])


# --------------------------------------------------------------- lora_update
def test_lora_adds_the_low_rank_product_to_the_weights():
    assert flat(lora_update([[1.0, 1.0]], [[1.0]], [[2.0, 0.0]])) == APPROX([3.0, 1.0])


def test_alpha_zero_leaves_the_base_model_exactly_as_it_was():
    """lora_scale = 0.0 — адаптер выключен, база не тронута."""
    W = [[1.0, 1.0], [2.0, -2.0]]
    out = lora_update(W, [[1.0], [3.0]], [[2.0, 0.0]], alpha=0.0)
    assert flat(out) == APPROX(flat(W))


def test_alpha_scales_the_adapter_linearly():
    """Ровно то, чем крутят lora_scale между 0 и 1."""
    W = [[0.0, 0.0], [0.0, 0.0]]
    A, B = [[1.0], [2.0]], [[3.0, -1.0]]
    half = flat(lora_update(W, A, B, alpha=0.5))
    full = flat(lora_update(W, A, B, alpha=1.0))
    assert half == APPROX([v / 2 for v in full])


def test_lora_does_not_mutate_the_frozen_base_weights():
    """База заморожена: обновление обязано вернуть новую матрицу."""
    W = [[1.0, 1.0], [1.0, 1.0]]
    lora_update(W, [[1.0], [1.0]], [[5.0, 5.0]])
    assert flat(W) == APPROX([1.0, 1.0, 1.0, 1.0])


def test_rank_one_adapter_can_only_add_proportional_rows():
    """Поправка ранга r не может изменить веса как угодно — в этом её защита."""
    zeros = [[0.0] * 3 for _ in range(3)]
    A = [[1.0], [2.0], [-3.0]]
    B = [[4.0, 5.0, 6.0]]
    delta = lora_update(zeros, A, B)
    for i in range(3):
        # строка i пропорциональна строке 0, значит ранг ровно 1
        cross = delta[i][0] * delta[0][1] - delta[i][1] * delta[0][0]
        assert cross == APPROX(0.0)


def test_rank_two_adapter_reaches_what_rank_one_cannot():
    """Ранг 2 уже склеивает две независимые строки — единичную матрицу видно."""
    zeros = [[0.0, 0.0], [0.0, 0.0]]
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[1.0, 0.0], [0.0, 1.0]]
    assert flat(lora_update(zeros, A, B)) == APPROX([1.0, 0.0, 0.0, 1.0])
