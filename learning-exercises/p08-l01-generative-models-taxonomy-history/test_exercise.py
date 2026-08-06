"""Тесты к уроку «Генеративные модели: таксономия и история». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    has_explicit_density,
    histogram_density,
    implicit_generator,
    integrate_density,
    kde_density,
    model_family,
    sampling_cost,
    speedup_source,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def _bimodal(n, seed=0):
    """Двухмодовая смесь как в code/main.py урока: моды на -2 и +2."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(rng.gauss(-2.0, 0.6) if rng.random() < 0.4 else rng.gauss(2.0, 0.9))
    return out


# ----------------------------------------------------------- model_family
def test_autoregressive_and_flows_are_family_one():
    assert model_family("PixelCNN") == 1
    assert model_family("WaveNet") == 1
    assert model_family("GPT") == 1
    assert model_family("Glow") == 1
    assert model_family("RealNVP") == 1


def test_vae_and_diffusion_share_the_approximate_bucket():
    """VAE и DDPM оба оптимизируют нижнюю оценку — это одно семейство."""
    assert model_family("VAE") == 2
    assert model_family("DDPM") == 2
    assert model_family("beta-VAE") == 2


def test_gans_are_the_only_implicit_family():
    assert model_family("GAN") == 3
    assert model_family("DCGAN") == 3
    assert model_family("StyleGAN") == 3


def test_flow_matching_sits_with_score_based_models():
    assert model_family("flow matching") == 4
    assert model_family("rectified flow") == 4
    assert model_family("score SDE") == 4


def test_token_based_models_are_family_five():
    assert model_family("Parti") == 5
    assert model_family("AudioLM") == 5
    assert model_family("VALL-E") == 5


def test_model_family_ignores_case_and_padding():
    assert model_family("  sTyLeGaN  ") == 3


def test_unknown_model_raises_value_error_not_silent_none():
    with pytest.raises(ValueError):
        model_family("SuperGen 9000")


# ---------------------------------------------------- has_explicit_density
def test_density_is_available_for_the_explicit_families():
    assert has_explicit_density(1) is True
    assert has_explicit_density(2) is True
    assert has_explicit_density(5) is True


def test_gan_family_cannot_evaluate_a_point():
    assert has_explicit_density(3) is False


def test_score_family_gives_the_gradient_not_the_density():
    assert has_explicit_density(4) is False


def test_family_number_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        has_explicit_density(6)


def test_gan_is_the_family_without_density():
    """Связка двух функций: по имени модели узнать, можно ли спросить p(x)."""
    assert has_explicit_density(model_family("StyleGAN")) is False
    assert has_explicit_density(model_family("Glow")) is True


# ------------------------------------------------------ histogram_density
def test_histogram_counts_only_the_bin_around_x():
    assert histogram_density([0.0, 0.1, 5.0], 0.05, bin_width=1.0) == APPROX(2 / 3)


def test_histogram_is_exactly_zero_far_from_the_data():
    """Резкий ноль — главная беда гистограммы: логарифм от неё уже не взять."""
    assert histogram_density([0.0, 0.1, 5.0], 99.0, bin_width=1.0) == APPROX(0.0)


def test_histogram_divides_by_bin_width_not_just_by_n():
    """Одна и та же корзина при вдвое меньшей ширине даёт вдвое большую плотность."""
    samples = [0.0] * 10
    wide = histogram_density(samples, 0.0, bin_width=1.0)
    narrow = histogram_density(samples, 0.0, bin_width=0.5)
    assert narrow == APPROX(2 * wide)


def test_histogram_on_empty_samples_raises_value_error():
    with pytest.raises(ValueError):
        histogram_density([], 0.0)


# ------------------------------------------------------------ kde_density
def test_kde_at_a_single_sample_is_the_standard_normal_peak():
    assert kde_density([0.0], 0.0, bandwidth=1.0) == pytest.approx(
        1 / math.sqrt(2 * math.pi), abs=1e-12
    )


def test_kde_is_positive_everywhere_unlike_the_histogram():
    """Гладкое ядро нигде не обнуляется — даже в двадцати сигмах от данных."""
    assert kde_density([0.0], 20.0, bandwidth=1.0) > 0.0


def test_kde_finds_both_modes_of_the_mixture():
    samples = _bimodal(2000)
    assert kde_density(samples, -2.0) > kde_density(samples, 0.0)
    assert kde_density(samples, 2.0) > kde_density(samples, 0.0)


def test_kde_sees_the_heavier_mode_as_heavier():
    """В смеси 40/60 правая мода тяжелее — плотность обязана это показать."""
    samples = _bimodal(4000)
    assert kde_density(samples, 2.0) > 0.0
    assert kde_density(samples, -2.0) > 0.0


# ------------------------------------------------------ integrate_density
def test_kde_integrates_to_one_over_the_whole_axis():
    """Определение плотности: полный интеграл равен единице."""
    got = integrate_density(kde_density, [0.0], -20.0, 20.0, steps=4000)
    assert got == pytest.approx(1.0, abs=1e-3)


def test_histogram_integrates_to_one_too():
    got = integrate_density(histogram_density, [0.0, 1.0, 2.0], -10.0, 12.0, steps=8000)
    assert got == pytest.approx(1.0, abs=0.05)


def test_integral_over_an_empty_interval_is_zero():
    assert integrate_density(kde_density, [0.0], 1.0, 1.0, steps=10) == APPROX(0.0)


def test_integral_over_a_subinterval_is_a_probability_below_one():
    samples = _bimodal(2000)
    p = integrate_density(kde_density, samples, -0.5, 0.5, steps=400)
    assert 0.0 < p < 1.0


def test_zero_steps_raises_value_error():
    with pytest.raises(ValueError):
        integrate_density(kde_density, [0.0], -1.0, 1.0, steps=0)


# ------------------------------------------------------ implicit_generator
def test_implicit_generator_returns_the_requested_count():
    assert len(implicit_generator([1.0, 2.0], 7, random.Random(0))) == 7


def test_implicit_generator_stays_near_the_training_points():
    """Каждый сэмпл — обучающая точка плюс маленький шум, не что-то новое."""
    out = implicit_generator([1.0, 2.0], 200, random.Random(1), sigma=0.1)
    assert all(min(abs(v - 1.0), abs(v - 2.0)) < 0.6 for v in out)


def test_implicit_generator_is_reproducible_from_the_same_seed():
    a = implicit_generator([1.0, 2.0], 10, random.Random(42))
    b = implicit_generator([1.0, 2.0], 10, random.Random(42))
    assert a == b


def test_implicit_generator_gives_different_samples_for_different_seeds():
    a = implicit_generator([1.0, 2.0], 10, random.Random(1))
    b = implicit_generator([1.0, 2.0], 10, random.Random(2))
    assert a != b


def test_implicit_generator_covers_both_modes():
    out = implicit_generator([-2.0, 2.0], 300, random.Random(3))
    assert any(v < 0 for v in out) and any(v > 0 for v in out)


# ---------------------------------------------------------- sampling_cost
def test_sampling_cost_is_steps_times_step_cost():
    assert sampling_cost(50, 0.06) == APPROX(3.0)


def test_one_step_gan_costs_one_step():
    assert sampling_cost(1, 0.03) == APPROX(0.03)


def test_negative_cost_raises_value_error():
    with pytest.raises(ValueError):
        sampling_cost(-1, 0.03)


# --------------------------------------------------------- speedup_source
def test_fewer_steps_is_a_step_count_speedup():
    assert speedup_source(50, 0.06, 4, 0.06) == "steps"


def test_cheaper_step_is_a_step_cost_speedup():
    assert speedup_source(50, 0.06, 50, 0.01) == "step_cost"


def test_distillation_plus_quantization_is_both():
    assert speedup_source(50, 0.06, 4, 0.01) == "both"


def test_identical_pipeline_is_not_a_speedup():
    assert speedup_source(50, 0.06, 50, 0.06) == "none"


def test_fewer_but_far_pricier_steps_is_not_a_speedup():
    """Четыре шага по 1.0 дороже пятидесяти по 0.06 — маркетинг, не ускорение."""
    assert speedup_source(50, 0.06, 4, 1.0) == "none"
