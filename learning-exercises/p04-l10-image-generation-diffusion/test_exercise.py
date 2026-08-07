"""Тесты к уроку «Генерация изображений — диффузионные модели». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    alphas_cumprod,
    ddim_step,
    ddpm_step,
    linear_beta_schedule,
    predict_x0,
    q_sample,
    q_step,
    timestep_embedding,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


# --------------------------------------------------- linear_beta_schedule
def test_schedule_hits_both_endpoints_exactly():
    """Шаг делится на T-1, поэтому beta_end обязан попасть в список."""
    betas = linear_beta_schedule(5, 0.0, 1.0)
    assert betas == APPROX([0.0, 0.25, 0.5, 0.75, 1.0])


def test_schedule_has_exactly_T_values():
    assert len(linear_beta_schedule(1000)) == 1000
    assert len(linear_beta_schedule(7, 0.1, 0.2)) == 7


def test_schedule_is_monotonically_increasing():
    """Портим картинку всё смелее: beta растёт от шага к шагу."""
    betas = linear_beta_schedule(50, 1e-4, 2e-2)
    assert all(b < c for b, c in zip(betas, betas[1:]))


def test_schedule_of_one_step_does_not_divide_by_zero():
    """Ловушка: T-1 = 0. Одиночный шаг — это просто [beta_start]."""
    assert linear_beta_schedule(1, 0.1, 0.2) == APPROX([0.1])


def test_default_schedule_stays_small_enough_to_be_gaussian_per_step():
    """Дефолт DDPM: beta везде сильно меньше 1, иначе обратный шаг не гауссов."""
    betas = linear_beta_schedule()
    assert betas[0] == pytest.approx(1e-4, abs=1e-12)
    assert betas[-1] == pytest.approx(2e-2, abs=1e-12)
    assert max(betas) < 0.05


# --------------------------------------------------------- alphas_cumprod
def test_cumprod_of_zero_betas_keeps_the_whole_signal():
    assert alphas_cumprod([0.0, 0.0, 0.0]) == APPROX([1.0, 1.0, 1.0])


def test_cumprod_multiplies_step_by_step():
    assert alphas_cumprod([0.1, 0.2]) == APPROX([0.9, 0.72])
    assert alphas_cumprod([0.5, 0.5]) == APPROX([0.5, 0.25])


def test_alpha_bar_decreases_monotonically():
    """Доля выжившего сигнала может только падать."""
    bars = alphas_cumprod(linear_beta_schedule(200, 1e-4, 2e-2))
    assert all(a > b for a, b in zip(bars, bars[1:]))


def test_alpha_bar_reaches_almost_zero_at_the_end_of_the_default_schedule():
    """x_T обязан быть неотличим от чистого шума — иначе не с чего начинать."""
    bars = alphas_cumprod(linear_beta_schedule())
    assert bars[-1] < 1e-4
    assert bars[-1] > 0.0


# ----------------------------------------------------------------- q_step
def test_zero_beta_leaves_the_image_untouched():
    assert q_step([1.0, -2.0], 0.0, [5.0, 5.0]) == APPROX([1.0, -2.0])


def test_beta_one_replaces_the_image_with_pure_noise():
    assert q_step([1.0, -2.0], 1.0, [5.0, 7.0]) == APPROX([5.0, 7.0])


def test_q_step_shrinks_the_signal_by_the_square_root():
    assert q_step([2.0], 0.75, [0.0]) == APPROX([1.0])


def test_q_step_preserves_variance():
    """sqrt(1-b)^2 + sqrt(b)^2 = 1: за тысячу шагов числа не разлетаются."""
    rng = random.Random(0)
    x_prev = [rng.gauss(0, 1) for _ in range(4000)]
    noise = [rng.gauss(0, 1) for _ in range(4000)]
    out = q_step(x_prev, 0.3, noise)
    assert variance(out) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------- q_sample
def test_alpha_bar_one_returns_the_clean_image():
    assert q_sample([1.0, 2.0], [9.0, 9.0], 1.0) == APPROX([1.0, 2.0])


def test_alpha_bar_zero_returns_pure_noise():
    assert q_sample([1.0, 2.0], [9.0, -3.0], 0.0) == APPROX([9.0, -3.0])


def test_q_sample_interpolates_by_square_roots_not_linearly():
    assert q_sample([4.0], [0.0], 0.25) == APPROX([2.0])
    assert q_sample([0.0], [4.0], 0.25) == APPROX([math.sqrt(0.75) * 4.0])


def test_closed_form_matches_the_iterative_chain_without_noise():
    """Прыжок на шаг t совпадает с t шагами подряд — вся суть замкнутой формы."""
    betas = linear_beta_schedule(30, 0.01, 0.2)
    bars = alphas_cumprod(betas)
    x = [3.0, -1.0]
    zeros = [0.0, 0.0]
    for t, beta in enumerate(betas):
        x = q_step(x, beta, zeros)
        assert x == pytest.approx(q_sample([3.0, -1.0], zeros, bars[t]), abs=1e-12)


def test_closed_form_matches_the_iterative_chain_in_distribution():
    """С шумом совпадение статистическое: та же средняя и та же дисперсия."""
    betas = linear_beta_schedule(40, 0.01, 0.2)
    bars = alphas_cumprod(betas)
    rng = random.Random(0)
    n = 4000
    x0 = 2.0

    chain = [x0] * n
    for beta in betas:
        chain = q_step(chain, beta, [rng.gauss(0, 1) for _ in range(n)])
    direct = q_sample([x0] * n, [rng.gauss(0, 1) for _ in range(n)], bars[-1])

    assert mean(chain) == pytest.approx(mean(direct), abs=0.1)
    assert variance(chain) == pytest.approx(variance(direct), abs=0.1)


# -------------------------------------------------------------- predict_x0
def test_predict_x0_inverts_q_sample_exactly():
    """Подставили истинный шум — вернули исходную картинку."""
    x0 = [1.0, -0.5, 3.0]
    noise = [0.3, -1.2, 0.7]
    for bar in (0.9, 0.5, 0.05):
        x_t = q_sample(x0, noise, bar)
        assert predict_x0(x_t, noise, bar) == pytest.approx(x0, abs=1e-9)


def test_predict_x0_divides_by_the_square_root_of_alpha_bar():
    assert predict_x0([2.0], [0.0], 0.25) == APPROX([4.0])


def test_wrong_noise_prediction_shows_up_as_error_in_x0():
    """Ошибка в eps не исчезает: она масштабируется и уезжает в оценку x0."""
    x0 = [0.0]
    x_t = q_sample(x0, [1.0], 0.5)
    assert predict_x0(x_t, [1.0], 0.5) == APPROX([0.0])
    assert abs(predict_x0(x_t, [0.0], 0.5)[0]) > 0.5


# --------------------------------------------------- timestep_embedding
def test_embedding_has_the_requested_length():
    assert len(timestep_embedding(7, 64)) == 64
    assert len(timestep_embedding(0, 8)) == 8


def test_embedding_of_step_zero_is_all_sines_then_all_ones():
    assert timestep_embedding(0, 4) == APPROX([0.0, 0.0, 1.0, 1.0])


def test_first_frequency_is_one_so_the_first_coordinates_are_plain_sin_and_cos():
    emb = timestep_embedding(3.0, 8)
    half = 4
    assert emb[0] == APPROX(math.sin(3.0))
    assert emb[half] == APPROX(math.cos(3.0))


def test_embedding_norm_does_not_depend_on_the_timestep():
    """sin^2 + cos^2 = 1 на каждой частоте: длина всегда sqrt(dim/2)."""
    for t in (0, 1, 137, 999):
        emb = timestep_embedding(t, 64)
        assert math.sqrt(sum(v * v for v in emb)) == pytest.approx(
            math.sqrt(32), abs=1e-9
        )


def test_different_timesteps_get_different_embeddings():
    """Иначе сеть не смогла бы отличить уровень зашумления."""
    embeddings = {tuple(timestep_embedding(t, 64)) for t in range(0, 1000, 97)}
    assert len(embeddings) == len(range(0, 1000, 97))


def test_high_frequency_coordinates_change_slowly():
    """Частоты падают геометрически: последняя почти не двигается за шаг."""
    a = timestep_embedding(0, 64)
    b = timestep_embedding(1, 64)
    assert abs(a[0] - b[0]) > 0.5
    assert abs(a[31] - b[31]) < 1e-3


# -------------------------------------------------------------- ddpm_step
def test_ddpm_last_step_equals_predict_x0():
    """На t = 0 байесовская формула сворачивается в оценку чистой картинки."""
    betas = [0.2]
    bars = alphas_cumprod(betas)
    x_t, eps = [0.7], [0.4]
    assert ddpm_step(x_t, eps, 0, betas, bars) == pytest.approx(
        predict_x0(x_t, eps, bars[0]), abs=1e-12
    )


def test_ddpm_ignores_noise_on_the_final_step():
    """Добавить шум на t = 0 значит испортить уже готовую картинку."""
    betas = [0.2, 0.3]
    bars = alphas_cumprod(betas)
    quiet = ddpm_step([0.7], [0.4], 0, betas, bars)
    loud = ddpm_step([0.7], [0.4], 0, betas, bars, z=[10.0])
    assert loud == pytest.approx(quiet, abs=1e-12)


def test_ddpm_adds_noise_scaled_by_sqrt_beta_on_earlier_steps():
    betas = [0.2, 0.36]
    bars = alphas_cumprod(betas)
    quiet = ddpm_step([0.0], [0.0], 1, betas, bars)
    loud = ddpm_step([0.0], [0.0], 1, betas, bars, z=[1.0])
    assert loud[0] - quiet[0] == pytest.approx(math.sqrt(0.36), abs=1e-12)


def test_ddpm_step_with_the_true_noise_returns_the_signal_to_the_previous_level():
    """Шаг назад с угаданным eps меняет вес сигнала ровно на sqrt(alpha_bar_{t-1}).

    Проверяем по частям, пользуясь линейностью: сначала картинка без шума,
    потом шум без картинки.
    """
    betas = linear_beta_schedule(100, 0.01, 0.2)
    bars = alphas_cumprod(betas)
    t = 60

    signal_only = ddpm_step(q_sample([1.0], [0.0], bars[t]), [0.0], t, betas, bars)
    assert signal_only[0] == pytest.approx(math.sqrt(bars[t - 1]), abs=1e-12)

    noise_only = ddpm_step(q_sample([0.0], [1.0], bars[t]), [1.0], t, betas, bars)
    assert 0.0 < noise_only[0] < math.sqrt(1.0 - bars[t])


# -------------------------------------------------------------- ddim_step
def test_ddim_step_to_alpha_bar_one_returns_the_predicted_image():
    assert ddim_step([2.0], [0.0], 0.25, 1.0) == APPROX([4.0])


def test_ddim_step_that_goes_nowhere_is_the_identity():
    """alpha_bar_prev = alpha_bar_t: вычли eps и тут же вернули с тем же весом."""
    assert ddim_step([1.0], [2.0], 0.5, 0.5) == APPROX([1.0])
    assert ddim_step([-3.0, 0.4], [0.1, 0.2], 0.3, 0.3) == pytest.approx(
        [-3.0, 0.4], abs=1e-9
    )


def test_ddim_stays_on_the_forward_trajectory_when_eps_is_exact():
    """С точным eps шаг DDIM ровно равен q_sample на новом уровне шума."""
    x0, noise = [1.0, -2.0], [0.5, 0.25]
    bar_t, bar_prev = 0.2, 0.6
    x_t = q_sample(x0, noise, bar_t)
    assert ddim_step(x_t, noise, bar_t, bar_prev) == pytest.approx(
        q_sample(x0, noise, bar_prev), abs=1e-9
    )


def test_skipping_steps_lands_where_the_dense_walk_lands():
    """Ровно то, ради чего DDIM: 50 шагов вместо 1000 без потери траектории."""
    x0, noise = [1.0, 0.0, -0.5], [0.2, -0.9, 0.4]
    bars = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    x = q_sample(x0, noise, bars[0])

    dense = x
    for prev in bars[1:]:
        dense = ddim_step(dense, noise, bars[bars.index(prev) - 1], prev)

    sparse = ddim_step(x, noise, bars[0], bars[-1])
    assert sparse == pytest.approx(dense, abs=1e-9)
