"""Тесты к уроку «Flow matching и rectified flows». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    cfg_velocity,
    euler_sample,
    flow_matching_loss,
    flow_target,
    interpolate,
    logit_normal_t,
    path_curvature,
    reflow_pairs,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- interpolate
def test_interpolate_at_zero_returns_the_data_point():
    assert interpolate(2.0, -1.0, 0.0) == APPROX(2.0)


def test_interpolate_at_one_returns_the_noise_point():
    assert interpolate(2.0, -1.0, 1.0) == APPROX(-1.0)


def test_interpolate_at_half_is_the_midpoint():
    assert interpolate(2.0, -1.0, 0.5) == APPROX(0.5)


def test_interpolate_is_linear_in_time():
    """Путь прямой: приращение за одинаковый шаг по t всюду одно и то же."""
    x0, x1 = 3.0, -5.0
    d1 = interpolate(x0, x1, 0.3) - interpolate(x0, x1, 0.1)
    d2 = interpolate(x0, x1, 0.9) - interpolate(x0, x1, 0.7)
    assert d1 == APPROX(d2)


# ------------------------------------------------------------- flow_target
def test_flow_target_points_from_data_to_noise():
    assert flow_target(2.0, -1.0) == APPROX(-3.0)


def test_flow_target_is_zero_when_data_and_noise_coincide():
    assert flow_target(1.7, 1.7) == APPROX(0.0)


def test_flow_target_is_the_time_derivative_of_the_path():
    """Аналитическая цель обязана совпасть с центральной разностью по t."""
    x0, x1, t, h = 1.5, -2.0, 0.37, 1e-6
    numeric = (interpolate(x0, x1, t + h) - interpolate(x0, x1, t - h)) / (2 * h)
    assert flow_target(x0, x1) == pytest.approx(numeric, abs=1e-6)


# ------------------------------------------------------ flow_matching_loss
def test_loss_is_zero_for_the_exact_velocity_field():
    v = lambda x, t: 2.0
    assert flow_matching_loss(v, [1.0], [3.0], [0.5]) == APPROX(0.0)


def test_loss_is_the_squared_miss():
    v = lambda x, t: 2.0
    assert flow_matching_loss(v, [1.0], [5.0], [0.5]) == APPROX(4.0)


def test_loss_averages_over_the_batch():
    v = lambda x, t: 0.0
    # цели 2.0 и 4.0, поле нулевое -> средний квадрат (4 + 16) / 2
    assert flow_matching_loss(v, [0.0, 0.0], [2.0, 4.0], [0.3, 0.7]) == APPROX(10.0)


def test_loss_feeds_the_interpolant_not_the_endpoints():
    """Ловушка: подать в velocity x0 или x1 вместо x_t — и обучение мимо."""
    v = lambda x, t: x
    x0, x1, t = 1.0, 5.0, 0.25
    expected = (interpolate(x0, x1, t) - flow_target(x0, x1)) ** 2
    assert flow_matching_loss(v, [x0], [x1], [t]) == APPROX(expected)


def test_loss_of_an_empty_batch_is_zero():
    assert flow_matching_loss(lambda x, t: 1.0, [], [], []) == APPROX(0.0)


# ------------------------------------------------------------ euler_sample
def test_euler_sample_on_a_constant_field():
    assert euler_sample(lambda x, t: 1.5, 4.0, 1) == APPROX(2.5)


def test_straight_path_is_integrated_exactly_by_one_euler_step():
    """Главное обещание flow matching: прямая берётся одним шагом."""
    x0, x1 = 2.0, -1.0
    v = lambda x, t: flow_target(x0, x1)
    assert euler_sample(v, x1, 1) == APPROX(x0)


def test_straight_path_gives_the_same_answer_at_any_step_count():
    x0, x1 = 2.0, -1.0
    v = lambda x, t: flow_target(x0, x1)
    assert euler_sample(v, x1, 20) == APPROX(euler_sample(v, x1, 1))


def test_curved_path_needs_many_euler_steps():
    """А кривое поле одним шагом не берётся: ошибка падает только с шагами."""
    v = lambda x, t: x
    exact = math.exp(-1.0)  # решение dx/dt = x на пути от t=1 к t=0
    e1 = abs(euler_sample(v, 1.0, 1) - exact)
    e4 = abs(euler_sample(v, 1.0, 4) - exact)
    e50 = abs(euler_sample(v, 1.0, 50) - exact)
    assert e1 > e4 > e50


def test_euler_sample_evaluates_time_from_one_down_to_zero():
    """Поле должно опрашиваться в t = 1, 1-dt, ..., а не в 0, dt, ..."""
    seen = []
    euler_sample(lambda x, t: seen.append(t) or 0.0, 1.0, 4)
    assert seen == pytest.approx([1.0, 0.75, 0.5, 0.25])


# ----------------------------------------------------------- path_curvature
def test_constant_field_has_no_curvature():
    assert path_curvature(lambda x, t: 2.0, 1.0, 10) < 1e-12


def test_state_dependent_field_bends_the_path():
    assert path_curvature(lambda x, t: x, 1.0, 20) > 0.05


def test_curvature_is_never_negative():
    assert path_curvature(lambda x, t: -3.0 * x, 2.0, 16) >= 0.0


def test_zero_curvature_predicts_the_one_step_shortcut():
    """Прямое поле: и кривизна ноль, и один шаг совпадает с двадцатью."""
    v = lambda x, t: 0.7
    assert path_curvature(v, 3.0, 12) < 1e-12
    assert euler_sample(v, 3.0, 1) == APPROX(euler_sample(v, 3.0, 12))


# ------------------------------------------------------------ reflow_pairs
def test_reflow_returns_data_first_noise_second():
    pairs = reflow_pairs(lambda x, t: x, [1.0], 100)
    x0, x1 = pairs[0]
    assert x1 == APPROX(1.0)
    assert x0 == pytest.approx(math.exp(-1.0), abs=1e-2)


def test_reflow_keeps_every_noise_sample():
    noise = [1.0, 2.0, -0.5]
    pairs = reflow_pairs(lambda x, t: x, noise, 20)
    assert [x1 for _, x1 in pairs] == APPROX(noise)


def test_reflow_is_deterministic():
    """Никакой случайности внутри: те же вход и поле — тот же ответ."""
    v = lambda x, t: 0.3 * x + 0.2
    assert reflow_pairs(v, [1.0, -2.0], 10) == reflow_pairs(v, [1.0, -2.0], 10)


def test_reflow_makes_one_step_euler_exact():
    """Ради этого reflow и затевается: пара задаёт прямую, а прямая берётся шагом."""
    v = lambda x, t: x
    for x0, x1 in reflow_pairs(v, [1.0, 2.0, -0.5], 50):
        straight = lambda x, t, d=flow_target(x0, x1): d
        assert euler_sample(straight, x1, 1) == APPROX(x0)


def test_reflow_straightens_a_curved_path():
    v = lambda x, t: x
    x1 = 1.0
    before = path_curvature(v, x1, 20)
    x0, _ = reflow_pairs(v, [x1], 20)[0]
    after = path_curvature(lambda x, t: flow_target(x0, x1), x1, 20)
    assert after < 1e-12 < before


# ---------------------------------------------------------- logit_normal_t
def test_logit_normal_t_stays_strictly_inside_the_unit_interval():
    rng = random.Random(0)
    assert all(0.0 < logit_normal_t(rng) < 1.0 for _ in range(500))


def test_logit_normal_t_is_centred_on_a_half():
    rng = random.Random(1)
    samples = [logit_normal_t(rng) for _ in range(4000)]
    assert sum(samples) / len(samples) == pytest.approx(0.5, abs=0.02)


def test_logit_normal_t_concentrates_on_the_middle():
    """Именно за это его берёт SD3: середина шкалы получает больше сэмплов."""
    rng = random.Random(2)
    mid = sum(1 for _ in range(4000) if 0.25 < logit_normal_t(rng) < 0.75)
    assert mid / 4000 > 0.6  # у равномерного t было бы ровно 0.5


def test_larger_std_pushes_t_towards_the_edges():
    narrow = random.Random(3)
    wide = random.Random(3)
    mid_narrow = sum(1 for _ in range(3000) if 0.25 < logit_normal_t(narrow, std=1.0) < 0.75)
    mid_wide = sum(1 for _ in range(3000) if 0.25 < logit_normal_t(wide, std=3.0) < 0.75)
    assert mid_wide < mid_narrow


def test_logit_normal_t_survives_an_extreme_draw():
    """Наивная 1/(1+exp(-z)) падает с OverflowError на большом отрицательном z."""
    class FixedRng:
        def gauss(self, mu, sigma):
            return -800.0

    assert logit_normal_t(FixedRng()) == APPROX(0.0)


# ----------------------------------------------------------- cfg_velocity
def test_cfg_with_zero_weight_is_the_conditional_field():
    assert cfg_velocity(2.0, 1.0, 0.0) == APPROX(2.0)


def test_cfg_extrapolates_away_from_the_unconditional_field():
    assert cfg_velocity(2.0, 1.0, 1.0) == APPROX(3.0)


def test_cfg_does_nothing_when_both_fields_agree():
    """Если условие ни на что не влияет — усиливать нечего при любом w."""
    assert cfg_velocity(2.0, 2.0, 5.0) == APPROX(2.0)


def test_cfg_grows_with_the_guidance_weight():
    weak = cfg_velocity(2.0, 1.0, 1.0)
    strong = cfg_velocity(2.0, 1.0, 4.0)
    assert strong > weak > 2.0
