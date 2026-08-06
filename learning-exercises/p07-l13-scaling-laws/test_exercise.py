"""Тесты к уроку «Законы масштабирования». Правь exercise.py."""

import pytest

from exercise import (
    chinchilla_loss,
    compute_flops,
    compute_optimal,
    emergence_curves,
    min_compute_for_loss,
    optimal_exponents,
    overtraining_tradeoff,
    tokens_for_budget,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Аналитическое решение задачи «минимум L при 6ND = C», выведенное руками.
# Упражнение просит перебор по сетке; здесь мы им проверяем перебор.
A_FIT, B_FIT, ALPHA_FIT, BETA_FIT = 406.4, 410.7, 0.34, 0.28


def analytic_optimal_n(C):
    """N* = (alpha*A / (beta*B))^(1/(alpha+beta)) * (C/6)^(beta/(alpha+beta))."""
    ratio = (ALPHA_FIT * A_FIT) / (BETA_FIT * B_FIT)
    total = ALPHA_FIT + BETA_FIT
    return ratio ** (1.0 / total) * (C / 6.0) ** (BETA_FIT / total)


# -------------------------------------------------------- chinchilla_loss
def test_chinchilla_loss_on_the_chinchilla_model():
    assert chinchilla_loss(70e9, 1400e9) == pytest.approx(1.93664547, abs=1e-6)


def test_loss_falls_when_the_model_grows():
    assert chinchilla_loss(140e9, 1400e9) < chinchilla_loss(70e9, 1400e9)


def test_loss_falls_when_the_data_grows():
    assert chinchilla_loss(70e9, 2800e9) < chinchilla_loss(70e9, 1400e9)


def test_irreducible_loss_is_the_floor():
    """Сколько ни лей компьюта, ниже E не опустишься: это энтропия данных."""
    assert chinchilla_loss(1e30, 1e30) == pytest.approx(1.69, abs=1e-4)
    assert chinchilla_loss(1e12, 1e14) > 1.69


def test_gpt3_is_undertrained_relative_to_chinchilla():
    """175B на 300B токенов проигрывает 70B на 1.4T — весь смысл статьи 2022."""
    assert chinchilla_loss(175e9, 300e9) > chinchilla_loss(70e9, 1400e9)


def test_loss_terms_are_separable():
    """Слагаемое по N не зависит от D: разность двух D при равных N одна и та же."""
    d1 = chinchilla_loss(1e9, 1e11) - chinchilla_loss(1e9, 1e12)
    d2 = chinchilla_loss(1e10, 1e11) - chinchilla_loss(1e10, 1e12)
    assert d1 == pytest.approx(d2, abs=1e-9)


# ------------------------------------------ compute_flops/tokens_for_budget
def test_compute_flops_uses_the_factor_six():
    assert compute_flops(70e9, 1400e9) == APPROX(5.88e23)


def test_tokens_for_budget_inverts_compute_flops():
    """Круговой прогон: 6ND -> D и обратно. Шестёрка не должна потеряться."""
    N, D = 8e9, 15e12
    C = compute_flops(N, D)
    assert tokens_for_budget(C, N) == pytest.approx(D, rel=1e-12)


def test_bigger_model_gets_fewer_tokens_at_fixed_budget():
    C = 1e24
    assert tokens_for_budget(C, 1e10) == pytest.approx(tokens_for_budget(C, 1e9) / 10)


# --------------------------------------------------------- compute_optimal
def test_compute_optimal_respects_the_budget_constraint():
    """Найденная пара обязана лежать точно на 6ND = C."""
    C = 1e23
    N, D, _ = compute_optimal(C)
    assert compute_flops(N, D) == pytest.approx(C, rel=1e-9)


def test_compute_optimal_returns_the_loss_of_its_own_pair():
    N, D, loss = compute_optimal(1e23)
    assert loss == pytest.approx(chinchilla_loss(N, D), abs=1e-12)


def test_compute_optimal_beats_its_neighbours_on_the_constraint():
    """Смысловая проверка минимума: сдвинь N в любую сторону — loss вырастет."""
    C = 1e23
    N, _, loss = compute_optimal(C, n_grid=4000)
    for factor in (0.5, 0.8, 1.25, 2.0):
        moved = N * factor
        assert chinchilla_loss(moved, tokens_for_budget(C, moved)) > loss


def test_compute_optimal_has_zero_slope_at_the_optimum():
    """Численная производная dL/d(ln N) вдоль связи 6ND = C в оптимуме = 0."""
    C = 1e23
    N, _, _ = compute_optimal(C, n_grid=4000)
    h = 1e-3

    def along(scale):
        n = N * scale
        return chinchilla_loss(n, tokens_for_budget(C, n))

    slope = (along(1 + h) - along(1 - h)) / (2 * h)
    assert slope == pytest.approx(0.0, abs=1e-3)


def test_compute_optimal_matches_the_analytic_solution():
    """Перебор по сетке обязан сойтись к формуле, выведенной из dL/dN = 0."""
    for C in (1e20, 1e22, 1e24):
        N, _, _ = compute_optimal(C, n_grid=4000)
        assert N == pytest.approx(analytic_optimal_n(C), rel=0.01)


def test_bigger_budget_always_buys_lower_loss():
    losses = [compute_optimal(10.0 ** k)[2] for k in range(18, 27)]
    assert losses == sorted(losses, reverse=True)
    assert len(set(losses)) == len(losses)


# ------------------------------------------------------- optimal_exponents
def test_optimal_exponents_sum_to_one():
    """a + b = 1, иначе N*D перестало бы быть пропорционально C."""
    a, b = optimal_exponents()
    assert a + b == APPROX(1.0)


def test_optimal_exponents_for_the_chinchilla_fit():
    a, b = optimal_exponents(0.34, 0.28)
    assert (a, b) == pytest.approx((0.28 / 0.62, 0.34 / 0.62))


def test_symmetric_exponents_split_the_budget_evenly():
    assert optimal_exponents(0.5, 0.5) == pytest.approx((0.5, 0.5))


def test_grid_optimum_scales_with_the_predicted_exponents():
    """N_opt(100C)/N_opt(C) обязано быть 100^a, а D_opt — 100^b."""
    a, b = optimal_exponents()
    N1, D1, _ = compute_optimal(1e22, n_grid=4000)
    N2, D2, _ = compute_optimal(1e24, n_grid=4000)
    assert N2 / N1 == pytest.approx(100.0 ** a, rel=0.02)
    assert D2 / D1 == pytest.approx(100.0 ** b, rel=0.02)


def test_token_per_param_ratio_is_not_a_constant():
    """С этими показателями D/N растёт по C, а не сидит на 20."""
    r1 = (lambda t: t[1] / t[0])(compute_optimal(1e20, n_grid=4000)[:2])
    r2 = (lambda t: t[1] / t[0])(compute_optimal(1e25, n_grid=4000)[:2])
    assert r2 > 2 * r1


# --------------------------------------------------- overtraining_tradeoff
def test_overtraining_keeps_the_same_compute_budget():
    C = 1e24
    N_small, D_big, _, _ = overtraining_tradeoff(C, 10.0)
    assert compute_flops(N_small, D_big) == pytest.approx(C, rel=1e-9)


def test_overtraining_always_costs_some_loss():
    """Оптимум есть оптимум: любой уход от него платит положительным штрафом."""
    for shrink in (2.0, 5.0, 10.0, 50.0):
        assert overtraining_tradeoff(1e24, shrink)[2] > 0.0


def test_overtraining_penalty_grows_with_shrink():
    penalties = [overtraining_tradeoff(1e24, s)[2] for s in (2.0, 5.0, 10.0, 50.0)]
    assert penalties == sorted(penalties)


def test_overtraining_buys_exactly_shrink_times_cheaper_inference():
    assert overtraining_tradeoff(1e24, 10.0)[3] == APPROX(10.0)


def test_llama_style_tradeoff_is_a_fraction_of_a_nat():
    """Десятикратно меньшая модель стоит меньше 0.2 кросс-энтропии — вот и весь секрет."""
    penalty = overtraining_tradeoff(1e24, 10.0)[2]
    assert 0.0 < penalty < 0.2


# --------------------------------------------------- min_compute_for_loss
def test_loss_below_the_irreducible_floor_is_unreachable():
    """1.5 < E = 1.69, поэтому ответ None, а не «очень много флопсов»."""
    assert min_compute_for_loss(1.5) is None


def test_min_compute_actually_reaches_the_target():
    C = min_compute_for_loss(2.5)
    assert C is not None
    assert compute_optimal(C)[2] <= 2.5


def test_stricter_target_costs_more_compute():
    assert min_compute_for_loss(2.0) > min_compute_for_loss(2.5)


def test_each_tenth_of_a_nat_costs_more_than_the_previous_one():
    """Срезать 2.0 -> 1.9 дороже, чем 2.1 -> 2.0. Так работает степенной закон."""
    step_1 = min_compute_for_loss(2.0, n_grid=600) / min_compute_for_loss(2.1, n_grid=600)
    step_2 = min_compute_for_loss(1.9, n_grid=600) / min_compute_for_loss(2.0, n_grid=600)
    assert step_2 > step_1 > 5.0


# -------------------------------------------------------- emergence_curves
def test_emergence_curves_return_one_value_per_budget():
    budgets = [10.0 ** k for k in range(18, 27)]
    smooth, stepped = emergence_curves(budgets, 2.0)
    assert len(smooth) == len(stepped) == len(budgets)


def test_continuous_metric_improves_smoothly():
    """Непрерывный loss падает маленькими шагами — никакого скачка нет."""
    budgets = [10.0 ** k for k in range(18, 27)]
    smooth, _ = emergence_curves(budgets, 2.0)
    steps = [a - b for a, b in zip(smooth, smooth[1:])]
    assert all(s > 0 for s in steps)
    assert max(steps) < 0.7


def test_thresholded_metric_jumps_by_a_whole_point_in_one_step():
    """Тот же прогресс пороговой метрикой: 0 -> 1 за один шаг. Скачок в метрике."""
    budgets = [10.0 ** k for k in range(18, 27)]
    smooth, stepped = emergence_curves(budgets, 2.0)
    jumps = [b - a for a, b in zip(stepped, stepped[1:])]
    assert max(jumps) == APPROX(1.0)
    assert max(a - b for a, b in zip(smooth, smooth[1:])) < max(jumps)


def test_thresholded_metric_switches_exactly_once():
    budgets = [10.0 ** k for k in range(18, 27)]
    _, stepped = emergence_curves(budgets, 2.0)
    switches = sum(1 for a, b in zip(stepped, stepped[1:]) if a != b)
    assert switches == 1


def test_impossible_threshold_never_fires():
    """Порог ниже E не берётся ни на каком бюджете — метрика остаётся нулевой."""
    _, stepped = emergence_curves([10.0 ** k for k in range(18, 31)], 1.5)
    assert set(stepped) == {0.0}
