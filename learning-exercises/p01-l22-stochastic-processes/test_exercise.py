"""Тесты к уроку «Случайные процессы». Правь exercise.py."""

import random
import statistics

import pytest

from exercise import (
    brownian_motion,
    distribution_after_n_steps,
    empirical_distribution,
    langevin_dynamics,
    random_walk_1d,
    simulate_markov_chain,
    stationary_distribution,
    step_distribution,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# погода из урока: солнечно (0), дождь (1), облачно (2)
WEATHER = [[0.7, 0.1, 0.2],
           [0.3, 0.4, 0.3],
           [0.4, 0.2, 0.4]]

STICKY = [[0.9, 0.1], [0.5, 0.5]]          # обычная цепь из двух состояний
ABSORBING = [[1.0, 0.0], [0.5, 0.5]]       # состояние 0 не отпускает
SWAP = [[0.0, 1.0], [1.0, 0.0]]            # детерминированный перескок


# --------------------------------------------------------- step_distribution
def test_step_distribution_of_a_point_mass_is_the_matrix_row():
    """Вся масса в состоянии 0 — ответ это ровно строка 0 матрицы."""
    assert step_distribution([1.0, 0.0], STICKY) == APPROX([0.9, 0.1])


def test_step_distribution_mixes_two_states():
    assert step_distribution([0.5, 0.5], STICKY) == APPROX([0.7, 0.3])


def test_step_distribution_stays_a_distribution():
    """Строки P суммируются в 1, значит и результат обязан суммироваться в 1."""
    result = step_distribution([0.2, 0.5, 0.3], WEATHER)
    assert sum(result) == APPROX(1.0)
    assert all(p >= 0.0 for p in result)


def test_step_distribution_uses_rows_not_columns():
    """Ловушка: если суммировать по столбцам, ответ перестанет быть распределением."""
    result = step_distribution([1.0, 0.0], STICKY)
    transposed = [[STICKY[j][i] for j in range(2)] for i in range(2)]
    assert result != pytest.approx(step_distribution([1.0, 0.0], transposed))
    assert sum(result) == APPROX(1.0)


def test_step_distribution_does_not_mutate_the_input():
    dist = [0.25, 0.75]
    step_distribution(dist, STICKY)
    assert dist == [0.25, 0.75]


# ------------------------------------------------- distribution_after_n_steps
def test_zero_steps_returns_the_same_numbers_in_a_new_list():
    """При n = 0 значения те же, но список обязан быть новым."""
    dist = [1.0, 0.0]
    out = distribution_after_n_steps(dist, STICKY, 0)
    assert out == APPROX([1.0, 0.0])
    assert out is not dist


def test_two_steps_equal_two_single_steps():
    once = step_distribution([1.0, 0.0], STICKY)
    twice = step_distribution(once, STICKY)
    assert distribution_after_n_steps([1.0, 0.0], STICKY, 2) == APPROX(twice)
    assert twice == APPROX([0.86, 0.14])


def test_many_steps_still_sum_to_one():
    """Ловушка: ошибка округления копится, но распределением остаться обязано."""
    out = distribution_after_n_steps([1.0, 0.0, 0.0], WEATHER, 500)
    assert sum(out) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------- stationary_distribution
def test_stationary_distribution_is_a_fixed_point():
    """Главное свойство: ещё один шаг ничего не меняет, pi = pi * P."""
    pi = stationary_distribution(WEATHER)
    assert step_distribution(pi, WEATHER) == pytest.approx(pi, abs=1e-9)


def test_stationary_distribution_of_the_two_state_chain():
    assert stationary_distribution(STICKY) == pytest.approx([5 / 6, 1 / 6], abs=1e-9)


def test_stationary_distribution_sums_to_one():
    pi = stationary_distribution(WEATHER)
    assert sum(pi) == pytest.approx(1.0, abs=1e-12)


def test_absorbing_state_takes_all_the_mass():
    """Из поглощающего состояния выхода нет — вся вероятность стекает туда."""
    assert stationary_distribution(ABSORBING) == pytest.approx([1.0, 0.0], abs=1e-6)


def test_stationary_matches_the_long_run_power_iteration():
    """Тот же ответ должен получаться простым прогоном многих шагов из угла."""
    long_run = distribution_after_n_steps([1.0, 0.0, 0.0], WEATHER, 500)
    assert stationary_distribution(WEATHER) == pytest.approx(long_run, abs=1e-9)


def test_stationary_distribution_converges_for_a_periodic_chain():
    """Ленивая итерация гасит цикл между долями двудольной цепи."""
    periodic = [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert stationary_distribution(periodic) == pytest.approx([0.5, 0.25, 0.25], abs=1e-9)


def test_stationary_distribution_reports_an_exhausted_iteration_budget():
    # Контраст не даёт NotImplementedError из заготовки притвориться ожидаемым
    # RuntimeError: рабочий вызов обязан сначала вернуть настоящий ответ.
    pi = stationary_distribution(WEATHER)
    assert step_distribution(pi, WEATHER) == pytest.approx(pi, abs=1e-9)
    with pytest.raises(RuntimeError, match="не сошлось"):
        stationary_distribution(WEATHER, tol=1e-15, max_steps=1)


# ---------------------------------------------------- empirical_distribution
def test_empirical_distribution_counts_shares():
    assert empirical_distribution([0, 1, 1, 1], 2) == APPROX([0.25, 0.75])


def test_unvisited_state_still_gets_a_slot():
    """Ловушка: длина ответа задаётся n_states, а не набором встреченных состояний."""
    out = empirical_distribution([0, 0, 0], 3)
    assert len(out) == 3
    assert out == APPROX([1.0, 0.0, 0.0])


def test_empty_trajectory_raises():
    with pytest.raises(ValueError):
        empirical_distribution([], 3)


# --------------------------------------------------- simulate_markov_chain
def test_trajectory_length_is_n_steps_plus_one():
    traj = simulate_markov_chain(WEATHER, 0, 10, random.Random(1))
    assert len(traj) == 11


def test_trajectory_starts_at_the_start_state_and_stays_in_range():
    traj = simulate_markov_chain(WEATHER, 2, 50, random.Random(2))
    assert traj[0] == 2
    assert all(0 <= s < 3 for s in traj)


def test_absorbing_state_is_never_left():
    traj = simulate_markov_chain(ABSORBING, 0, 30, random.Random(3))
    assert traj == [0] * 31


def test_deterministic_chain_alternates():
    """Вероятности 0 и 1 не оставляют выбора: 0, 1, 0, 1, ..."""
    traj = simulate_markov_chain(SWAP, 0, 5, random.Random(4))
    assert traj == [0, 1, 0, 1, 0, 1]


def test_same_seed_reproduces_the_trajectory_and_different_seeds_do_not():
    a = simulate_markov_chain(WEATHER, 0, 100, random.Random(9))
    b = simulate_markov_chain(WEATHER, 0, 100, random.Random(9))
    c = simulate_markov_chain(WEATHER, 0, 100, random.Random(10))
    assert a == b
    assert a != c


def test_long_run_frequencies_converge_to_the_stationary_distribution():
    """Смысловая проверка: симуляция и теория обязаны сойтись."""
    traj = simulate_markov_chain(WEATHER, 0, 200000, random.Random(12345))
    empirical = empirical_distribution(traj, 3)
    assert empirical == pytest.approx(stationary_distribution(WEATHER), abs=0.02)


# --------------------------------------------------------- random_walk_1d
def test_walk_starts_at_zero_and_records_every_position():
    walk = random_walk_1d(20, random.Random(1))
    assert walk[0] == 0
    assert len(walk) == 21


def test_every_step_moves_by_exactly_one():
    walk = random_walk_1d(200, random.Random(2))
    assert all(abs(b - a) == 1 for a, b in zip(walk, walk[1:]))


def test_same_seed_reproduces_the_walk_and_different_seeds_do_not():
    assert random_walk_1d(50, random.Random(5)) == random_walk_1d(50, random.Random(5))
    assert random_walk_1d(50, random.Random(5)) != random_walk_1d(50, random.Random(6))


def test_mean_final_position_is_near_zero():
    """Блуждание честное: сноса нет, среднее смещение около нуля."""
    finals = [random_walk_1d(100, random.Random(1000 + i))[-1] for i in range(500)]
    assert abs(statistics.fmean(finals)) < 3.0


def test_random_walk_variance_grows_linearly_with_time():
    """Дисперсия позиции после n шагов равна n, то есть расстояние растёт как sqrt(n)."""
    short = [random_walk_1d(100, random.Random(1000 + i))[-1] for i in range(500)]
    long = [random_walk_1d(400, random.Random(1000 + i))[-1] for i in range(500)]
    assert statistics.variance(short) == pytest.approx(100, rel=0.3)
    assert statistics.variance(long) == pytest.approx(400, rel=0.3)


# -------------------------------------------------------- brownian_motion
def test_brownian_path_starts_at_zero_with_the_right_length():
    path = brownian_motion(100, 0.01, random.Random(1))
    assert path[0] == 0.0
    assert len(path) == 101


def test_brownian_variance_equals_elapsed_time():
    """B(t) ~ N(0, t): за время 1 дисперсия конечной точки около 1."""
    rng = random.Random(3)
    ends = [brownian_motion(100, 0.01, rng)[-1] for _ in range(600)]
    assert statistics.fmean(ends) == pytest.approx(0.0, abs=0.2)
    assert statistics.variance(ends) == pytest.approx(1.0, rel=0.25)


def test_brownian_scaling_is_sqrt_dt_not_dt():
    """Ловушка: вчетверо мельче шаг при вчетверо большем числе шагов — та же дисперсия."""
    rng = random.Random(3)
    coarse = [brownian_motion(100, 0.01, rng)[-1] for _ in range(600)]
    rng = random.Random(3)
    fine = [brownian_motion(400, 0.0025, rng)[-1] for _ in range(600)]
    assert statistics.variance(fine) == pytest.approx(statistics.variance(coarse), rel=0.3)


def test_same_seed_reproduces_the_brownian_path():
    assert brownian_motion(50, 0.01, random.Random(8)) == brownian_motion(
        50, 0.01, random.Random(8)
    )


# ------------------------------------------------------ langevin_dynamics
def test_langevin_trajectory_length_and_start():
    traj = langevin_dynamics(lambda x: x, 2.0, 0.01, 1.0, 100, random.Random(1))
    assert len(traj) == 101
    assert traj[0] == APPROX(2.0)


def test_zero_temperature_is_plain_gradient_descent():
    """Без шума остаётся спуск: скатывается в минимум U(x) = (x - 3)^2 / 2."""
    traj = langevin_dynamics(lambda x: x - 3.0, 0.0, 0.1, 0.0, 300, random.Random(5))
    assert traj[-1] == pytest.approx(3.0, abs=1e-6)


def test_langevin_variance_matches_the_temperature():
    """Для U(x) = x^2 / 2 равновесие это N(0, T): дисперсия хвоста около T."""
    traj = langevin_dynamics(lambda x: x, 0.0, 0.05, 1.0, 40000, random.Random(11))
    tail = traj[8000:]  # выбрасываем разогрев
    assert statistics.fmean(tail) == pytest.approx(0.0, abs=0.2)
    assert statistics.variance(tail) == pytest.approx(1.0, rel=0.25)


def test_higher_temperature_spreads_wider():
    cold = langevin_dynamics(lambda x: x, 0.0, 0.05, 0.25, 40000, random.Random(11))
    hot = langevin_dynamics(lambda x: x, 0.0, 0.05, 1.0, 40000, random.Random(11))
    assert statistics.variance(cold[8000:]) < statistics.variance(hot[8000:])
