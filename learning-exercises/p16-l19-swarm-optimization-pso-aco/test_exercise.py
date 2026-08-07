"""Тесты к уроку «Роевая оптимизация: PSO и ACO». Правь exercise.py."""

import itertools
import math
import random

import pytest

from exercise import (
    pso_step,
    pso_velocity,
    rastrigin,
    run_aco,
    run_pso,
    tour_length,
    transition_probabilities,
    update_pheromone,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# pytest.approx не умеет вложенные списки — матрицы сравниваем плоскими
flat = lambda m: [v for row in m for v in row]

CITIES = [(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0), (2.0, 6.0)]
DIST = [[math.dist(a, b) for b in CITIES] for a in CITIES]


def _brute_force_optimum():
    """Эталонный оптимум перебором. Своя реализация, из exercise ничего не берёт.

    Вызывать на этапе сбора тестов что-либо из exercise нельзя: заготовка
    упадёт при импорте, pytest покажет один error вместо N failed, и
    проверка «N failed == N passed» ничего не проверит.
    """
    best = math.inf
    for perm in itertools.permutations(range(1, 5)):
        tour = (0,) + perm
        best = min(best, sum(DIST[tour[k]][tour[(k + 1) % 5]] for k in range(5)))
    return best


OPTIMUM = _brute_force_optimum()

UNIFORM3 = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]


def particle(x, v, p_best=None, p_best_fit=math.inf):
    return {
        "x": list(x),
        "v": list(v),
        "p_best": list(p_best if p_best is not None else x),
        "p_best_fit": p_best_fit,
    }


# --------------------------------------------------------------- rastrigin
def test_rastrigin_is_zero_at_the_global_minimum():
    assert rastrigin([0.0, 0.0]) == APPROX(0.0)


def test_rastrigin_has_a_local_minimum_at_every_integer_point():
    """Ямы стоят решёткой: в целых точках значение равно сумме квадратов."""
    assert rastrigin([1.0]) == APPROX(1.0)
    assert rastrigin([2.0]) == APPROX(4.0)
    assert rastrigin([1.0, 1.0]) == APPROX(2.0)


def test_rastrigin_ridge_between_wells_is_high():
    """Полуцелая точка — гребень, а не яма: 20.25 против 0 и 1 рядом."""
    assert rastrigin([0.5]) == APPROX(20.25)


def test_rastrigin_grows_with_dimension_offset():
    """Смещение по одной координате не гасится другими."""
    assert rastrigin([0.0, 0.5]) > rastrigin([0.0, 0.0])


# ------------------------------------------------------------ pso_velocity
def test_pso_velocity_sums_inertia_cognitive_and_social():
    assert pso_velocity([0.0], [0.0], [1.0], [2.0], 0.5, 1.0, 1.0, 1.0, 1.0) == APPROX(
        [3.0]
    )


def test_pso_velocity_with_zero_social_weight_ignores_g_best():
    """c2=0 — рой перестаёт быть роем: частица не видит чужих находок."""
    a = pso_velocity([1.0], [0.0], [2.0], [99.0], 0.7, 1.5, 0.0, 0.5, 0.5)
    b = pso_velocity([1.0], [0.0], [2.0], [-99.0], 0.7, 1.5, 0.0, 0.5, 0.5)
    assert a == APPROX(b)


def test_pso_velocity_at_both_bests_keeps_only_inertia():
    assert pso_velocity([1.0], [5.0], [5.0], [5.0], 0.7, 1.5, 1.5, 0.5, 0.5) == APPROX(
        [0.7]
    )


def test_pso_velocity_with_zero_inertia_forgets_previous_motion():
    assert pso_velocity([100.0], [0.0], [0.0], [0.0], 0.0, 1.5, 1.5, 0.5, 0.5) == APPROX(
        [0.0]
    )


def test_pso_velocity_works_coordinatewise():
    v = pso_velocity([0.0, 0.0], [0.0, 0.0], [1.0, -1.0], [0.0, 0.0], 0.0, 1.0, 0.0, 1.0, 1.0)
    assert v == APPROX([1.0, -1.0])


# ---------------------------------------------------------------- pso_step
def test_pso_step_moves_position_by_velocity():
    swarm = [particle([0.0], [0.0], p_best=[0.0], p_best_fit=0.0)]
    rng = random.Random(0)
    out = pso_step(swarm, [10.0], lambda p: p[0] ** 2, [(-100.0, 100.0)], 1.0, 0.0, 1.0, rng)
    assert out[0]["x"][0] > 0.0


def test_pso_step_clamps_position_into_bounds():
    swarm = [particle([0.9], [0.0], p_best=[0.9], p_best_fit=0.0)]
    rng = random.Random(0)
    out = pso_step(swarm, [1000.0], lambda p: 0.0, [(0.0, 1.0)], 1.0, 0.0, 5.0, rng)
    assert 0.0 <= out[0]["x"][0] <= 1.0


def test_pso_step_updates_personal_best_only_on_improvement():
    swarm = [particle([5.0], [0.0], p_best=[5.0], p_best_fit=0.0)]
    rng = random.Random(1)
    out = pso_step(swarm, [0.0], lambda p: 42.0, [(-10.0, 10.0)], 0.5, 1.0, 1.0, rng)
    assert out[0]["p_best_fit"] == APPROX(0.0)
    assert out[0]["p_best"] == APPROX([5.0])


def test_pso_step_does_not_mutate_the_input_swarm():
    swarm = [particle([1.0], [1.0], p_best=[1.0], p_best_fit=1.0)]
    rng = random.Random(2)
    pso_step(swarm, [0.0], lambda p: p[0] ** 2, [(-10.0, 10.0)], 0.5, 1.0, 1.0, rng)
    assert swarm[0]["x"] == APPROX([1.0])
    assert swarm[0]["v"] == APPROX([1.0])


def test_pso_step_without_social_pull_is_independent_local_search():
    """c2=0 вырождает рой в набор независимых локальных поисков.

    Один и тот же рой, тот же seed, но совершенно разный g_best — результат
    обязан совпасть до последнего бита. Если не совпал, социальная
    компонента протекла туда, где её быть не должно.
    """
    swarm = [particle([1.0], [0.2], p_best=[1.0], p_best_fit=1.0),
             particle([-2.0], [0.1], p_best=[-2.0], p_best_fit=4.0)]
    fit = lambda p: p[0] ** 2
    a = pso_step(swarm, [50.0], fit, [(-10.0, 10.0)], 0.7, 1.5, 0.0, random.Random(7))
    b = pso_step(swarm, [-50.0], fit, [(-10.0, 10.0)], 0.7, 1.5, 0.0, random.Random(7))
    assert [p["x"][0] for p in a] == APPROX([p["x"][0] for p in b])


# ----------------------------------------------------------------- run_pso
def test_run_pso_finds_the_bottom_of_a_bowl():
    point, value = run_pso(lambda p: p[0] ** 2, [(-5.0, 5.0)], 20, 60, random.Random(0))
    assert value == pytest.approx(0.0, abs=1e-4)
    assert point[0] == pytest.approx(0.0, abs=1e-2)


def test_run_pso_escapes_local_minima_of_rastrigin():
    """Смысл роя: 30 частиц находят глобальную яму там, где спуск сел бы в первую."""
    _, value = run_pso(rastrigin, [(-5.12, 5.12)] * 2, 30, 150, random.Random(0))
    assert value < 1e-3


def test_run_pso_is_reproducible_for_the_same_seed():
    args = (rastrigin, [(-5.12, 5.12)] * 2, 10, 20)
    a = run_pso(*args, random.Random(3))
    b = run_pso(*args, random.Random(3))
    assert a[1] == APPROX(b[1])
    assert a[0] == APPROX(b[0])


def test_run_pso_never_reports_a_worse_value_than_it_found():
    """Возвращённая точка и её значение обязаны соответствовать друг другу."""
    point, value = run_pso(rastrigin, [(-5.12, 5.12)] * 2, 15, 40, random.Random(5))
    assert rastrigin(point) == APPROX(value)


# ------------------------------------------------------------- tour_length
def test_tour_length_closes_the_cycle():
    d = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    assert tour_length([0, 1, 2], d) == APPROX(6.0)


def test_tour_length_of_single_city_is_zero():
    assert tour_length([0], [[0, 1], [1, 0]]) == APPROX(0.0)


def test_tour_length_is_invariant_to_rotation():
    """Замкнутый цикл не имеет начала — сдвиг стартового города ничего не меняет."""
    assert tour_length([0, 1, 2, 3], DIST) == APPROX(tour_length([2, 3, 0, 1], DIST))


def test_tour_length_is_invariant_to_direction_on_symmetric_distances():
    assert tour_length([0, 1, 2, 3], DIST) == APPROX(tour_length([3, 2, 1, 0], DIST))


# --------------------------------------------- transition_probabilities
def test_transition_probabilities_sum_to_one():
    p = transition_probabilities([0, 1, 3, 2], [0, 1, 2, 4], [1, 2, 3])
    assert sum(p.values()) == APPROX(1.0)


def test_transition_probabilities_cover_only_allowed_cities():
    """Нормировка идёт по allowed. Посещённые города не должны всплывать."""
    p = transition_probabilities([0, 1, 3], [0, 1, 1], [1, 2])
    assert sorted(p) == [1, 2]


def test_transition_probabilities_with_beta_zero_follow_pheromone_only():
    assert transition_probabilities([0, 1, 3], [0, 1, 1], [1, 2], 1.0, 0.0) == {
        1: APPROX(0.25),
        2: APPROX(0.75),
    }


def test_transition_probabilities_with_alpha_zero_are_pure_greed():
    """alpha=0 — муравей без памяти: ближе значит вероятнее, феромон не важен."""
    assert transition_probabilities([0, 1, 3], [0, 1, 3], [1, 2], 0.0, 1.0) == {
        1: APPROX(0.75),
        2: APPROX(0.25),
    }


def test_transition_probabilities_reject_an_all_zero_row():
    """Ловушка нулевого tau0: сумма весов нулевая, делить не на что."""
    with pytest.raises(ValueError):
        transition_probabilities([0, 0, 0], [0, 1, 1], [1, 2], 1.0, 0.0)


# --------------------------------------------------------- update_pheromone
def test_update_pheromone_evaporates_every_edge():
    assert flat(update_pheromone(UNIFORM3, [], 0.5)) == APPROX(
        flat([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    )


def test_update_pheromone_deposits_symmetrically():
    got = update_pheromone(UNIFORM3, [([0, 1, 2], 0.5)], 0.5)
    assert flat(got) == APPROX(flat([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]))


def test_update_pheromone_gate_blocks_low_quality_runs():
    """Quality gate из AMRO-S: быстрый и неправильный агент не копит феромон."""
    gated = update_pheromone(UNIFORM3, [([0, 1, 2], 0.3)], 0.5, gate=0.6)
    assert flat(gated) == APPROX(flat(update_pheromone(UNIFORM3, [], 0.5)))


def test_update_pheromone_does_not_mutate_the_input_matrix():
    update_pheromone(UNIFORM3, [([0, 1, 2], 1.0)], 0.5)
    assert flat(UNIFORM3) == APPROX(flat([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))


def test_evaporation_lets_a_late_route_overtake_an_entrenched_one():
    """Без испарения колония залипает на первом найденном маршруте.

    Сценарий: 30 раз подкрепили старый маршрут, потом 5 раз новый.
    rho=0   — старое ребро всё ещё сильнее, колония заперта.
    rho=0.5 — старый след успел выветриться, новый маршрут перехватывает.
    """
    def run(rho):
        tau = [[0.0 if i == j else 1.0 for j in range(4)] for i in range(4)]
        for _ in range(30):
            tau = update_pheromone(tau, [([0, 1, 2, 3], 1.0)], rho)
        for _ in range(5):
            tau = update_pheromone(tau, [([0, 2, 1, 3], 1.0)], rho)
        return tau[0][1], tau[0][2]

    old_no_decay, new_no_decay = run(0.0)
    assert old_no_decay > new_no_decay

    old_decay, new_decay = run(0.5)
    assert new_decay > old_decay


# ----------------------------------------------------------------- run_aco
def test_run_aco_finds_the_optimal_tour_on_five_cities():
    _, length = run_aco(DIST, 8, 30, random.Random(0))
    assert length == APPROX(OPTIMUM)


def test_run_aco_returns_a_valid_permutation_of_all_cities():
    tour, _ = run_aco(DIST, 6, 10, random.Random(1))
    assert sorted(tour) == list(range(len(DIST)))


def test_run_aco_reports_the_length_of_the_tour_it_returns():
    tour, length = run_aco(DIST, 6, 15, random.Random(2))
    assert tour_length(tour, DIST) == APPROX(length)


def test_run_aco_is_reproducible_for_the_same_seed():
    a = run_aco(DIST, 6, 10, random.Random(4))
    b = run_aco(DIST, 6, 10, random.Random(4))
    assert a[0] == b[0]
    assert a[1] == APPROX(b[1])
