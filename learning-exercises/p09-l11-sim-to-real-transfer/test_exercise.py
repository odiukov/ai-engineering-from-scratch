"""Тесты к уроку «Sim-to-real transfer». Правь exercise.py."""

import random

import pytest

from exercise import (
    epsilon_greedy,
    evaluate,
    perpendicular,
    randomize,
    slip_step,
    sweep,
    train_q,
    widen_range,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ACTIONS = ("up", "down", "left", "right")
START = (3, 0)
GOAL = (3, 5)
CLIFF = {(3, 1), (3, 2), (3, 3), (3, 4)}
ABOVE_CLIFF = {(2, 1), (2, 2), (2, 3), (2, 4)}

_CACHE = {}


def _policy(slip_range):
    """Обучаем один раз на весь модуль: тесты читают один и тот же прогон."""
    if slip_range not in _CACHE:
        _CACHE[slip_range] = train_q(slip_range, rng=random.Random(0))
    return _CACHE[slip_range]


def greedy_path(Q, max_steps=25):
    """Куда ведёт жадная политика при нулевом slip."""
    rng = random.Random(0)
    state = START
    path = [state]
    for _ in range(max_steps):
        row = Q.get(state, {a: 0.0 for a in ACTIONS})
        action = epsilon_greedy(row, rng, 0.0)
        state, _, done = slip_step(state, action, 0.0, rng)
        path.append(state)
        if done:
            break
    return path


# ---------------------------------------------------------- perpendicular
def test_vertical_actions_slip_sideways():
    assert set(perpendicular("up")) == {"left", "right"}
    assert set(perpendicular("down")) == {"left", "right"}


def test_horizontal_actions_slip_vertically():
    assert set(perpendicular("left")) == {"up", "down"}
    assert set(perpendicular("right")) == {"up", "down"}


def test_an_action_is_never_perpendicular_to_itself():
    for action in ACTIONS:
        assert action not in perpendicular(action)


def test_perpendicular_order_is_stable():
    """rng.choice по этому кортежу обязан быть воспроизводим."""
    assert perpendicular("up") == perpendicular("up")


# ------------------------------------------------------------- slip_step
def test_an_ordinary_step_costs_one():
    state, reward, done = slip_step((2, 0), "right", 0.0, random.Random(0))
    assert (state, reward, done) == ((2, 1), -1.0, False)


def test_falling_into_the_cliff_costs_twenty_and_teleports_home():
    """Обрыв не завершает эпизод — он возвращает агента на старт с долгом."""
    state, reward, done = slip_step((2, 1), "down", 0.0, random.Random(0))
    assert state == START
    assert reward == APPROX(-20.0)
    assert done is False


def test_reaching_the_goal_ends_the_episode():
    state, reward, done = slip_step((2, 5), "down", 0.0, random.Random(0))
    assert (state, reward, done) == (GOAL, -1.0, True)


def test_walls_hold_the_agent_inside_the_grid():
    assert slip_step((0, 0), "up", 0.0, random.Random(0))[0] == (0, 0)
    assert slip_step((0, 0), "left", 0.0, random.Random(0))[0] == (0, 0)
    assert slip_step((1, 5), "right", 0.0, random.Random(0))[0] == (1, 5)


def test_zero_slip_is_fully_deterministic():
    rng = random.Random(5)
    assert all(slip_step((1, 1), "right", 0.0, rng)[0] == (1, 2) for _ in range(50))


def test_slip_one_always_derails_the_action():
    """slip = 1.0 — моторы не слушаются вообще: агент уходит только вбок."""
    rng = random.Random(5)
    landed = {slip_step((1, 1), "right", 1.0, rng)[0] for _ in range(50)}
    assert landed == {(0, 1), (2, 1)}


def test_the_slip_dice_is_rolled_even_when_slip_is_zero():
    """Ловушка: иначе один seed даст разные прогоны на разных slip.

    Если кубик бросается только при slip > 0, поток rng расходится, и
    сравнение двух политик на одном seed перестаёт быть честным.
    """
    control = random.Random(9)
    control.random()  # ровно один брошенный кубик
    expected = control.random()

    rng = random.Random(9)
    slip_step((1, 1), "right", 0.0, rng)
    assert rng.random() == expected


# ------------------------------------------------------------- randomize
def test_randomize_samples_inside_every_range():
    rng = random.Random(0)
    for _ in range(100):
        params = randomize(rng, {"slip": (0.1, 0.3), "mass": (1.0, 2.0)})
        assert 0.1 <= params["slip"] <= 0.3
        assert 1.0 <= params["mass"] <= 2.0


def test_a_degenerate_range_returns_its_single_value():
    assert randomize(random.Random(0), {"slip": (0.2, 0.2)})["slip"] == APPROX(0.2)


def test_randomize_is_reproducible_from_the_same_seed():
    ranges = {"slip": (0.0, 0.4), "mass": (1.0, 5.0)}
    a = [randomize(random.Random(3), ranges) for _ in range(3)]
    b = [randomize(random.Random(3), ranges) for _ in range(3)]
    assert a == b


def test_a_wider_range_produces_a_wider_spread():
    """Ровно то, за что платят в DR: разброс параметров растёт вместе с диапазоном."""

    def spread(low, high):
        rng = random.Random(0)
        xs = [randomize(rng, {"slip": (low, high)})["slip"] for _ in range(400)]
        return max(xs) - min(xs)

    assert spread(0.0, 0.05) < spread(0.0, 0.3) < spread(0.0, 0.9)


# --------------------------------------------------------- epsilon_greedy
def test_epsilon_zero_always_takes_the_best_action():
    row = {"up": 1.0, "down": 5.0, "left": -2.0, "right": 0.0}
    assert all(epsilon_greedy(row, random.Random(s), 0.0) == "down" for s in range(20))


def test_epsilon_one_eventually_tries_every_action():
    row = {a: 0.0 for a in ACTIONS}
    rng = random.Random(0)
    assert {epsilon_greedy(row, rng, 1.0) for _ in range(200)} == set(ACTIONS)


def test_epsilon_greedy_is_reproducible_from_the_same_seed():
    row = {"up": 1.0, "down": 5.0, "left": -2.0, "right": 0.0}
    a = [epsilon_greedy(row, rng, 0.5) for rng in [random.Random(1)] for _ in range(40)]
    b = [epsilon_greedy(row, rng, 0.5) for rng in [random.Random(1)] for _ in range(40)]
    assert a == b


# --------------------------------------------------------------- train_q
def test_narrow_training_finds_the_shortest_path():
    """slip = 0 в «симуляторе» — и политика идёт вплотную над обрывом."""
    assert greedy_path(_policy((0.0, 0.0))) == [
        START, (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), GOAL
    ]


def test_domain_randomization_learns_a_detour_away_from_the_cliff():
    """DR-политика платит лишними шагами за то, что скольжение её не убьёт."""
    dr_path = greedy_path(_policy((0.0, 0.3)))
    assert dr_path[-1] == GOAL
    assert not (set(dr_path) & ABOVE_CLIFF)
    assert len(dr_path) > 8


def test_training_never_leaves_the_agent_standing_in_the_cliff():
    Q = _policy((0.0, 0.3))
    assert not (set(Q) & CLIFF)


def test_training_is_reproducible_from_the_same_seed():
    a = train_q((0.0, 0.3), episodes=200, rng=random.Random(4))
    b = train_q((0.0, 0.3), episodes=200, rng=random.Random(4))
    assert a == b


# -------------------------------------------------------------- evaluate
def test_the_narrow_policy_is_optimal_in_the_domain_it_was_trained_on():
    assert evaluate(_policy((0.0, 0.0)), 0.0, random.Random(1)) == APPROX(-7.0)


def test_domain_randomization_costs_something_at_home():
    """Честная цена робастности: без скольжения объезд просто длиннее."""
    narrow = evaluate(_policy((0.0, 0.0)), 0.0, random.Random(1))
    dr = evaluate(_policy((0.0, 0.3)), 0.0, random.Random(1))
    assert dr < narrow


def test_domain_randomization_wins_by_a_lot_out_of_distribution():
    """Главный результат урока: на «железе» узкая политика рассыпается."""
    narrow = evaluate(_policy((0.0, 0.0)), 0.4, random.Random(1))
    dr = evaluate(_policy((0.0, 0.3)), 0.4, random.Random(1))
    assert dr > narrow
    assert dr > 2 * narrow  # награды отрицательные: разрыв больше двух раз


def test_return_degrades_as_the_real_slip_grows():
    Q = _policy((0.0, 0.3))
    rng = random.Random(1)
    assert evaluate(Q, 0.0, rng) > evaluate(Q, 0.3, rng) > evaluate(Q, 0.6, rng)


def test_evaluate_is_reproducible_from_the_same_seed():
    Q = _policy((0.0, 0.3))
    assert evaluate(Q, 0.3, random.Random(2), episodes=30) == APPROX(
        evaluate(Q, 0.3, random.Random(2), episodes=30)
    )


# ----------------------------------------------------------------- sweep
def test_sweep_reports_one_number_per_slip():
    got = sweep(_policy((0.0, 0.3)), [0.0, 0.2, 0.5], random.Random(1), episodes=20)
    assert sorted(got) == [0.0, 0.2, 0.5]


def test_sweep_is_monotone_for_a_sane_policy():
    got = sweep(_policy((0.0, 0.3)), [0.0, 0.3, 0.6], random.Random(1), episodes=100)
    assert got[0.0] > got[0.3] > got[0.6]


def test_sweep_of_an_empty_slip_list_is_empty():
    assert sweep(_policy((0.0, 0.0)), [], random.Random(1)) == {}


# ----------------------------------------------------------- widen_range
def test_a_successful_policy_earns_a_wider_range():
    assert widen_range((0.0, 0.1), -9.0, -12.0) == pytest.approx((0.0, 0.15))


def test_a_failing_policy_keeps_the_range_it_has():
    """Знак важен: награды отрицательные, «справилась» это score БОЛЬШЕ порога."""
    assert widen_range((0.0, 0.1), -30.0, -12.0) == pytest.approx((0.0, 0.1))


def test_the_range_never_grows_past_the_cap():
    assert widen_range((0.0, 0.88), -9.0, -12.0, cap=0.9) == pytest.approx((0.0, 0.9))


def test_the_lower_bound_never_moves():
    assert widen_range((0.1, 0.2), -1.0, -5.0)[0] == APPROX(0.1)
    assert widen_range((0.1, 0.2), -99.0, -5.0)[0] == APPROX(0.1)


def test_repeated_success_walks_the_curriculum_up_to_the_cap():
    r = (0.0, 0.0)
    for _ in range(50):
        r = widen_range(r, -9.0, -12.0, step=0.05, cap=0.3)
    assert r == pytest.approx((0.0, 0.3))
