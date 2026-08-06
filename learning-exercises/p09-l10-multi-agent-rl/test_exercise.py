"""Тесты к уроку «Multi-agent RL». Правь exercise.py."""

import random

import pytest

from exercise import (
    counterfactual_advantage,
    epsilon_greedy,
    joint_actions,
    joint_step,
    move,
    q_learning_update,
    train_independent_q,
    train_joint_q,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

GRID = 4
GOAL = (GRID - 1, GRID - 1)
ACTIONS = ("up", "down", "left", "right")
START = ((0, 0), (GRID - 1, 0))
EPISODES = 800

_CACHE = {}


def _independent():
    """Обучаем один раз на весь модуль: тесты читают один и тот же прогон."""
    if "ind" not in _CACHE:
        _CACHE["ind"] = train_independent_q(episodes=EPISODES, rng=random.Random(0))
    return _CACHE["ind"]


def _joint():
    if "joint" not in _CACHE:
        _CACHE["joint"] = train_joint_q(episodes=EPISODES, rng=random.Random(0))
    return _CACHE["joint"]


def zeros(keys):
    return {k: 0.0 for k in keys}


def greedy_return(pick, max_steps=60):
    """Прогон жадной политики: pick(state) -> совместное действие."""
    state, total = START, 0.0
    for _ in range(max_steps):
        state, reward, done = joint_step(state, pick(state))
        total += reward
        if done:
            return total, True
    return total, False


def head_tail(log):
    n = len(log) // 5
    return sum(log[:n]) / n, sum(log[-n:]) / n


# ------------------------------------------------------------------- move
def test_move_goes_one_cell_in_each_direction():
    assert move((1, 1), "up") == (0, 1)
    assert move((1, 1), "down") == (2, 1)
    assert move((1, 1), "left") == (1, 0)
    assert move((1, 1), "right") == (1, 2)


def test_move_is_blocked_by_the_top_left_walls():
    """Без обрезки агент уедет в (-1, 0) и засорит Q-таблицу небывшими клетками."""
    assert move((0, 0), "up") == (0, 0)
    assert move((0, 0), "left") == (0, 0)


def test_move_is_blocked_by_the_bottom_right_walls():
    assert move((GRID - 1, GRID - 1), "down") == (GRID - 1, GRID - 1)
    assert move((GRID - 1, GRID - 1), "right") == (GRID - 1, GRID - 1)


def test_move_respects_a_custom_grid_size():
    assert move((1, 1), "down", size=2) == (1, 1)


# ------------------------------------------------------------- joint_step
def test_every_ordinary_step_costs_one():
    _, reward, done = joint_step(((0, 0), (3, 0)), ("down", "right"))
    assert reward == APPROX(-1.0)
    assert done is False


def test_both_agents_move_independently_on_the_same_step():
    state, _, _ = joint_step(((0, 0), (3, 0)), ("down", "right"))
    assert state == ((1, 0), (3, 1))


def test_the_bonus_arrives_only_when_both_agents_stand_on_the_goal():
    state, reward, done = joint_step(((2, 3), (3, 2)), ("down", "right"))
    assert state == (GOAL, GOAL)
    assert reward == APPROX(10.0)
    assert done is True


def test_one_agent_on_the_goal_is_not_enough():
    """Из этого условия и рождается кооперация: пришедший первым обязан ждать."""
    _, reward, done = joint_step(((2, 3), (0, 0)), ("down", "down"))
    assert done is False
    assert reward == APPROX(-1.0)


def test_an_agent_can_wait_on_the_goal_by_pushing_into_a_wall():
    """Ждать нечем, кроме упора в стену, — и это должно работать."""
    state, _, _ = joint_step((GOAL, (0, 0)), ("right", "down"))
    assert state[0] == GOAL


# ---------------------------------------------------------- joint_actions
def test_joint_action_space_is_the_cartesian_power():
    assert joint_actions(("a", "b"), 2) == [
        ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")
    ]


def test_joint_action_space_grows_exponentially_in_the_number_of_agents():
    """|A|^n — главный барьер MARL, и он виден уже на трёх агентах."""
    assert len(joint_actions(("a", "b"), 2)) == 4
    assert len(joint_actions(("a", "b"), 3)) == 8
    assert len(joint_actions(ACTIONS, 2)) == 16


def test_joint_actions_has_no_duplicates():
    js = joint_actions(ACTIONS, 2)
    assert len(set(js)) == len(js)


# --------------------------------------------------------- epsilon_greedy
def test_epsilon_zero_always_takes_the_best_action():
    row = {"a": 1.0, "b": 5.0, "c": -2.0}
    assert all(epsilon_greedy(row, random.Random(s), 0.0) == "b" for s in range(20))


def test_epsilon_one_eventually_tries_every_action():
    row = {"a": 1.0, "b": 5.0, "c": -2.0}
    rng = random.Random(0)
    seen = {epsilon_greedy(row, rng, 1.0) for _ in range(200)}
    assert seen == {"a", "b", "c"}


def test_epsilon_greedy_never_invents_an_action():
    rng = random.Random(3)
    row = zeros(ACTIONS)
    assert all(epsilon_greedy(row, rng, 0.5) in ACTIONS for _ in range(100))


def test_epsilon_greedy_is_reproducible_from_the_same_seed():
    """Без rng-параметра расходящийся MARL-прогон невозможно повторить."""
    row = {"a": 1.0, "b": 5.0, "c": -2.0}
    first = [epsilon_greedy(row, random.Random(7), 0.5) for _ in range(1)]
    rng_a, rng_b = random.Random(7), random.Random(7)
    seq_a = [epsilon_greedy(row, rng_a, 0.5) for _ in range(50)]
    seq_b = [epsilon_greedy(row, rng_b, 0.5) for _ in range(50)]
    assert seq_a == seq_b
    assert first[0] == seq_a[0]


# ------------------------------------------------------ q_learning_update
def test_q_learning_update_worked_example():
    got = q_learning_update({"a": 0.0}, "a", 1.0, {"a": 0.0}, alpha=0.5, done=True)
    assert got["a"] == APPROX(0.5)


def test_q_learning_update_bootstraps_through_the_next_state():
    got = q_learning_update({"a": 0.0}, "a", 0.0, {"a": 10.0}, alpha=1.0, gamma=0.5)
    assert got["a"] == APPROX(5.0)


def test_a_terminal_step_does_not_bootstrap():
    """Ловушка: bootstrap за концом эпизода уводит значения в бесконечность."""
    row = {"a": 0.0}
    got = q_learning_update(row, "a", 1.0, {"a": 100.0}, alpha=1.0, gamma=0.9, done=True)
    assert got["a"] == APPROX(1.0)


def test_zero_alpha_changes_nothing():
    row = {"a": 3.0, "b": -1.0}
    assert q_learning_update(row, "a", 99.0, {"a": 99.0}, alpha=0.0) == APPROX(row)


def test_q_learning_update_touches_only_the_taken_action():
    row = {"a": 0.0, "b": 7.0}
    got = q_learning_update(row, "a", 1.0, zeros(("a", "b")), alpha=1.0, done=True)
    assert got["b"] == APPROX(7.0)


def test_q_learning_update_does_not_mutate_the_row_it_was_given():
    row = {"a": 0.0}
    q_learning_update(row, "a", 1.0, {"a": 0.0}, alpha=1.0, done=True)
    assert row == {"a": 0.0}


# --------------------------------------------------- train_independent_q
def test_independent_agents_improve_over_training():
    _, _, log = _independent()
    first, last = head_tail(log)
    assert last > first


def test_independent_greedy_policy_reaches_the_shared_goal():
    Q1, Q2, _ = _independent()
    rng = random.Random(1)

    def pick(state):
        r1 = Q1.get(state, zeros(ACTIONS))
        r2 = Q2.get(state, zeros(ACTIONS))
        return (epsilon_greedy(r1, rng, 0.0), epsilon_greedy(r2, rng, 0.0))

    total, done = greedy_return(pick)
    assert done is True
    assert total > 0.0


def test_each_independent_agent_keeps_its_own_table():
    """Обе таблицы ключуются одним joint state, но учат РАЗНЫЕ действия."""
    Q1, Q2, _ = _independent()
    assert Q1[START] != Q2[START]
    assert set(Q1[START]) == set(ACTIONS)


def test_independent_training_is_reproducible():
    _, _, log_a = train_independent_q(episodes=50, rng=random.Random(11))
    _, _, log_b = train_independent_q(episodes=50, rng=random.Random(11))
    assert log_a == log_b


# --------------------------------------------------------- train_joint_q
def test_joint_q_improves_over_training():
    _, log = _joint()
    first, last = head_tail(log)
    assert last > first


def test_joint_q_rows_are_keyed_by_joint_actions():
    """Цена правильного глобального взгляда: строка в |A|^n раз шире."""
    Q, _ = _joint()
    Q1, _, _ = _independent()
    assert set(Q[START]) == set(joint_actions(ACTIONS, 2))
    assert len(Q[START]) == len(Q1[START]) ** 2


def test_joint_greedy_policy_reaches_the_shared_goal():
    Q, _ = _joint()
    rng = random.Random(1)
    all_joint = joint_actions(ACTIONS, 2)

    def pick(state):
        return epsilon_greedy(Q.get(state, zeros(all_joint)), rng, 0.0)

    total, done = greedy_return(pick)
    assert done is True
    assert total > 0.0


def test_joint_training_is_reproducible():
    _, log_a = train_joint_q(episodes=50, rng=random.Random(11))
    _, log_b = train_joint_q(episodes=50, rng=random.Random(11))
    assert log_a == log_b


# ------------------------------------------------ counterfactual_advantage
def test_counterfactual_advantage_worked_example():
    q_row = {("a", "x"): 4.0, ("b", "x"): 0.0}
    got = counterfactual_advantage(q_row, ("a", "x"), 0, {"a": 0.5, "b": 0.5})
    assert got == APPROX(2.0)


def test_an_agent_that_could_not_change_the_outcome_gets_no_credit():
    """Ровно это и решает credit assignment общей награды."""
    q_row = {("a", "x"): 7.0, ("b", "x"): 7.0}
    got = counterfactual_advantage(q_row, ("a", "x"), 0, {"a": 0.3, "b": 0.7})
    assert got == APPROX(0.0)


def test_the_best_action_gets_a_positive_advantage_and_the_worst_a_negative_one():
    q_row = {("a", "x"): 9.0, ("b", "x"): 1.0, ("c", "x"): 2.0}
    probs = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert counterfactual_advantage(q_row, ("a", "x"), 0, probs) > 0
    assert counterfactual_advantage(q_row, ("b", "x"), 0, probs) < 0


def test_advantage_ignores_columns_where_the_neighbour_acted_differently():
    """Baseline маргинализует только МОЁ действие, ход соседа заморожен."""
    base = {("a", "x"): 4.0, ("b", "x"): 0.0, ("a", "y"): 100.0, ("b", "y"): -100.0}
    probs = {"a": 0.5, "b": 0.5}
    assert counterfactual_advantage(base, ("a", "x"), 0, probs) == APPROX(2.0)


def test_advantage_works_for_the_second_agent_too():
    q_row = {("a", "x"): 4.0, ("a", "y"): 0.0}
    got = counterfactual_advantage(q_row, ("a", "y"), 1, {"x": 0.5, "y": 0.5})
    assert got == APPROX(-2.0)
