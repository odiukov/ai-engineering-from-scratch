"""Тесты к уроку «MARL: MADDPG, QMIX, MAPPO». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    assignment_cost,
    centralized_advantage,
    centralized_assignment,
    decentralized_argmax,
    independent_assignment,
    joint_argmax,
    mix,
    mix_gradient,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def numeric_mix_gradient(q_values, weights, bias, monotone=True, h=1e-6):
    """Численный градиент центральной разностью. Своя реализация, не из exercise."""
    grad = []
    for i in range(len(q_values)):
        up, down = list(q_values), list(q_values)
        up[i] += h
        down[i] -= h
        grad.append(
            (mix(up, weights, bias, monotone) - mix(down, weights, bias, monotone))
            / (2 * h)
        )
    return grad


# --------------------------------------------------------- assignment_cost
def test_assignment_cost_sums_manhattan_distances():
    assert assignment_cost([(0, 0), (2, 0)], [(3, 0), (10, 0)], [0, 1]) == APPROX(11.0)


def test_assignment_cost_is_infinite_when_a_pellet_stays_uncollected():
    """Два агента на одну цель — задача не выполнена, а не «выполнена дёшево»."""
    assert assignment_cost([(0, 0), (2, 0)], [(3, 0), (10, 0)], [0, 0]) == math.inf


def test_assignment_cost_counts_both_axes():
    assert assignment_cost([(0, 0)], [(3, 4)], [0]) == APPROX(7.0)


def test_assignment_cost_of_agents_already_on_targets_is_zero():
    assert assignment_cost([(1, 1), (2, 2)], [(1, 1), (2, 2)], [0, 1]) == APPROX(0.0)


# --------------------------------------------------- independent_assignment
def test_independent_assignment_picks_the_nearest_target_for_each_agent():
    assert independent_assignment([(0, 0), (9, 0)], [(1, 0), (8, 0)]) == ([0, 1], 2.0)


def test_independent_assignment_can_send_both_agents_to_the_same_target():
    """Локально оптимальный выбор — совместно недопустимый исход."""
    assignment, cost = independent_assignment([(0, 0), (2, 0)], [(3, 0), (10, 0)])
    assert assignment == [0, 0]
    assert cost == math.inf


def test_independent_assignment_breaks_ties_toward_the_lower_index():
    assignment, _ = independent_assignment([(0, 0)], [(1, 0), (0, 1)])
    assert assignment == [0]


def test_independent_assignment_ignores_the_other_agent_entirely():
    """Decentralized execution: сдвиг соседа не меняет мой выбор."""
    a, _ = independent_assignment([(0, 0), (9, 0)], [(1, 0), (8, 0)])
    b, _ = independent_assignment([(0, 0), (100, 0)], [(1, 0), (8, 0)])
    assert a[0] == b[0]


# --------------------------------------------------- centralized_assignment
def test_centralized_assignment_never_leaves_a_target_uncollected():
    _, cost = centralized_assignment([(0, 0), (2, 0)], [(3, 0), (10, 0)])
    assert cost == APPROX(11.0)


def test_centralized_assignment_matches_the_obvious_pairing():
    assert centralized_assignment([(0, 0), (9, 0)], [(1, 0), (8, 0)]) == ([0, 1], 2.0)


def test_centralized_assignment_is_never_worse_than_independent():
    """Централизованное обучение видит всё — хуже локального быть не может."""
    rng = random.Random(11)
    for _ in range(50):
        starts = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(3)]
        pellets = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(3)]
        assert centralized_assignment(starts, pellets)[1] <= (
            independent_assignment(starts, pellets)[1]
        )


def test_independent_assignment_is_either_invalid_or_already_optimal():
    """Жадность по отдельности решает РАССЦЕПЛЁННУЮ задачу точно.

    Пока связывающее ограничение не мешает, независимый выбор совпадает с
    централизованным. Всё расхождение живёт ровно в ограничении — это и
    есть граница применимости value decomposition.
    """
    rng = random.Random(12)
    for _ in range(50):
        starts = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(3)]
        pellets = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(3)]
        ind = independent_assignment(starts, pellets)[1]
        cen = centralized_assignment(starts, pellets)[1]
        assert ind == math.inf or ind == APPROX(cen)


# --------------------------------------------------------------------- mix
def test_mix_is_a_weighted_sum_plus_bias():
    assert mix([1.0, 2.0], [0.5, 2.0], 0.5) == APPROX(5.0)


def test_mix_takes_weights_by_absolute_value_when_monotone():
    assert mix([1.0, 2.0], [0.5, -2.0], 0.0) == APPROX(4.5)


def test_mix_keeps_the_sign_of_weights_when_not_monotone():
    assert mix([1.0, 2.0], [0.5, -2.0], 0.0, monotone=False) == APPROX(-3.5)


def test_mix_is_increasing_in_every_individual_q():
    """Монотонность в лоб: подняли Q_i — Q_tot не упал."""
    base = mix([1.0, 1.0], [0.3, 1.7], -2.0)
    assert mix([1.5, 1.0], [0.3, 1.7], -2.0) > base
    assert mix([1.0, 1.5], [0.3, 1.7], -2.0) > base


# ------------------------------------------------------------ mix_gradient
def test_mix_gradient_matches_the_numeric_central_difference():
    q, w, b = [1.0, 2.0, -3.0], [0.5, -2.0, 1.25], 0.7
    assert mix_gradient(q, w) == pytest.approx(
        numeric_mix_gradient(q, w, b), abs=1e-6
    )


def test_mix_gradient_matches_numeric_difference_without_monotonicity():
    q, w, b = [1.0, 2.0, -3.0], [0.5, -2.0, 1.25], 0.7
    assert mix_gradient(q, w, monotone=False) == pytest.approx(
        numeric_mix_gradient(q, w, b, monotone=False), abs=1e-6
    )


def test_mix_gradient_is_non_negative_under_monotonicity():
    """Всё требование QMIX умещается в эту строку."""
    assert all(g >= 0 for g in mix_gradient([9.0] * 4, [-3.0, 0.0, 2.5, -0.1]))


def test_mix_gradient_can_go_negative_without_monotonicity():
    assert any(g < 0 for g in mix_gradient([9.0, 9.0], [1.0, -1.0], monotone=False))


def test_mix_gradient_does_not_depend_on_the_q_values():
    assert mix_gradient([0.0, 0.0], [1.0, 2.0]) == APPROX(
        mix_gradient([100.0, -100.0], [1.0, 2.0])
    )


# ------------------------------------- joint_argmax / decentralized_argmax
def test_joint_argmax_enumerates_the_whole_joint_action_space():
    assert joint_argmax([[0.0, 1.0], [0.0, 1.0]], [1.0, 1.0], 0.0) == (1, 1)


def test_decentralized_argmax_looks_only_at_its_own_row():
    assert decentralized_argmax([[0.0, 1.0], [3.0, 2.0]]) == (1, 0)


def test_decentralized_argmax_handles_a_single_agent_with_one_action():
    assert decentralized_argmax([[5.0]]) == (0,)


def test_monotone_mixing_preserves_the_joint_argmax():
    """Ради этого свойства QMIX и жмёт веса через abs.

    Совместный argmax по Q_tot совпадает с покоординатным argmax по Q_i —
    значит агентам не нужно согласовывать действия на исполнении.
    """
    rng = random.Random(21)
    for _ in range(40):
        q_tables = [[rng.uniform(-5, 5) for _ in range(4)] for _ in range(3)]
        weights = [rng.uniform(-3, 3) for _ in range(3)]
        bias = rng.uniform(-10, 10)
        assert joint_argmax(q_tables, weights, bias) == decentralized_argmax(q_tables)


def test_a_negative_mixing_weight_breaks_decentralized_execution():
    """Контрпример: без монотонности агенты, выбирая порознь, промахиваются."""
    q_tables = [[0.0, 1.0], [0.0, 1.0]]
    assert joint_argmax(q_tables, [1.0, -1.0], 0.0, monotone=False) == (1, 0)
    assert decentralized_argmax(q_tables) == (1, 1)


def test_joint_argmax_ignores_the_bias():
    """Постоянная добавка не двигает argmax — сдвигает только шкалу Q_tot."""
    q_tables = [[0.2, 0.9], [0.7, 0.1]]
    assert joint_argmax(q_tables, [1.0, 4.0], 0.0) == joint_argmax(
        q_tables, [1.0, 4.0], 1000.0
    )


# --------------------------------------------------- centralized_advantage
def test_centralized_advantage_subtracts_the_mean_by_default():
    assert centralized_advantage([1.0, 2.0, 3.0]) == APPROX([-1.0, 0.0, 1.0])


def test_centralized_advantage_accepts_an_explicit_baseline():
    assert centralized_advantage([1.0, 2.0, 3.0], 0.0) == APPROX([1.0, 2.0, 3.0])


def test_centralized_advantage_sums_to_zero():
    assert sum(centralized_advantage([4.0, -1.0, 0.5, 9.0])) == APPROX(0.0)


def test_centralized_baseline_minimises_the_squared_magnitude():
    """Ровно то гашение дисперсии, ради которого MAPPO ставит централизованную V."""
    returns = [4.0, -1.0, 0.5, 9.0]
    centred = sum(a * a for a in centralized_advantage(returns))
    for other in (0.0, 1.0, -3.0, 10.0):
        assert centred <= sum(a * a for a in centralized_advantage(returns, other))


def test_centralized_advantage_of_empty_input_is_empty():
    """Пустой батч не должен делить на ноль."""
    assert centralized_advantage([]) == []
