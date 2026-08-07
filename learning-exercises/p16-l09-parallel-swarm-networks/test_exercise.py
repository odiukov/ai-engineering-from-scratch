"""Тесты к уроку «Параллельные и роевые архитектуры». Правь exercise.py."""

import pytest

from exercise import (
    TOPOLOGIES,
    aging_order,
    build_topology,
    channel_count,
    hot_spot_ratio,
    is_connected,
    simulate_fixed,
    simulate_swarm,
    speedup,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# 4 медленные задачи по 0.4с и 4 быстрые по 0.1с — нагрузка из урока.
DURATIONS = [0.4] * 4 + [0.1] * 4
# Пессимальное закрепление: все медленные повешены на воркера 0.
PESSIMAL = [0, 0, 0, 0, 1, 2, 3, 0]


# ----------------------------------------------------------- build_topology
def test_mesh_connects_every_pair():
    assert build_topology(3, "mesh") == [(0, 1), (0, 2), (1, 2)]


def test_star_connects_everyone_to_the_hub_only():
    assert build_topology(4, "star") == [(0, 1), (0, 2), (0, 3)]


def test_ring_closes_the_loop_without_duplicating_the_last_edge():
    """Ребро (n-1, 0) нормализуется в (0, n-1), а не считается вторым."""
    assert build_topology(4, "ring") == [(0, 1), (0, 3), (1, 2), (2, 3)]


def test_ring_of_two_agents_is_a_single_edge():
    assert build_topology(2, "ring") == [(0, 1)]


def test_a_single_agent_has_no_channels_in_any_topology():
    assert all(build_topology(1, kind) == [] for kind in TOPOLOGIES)


def test_build_topology_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        build_topology(3, "hypercube")


def test_edges_are_sorted_pairs_with_the_smaller_index_first():
    for kind in TOPOLOGIES:
        edges = build_topology(6, kind)
        assert all(i < j for i, j in edges), kind
        assert edges == sorted(edges), kind


# ------------------------------------------------------------ channel_count
def test_mesh_costs_n_times_n_minus_one_over_two():
    assert [channel_count(n, "mesh") for n in (2, 3, 5, 10)] == [1, 3, 10, 45]


def test_star_costs_n_minus_one():
    assert [channel_count(n, "star") for n in (2, 3, 5, 10)] == [1, 2, 4, 9]


def test_mesh_grows_quadratically_while_star_grows_linearly():
    """Сто агентов: 4950 каналов у полносвязного роя против 99 у звезды."""
    assert channel_count(100, "mesh") == 4950
    assert channel_count(100, "star") == 99


def test_doubling_the_swarm_roughly_quadruples_the_mesh_channels():
    assert channel_count(200, "mesh") / channel_count(100, "mesh") == APPROX(19900 / 4950)
    assert channel_count(200, "star") / channel_count(100, "star") == APPROX(199 / 99)


# ------------------------------------------------------------- is_connected
def test_star_is_connected_with_the_minimum_number_of_edges():
    assert is_connected(5, build_topology(5, "star")) is True


def test_ring_is_connected():
    assert is_connected(5, build_topology(5, "ring")) is True


def test_a_swarm_split_in_two_is_not_connected():
    """Два роя, которые не знают друг о друге."""
    assert is_connected(4, [(0, 1), (2, 3)]) is False


def test_losing_the_hub_disconnects_the_star():
    """Звезда дёшева по каналам и хрупка: без центра остаются одиночки."""
    without_hub = [e for e in build_topology(4, "star") if 0 not in e]
    assert is_connected(4, without_hub) is False


# ------------------------------------------------------------ simulate_fixed
def test_fixed_makespan_is_the_slowest_worker_not_the_total_work():
    assert simulate_fixed([1.0, 1.0], [0, 1], 2)[0] == APPROX(1.0)


def test_fixed_piles_up_when_the_assignment_is_bad():
    makespan, counts = simulate_fixed([1.0, 1.0], [0, 0], 2)
    assert makespan == APPROX(2.0)
    assert counts == {0: 2, 1: 0}


def test_one_worker_reproduces_the_sequential_baseline():
    assert simulate_fixed(DURATIONS, [0] * 8, 1)[0] == APPROX(sum(DURATIONS))


def test_fixed_rejects_a_worker_id_out_of_range():
    with pytest.raises(ValueError):
        simulate_fixed([1.0], [7], 2)


# ------------------------------------------------------------ simulate_swarm
def test_swarm_hands_the_next_task_to_whoever_frees_up_first():
    makespan, counts = simulate_swarm([4.0, 1.0, 1.0, 1.0], 2)
    assert makespan == APPROX(4.0)
    assert counts == {0: 1, 1: 3}


def test_swarm_counts_are_uneven_but_the_makespan_is_not():
    """Рой не раздаёт поровну — он раздаёт так, чтобы все закончили вместе."""
    makespan, counts = simulate_swarm([3.0, 1.0, 1.0, 1.0], 2)
    assert counts == {0: 1, 1: 3}
    assert makespan == APPROX(3.0)


def test_swarm_beats_the_pessimal_fixed_assignment_on_the_lesson_workload():
    swarm_time, swarm_counts = simulate_swarm(DURATIONS, 4)
    fixed_time, fixed_counts = simulate_fixed(DURATIONS, PESSIMAL, 4)
    assert swarm_time == APPROX(0.5)
    assert swarm_counts == {0: 2, 1: 2, 2: 2, 3: 2}
    assert fixed_time == APPROX(1.7)
    assert fixed_counts == {0: 5, 1: 1, 2: 1, 3: 1}


def test_swarm_reaches_the_ideal_parallel_time_on_a_divisible_workload():
    ideal = sum(DURATIONS) / 4
    assert simulate_swarm(DURATIONS, 4)[0] == APPROX(ideal)


def test_swarm_can_never_beat_the_ideal_parallel_time():
    """Нижняя граница: суммарная работа, делённая на число воркеров."""
    durations = [0.7, 0.3, 0.5, 0.9, 0.2]
    makespan, _ = simulate_swarm(durations, 3)
    assert makespan >= sum(durations) / 3 - 1e-9


def test_one_worker_swarm_degenerates_to_sequential():
    assert simulate_swarm(DURATIONS, 1)[0] == APPROX(sum(DURATIONS))


# -------------------------------------------------------------------- speedup
def test_speedup_on_the_lesson_workload_is_four_times_over_sequential():
    sequential = simulate_fixed(DURATIONS, [0] * 8, 1)[0]
    swarm_time = simulate_swarm(DURATIONS, 4)[0]
    assert speedup(sequential, swarm_time) == APPROX(4.0)


def test_speedup_rejects_a_zero_candidate_time():
    with pytest.raises(ValueError):
        speedup(2.0, 0.0)


# ------------------------------------------------------------- hot_spot_ratio
def test_hot_spot_ratio_is_one_on_a_balanced_swarm():
    assert hot_spot_ratio(simulate_swarm(DURATIONS, 4)[1]) == APPROX(1.0)


def test_hot_spot_ratio_flags_the_pessimal_fixed_assignment():
    assert hot_spot_ratio(simulate_fixed(DURATIONS, PESSIMAL, 4)[1]) == APPROX(5.0)


def test_an_idle_worker_makes_the_hot_spot_ratio_infinite():
    assert hot_spot_ratio({0: 5, 1: 0}) == float("inf")


# ---------------------------------------------------------------- aging_order
def test_without_aging_the_low_priority_task_stays_last():
    tasks = [
        {"id": "old-and-boring", "priority": 1, "arrival": 0},
        {"id": "fresh-and-urgent", "priority": 5, "arrival": 9},
    ]
    assert aging_order(tasks, 10, 0.0) == ["fresh-and-urgent", "old-and-boring"]


def test_aging_lets_a_starving_task_overtake():
    tasks = [
        {"id": "old-and-boring", "priority": 1, "arrival": 0},
        {"id": "fresh-and-urgent", "priority": 5, "arrival": 9},
    ]
    assert aging_order(tasks, 10, 0.5) == ["old-and-boring", "fresh-and-urgent"]


def test_starvation_persists_under_continuous_load_without_aging():
    """Каждый тик приходит новая срочная задача — старая не дождётся никогда."""
    tasks = [{"id": "starving", "priority": 1, "arrival": 0}]
    for now in range(1, 30):
        tasks.append({"id": f"urgent-{now}", "priority": 5, "arrival": now})
        assert aging_order(tasks, now, 0.0)[0] != "starving"


def test_aging_ties_are_broken_by_arrival_then_id():
    tasks = [
        {"id": "b", "priority": 2, "arrival": 4},
        {"id": "a", "priority": 2, "arrival": 4},
        {"id": "c", "priority": 2, "arrival": 1},
    ]
    assert aging_order(tasks, 10, 0.0) == ["c", "a", "b"]


def test_aging_order_rejects_a_negative_aging_rate():
    with pytest.raises(ValueError):
        aging_order([{"id": "a", "priority": 1, "arrival": 0}], 5, -0.1)
