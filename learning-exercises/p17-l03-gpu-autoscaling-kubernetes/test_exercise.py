"""Тесты к уроку «GPU-автоскейлинг в Kubernetes». Правь exercise.py."""

import pytest

from exercise import (
    UNDERUTILIZED_THRESHOLD,
    EmptyDeployment,
    GangSchedulingFailure,
    consolidation_plan,
    count_scale_events,
    desired_replicas,
    duty_cycle_util,
    gang_schedule,
    queue_depth_per_replica,
    run_autoscaler,
    stabilize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SAWTOOTH = [40, 4] * 8

CLUSTER = {
    "busy-low-util": {"running_requests": 12, "empty_since": None, "utilization": 30.0},
    "busy-high-util": {"running_requests": 40, "empty_since": None, "utilization": 90.0},
    "idle-fresh": {"running_requests": 0, "empty_since": 7000.0, "utilization": 0.0},
    "idle-old": {"running_requests": 0, "empty_since": 0.0, "utilization": 0.0},
}


# --------------------------------------------------------- duty_cycle_util
def test_duty_cycle_reads_below_saturation():
    assert duty_cycle_util(2, 1, 4) == APPROX(50.0)


def test_duty_cycle_saturates_at_a_hundred():
    assert duty_cycle_util(10, 1, 4) == APPROX(100.0)


def test_duty_cycle_cannot_tell_ten_requests_from_a_hundred():
    """Главная ловушка урока: сигнал одинаков при десятикратной разнице нагрузки."""
    assert duty_cycle_util(10, 1, 4) == duty_cycle_util(100, 1, 4)


def test_duty_cycle_refuses_an_empty_deployment():
    with pytest.raises(EmptyDeployment):
        duty_cycle_util(10, 0, 4)


# -------------------------------------------------- queue_depth_per_replica
def test_queue_depth_grows_with_the_backlog():
    assert queue_depth_per_replica(10, 1) == APPROX(10.0)


def test_queue_depth_distinguishes_what_duty_cycle_cannot():
    """Тот же вход, где duty cycle показал одно и то же — очередь различает."""
    assert queue_depth_per_replica(10, 1) != queue_depth_per_replica(100, 1)


def test_queue_depth_falls_as_replicas_are_added():
    assert queue_depth_per_replica(10, 5) == APPROX(2.0)


def test_queue_depth_refuses_an_empty_deployment():
    with pytest.raises(EmptyDeployment):
        queue_depth_per_replica(10, 0)


# -------------------------------------------------------- desired_replicas
def test_desired_replicas_scales_proportionally_to_the_metric():
    assert desired_replicas(2, 40.0, 10.0, 1, 16) == 8


def test_desired_replicas_rounds_up():
    """ceil, а не round: недобрать реплику значит остаться выше цели навсегда."""
    assert desired_replicas(1, 11.0, 10.0, 1, 16) == 2


def test_desired_replicas_respects_the_lower_bound():
    assert desired_replicas(4, 0.0, 10.0, 1, 16) == 1


def test_desired_replicas_respects_the_upper_bound():
    assert desired_replicas(2, 400.0, 10.0, 1, 16) == 16


# --------------------------------------------------------------- stabilize
def test_stabilize_without_a_window_trusts_the_last_sample():
    assert stabilize([4, 1], 1) == 1


def test_stabilize_remembers_the_recent_peak():
    assert stabilize([4, 1], 3) == 4


def test_stabilize_reacts_to_growth_immediately():
    """Асимметрия как в Kubernetes: вниз медленно, вверх сразу."""
    assert stabilize([1, 1, 4], 3) == 4


def test_stabilize_forgets_the_peak_once_it_leaves_the_window():
    assert stabilize([4, 1, 1, 1], 3) == 1


def test_stabilize_rejects_a_zero_window():
    with pytest.raises(ValueError):
        stabilize([4, 1], 0)


# ----------------------------------------------------------- run_autoscaler
def test_run_autoscaler_without_a_window_flaps_on_a_sawtooth():
    series = run_autoscaler(SAWTOOTH, 10.0, 1, 1, 16)
    assert count_scale_events(series) == len(SAWTOOTH) - 1


def test_run_autoscaler_with_a_window_does_not_flap_on_the_same_load():
    """Тот же ряд, окно в три тика — ни одного события масштабирования."""
    series = run_autoscaler(SAWTOOTH, 10.0, 3, 1, 16)
    assert count_scale_events(series) == 0


def test_run_autoscaler_with_a_window_holds_the_peak_capacity():
    """Окно не занижает мощность: держится уровень пика, а не средний."""
    assert run_autoscaler(SAWTOOTH, 10.0, 3, 1, 16) == [4] * len(SAWTOOTH)


def test_run_autoscaler_scales_up_on_the_very_first_tick():
    assert run_autoscaler([40], 10.0, 5, 1, 16)[0] == 4


def test_run_autoscaler_eventually_scales_down_after_the_window_expires():
    """Нагрузка упала и держится — через окно реплики всё-таки уезжают вниз."""
    series = run_autoscaler([40, 40, 40] + [4] * 6, 10.0, 3, 1, 16)
    assert series[0] == 4
    assert series[-1] == 1


def test_run_autoscaler_never_goes_below_the_floor():
    series = run_autoscaler([0] * 10, 10.0, 1, 2, 16)
    assert min(series) == 2


# ------------------------------------------------------ count_scale_events
def test_count_scale_events_counts_every_change():
    assert count_scale_events([4, 1, 4, 1]) == 3


def test_count_scale_events_of_a_flat_series_is_zero():
    assert count_scale_events([4, 4, 4, 4]) == 0


# ------------------------------------------------------------ gang_schedule
def test_gang_schedule_spreads_across_two_nodes_when_it_must():
    assert gang_schedule({"n1": 4, "n2": 4}, 8) == {"n1": 4, "n2": 4}


def test_gang_schedule_prefers_a_single_node_for_topology():
    assert gang_schedule({"n1": 8, "n2": 4}, 8) == {"n1": 8}


def test_gang_schedule_allocates_exactly_what_was_asked():
    plan = gang_schedule({"n1": 4, "n2": 4}, 6)
    assert sum(plan.values()) == 6


def test_gang_schedule_places_nothing_when_it_cannot_place_everything():
    """Ловушка «7 из 8»: обычный планировщик занял бы семь GPU и ждал восьмую."""
    with pytest.raises(GangSchedulingFailure) as err:
        gang_schedule({"n1": 4, "n2": 3}, 8)
    assert err.value.stranded == 7


def test_gang_schedule_is_deterministic_regardless_of_dict_order():
    a = gang_schedule({"n1": 4, "n2": 4, "n3": 8}, 8)
    b = gang_schedule({"n3": 8, "n2": 4, "n1": 4}, 8)
    assert a == b == {"n3": 8}


# -------------------------------------------------------- consolidation_plan
def test_when_empty_terminates_only_long_idle_nodes():
    plan = consolidation_plan(CLUSTER, "WhenEmpty", 7200.0, 3600.0)
    assert plan["terminate"] == ["idle-old"]


def test_when_empty_evicts_no_running_requests():
    """Безопасная политика урока: узлы гаснут, запросы не умирают."""
    plan = consolidation_plan(CLUSTER, "WhenEmpty", 7200.0, 3600.0)
    assert plan["evicted_requests"] == 0


def test_when_empty_waits_out_consolidate_after():
    """Узел пуст всего 200 секунд из 3600 — трогать рано."""
    plan = consolidation_plan(CLUSTER, "WhenEmpty", 7200.0, 3600.0)
    assert "idle-fresh" not in plan["terminate"]


def test_when_empty_or_underutilized_kills_running_requests():
    """Дефолт Karpenter выбивает работающий узел с утилизацией ниже порога."""
    plan = consolidation_plan(CLUSTER, "WhenEmptyOrUnderutilized", 7200.0, 3600.0)
    assert "busy-low-util" in plan["terminate"]
    assert plan["evicted_requests"] == 12
    # а по-настоящему загруженный узел (90% > порога) политика не трогает
    assert "busy-high-util" not in plan["terminate"]
    assert CLUSTER["busy-high-util"]["utilization"] > UNDERUTILIZED_THRESHOLD


def test_consolidation_rejects_an_unknown_policy():
    with pytest.raises(ValueError):
        consolidation_plan(CLUSTER, "WheneverIFeelLikeIt", 7200.0, 3600.0)
