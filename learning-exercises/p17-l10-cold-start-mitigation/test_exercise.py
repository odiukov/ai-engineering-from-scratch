"""Тесты к уроку «Холодный старт serverless-LLM и как его лечить». Правь exercise.py."""

import pytest

from exercise import (
    PHASES_70B,
    UnknownMitigationError,
    available_replicas,
    cold_start_seconds,
    min_warm_pool_for,
    mitigation_savings,
    ready_at,
    simulate_arrivals,
    warm_pool_monthly_cost,
    weights_load_seconds,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Утренний всплеск: ночь, потом пик, потом спад. Слот — минута.
MORNING_SPIKE = [10, 100, 100, 10]
CAPACITY = 10
SLOT = 60.0


# -------------------------------------------------- weights_load_seconds
def test_seventy_b_in_bf16_from_nvme_takes_twenty_seconds():
    assert weights_load_seconds(140.0, 7.0) == APPROX(20.0)


def test_quantized_weights_load_proportionally_faster():
    """INT4 — вчетверо меньше байт, значит вчетверо короче загрузка."""
    assert weights_load_seconds(35.0, 7.0) == APPROX(weights_load_seconds(140.0, 7.0) / 4)


def test_zero_read_speed_is_a_call_error_not_infinity():
    with pytest.raises(ValueError):
        weights_load_seconds(140.0, 0.0)


# ---------------------------------------------------- cold_start_seconds
def test_raw_cold_start_of_a_70b_is_the_published_328_seconds():
    assert cold_start_seconds(PHASES_70B) == APPROX(328.0)


def test_pre_seeded_image_removes_the_image_pull_entirely():
    assert cold_start_seconds(PHASES_70B, ["pre_seeded"]) == APPROX(148.0)


def test_streamer_halves_only_the_weights_phase():
    assert cold_start_seconds(PHASES_70B, ["streamer"]) == APPROX(328.0 - 37.5)


def test_mitigations_stack_multiplicatively():
    both = cold_start_seconds(PHASES_70B, ["pre_seeded", "streamer"])
    assert both == APPROX(50.0 + 0.0 + 37.5 + 20.0 + 3.0)


def test_mitigation_order_does_not_matter():
    a = cold_start_seconds(PHASES_70B, ["pre_seeded", "streamer"])
    b = cold_start_seconds(PHASES_70B, ["streamer", "pre_seeded"])
    assert a == APPROX(b)


def test_gpu_snapshot_leaves_only_node_provisioning():
    """Снапшот убирает загрузку и прогрев, но ноду всё равно кто-то выдаёт."""
    total = cold_start_seconds(PHASES_70B, ["gpu_snapshot"])
    assert total == APPROX(55.0)
    assert total > PHASES_70B["node provision"]


def test_a_typo_in_the_mitigation_name_is_not_silently_ignored():
    with pytest.raises(UnknownMitigationError):
        cold_start_seconds(PHASES_70B, ["pre-seeded"])


# --------------------------------------------------- mitigation_savings
def test_pre_seeding_saves_exactly_the_image_pull():
    assert mitigation_savings(PHASES_70B, ["pre_seeded"]) == APPROX(180.0)


def test_no_mitigations_save_nothing():
    assert mitigation_savings(PHASES_70B, []) == APPROX(0.0)


# ---------------------------------------------- ready_at / available_replicas
def test_replica_is_ready_a_full_cold_start_after_launch():
    assert ready_at(600.0, 15.0) == APPROX(615.0)


def test_only_finished_replicas_answer_traffic():
    assert available_replicas([0.0, 328.0, 700.0], 400.0) == 2


def test_a_replica_that_just_finished_counts_as_available():
    assert available_replicas([328.0], 328.0) == 1


def test_nothing_is_available_before_the_first_replica_warms_up():
    assert available_replicas([328.0], 327.9) == 0


# ---------------------------------------------------- simulate_arrivals
def test_scale_to_zero_makes_the_first_requests_cold():
    result = simulate_arrivals([10, 10], CAPACITY, 0, 300.0, SLOT)
    assert result["cold"] == 20
    assert result["cold_share"] == APPROX(1.0)


def test_one_warm_replica_covers_a_flat_load():
    result = simulate_arrivals([10, 10], CAPACITY, 1, 300.0, SLOT)
    assert result["cold"] == 0


def test_a_warm_pool_the_size_of_the_peak_removes_cold_starts():
    peak = simulate_arrivals(MORNING_SPIKE, CAPACITY, 0, 300.0, SLOT)["peak_replicas"]
    assert peak == 10
    result = simulate_arrivals(MORNING_SPIKE, CAPACITY, peak, 300.0, SLOT)
    assert result["cold_share"] == APPROX(0.0)


def test_a_warm_pool_below_the_peak_does_not():
    """Девять реплик вместо десяти — и утренний пик всё равно частично холодный."""
    result = simulate_arrivals(MORNING_SPIKE, CAPACITY, 9, 300.0, SLOT)
    assert result["cold"] > 0


def test_shorter_cold_start_means_fewer_cold_requests():
    """Тот же профиль трафика, но реплика поднимается за полминуты, а не за пять."""
    slow = simulate_arrivals(MORNING_SPIKE, CAPACITY, 1, 300.0, SLOT)
    fast = simulate_arrivals(MORNING_SPIKE, CAPACITY, 1, 30.0, SLOT)
    assert fast["cold"] < slow["cold"]


def test_mitigations_measured_in_seconds_turn_into_saved_requests():
    """Снапшот сокращает старт с 328 до 55 секунд — и хвост холодных запросов тает."""
    raw = simulate_arrivals(MORNING_SPIKE, CAPACITY, 1,
                            cold_start_seconds(PHASES_70B), SLOT)
    snapshot = simulate_arrivals(MORNING_SPIKE, CAPACITY, 1,
                                 cold_start_seconds(PHASES_70B, ["gpu_snapshot"]), SLOT)
    assert snapshot["cold"] < raw["cold"]


def test_total_requests_are_never_lost():
    result = simulate_arrivals(MORNING_SPIKE, CAPACITY, 3, 300.0, SLOT)
    assert result["total"] == sum(MORNING_SPIKE)
    assert 0 <= result["cold"] <= result["total"]


def test_replica_of_zero_capacity_is_a_call_error():
    with pytest.raises(ValueError):
        simulate_arrivals(MORNING_SPIKE, 0, 1, 300.0, SLOT)


# ------------------------------------------------- warm_pool_monthly_cost
def test_one_warm_h100_costs_a_full_month_of_rent():
    assert warm_pool_monthly_cost(1, 4.50) == APPROX(3240.0)


def test_cost_is_linear_in_the_number_of_warm_replicas():
    assert warm_pool_monthly_cost(5, 4.50) == APPROX(5 * warm_pool_monthly_cost(1, 4.50))


def test_scale_to_zero_costs_nothing_while_idle():
    assert warm_pool_monthly_cost(0, 4.50) == APPROX(0.0)


# ---------------------------------------------------- min_warm_pool_for
def test_zero_cold_starts_requires_a_pool_the_size_of_the_peak():
    assert min_warm_pool_for(0.0, MORNING_SPIKE, CAPACITY, 300.0, SLOT) == 10


def test_tolerating_a_cold_tail_buys_back_most_of_the_pool():
    """Согласиться на 10% холодных — и платить надо за втрое меньший пул."""
    strict = min_warm_pool_for(0.0, MORNING_SPIKE, CAPACITY, 300.0, SLOT)
    relaxed = min_warm_pool_for(0.1, MORNING_SPIKE, CAPACITY, 300.0, SLOT)
    assert relaxed < strict


def test_faster_cold_start_lowers_the_required_warm_pool():
    slow = min_warm_pool_for(0.1, MORNING_SPIKE, CAPACITY, 300.0, SLOT)
    fast = min_warm_pool_for(0.1, MORNING_SPIKE, CAPACITY, 30.0, SLOT)
    assert fast <= slow


def test_a_share_above_one_is_a_call_error():
    with pytest.raises(ValueError):
        min_warm_pool_for(1.5, MORNING_SPIKE, CAPACITY, 300.0, SLOT)
