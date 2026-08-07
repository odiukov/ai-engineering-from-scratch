"""Тесты к уроку «METR: временные горизонты и внешняя оценка возможностей».
Правь exercise.py."""

import random

import pytest

from exercise import (
    DEPLOYMENT_DISCOUNTS,
    METR_ENGAGEMENT,
    SUITES,
    deployment_gap,
    doubling_time_days,
    horizon_at,
    inject_gaming,
    resolve_access,
    run_manifest,
    sample_tasks,
    success_curve,
)

APPROX = lambda x: pytest.approx(x, abs=1e-6)

# Чистая кривая: доля успеха падает с длиной задачи.
CURVE = [(0.25, 1.0), (1.0, 0.9), (4.0, 0.7), (14.0, 0.5), (32.0, 0.3), (64.0, 0.1)]

# Сырой прогон: по десять задач на каждую длительность.
HOURS = (0.25, 1.0, 4.0, 14.0, 32.0, 64.0)
HITS = (10, 9, 7, 5, 2, 0)
RAW = [(h, i < k) for h, k in zip(HOURS, HITS) for i in range(10)]


# ------------------------------------------------------------- resolve_access
def test_the_evaluator_gets_exactly_the_agreed_scopes():
    got = resolve_access(METR_ENGAGEMENT, ["model_api", "task_scaffold"])
    assert got["granted"] == ["model_api", "task_scaffold"]
    assert got["refused"] == []


def test_a_scope_outside_the_agreement_is_refused_not_granted():
    """Независимость оценщика держится на том, что доступ не расширяется."""
    got = resolve_access(METR_ENGAGEMENT, ["model_api", "model_weights"])
    assert got["granted"] == ["model_api"]
    assert got["refused"] == ["model_weights"]


def test_an_agreement_wider_than_the_request_shows_up_as_unused():
    """Выданные и не использованные веса остаются выданными весами."""
    got = resolve_access(
        ["model_api", "prerelease_checkpoint", "model_weights"], ["model_api"]
    )
    assert got["unused"] == ["model_weights", "prerelease_checkpoint"]


def test_an_unknown_scope_name_is_an_error():
    with pytest.raises(ValueError):
        resolve_access(METR_ENGAGEMENT, ["model_apy"])
    with pytest.raises(ValueError):
        resolve_access(["everything"], ["model_api"])


# --------------------------------------------------------------- sample_tasks
def test_the_same_seed_reproduces_the_same_task_sample():
    """Прогон, который нельзя повторить, — не измерение, а слайд."""
    first = sample_tasks("HCAST", 12, random.Random(4))
    second = sample_tasks("HCAST", 12, random.Random(4))
    assert first == second


def test_a_different_seed_picks_a_different_sample():
    first = sample_tasks("HCAST", 12, random.Random(4))
    other = sample_tasks("HCAST", 12, random.Random(5))
    assert first != other


def test_sampled_tasks_are_sorted_and_stay_inside_the_suite_span():
    spec = SUITES["RE-Bench"]
    tasks = sample_tasks("RE-Bench", 20, random.Random(1))
    times = [hours for _, hours in tasks]
    assert times == sorted(times)
    assert len(set(tid for tid, _ in tasks)) == 20
    assert spec["min_hours"] - 1e-9 <= min(times)
    assert max(times) <= spec["max_hours"] + 1e-9


def test_an_unknown_suite_or_an_oversized_request_is_an_error():
    """Молча выдать меньше задач — отчитаться о прогоне, которого не было."""
    with pytest.raises(ValueError):
        sample_tasks("HCAST-2", 5, random.Random(0))
    with pytest.raises(ValueError):
        sample_tasks("RE-Bench", SUITES["RE-Bench"]["n_tasks"] + 1, random.Random(0))


# --------------------------------------------------------------- run_manifest
def test_two_runs_with_the_same_seed_share_a_digest():
    a = run_manifest("HCAST", 15, 11, METR_ENGAGEMENT, ["model_api"])
    b = run_manifest("HCAST", 15, 11, METR_ENGAGEMENT, ["model_api"])
    assert a["digest"] == b["digest"]
    assert a["tasks"] == b["tasks"]


def test_changing_the_seed_changes_the_digest():
    a = run_manifest("HCAST", 15, 11, METR_ENGAGEMENT, ["model_api"])
    b = run_manifest("HCAST", 15, 12, METR_ENGAGEMENT, ["model_api"])
    assert a["digest"] != b["digest"]


def test_the_manifest_records_granted_access_and_the_digest_follows_it():
    """Запрошенные, но не выданные веса не попадают в паспорт как выданные."""
    asked = run_manifest("HCAST", 15, 11, METR_ENGAGEMENT,
                         ["model_api", "model_weights"])
    assert asked["access"]["granted"] == ["model_api"]
    assert asked["access"]["refused"] == ["model_weights"]

    wider = run_manifest("HCAST", 15, 11,
                         list(METR_ENGAGEMENT) + ["model_weights"],
                         ["model_api", "model_weights"])
    assert wider["access"]["granted"] == ["model_api", "model_weights"]
    assert wider["digest"] != asked["digest"]


def test_the_manifest_carries_the_run_it_describes():
    m = run_manifest("RE-Bench", 9, 3, METR_ENGAGEMENT, list(METR_ENGAGEMENT))
    assert m["suite"] == "RE-Bench"
    assert m["seed"] == 3
    assert m["n_tasks"] == 9
    assert len(m["tasks"]) == 9


# -------------------------------------------------------------- success_curve
def test_tasks_of_equal_length_collapse_into_one_point():
    assert success_curve([(1.0, True), (1.0, False), (4.0, False)]) == [
        (1.0, APPROX(0.5)),
        (4.0, APPROX(0.0)),
    ]


def test_the_curve_is_sorted_by_expert_time():
    curve = success_curve([(64.0, False), (0.25, True), (4.0, True)])
    assert [h for h, _ in curve] == [0.25, 4.0, 64.0]


def test_the_curve_reproduces_the_bucket_rates():
    curve = success_curve(RAW)
    assert [h for h, _ in curve] == list(HOURS)
    assert [r for _, r in curve] == [APPROX(k / 10) for k in HITS]


def test_an_empty_result_set_is_an_error_not_an_empty_curve():
    """Пустая кривая читалась бы как «замер был», а его не было."""
    with pytest.raises(ValueError):
        success_curve([])


# ----------------------------------------------------------------- horizon_at
def test_the_fifty_percent_point_is_read_off_the_curve():
    assert horizon_at(CURVE, 0.50) == APPROX(14.0)


def test_interpolation_is_geometric_not_arithmetic():
    """Между 4 и 14 середина — семь с половиной, а не девять."""
    assert horizon_at(CURVE, 0.60) == APPROX((4.0 * 14.0) ** 0.5)


def test_a_stricter_reliability_target_gives_a_shorter_horizon():
    assert horizon_at(CURVE, 0.90) < horizon_at(CURVE, 0.50) < horizon_at(CURVE, 0.10)


def test_a_target_the_curve_never_reaches_is_an_error():
    """Горизонт, продлённый за последнюю измеренную точку, — не измерение."""
    with pytest.raises(ValueError):
        horizon_at(CURVE, 0.05)


def test_an_unsorted_curve_is_an_error():
    with pytest.raises(ValueError):
        horizon_at(list(reversed(CURVE)), 0.50)


def test_a_probability_outside_the_open_unit_interval_is_an_error():
    with pytest.raises(ValueError):
        horizon_at(CURVE, 1.0)
    with pytest.raises(ValueError):
        horizon_at(CURVE, 0.0)


# --------------------------------------------------------------- inject_gaming
def test_zero_rate_changes_nothing_and_full_rate_flips_every_failure():
    assert inject_gaming(RAW, 0.0, random.Random(0)) == RAW
    flipped = inject_gaming(RAW, 1.0, random.Random(0))
    assert all(success for _, success in flipped)


def test_successes_are_never_turned_into_failures():
    """Gaming — про занижение, а не про шум в обе стороны."""
    gamed = inject_gaming(RAW, 0.5, random.Random(2))
    for (h_raw, s_raw), (h_new, s_new) in zip(RAW, gamed):
        assert h_raw == h_new
        assert s_new >= s_raw


def test_gaming_never_lowers_the_measured_horizon():
    """Занижение модели на оценках занижает пороги всех трёх политик."""
    clean = horizon_at(success_curve(RAW), 0.50)
    for seed in (1, 2, 3, 7):
        gamed = horizon_at(success_curve(inject_gaming(RAW, 0.3, random.Random(seed))), 0.50)
        assert gamed >= clean


def test_the_same_seed_reproduces_the_gamed_set_and_the_input_survives():
    before = list(RAW)
    a = inject_gaming(RAW, 0.4, random.Random(9))
    b = inject_gaming(RAW, 0.4, random.Random(9))
    assert a == b
    assert RAW == before


# ---------------------------------------------------------- doubling_time_days
def test_a_doubling_over_the_window_is_the_window_itself():
    assert doubling_time_days(1.0, 2.0, 130.8) == APPROX(130.8)


def test_only_the_ratio_matters_not_the_absolute_horizons():
    assert doubling_time_days(7.0, 14.0, 130.8) == APPROX(
        doubling_time_days(1.0, 2.0, 130.8)
    )


def test_faster_growth_means_a_shorter_doubling_time():
    assert doubling_time_days(1.0, 4.0, 130.8) == APPROX(65.4)
    assert doubling_time_days(1.0, 4.0, 130.8) < doubling_time_days(1.0, 2.0, 130.8)


def test_no_growth_or_nonpositive_inputs_are_an_error():
    """При равных горизонтах время удвоения не бесконечно, оно неизвестно."""
    with pytest.raises(ValueError):
        doubling_time_days(14.0, 14.0, 130.8)
    with pytest.raises(ValueError):
        doubling_time_days(0.0, 14.0, 130.8)
    with pytest.raises(ValueError):
        doubling_time_days(7.0, 14.0, 0.0)


# --------------------------------------------------------------- deployment_gap
def test_no_discounts_leaves_the_horizon_untouched():
    gap = deployment_gap(14.0, 8.0, [])
    assert gap["effective_hours"] == APPROX(14.0)
    assert gap["within_horizon"] is True


def test_each_discount_can_only_lower_the_effective_horizon():
    """«Горизонт — верхняя граница» проверяемо, а не только декларируемо."""
    for name in DEPLOYMENT_DISCOUNTS:
        gap = deployment_gap(14.0, 1.0, [name])
        assert gap["effective_hours"] < 14.0


def test_a_task_inside_the_raw_horizon_can_fall_outside_the_effective_one():
    raw = deployment_gap(14.0, 8.0, [])
    real = deployment_gap(14.0, 8.0, list(DEPLOYMENT_DISCOUNTS))
    assert raw["within_horizon"] is True
    assert real["within_horizon"] is False
    assert real["effective_hours"] == APPROX(14.0 * 0.70 * 0.80 * 0.75 * 0.80)


def test_the_reason_names_the_effective_hours_and_the_applied_discounts():
    gap = deployment_gap(14.0, 8.0, ["idealized_tooling", "user_variance"])
    assert f"{gap['effective_hours']:.3f}" in gap["reason"]
    assert "idealized_tooling" in gap["reason"]
    assert "user_variance" in gap["reason"]


def test_an_unknown_discount_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError):
        deployment_gap(14.0, 8.0, ["messy_prompts"])
