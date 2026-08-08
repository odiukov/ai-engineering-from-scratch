"""Тесты к уроку «Batch API: очередь, скидка и SLA завершения». Правь exercise.py."""

import pytest

from exercise import (
    BATCH_DISCOUNT,
    BATCH_SLA_H,
    BatchError,
    batch_cost,
    cached_cost,
    drain_window,
    lane_decision,
    sla_report,
    submit,
    sync_cost,
    triage,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, rel=1e-6)


def night_queue():
    """Три задания: одно в окно, одно мимо окна, одно длиннее окна."""
    q = submit([], "fits", 20_000, 1.0)
    q = submit(q, "late", 10_000, 10.0)
    q = submit(q, "huge", 100_000, 1.0)
    return q


# ---------------------------------------------------------------- sync_cost
def test_single_synchronous_call():
    assert sync_cost(1, 4000, 2000, 200) == APPROX(0.021)


def test_synchronous_cost_is_linear_in_the_number_of_calls():
    assert sync_cost(50_000, 4000, 2000, 200) == ROUGH(1050.0)


def test_shared_prefix_is_paid_again_on_every_call():
    """Базовая линия не знает про кэш: 50k раз оплачен один и тот же промпт."""
    with_prefix = sync_cost(1000, 4000, 0, 0)
    without = sync_cost(1000, 0, 0, 0)
    assert with_prefix - without == APPROX(1000 * 4000 / 1e6 * 3.00)


def test_empty_run_costs_nothing():
    assert sync_cost(0, 4000, 2000, 200) == APPROX(0.0)


def test_negative_call_count_is_rejected():
    with pytest.raises(BatchError):
        sync_cost(-1, 4000, 2000, 200)


# -------------------------------------------------------------- cached_cost
def test_cached_run_of_fifty_thousand_documents():
    assert cached_cost(50_000, 4000, 2000, 200) == ROUGH(510.0138)


def test_cache_only_helps_the_shared_prefix():
    """Уникальная часть и выход стоят столько же, что и без кэша."""
    no_prefix_sync = sync_cost(100, 0, 2000, 200)
    no_prefix_cached = cached_cost(100, 0, 2000, 200)
    assert no_prefix_cached == APPROX(no_prefix_sync)


def test_first_call_pays_the_premium_instead_of_the_base_price():
    """Ловушка: премия ВМЕСТО базовой цены, а не вдобавок."""
    assert cached_cost(1, 4000, 0, 0) == APPROX(4000 / 1e6 * 3.00 * 1.25)


def test_a_single_call_is_more_expensive_with_caching_than_without():
    """Кэш из одного вызова — чистый убыток: записали и никто не прочитал."""
    assert cached_cost(1, 4000, 0, 0) > sync_cost(1, 4000, 0, 0)


def test_caching_pays_off_from_the_second_call():
    assert cached_cost(2, 4000, 0, 0) < sync_cost(2, 4000, 0, 0)


def test_cached_zero_calls_is_free():
    assert cached_cost(0, 4000, 2000, 200) == APPROX(0.0)


# --------------------------------------------------------------- batch_cost
def test_batch_is_exactly_half_of_synchronous():
    assert batch_cost(50_000, 4000, 2000, 200, False) == ROUGH(
        sync_cost(50_000, 4000, 2000, 200) * BATCH_DISCOUNT)


def test_anthropic_stacks_batch_and_cache_discounts():
    stacked = batch_cost(50_000, 4000, 2000, 200, True, provider="anthropic")
    assert stacked < batch_cost(50_000, 4000, 2000, 200, False)
    assert stacked < cached_cost(50_000, 4000, 2000, 200)


def test_vertex_cached_prefix_price_takes_precedence_over_batch_discount():
    n, prefix, unique, output = 50_000, 4000, 2000, 200
    vertex = batch_cost(n, prefix, unique, output, True, provider="vertex-gemini")
    cached_prefix = cached_cost(n, prefix, 0, 0)
    discounted_tail = sync_cost(n, 0, unique, output) * BATCH_DISCOUNT
    assert vertex == APPROX(cached_prefix + discounted_tail)
    assert vertex > batch_cost(n, prefix, unique, output, True, provider="anthropic")


def test_unknown_provider_policy_is_rejected():
    with pytest.raises(BatchError):
        batch_cost(1, 100, 100, 10, True, provider="mystery-cloud")


def test_long_shared_prefix_gets_close_to_the_advertised_ten_percent():
    """4000 общих токенов, 200 уникальных, короткий ответ — вот тут и ~10%."""
    baseline = sync_cost(50_000, 4000, 200, 100)
    stacked = batch_cost(50_000, 4000, 200, 100, True)
    assert 0.10 < stacked / baseline < 0.13


def test_heavy_unique_content_never_reaches_ten_percent():
    """Уникальные 15k токенов и ответ на 2k: стек даёт около 40%, не 10%."""
    baseline = sync_cost(1000, 6000, 15_000, 2000)
    stacked = batch_cost(1000, 6000, 15_000, 2000, True)
    assert 0.40 < stacked / baseline < 0.42


# ------------------------------------------------------------------- submit
def test_submit_returns_a_new_queue_and_leaves_the_old_one_alone():
    original = []
    grown = submit(original, "a", 10, 0.0)
    assert original == []
    assert len(grown) == 1


def test_duplicate_job_id_is_rejected():
    q = submit([], "a", 10, 0.0)
    with pytest.raises(BatchError):
        submit(q, "a", 10, 1.0)


def test_empty_job_is_rejected():
    with pytest.raises(BatchError):
        submit([], "a", 0, 0.0)


def test_negative_submission_time_is_rejected():
    with pytest.raises(BatchError):
        submit([], "a", 10, -1.0)


# ------------------------------------------------------------- drain_window
def test_job_submitted_inside_the_window_runs_immediately():
    q = submit([], "fits", 20_000, 1.0)
    assert drain_window(q, 0, 6, 10_000)[0] == {
        "job_id": "fits", "finished_h": 3.0, "wait_h": 2.0}


def test_job_submitted_outside_the_window_waits_for_the_next_night():
    q = submit([], "late", 10_000, 10.0)
    done = drain_window(q, 0, 6, 10_000)[0]
    assert done["finished_h"] == APPROX(25.0)
    assert done["wait_h"] == APPROX(15.0)


def test_work_longer_than_the_window_continues_the_next_night():
    """Ловушка: остаток продолжают, а не начинают заново и не считают за окном."""
    q = submit([], "huge", 100_000, 1.0)
    done = drain_window(q, 0, 6, 10_000)[0]
    assert done["finished_h"] == APPROX(29.0)


def test_jobs_run_one_after_another_not_in_parallel():
    q = submit(submit([], "a", 10_000, 0.0), "b", 10_000, 0.0)
    done = drain_window(q, 0, 6, 10_000)
    assert [c["finished_h"] for c in done] == [APPROX(1.0), APPROX(2.0)]


def test_higher_throughput_finishes_sooner():
    q = submit([], "a", 40_000, 0.0)
    slow = drain_window(q, 0, 6, 10_000)[0]["finished_h"]
    fast = drain_window(q, 0, 6, 40_000)[0]["finished_h"]
    assert fast < slow


def test_wider_window_removes_the_overnight_wait():
    q = submit([], "huge", 100_000, 1.0)
    narrow = drain_window(q, 0, 6, 10_000)[0]["wait_h"]
    wide = drain_window(q, 0, 24, 10_000)[0]["wait_h"]
    assert wide == APPROX(10.0)
    assert narrow > wide


def test_broken_window_is_rejected():
    with pytest.raises(BatchError):
        drain_window(submit([], "a", 10, 0.0), 6, 6, 10_000)


def test_zero_throughput_is_rejected():
    with pytest.raises(BatchError):
        drain_window(submit([], "a", 10, 0.0), 0, 6, 0)


# --------------------------------------------------------------- sla_report
def test_empty_report_does_not_divide_by_zero():
    report = sla_report([])
    assert report["met_fraction"] == APPROX(0.0)
    assert report["worst_job"] is None


def test_one_oversized_job_pushes_the_whole_queue_past_the_sla():
    """Очередь последовательная: гигант тормозит и того, кто подан позже него."""
    report = sla_report(drain_window(night_queue(), 0, 6, 10_000))
    assert report["met"] == 1
    assert report["missed"] == 2
    assert report["max_wait_h"] > BATCH_SLA_H


def test_exactly_at_the_sla_boundary_counts_as_met():
    assert sla_report([{"job_id": "a", "finished_h": 24.0, "wait_h": 24.0}])["met"] == 1


def test_a_job_that_waited_two_nights_misses_the_promise():
    q = submit([], "huge", 300_000, 1.0)
    report = sla_report(drain_window(q, 0, 6, 10_000))
    assert report["missed"] == 1
    assert report["max_wait_h"] > BATCH_SLA_H


def test_worst_job_is_named():
    report = sla_report(drain_window(night_queue(), 0, 6, 10_000))
    assert report["worst_job"] == "huge"


# ------------------------------------------------------------------- triage
def test_a_user_watching_a_spinner_is_interactive():
    assert triage(5) == "interactive"


def test_a_few_minutes_is_semi_interactive():
    assert triage(600) == "semi"


def test_by_morning_is_batch():
    assert triage(86_400) == "batch"


def test_lane_boundaries_are_inclusive():
    assert triage(60) == "interactive"
    assert triage(3600) == "semi"
    assert triage(3601) == "batch"


def test_non_positive_budget_is_rejected():
    with pytest.raises(BatchError):
        triage(0)


# ------------------------------------------------------------ lane_decision
def test_batch_lane_leaves_nothing_on_the_table():
    d = lane_decision(50_000, 4000, 200, 100, 86_400)
    assert d["lane"] == "batch"
    assert d["forgone_usd"] == APPROX(0.0)


def test_vertex_lane_decision_uses_cache_precedence_policy():
    d = lane_decision(50_000, 4000, 2000, 200, 86_400, provider="vertex-gemini")
    assert d["cost"] == APPROX(
        batch_cost(50_000, 4000, 2000, 200, True, provider="vertex-gemini")
    )


def test_interactive_lane_pays_double_the_achievable_minimum():
    """Скидка недоступна: тот же workload стоит ровно вдвое дороже дна."""
    d = lane_decision(50_000, 4000, 200, 100, 5)
    assert d["lane"] == "interactive"
    assert d["forgone_usd"] > 0
    assert d["cost"] / d["best_cost"] == ROUGH(1 / BATCH_DISCOUNT)


def test_semi_lane_also_forgoes_the_discount():
    d = lane_decision(50_000, 4000, 200, 100, 600)
    assert d["lane"] == "semi"
    assert d["forgone_usd"] > 0


def test_batch_is_worth_it_only_when_latency_allows_it():
    """Главное свойство: экономия появляется от смены SLA, а не от кода."""
    tight = lane_decision(50_000, 4000, 200, 100, 5)
    loose = lane_decision(50_000, 4000, 200, 100, 86_400)
    assert loose["cost"] < tight["cost"]
    assert loose["saving_pct"] > tight["saving_pct"]


def test_caching_still_helps_the_interactive_lane():
    """Полоса interactive не безнадёжна: кэш работает и там."""
    d = lane_decision(50_000, 4000, 200, 100, 5)
    assert d["saving_usd"] > 0
    assert d["cost"] < d["baseline_cost"]


def test_workload_without_a_shared_prefix_gains_only_from_the_discount():
    d = lane_decision(1000, 0, 5000, 500, 86_400)
    assert d["saving_pct"] == ROUGH(50.0)
