"""Тесты к уроку «Нагрузочное тестирование LLM API». Правь exercise.py."""

import random
import statistics

import pytest

from exercise import (
    PATTERNS,
    TPOT_MS,
    TTFT_CACHE_HIT_MS,
    TTFT_CACHE_MISS_MS,
    apparent_itl,
    arrival_schedule,
    ci_gate,
    make_workload,
    percentile,
    prompt_lengths,
    run_load,
    summarize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------- percentile
def test_median_of_four_values():
    assert percentile([1, 2, 3, 4], 0.5) == 2


def test_top_percentile_is_the_worst_observation():
    assert percentile([1, 2, 3, 4], 1.0) == 4


def test_percentile_returns_an_observed_value_not_an_interpolation():
    """P99, которого не показал ни один запрос, — выдуманное число в отчёте."""
    samples = [80.0] * 99 + [800.0]
    assert percentile(samples, 0.99) in samples
    assert percentile(samples, 0.99) == APPROX(80.0)


def test_percentiles_are_monotone_in_q():
    samples = list(range(100))
    assert percentile(samples, 0.5) <= percentile(samples, 0.95) <= percentile(samples, 0.99)


def test_percentile_of_nothing_is_refused():
    with pytest.raises(ValueError):
        percentile([], 0.5)


# ------------------------------------------------------------ prompt_lengths
def test_zero_stddev_gives_one_prompt_over_and_over():
    """Это и есть «loop with one prompt», против которого написан урок."""
    assert prompt_lengths(3, 500, 0, None) == (500, 500, 500)


def test_lengths_are_reproducible_for_the_same_seed():
    a = prompt_lengths(200, 500, 150, random.Random(4))
    b = prompt_lengths(200, 500, 150, random.Random(4))
    assert a == b


def test_lengths_scatter_around_the_requested_mean():
    lengths = prompt_lengths(2000, 500, 150, random.Random(0))
    assert 480 < statistics.mean(lengths) < 520
    assert 130 < statistics.pstdev(lengths) < 170


def test_lengths_never_go_below_the_minimum():
    """Нормальное распределение спокойно выдаёт минус двести токенов."""
    lengths = prompt_lengths(2000, 100, 300, random.Random(0), minimum=1)
    assert min(lengths) >= 1


# ------------------------------------------------------------- make_workload
def test_uniform_workload_reuses_a_single_prefix():
    workload = make_workload(4, 2000, 0, 1, None)
    assert {r["prefix"] for r in workload} == {"prefix-0"}
    assert {r["prompt_tokens"] for r in workload} == {2000}


def test_realistic_workload_spreads_over_the_requested_prefixes():
    workload = make_workload(500, 500, 150, 80, random.Random(0))
    assert len({r["prefix"] for r in workload}) == 80
    assert len({r["prompt_tokens"] for r in workload}) > 100


def test_zero_distinct_prefixes_is_refused():
    with pytest.raises(ValueError):
        make_workload(10, 500, 0, 0, None)


# ---------------------------------------------------------- arrival_schedule
def test_steady_arrivals_are_evenly_spaced():
    assert arrival_schedule("steady", 4.0, 2.0) == APPROX((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5))


def test_arrivals_come_out_sorted():
    times = arrival_schedule("spike", 10.0, 10.0, 100.0)
    assert list(times) == sorted(times)


def test_steady_count_is_rate_times_duration():
    assert len(arrival_schedule("steady", 60.0, 5.0)) == 300


def test_ramp_puts_most_of_the_load_at_the_end():
    """Рамп ищет точку излома по ёмкости — темп обязан расти."""
    times = arrival_schedule("ramp", 10.0, 0.0, 10.0)
    first_half = sum(1 for t in times if t < 5.0)
    second_half = sum(1 for t in times if t >= 5.0)
    assert second_half > 2 * first_half


def test_spike_concentrates_the_load_in_the_middle():
    """Спайк проверяет, успеет ли автоскейлер, — всплеск должен быть резким."""
    times = arrival_schedule("spike", 10.0, 10.0, 100.0)
    inside = sum(1 for t in times if 4.0 <= t < 6.0)
    outside = len(times) - inside
    assert inside == 200
    assert inside > outside


def test_soak_runs_long_at_a_calm_rate():
    """Soak ловит утечки памяти — важна не интенсивность, а длительность."""
    times = arrival_schedule("soak", 3600.0, 1.0)
    assert len(times) == 3600
    assert max(times) > 3500.0


def test_typo_in_the_pattern_name_does_not_silently_become_steady():
    with pytest.raises(ValueError):
        arrival_schedule("burst", 10.0, 10.0)


def test_ramp_without_a_peak_is_refused():
    with pytest.raises(ValueError):
        arrival_schedule("ramp", 10.0, 1.0)


def test_every_documented_pattern_is_supported():
    for pattern in PATTERNS:
        assert len(arrival_schedule(pattern, 10.0, 2.0, peak_rps=20.0)) > 0


# ------------------------------------------------------------------ run_load
def test_light_load_never_waits_and_never_rejects():
    workload = make_workload(20, 500, 150, 20, random.Random(1))
    arrivals = arrival_schedule("steady", 20.0, 1.0)
    records = run_load(workload, arrivals, concurrency=4, queue_limit=5)
    assert all(r["rejected"] is False for r in records)
    assert all(r["wait_ms"] == APPROX(0.0) for r in records)


def test_first_request_with_a_prefix_misses_and_the_next_one_hits():
    workload = make_workload(2, 500, 0, 1, None)
    records = run_load(workload, (0.0, 10.0), concurrency=1, queue_limit=5)
    assert records[0]["cache_hit"] is False
    assert records[1]["cache_hit"] is True
    assert records[0]["ttft_ms"] == APPROX(TTFT_CACHE_MISS_MS)
    assert records[1]["ttft_ms"] == APPROX(TTFT_CACHE_HIT_MS)


def test_service_time_is_ttft_plus_tokens_times_tpot():
    workload = make_workload(1, 500, 0, 1, None, output_tokens=100)
    record = run_load(workload, (0.0,), concurrency=1, queue_limit=0)[0]
    assert record["total_ms"] == APPROX(TTFT_CACHE_MISS_MS + 100 * TPOT_MS)


def test_overload_grows_the_queue_and_then_starts_rejecting():
    workload = make_workload(60, 500, 150, 60, random.Random(1))
    arrivals = arrival_schedule("steady", 15.0, 4.0)
    records = run_load(workload, arrivals, concurrency=4, queue_limit=5)
    assert any(r["rejected"] for r in records)
    assert max(r["wait_ms"] for r in records if not r["rejected"]) > 1000.0


def test_a_request_that_can_start_now_is_never_rejected():
    """queue_limit = 0 не значит «отвергать всех»."""
    workload = make_workload(3, 500, 0, 3, None)
    records = run_load(workload, (0.0, 100.0, 200.0), concurrency=1, queue_limit=0)
    assert all(r["rejected"] is False for r in records)


def test_rejected_requests_carry_no_latency_at_all():
    workload = make_workload(30, 500, 150, 30, random.Random(1))
    arrivals = arrival_schedule("steady", 3.0, 10.0)
    rejected = [r for r in run_load(workload, arrivals, 1, 1) if r["rejected"]]
    assert rejected
    assert all(r["ttft_ms"] is None and r["wait_ms"] is None for r in rejected)


def test_schedule_and_workload_must_be_the_same_length():
    with pytest.raises(ValueError):
        run_load(make_workload(5, 500, 0, 1, None), (0.0, 1.0), 1, 1)


# ----------------------------------------------------------------- summarize
def test_summary_of_a_clean_run():
    workload = make_workload(20, 500, 150, 20, random.Random(1))
    report = summarize(run_load(workload, arrival_schedule("steady", 20.0, 1.0), 4, 5))
    assert (report["n"], report["ok"], report["rejected"]) == (20, 20, 0)
    assert report["reject_rate"] == APPROX(0.0)
    assert report["wait_p99"] == APPROX(0.0)


def test_uniform_prompts_make_the_endpoint_look_ten_times_faster():
    """Prompt-uniformity trap целиком, в двух строках отчёта.

    Один и тот же сервер, один и тот же темп. Разница только в том, что
    uniform-прогон бьёт в один префикс: prefix cache попадает почти всегда,
    и P99 получается в десять раз лучше реального.
    """
    arrivals = arrival_schedule("steady", 100.0, 2.0)
    uniform = summarize(run_load(make_workload(200, 2000, 0, 1, None), arrivals, 8, 10))
    realistic = summarize(
        run_load(make_workload(200, 500, 150, 80, random.Random(0)), arrivals, 8, 10)
    )
    assert uniform["cache_hit_rate"] > 0.95
    assert realistic["cache_hit_rate"] < 0.7
    assert uniform["ttft_p99"] == APPROX(TTFT_CACHE_HIT_MS)
    assert realistic["ttft_p99"] == APPROX(TTFT_CACHE_MISS_MS)


def test_rejections_beautify_the_percentiles_and_must_be_read_alongside_them():
    """Самый опасный отчёт в уроке: чем хуже сервису, тем красивее числа.

    Очередь длиной ноль отбрасывает всё, что не начинается мгновенно, — и в
    перцентили попадают только счастливчики. P99 идеальный, сервис лежит.
    """
    workload = make_workload(60, 500, 150, 60, random.Random(1))
    arrivals = arrival_schedule("steady", 15.0, 4.0)
    with_queue = summarize(run_load(workload, arrivals, 4, 5))
    no_queue = summarize(run_load(workload, arrivals, 4, 0))
    assert no_queue["ttft_p99"] < with_queue["ttft_p99"]
    assert no_queue["reject_rate"] > with_queue["reject_rate"]


def test_percentiles_ignore_the_rejected_requests():
    workload = make_workload(30, 500, 150, 30, random.Random(1))
    report = summarize(run_load(workload, arrival_schedule("steady", 3.0, 10.0), 1, 1))
    assert report["ok"] + report["rejected"] == report["n"]
    assert report["ok"] < report["n"]
    assert report["ttft_p50"] > 0.0


def test_a_run_where_everything_was_rejected_has_no_percentiles():
    """«P99 = 0» на полностью упавшем прогоне хуже, чем ошибка."""
    with pytest.raises(ValueError):
        summarize(({"rejected": True, "ttft_ms": None, "wait_ms": None, "cache_hit": None},))


# --------------------------------------------------------------- apparent_itl
def test_a_fast_client_reports_the_truth():
    assert apparent_itl(10.0, 0.5, 1, workers=1) == APPROX(10.0)


def test_gil_trap_inflates_the_reported_latency_under_concurrency():
    """Сервер не изменился. Изменился только клиент, и он же виноват."""
    assert apparent_itl(10.0, 0.5, 50, workers=1) == APPROX(25.0)


def test_moving_tokenization_to_processes_makes_the_regression_vanish():
    """Так отличают клиентскую «деградацию» от серверной: настоящая не лечится
    добавлением процессов на стороне теста."""
    assert apparent_itl(10.0, 0.5, 50, workers=8) == APPROX(10.0)


def test_zero_workers_is_refused():
    with pytest.raises(ValueError):
        apparent_itl(10.0, 0.5, 10, workers=0)


# -------------------------------------------------------------------- ci_gate
def test_healthy_run_passes_the_gate():
    summary = {"ttft_p95": 700.0, "reject_rate": 0.01}
    assert ci_gate(summary, {"ttft_p95": 800.0, "reject_rate": 0.05}) == (True, ())


def test_breached_latency_breaks_the_build():
    summary = {"ttft_p95": 900.0, "reject_rate": 0.01}
    assert ci_gate(summary, {"ttft_p95": 800.0, "reject_rate": 0.05}) == (False, ("ttft_p95",))


def test_exactly_at_the_threshold_still_passes():
    """«Не более 800 мс» обязано пропускать 800 мс."""
    assert ci_gate({"ttft_p95": 800.0}, {"ttft_p95": 800.0}) == (True, ())


def test_breaches_come_back_in_threshold_order():
    summary = {"ttft_p95": 900.0, "reject_rate": 0.5}
    thresholds = {"reject_rate": 0.05, "ttft_p95": 800.0}
    assert ci_gate(summary, thresholds) == (False, ("reject_rate", "ttft_p95"))


def test_a_gate_on_a_metric_nobody_measured_is_not_a_pass():
    with pytest.raises(KeyError):
        ci_gate({"ttft_p95": 700.0}, {"tpot_ms": 20.0})


def test_gate_reads_a_real_summary_end_to_end():
    workload = make_workload(60, 500, 150, 60, random.Random(1))
    report = summarize(run_load(workload, arrival_schedule("steady", 15.0, 4.0), 4, 5))
    passed, breaches = ci_gate(report, {"ttft_p95": 800.0, "reject_rate": 0.05})
    assert passed is False
    assert set(breaches) == {"ttft_p95", "reject_rate"}
