"""Тесты к уроку «Метрики инференса: TTFT, TPOT, ITL, goodput, P99». Правь exercise.py."""

import pytest

from exercise import (
    CONSUMER_SLO,
    TraceTooShortError,
    e2e_ms,
    goodput,
    itl_ms,
    latency_summary,
    percentile,
    slo_breakdown,
    throughput_tokens_per_s,
    ttft_ms,
)

APPROX = lambda x: pytest.approx(x, rel=1e-9)


def req(ttft, tpot, tokens):
    """Один запрос в том виде, в каком его отдаёт benchmark-клиент."""
    return {"ttft_ms": ttft, "tpot_ms": tpot, "output_tokens": tokens}


# ---------------------------------------------------------------- ttft_ms
def test_ttft_is_queue_plus_network_plus_prefill():
    assert ttft_ms(40, 12, 110) == APPROX(162.0)


def test_ttft_of_a_long_prompt_is_all_prefill():
    assert ttft_ms(0, 0, 800) == APPROX(800.0)


def test_queue_time_hurts_ttft_as_much_as_prefill():
    """Очередь и prefill входят в TTFT одинаково — пользователь их не различает."""
    assert ttft_ms(300, 0, 100) == ttft_ms(0, 0, 400)


# ----------------------------------------------------------------- e2e_ms
def test_e2e_matches_the_reference_llama_numbers():
    assert e2e_ms(162.0, 7.33, 127) == APPROX(162.0 + 7.33 * 126)


def test_ttft_already_contains_the_first_output_token():
    assert e2e_ms(162.0, 7.33, 1) == APPROX(162.0)


def test_short_answers_are_dominated_by_ttft():
    """10 токенов при TTFT 800 мс: decode — меньше десятой доли ответа."""
    total = e2e_ms(800.0, 7.0, 10)
    assert 800.0 / total > 0.9


def test_long_answers_are_dominated_by_tpot():
    total = e2e_ms(162.0, 7.33, 1000)
    assert (7.33 * 999) / total > 0.9


def test_response_network_time_adds_on_top():
    assert e2e_ms(100.0, 5.0, 10, network_response_ms=50.0) == APPROX(
        e2e_ms(100.0, 5.0, 10) + 50.0
    )


# ----------------------------------------------------------------- itl_ms
def test_genai_perf_divides_by_intervals_not_tokens():
    assert itl_ms(500, 700, 100, "genai-perf") == APPROX(700 / 99)


def test_llmperf_drags_ttft_into_the_number():
    assert itl_ms(500, 700, 100, "llmperf") == APPROX(12.0)


def test_the_two_tools_disagree_on_the_same_trace():
    """Один и тот же прогон, разница почти вдвое — всегда называй инструмент."""
    assert itl_ms(500, 700, 100, "llmperf") > itl_ms(500, 700, 100, "genai-perf")


def test_the_gap_shrinks_when_ttft_is_small():
    """Расхождение тулов — это по сути размазанный TTFT: нет TTFT — нет спора."""
    big = itl_ms(500, 700, 100, "llmperf") - itl_ms(500, 700, 100, "genai-perf")
    small = itl_ms(5, 700, 100, "llmperf") - itl_ms(5, 700, 100, "genai-perf")
    assert small < big


def test_single_token_trace_has_no_intervals_at_all():
    with pytest.raises(TraceTooShortError):
        itl_ms(500, 0, 1, "genai-perf")


def test_unknown_tool_is_rejected_instead_of_silently_guessed():
    with pytest.raises(ValueError):
        itl_ms(500, 700, 100, "vllm-bench")


# ------------------------------------------------------------- percentile
def test_percentile_uses_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_p99_of_a_hundred_values_is_the_ninety_ninth():
    assert percentile(list(range(1, 101)), 99) == 99


def test_percentile_100_is_the_worst_observation():
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_percentile_is_monotone_in_p():
    sample = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    assert percentile(sample, 50) <= percentile(sample, 90) <= percentile(sample, 99)


def test_percentile_of_an_empty_sample_is_undefined():
    with pytest.raises(ValueError):
        percentile([], 99)


# --------------------------------------------------------- latency_summary
def test_summary_reports_the_whole_triple_plus_mean():
    s = latency_summary([1, 2, 3, 4, 5])
    assert (s["p50"], s["p99"], s["mean"]) == (3, 5, APPROX(3.0))


def test_mean_does_not_reconstruct_the_tail():
    """Одинаковое среднее, P99 отличается в тридцать раз — вот почему P99 обязателен."""
    flat = latency_summary([10] * 100)
    spiky = latency_summary([4] * 98 + [304] * 2)
    assert flat["mean"] == APPROX(spiky["mean"])
    assert spiky["p99"] > 30 * flat["p99"]


# --------------------------------------------------- throughput_tokens_per_s
def test_throughput_counts_output_tokens_over_wall_clock():
    assert throughput_tokens_per_s([req(100, 7, 150)], 1.0) == APPROX(150.0)


def test_throughput_ignores_how_the_latency_was_distributed():
    fast = [req(100, 7, 150) for _ in range(100)]
    slow = [req(3000, 60, 150) for _ in range(100)]
    assert throughput_tokens_per_s(fast, 10.0) == throughput_tokens_per_s(slow, 10.0)


# ---------------------------------------------------------------- goodput
def test_a_healthy_request_counts_as_good():
    assert goodput([req(100, 7, 100)], CONSUMER_SLO) == APPROX(1.0)


def test_one_violated_constraint_is_enough_to_lose_the_request():
    """SLO — это И: 900 мс TTFT портит запрос, даже если decode идеален."""
    assert goodput([req(900, 7, 100)], CONSUMER_SLO) == APPROX(0.0)


def test_goodput_is_the_share_of_fully_conforming_requests():
    reqs = [req(100, 7, 100)] * 3 + [req(900, 7, 100)]
    assert goodput(reqs, CONSUMER_SLO) == APPROX(0.75)


def test_goodput_uses_tpot_only_after_the_first_token():
    slo = {"ttft_ms": 100.0, "tpot_ms": 50.0, "e2e_ms": 100.0}
    assert goodput([req(100.0, 50.0, 1)], slo) == APPROX(1.0)


def test_goodput_falls_while_throughput_stays_flat():
    """Тот же tok/s, та же длина ответов — а качество сервиса рухнуло на 40%."""
    healthy = [req(100, 7, 150) for _ in range(100)]
    degraded = [req(100, 7, 150) for _ in range(60)] + [req(2500, 7, 150) for _ in range(40)]
    assert throughput_tokens_per_s(healthy, 10.0) == throughput_tokens_per_s(degraded, 10.0)
    assert goodput(healthy, CONSUMER_SLO) == APPROX(1.0)
    assert goodput(degraded, CONSUMER_SLO) == APPROX(0.6)


def test_tightening_the_slo_can_only_lower_goodput():
    reqs = [req(100 + 10 * i, 7 + 0.1 * i, 100) for i in range(50)]
    loose = goodput(reqs, CONSUMER_SLO)
    tight = goodput(reqs, {"ttft_ms": 300.0, "tpot_ms": 10.0, "e2e_ms": 1500.0})
    assert tight <= loose


def test_goodput_of_an_empty_run_is_undefined():
    with pytest.raises(ValueError):
        goodput([], CONSUMER_SLO)


# ----------------------------------------------------------- slo_breakdown
def test_breakdown_names_every_violated_constraint():
    counts = slo_breakdown([req(900, 40, 10)], CONSUMER_SLO)
    assert (counts["ttft"], counts["tpot"], counts["e2e"], counts["any"]) == (1, 1, 0, 1)


def test_one_request_is_counted_once_in_any_however_many_constraints_it_broke():
    counts = slo_breakdown([req(900, 40, 200)], CONSUMER_SLO)
    assert counts["ttft"] + counts["tpot"] + counts["e2e"] > counts["any"]


def test_breakdown_agrees_with_goodput():
    reqs = [req(100, 7, 100)] * 3 + [req(900, 7, 100)] + [req(100, 40, 100)]
    counts = slo_breakdown(reqs, CONSUMER_SLO)
    assert counts["any"] / len(reqs) == APPROX(1 - goodput(reqs, CONSUMER_SLO))


def test_breakdown_points_at_the_dominant_failure():
    """Все нарушения по TPOT — чинить надо chunked prefill, а не очередь."""
    reqs = [req(100, 40, 100) for _ in range(10)]
    counts = slo_breakdown(reqs, CONSUMER_SLO)
    assert counts["tpot"] == 10
    assert counts["ttft"] == 0
