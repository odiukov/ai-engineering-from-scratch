"""Тесты к уроку «Observability агентов: Langfuse, Phoenix, Opik». Правь exercise.py."""

import pytest

from exercise import (
    categorize_failures,
    ingest_spans,
    judge_session,
    latency_percentile,
    redact_pii,
    session_latency_ms,
    summarize,
    worst_session,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

RUBRIC = {
    "has_final_answer": 1.0,
    "no_tool_errors": 1.0,
    "within_step_budget": 1.0,
    "no_pii_in_output": 1.0,
}


def span(session_id, name, start_ms, end_ms, status="ok", output=None):
    """Спан с временами в миллисекундах — читать удобнее, чем наносекунды."""
    return {
        "session_id": session_id,
        "name": name,
        "start_ns": int(start_ms * 1e6),
        "end_ns": int(end_ms * 1e6),
        "status": status,
        "output": output,
    }


def good_session(session_id, end_ms=10.0):
    """Сессия, проходящая всю рубрику."""
    return [
        span(session_id, "invoke_agent", 0.0, end_ms),
        span(session_id, "tool_call search", 1.0, 2.0),
        span(session_id, "final_answer", end_ms - 1.0, end_ms, output="all good"),
    ]


# ------------------------------------------------------------- ingest_spans
def test_ingest_groups_spans_by_session():
    spans = good_session("s1") + good_session("s2")
    assert sorted(ingest_spans(spans)) == ["s1", "s2"]


def test_ingest_orders_spans_by_start_time():
    spans = [span("s1", "b", 5.0, 9.0), span("s1", "a", 0.0, 3.0)]
    assert [s["name"] for s in ingest_spans(spans)["s1"]] == ["a", "b"]


def test_ingest_result_does_not_depend_on_arrival_order():
    """Спаны приезжают из сети как попало — метрика обязана быть той же."""
    spans = good_session("s1") + good_session("s2")
    assert ingest_spans(spans) == ingest_spans(list(reversed(spans)))


def test_ingest_orders_session_keys_too():
    spans = good_session("s2") + good_session("s1")
    assert list(ingest_spans(spans)) == ["s1", "s2"]


def test_ingest_rejects_a_span_that_ends_before_it_starts():
    with pytest.raises(ValueError):
        ingest_spans([span("s1", "a", 10.0, 1.0)])


# -------------------------------------------------------- session_latency_ms
def test_latency_of_one_span_is_its_duration():
    assert session_latency_ms([span("s1", "a", 0.0, 3.0)]) == APPROX(3.0)


def test_latency_is_wall_clock_not_the_sum_of_spans():
    """Вложенный tool-спан лежит внутри агентского — сумма посчитала бы его дважды."""
    spans = [span("s1", "invoke_agent", 0.0, 5.0), span("s1", "tool_call a", 1.0, 4.0)]
    assert session_latency_ms(spans) == APPROX(5.0)


def test_parallel_tool_calls_do_not_inflate_latency():
    spans = [
        span("s1", "invoke_agent", 0.0, 10.0),
        span("s1", "tool_call a", 1.0, 9.0),
        span("s1", "tool_call b", 1.0, 9.0),
        span("s1", "tool_call c", 1.0, 9.0),
    ]
    assert session_latency_ms(spans) == APPROX(10.0)


def test_latency_of_an_empty_session_is_an_error():
    with pytest.raises(ValueError):
        session_latency_ms([])


# ------------------------------------------------------ latency_percentile
def test_p50_of_five_values():
    assert latency_percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_ignores_input_order():
    assert latency_percentile([9, 1, 5, 3, 7], 80) == latency_percentile([1, 3, 5, 7, 9], 80)


def test_percentile_of_an_empty_sample_is_an_error():
    with pytest.raises(ValueError):
        latency_percentile([], 95)


def test_percentile_zero_is_undefined():
    with pytest.raises(ValueError):
        latency_percentile([1, 2, 3], 0)


# --------------------------------------------------------------- redact_pii
def test_email_is_masked():
    assert redact_pii("write to a.b@x.io") == "write to [email]"


def test_grouped_card_number_is_masked():
    assert redact_pii("card 4111 1111 1111 1111 ok") == "card [card] ok"


def test_short_numbers_survive():
    """Иначе из трейса пропадут номера шагов и коды ошибок."""
    assert redact_pii("step 3 of 10, code 429") == "step 3 of 10, code 429"


def test_redaction_is_idempotent():
    """Guardrail применяют и на приёме, и перед показом."""
    once = redact_pii("a@b.io paid 4111111111111111")
    assert redact_pii(once) == once


# ------------------------------------------------------------- judge_session
def test_clean_session_scores_one():
    verdict = judge_session(good_session("s1"), RUBRIC)
    assert verdict["score"] == APPROX(1.0) and verdict["passed"] is True


def test_tool_error_is_reported_as_a_reason():
    spans = good_session("s1")
    spans[1]["status"] = "error"
    verdict = judge_session(spans, RUBRIC)
    assert verdict["reasons"] == ["no_tool_errors"]
    assert verdict["score"] == APPROX(0.75)


def test_score_does_not_depend_on_the_weight_scale():
    """Иначе рубрику нельзя сравнить с порогом: максимум у каждой свой."""
    spans = good_session("s1")
    spans[1]["status"] = "error"
    doubled = {check: weight * 2 for check, weight in RUBRIC.items()}
    assert judge_session(spans, doubled)["score"] == judge_session(spans, RUBRIC)["score"]


def test_step_budget_is_enforced():
    spans = [span("s1", f"tool_call t{i}", i, i + 1) for i in range(9)]
    spans.append(span("s1", "final_answer", 9, 10, output="done"))
    assert "within_step_budget" in judge_session(spans, RUBRIC)["reasons"]
    assert judge_session(spans, RUBRIC, max_steps=20)["reasons"] == []


def test_reasons_are_sorted_so_two_identical_sessions_read_the_same():
    spans = good_session("s1")[:2]
    spans[1]["status"] = "error"
    assert judge_session(spans, RUBRIC)["reasons"] == ["has_final_answer", "no_tool_errors"]


def test_unknown_rubric_check_is_rejected():
    with pytest.raises(ValueError):
        judge_session(good_session("s1"), {"is_polite": 1.0})


# -------------------------------------------------------- categorize_failures
def test_categorize_counts_sessions_per_reason():
    spans = []
    for name in ("s1", "s2"):
        session = good_session(name)
        session[1]["status"] = "error"
        spans += session
    assert categorize_failures(ingest_spans(spans), RUBRIC) == {"no_tool_errors": 2}


def test_one_session_counts_once_per_reason_however_many_spans_failed():
    """Иначе одна длинная сессия перевесит в топе двадцать разных."""
    spans = good_session("s1")
    for i in range(20):
        spans.append(span("s1", f"tool_call e{i}", i, i + 1, status="error"))
    counts = categorize_failures(ingest_spans(spans), RUBRIC)
    assert counts["no_tool_errors"] == 1


def test_categorize_is_empty_when_everything_passes():
    assert categorize_failures(ingest_spans(good_session("s1")), RUBRIC) == {}


# ----------------------------------------------------------------- summarize
def test_failure_rate_is_a_fraction_of_sessions():
    spans = good_session("s1") + good_session("s2")
    spans[0]["status"] = "error"
    summary = summarize(ingest_spans(spans), RUBRIC, slow_ms=1000.0)
    assert summary["sessions"] == 2
    assert summary["failure_rate"] == APPROX(0.5)


def test_top_reasons_break_ties_alphabetically():
    spans = []
    for name in ("s1", "s2"):
        session = good_session(name)[:2]
        session[1]["status"] = "error"
        spans += session
    summary = summarize(ingest_spans(spans), RUBRIC, slow_ms=1000.0)
    assert summary["top_reasons"] == [("has_final_answer", 2), ("no_tool_errors", 2)]


def test_mean_hides_a_single_long_trace_but_max_and_slow_list_do_not():
    """Дашборд из одного среднего врёт: 310 ms «в среднем» на выброс 30 секунд."""
    spans = []
    for i in range(99):
        spans += good_session(f"s{i:03d}", end_ms=10.0)
    spans += good_session("s999", end_ms=30_000.0)
    summary = summarize(ingest_spans(spans), RUBRIC, slow_ms=1000.0)
    assert summary["latency_mean_ms"] < 400.0
    assert summary["latency_max_ms"] == APPROX(30_000.0)
    assert summary["slow_sessions"] == ["s999"]


def test_p95_catches_a_tail_of_several_slow_sessions():
    """А вот шесть медленных из ста уже видны в p95 — среднее их всё ещё гасит."""
    spans = []
    for i in range(94):
        spans += good_session(f"s{i:03d}", end_ms=10.0)
    for i in range(6):
        spans += good_session(f"z{i:03d}", end_ms=30_000.0)
    summary = summarize(ingest_spans(spans), RUBRIC, slow_ms=1000.0)
    assert summary["latency_mean_ms"] < 2000.0
    assert summary["latency_p95_ms"] == APPROX(30_000.0)


def test_summary_does_not_depend_on_span_arrival_order():
    spans = good_session("s1") + good_session("s2", end_ms=50.0)
    spans[1]["status"] = "error"
    forward = summarize(ingest_spans(spans), RUBRIC, slow_ms=20.0)
    backward = summarize(ingest_spans(list(reversed(spans))), RUBRIC, slow_ms=20.0)
    assert forward == backward


# -------------------------------------------------------------- worst_session
def test_worst_session_is_the_lowest_scoring_one():
    spans = good_session("s1")
    broken = good_session("s2")
    broken[1]["status"] = "error"
    spans += broken
    assert worst_session(ingest_spans(spans), RUBRIC)["session_id"] == "s2"


def test_equal_scores_are_broken_by_latency():
    spans = good_session("s1", end_ms=10.0) + good_session("s2", end_ms=900.0)
    spans[0]["status"] = "error"
    spans[3]["status"] = "error"
    assert worst_session(ingest_spans(spans), RUBRIC)["session_id"] == "s2"


def test_fully_equal_sessions_are_broken_by_id():
    """Без третьего ключа ссылка в тикете начнёт открывать разные прогоны."""
    spans = good_session("b", end_ms=10.0) + good_session("a", end_ms=10.0)
    for s in spans:
        if s["name"].startswith("tool_call"):
            s["status"] = "error"
    assert worst_session(ingest_spans(spans), RUBRIC)["session_id"] == "a"


def test_worst_session_reports_why_it_lost():
    spans = good_session("s1")
    spans[2]["output"] = "mail a.b@x.io"
    worst = worst_session(ingest_spans(spans), RUBRIC)
    assert worst["reasons"] == ["no_pii_in_output"]
    assert worst["latency_ms"] == APPROX(10.0)


