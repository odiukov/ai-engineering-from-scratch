"""Тесты к уроку «Наблюдаемость LLM: трассы, критический путь и сэмплирование». Правь exercise.py."""

import random

import pytest

from exercise import (
    BYTES_PER_TRACE,
    PRICE_PER_GB_MONTH_LAKE,
    PRICE_PER_GB_MONTH_MONOLITHIC,
    TraceError,
    critical_path,
    critical_path_ms,
    index_by_parent,
    keep_trace,
    make_span,
    retention_cost,
    sample_traces,
    trace_cost,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def rng():
    return random.Random(20260807)


def parallel_trace():
    """Корень [0,100] и два ребёнка, оба [0,100] — запущены параллельно."""
    return [
        make_span("root", None, "agent", 0, 100),
        make_span("a", "root", "tool.search", 0, 100),
        make_span("b", "root", "tool.fetch", 0, 100),
    ]


def sequential_trace():
    """Корень [0,100] и два ребёнка [0,50] и [50,100] — строго друг за другом."""
    return [
        make_span("root", None, "agent", 0, 100),
        make_span("a", "root", "tool.search", 0, 50),
        make_span("b", "root", "llm.chat", 50, 100),
    ]


def ok_trace(cost):
    return [make_span("r", None, "agent", 0, 10, cost_usd=cost)]


def error_trace(cost):
    return [
        make_span("r", None, "agent", 0, 10, cost_usd=cost),
        make_span("c", "r", "llm.chat", 1, 9, status="error"),
    ]


# ---------------------------------------------------------------- make_span
def test_span_duration_is_end_minus_start():
    assert make_span("a", None, "agent", 10, 45)["duration_ms"] == 35


def test_root_span_has_no_parent():
    assert make_span("a", None, "agent", 0, 1)["parent_id"] is None


def test_span_with_end_before_start_is_rejected():
    """Отрицательная длительность — сломанный инструментатор, а не данные."""
    with pytest.raises(TraceError):
        make_span("c", "a", "tool", 40, 10)


def test_span_with_unknown_status_is_rejected():
    with pytest.raises(TraceError):
        make_span("c", "a", "tool", 0, 10, status="maybe")


def test_zero_length_span_is_allowed():
    """Мгновенный спан — это нормально: кэш отдал ответ за доли миллисекунды."""
    assert make_span("c", "a", "cache.hit", 7, 7)["duration_ms"] == 0


# ---------------------------------------------------------- index_by_parent
def test_root_spans_are_indexed_under_none():
    index = index_by_parent(parallel_trace())
    assert [s["span_id"] for s in index[None]] == ["root"]


def test_children_are_grouped_under_their_parent():
    index = index_by_parent(sequential_trace())
    assert sorted(s["span_id"] for s in index["root"]) == ["a", "b"]


def test_children_are_sorted_by_end_time():
    """Порядок нужен критическому пути, он идёт от конца назад."""
    index = index_by_parent(sequential_trace())
    assert [s["span_id"] for s in index["root"]] == ["a", "b"]


def test_duplicate_span_id_is_rejected():
    spans = parallel_trace() + [make_span("a", "root", "dup", 0, 5)]
    with pytest.raises(TraceError):
        index_by_parent(spans)


def test_leaf_span_has_no_entry_in_the_index():
    index = index_by_parent(sequential_trace())
    assert "a" not in index


# --------------------------------------------------------------- trace_cost
def test_empty_trace_costs_nothing():
    assert trace_cost([]) == APPROX(0.0)


def test_trace_cost_sums_every_span():
    spans = [
        make_span("r", None, "agent", 0, 10, cost_usd=0.001),
        make_span("a", "r", "llm.chat", 1, 5, cost_usd=0.012),
        make_span("b", "r", "llm.chat", 5, 9, cost_usd=0.007),
    ]
    assert trace_cost(spans) == APPROX(0.02)


def test_expensive_span_is_not_the_root():
    """Бюджет жжёт не корень, а вложенные вызовы — сумма это показывает."""
    spans = [
        make_span("r", None, "agent", 0, 10, cost_usd=0.0),
        make_span("a", "r", "llm.chat", 1, 9, cost_usd=0.5),
    ]
    assert trace_cost(spans) > spans[0]["cost_usd"]


# ------------------------------------------------------------ critical_path
def test_parallel_siblings_leave_only_one_on_the_path():
    assert critical_path(parallel_trace(), "root") == ["root", "b"]


def test_sequential_siblings_all_land_on_the_path():
    assert critical_path(sequential_trace(), "root") == ["root", "b", "a"]


def test_critical_path_of_a_single_span_is_itself():
    assert critical_path([make_span("r", None, "agent", 0, 5)], "r") == ["r"]


def test_critical_path_descends_into_grandchildren():
    spans = [
        make_span("root", None, "agent", 0, 100),
        make_span("a", "root", "retrieve", 0, 40),
        make_span("a1", "a", "embed", 0, 40),
        make_span("b", "root", "llm.chat", 40, 100),
    ]
    assert critical_path(spans, "root") == ["root", "b", "a", "a1"]


def test_unknown_root_id_is_rejected():
    with pytest.raises(TraceError):
        critical_path(parallel_trace(), "nope")


# --------------------------------------------------------- critical_path_ms
def test_parallel_spans_make_the_path_shorter_than_the_sum():
    spans = parallel_trace()
    total = sum(s["duration_ms"] for s in spans if s["parent_id"] is not None)
    assert critical_path_ms(spans, "root") == APPROX(100.0)
    assert critical_path_ms(spans, "root") < total


def test_sequential_spans_make_the_path_equal_to_the_sum():
    spans = sequential_trace()
    total = sum(s["duration_ms"] for s in spans if s["parent_id"] is not None)
    assert critical_path_ms(spans, "root") == APPROX(total)


def test_parent_time_is_not_counted_twice():
    """Корень [0,100] и один ребёнок [0,100]: путь 100 мс, а не 200."""
    spans = [
        make_span("root", None, "agent", 0, 100),
        make_span("a", "root", "llm.chat", 0, 100),
    ]
    assert critical_path_ms(spans, "root") == APPROX(100.0)


def test_ten_parallel_tool_calls_do_not_add_up():
    """Десять параллельных инструментов по 2 с — ответ за 2 с, не за 20."""
    spans = [make_span("root", None, "agent", 0, 2000)]
    spans += [make_span(f"t{i}", "root", "tool", 0, 2000) for i in range(10)]
    assert critical_path_ms(spans, "root") == APPROX(2000.0)
    assert sum(s["duration_ms"] for s in spans) == 22000


# --------------------------------------------------------------- keep_trace
def test_error_trace_is_kept_even_at_zero_sample_rate():
    assert keep_trace(error_trace(0.0), rng(), 0.0, 1e9) == (True, "error")


def test_expensive_trace_is_kept_even_at_zero_sample_rate():
    assert keep_trace(ok_trace(5.0), rng(), 0.0, 1.0) == (True, "expensive")


def test_cheap_success_is_dropped_at_zero_sample_rate():
    assert keep_trace(ok_trace(0.001), rng(), 0.0, 1.0) == (False, "dropped")


def test_cheap_success_is_kept_at_full_sample_rate():
    assert keep_trace(ok_trace(0.001), rng(), 1.0, 1.0) == (True, "sampled")


def test_error_rule_does_not_consume_randomness():
    """Ловушка: монетку бросают только на третьем правиле.

    Если дёрнуть rng заранее, поток случайных чисел сместится и следующее
    решение окажется другим.
    """
    r = rng()
    keep_trace(error_trace(0.0), r, 0.5, 1e9)
    assert r.random() == rng().random()


# ------------------------------------------------------------ sample_traces
def test_no_error_trace_is_ever_dropped():
    traces = [error_trace(0.0) for _ in range(20)] + [ok_trace(0.0) for _ in range(20)]
    stats = sample_traces(traces, rng(), 0.05, 1e9)
    assert stats["dropped_errors"] == 0
    assert stats["kept_errors"] == 20


def test_kept_and_dropped_add_up_to_the_input():
    traces = [ok_trace(0.001) for _ in range(100)]
    stats = sample_traces(traces, rng(), 0.5, 1e9)
    assert stats["kept"] + stats["dropped"] == 100


def test_sampling_keeps_roughly_the_requested_fraction():
    traces = [ok_trace(0.001) for _ in range(2000)]
    stats = sample_traces(traces, rng(), 0.05, 1e9)
    assert 0.03 < stats["kept_fraction"] < 0.07


def test_dropped_cost_shows_the_blind_spot():
    """Отброшенные трассы — это часть счёта, которую больше не видно."""
    traces = [ok_trace(0.01) for _ in range(100)]
    stats = sample_traces(traces, rng(), 0.05, 1e9)
    assert stats["dropped_cost_usd"] > stats["kept_cost_usd"]
    assert stats["kept_cost_usd"] + stats["dropped_cost_usd"] == APPROX(1.0)


def test_empty_input_gives_zero_fraction_not_a_crash():
    assert sample_traces([], rng(), 0.05, 1.0)["kept_fraction"] == APPROX(0.0)


def test_reasons_are_counted_separately():
    traces = [error_trace(0.0), ok_trace(9.0), ok_trace(0.0)]
    stats = sample_traces(traces, rng(), 0.0, 1.0)
    assert stats["by_reason"] == {"error": 1, "expensive": 1, "dropped": 1}


# ----------------------------------------------------------- retention_cost
def test_full_retention_of_a_million_traces_is_monthly_not_daily():
    """4.5 ГБ в день на монолитном ингесте — $67.5 в месяц."""
    assert retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_MONOLITHIC) == APPROX(67.5)


def test_data_lake_price_is_two_orders_cheaper():
    mono = retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_MONOLITHIC)
    lake = retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_LAKE)
    assert mono / lake == APPROX(100.0)


def test_retention_cost_scales_with_the_kept_fraction():
    full = retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_MONOLITHIC)
    sampled = retention_cost(1_000_000, 0.05, PRICE_PER_GB_MONTH_MONOLITHIC)
    assert sampled == APPROX(full * 0.05)


def test_sampling_report_feeds_the_bill():
    """Связка: доля из sample_traces идёт прямо в счёт за хранение."""
    traces = [ok_trace(0.001) for _ in range(1000)]
    stats = sample_traces(traces, rng(), 0.05, 1e9)
    bill = retention_cost(1_000_000, stats["kept_fraction"], PRICE_PER_GB_MONTH_MONOLITHIC)
    expected = 1_000_000 * stats["kept_fraction"] * BYTES_PER_TRACE / 1e9 * 30 * 0.50
    assert bill == APPROX(expected)
