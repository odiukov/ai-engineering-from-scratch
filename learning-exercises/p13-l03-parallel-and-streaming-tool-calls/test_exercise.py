"""Тесты к уроку «Параллельные и потоковые вызовы инструментов». Правь exercise.py."""

import random

import pytest

from exercise import (
    accumulate_stream,
    correlate_results,
    parallel_batches,
    parallel_duration,
    sequential_duration,
    speedup,
    stream_completion_order,
    try_parse_arguments,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

LATENCIES = [400, 600, 800]

# Три параллельных вызова, чанки которых перемешаны на одном проводе.
# Так выглядит настоящий стрим OpenAI или Anthropic.
INTERLEAVED = [
    {"type": "call_start", "id": "A", "name": "get_weather"},
    {"type": "call_start", "id": "B", "name": "get_weather"},
    {"type": "call_start", "id": "C", "name": "get_weather"},
    {"type": "args_delta", "id": "A", "chunk": '{"city"'},
    {"type": "args_delta", "id": "B", "chunk": '{"city'},
    {"type": "args_delta", "id": "A", "chunk": ': "Tokyo"'},
    {"type": "args_delta", "id": "C", "chunk": '{"city": "Zu'},
    {"type": "args_delta", "id": "B", "chunk": '": "Bengaluru"}'},
    {"type": "call_stop", "id": "B"},
    {"type": "args_delta", "id": "A", "chunk": "}"},
    {"type": "call_stop", "id": "A"},
    {"type": "args_delta", "id": "C", "chunk": 'rich"}'},
    {"type": "call_stop", "id": "C"},
]

CALLS = [
    {"id": "A", "name": "get_weather", "arguments": {"city": "Tokyo"}},
    {"id": "B", "name": "get_weather", "arguments": {"city": "Bengaluru"}},
    {"id": "C", "name": "get_weather", "arguments": {"city": "Zurich"}},
]

RESULTS = [
    {"tool_call_id": "C", "content": "4C"},
    {"tool_call_id": "A", "content": "12C"},
    {"tool_call_id": "B", "content": "28C"},
]


# --------------------------------------------------- длительности и ускорение
def test_sequential_pays_the_sum_of_latencies():
    assert sequential_duration(LATENCIES) == 1800


def test_sequential_of_nothing_is_zero():
    assert sequential_duration([]) == 0


def test_parallel_pays_only_the_slowest_call():
    assert parallel_duration(LATENCIES) == 800


def test_parallel_of_nothing_is_zero_not_an_exception():
    """max([]) бросает ValueError — пустой веер обязан давать 0."""
    assert parallel_duration([]) == 0


def test_speedup_is_sum_over_max():
    assert speedup(LATENCIES) == APPROX(1800 / 800)


def test_equal_latencies_give_speedup_equal_to_call_count():
    assert speedup([500] * 4) == APPROX(4.0)


def test_one_dominating_call_makes_parallelism_pointless():
    """Один вызов на 10 секунд и два по 10 мс — веер почти ничего не даёт."""
    assert speedup([10000, 10, 10]) < 1.01


def test_speedup_of_nothing_is_one_not_a_division_by_zero():
    assert speedup([]) == APPROX(1.0)
    assert speedup([0, 0]) == APPROX(1.0)


# ------------------------------------------------------- try_parse_arguments
def test_complete_json_object_parses():
    assert try_parse_arguments('{"city": "Tokyo"}') == {"city": "Tokyo"}


def test_partial_json_returns_none_instead_of_raising():
    """Ловушка «разобрать слишком рано»: json.loads тут бросил бы исключение."""
    assert try_parse_arguments('{"city": "Tok') is None


def test_balanced_braces_inside_a_string_do_not_mean_complete():
    """Подсчёт скобок сказал бы «готово», а JSON невалиден."""
    assert try_parse_arguments('{"city": "a}"') is None


def test_empty_buffer_means_a_call_without_arguments():
    assert try_parse_arguments("") == {}


def test_non_object_json_is_rejected():
    """arguments по спецификации всегда объект, не массив и не число."""
    assert try_parse_arguments("[1, 2]") is None
    assert try_parse_arguments("42") is None


# --------------------------------------------------------- accumulate_stream
def test_interleaved_chunks_reassemble_per_id():
    assert accumulate_stream(INTERLEAVED) == {
        "A": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        "B": {"name": "get_weather", "arguments": {"city": "Bengaluru"}},
        "C": {"name": "get_weather", "arguments": {"city": "Zurich"}},
    }


def test_a_call_without_stop_is_not_returned():
    """Поток оборвался или вызов отменили — разбирать нечего."""
    assert accumulate_stream(INTERLEAVED[:-1]) == {
        "A": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        "B": {"name": "get_weather", "arguments": {"city": "Bengaluru"}},
    }


def test_call_with_no_deltas_yields_empty_arguments():
    events = [
        {"type": "call_start", "id": "A", "name": "get_time"},
        {"type": "call_stop", "id": "A"},
    ]
    assert accumulate_stream(events) == {"A": {"name": "get_time", "arguments": {}}}


def test_chunk_for_an_unknown_id_is_refused():
    with pytest.raises(ValueError):
        accumulate_stream([{"type": "args_delta", "id": "ghost", "chunk": "{}"}])


def test_unknown_event_type_is_refused():
    with pytest.raises(ValueError):
        accumulate_stream([{"type": "message_stop", "id": "A"}])


def test_stop_on_a_broken_buffer_is_refused():
    events = [
        {"type": "call_start", "id": "A", "name": "get_weather"},
        {"type": "args_delta", "id": "A", "chunk": '{"city": "Tok'},
        {"type": "call_stop", "id": "A"},
    ]
    with pytest.raises(ValueError):
        accumulate_stream(events)


# --------------------------------------------------- stream_completion_order
def test_completion_order_follows_stop_events_not_start_events():
    """B стартовал вторым, а закрылся первым — стартовать его можно раньше A."""
    assert stream_completion_order(INTERLEAVED) == ["B", "A", "C"]


def test_unfinished_calls_are_absent_from_the_completion_order():
    assert stream_completion_order(INTERLEAVED[:-1]) == ["B", "A"]


def test_completion_order_lists_exactly_the_ready_calls():
    ready = accumulate_stream(INTERLEAVED)
    assert sorted(stream_completion_order(INTERLEAVED)) == sorted(ready)


# ---------------------------------------------------------- correlate_results
def test_results_are_matched_by_id_not_by_position():
    matched = correlate_results(CALLS, RESULTS)
    assert [(m["tool_call_id"], m["content"]) for m in matched] == [
        ("A", "12C"),
        ("B", "28C"),
        ("C", "4C"),
    ]


def test_shuffling_the_completion_order_changes_nothing():
    """Веер завершается непредсказуемо — результат обязан быть тем же."""
    rng = random.Random(0)
    expected = correlate_results(CALLS, RESULTS)
    for _ in range(20):
        shuffled = list(RESULTS)
        rng.shuffle(shuffled)
        assert correlate_results(CALLS, shuffled) == expected


def test_two_parallel_calls_to_the_same_tool_stay_distinguishable():
    """Сопоставление по имени инструмента сломалось бы ровно здесь."""
    calls = [
        {"id": "A", "name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"id": "B", "name": "get_weather", "arguments": {"city": "Zurich"}},
    ]
    matched = correlate_results(
        calls,
        [{"tool_call_id": "B", "content": "4C"}, {"tool_call_id": "A", "content": "12C"}],
    )
    assert [m["content"] for m in matched] == ["12C", "4C"]


def test_missing_result_is_refused():
    with pytest.raises(ValueError):
        correlate_results(CALLS, RESULTS[:2])


def test_result_for_an_unknown_call_is_refused():
    with pytest.raises(ValueError):
        correlate_results(CALLS, RESULTS + [{"tool_call_id": "Z", "content": "?"}])


def test_duplicate_result_for_one_id_is_refused():
    with pytest.raises(ValueError):
        correlate_results(CALLS, RESULTS + [{"tool_call_id": "A", "content": "12C"}])


# ---------------------------------------------------------- parallel_batches
def test_independent_calls_all_fit_in_one_wave():
    assert parallel_batches(CALLS, {}) == [["A", "B", "C"]]


def test_a_dependency_pushes_the_call_into_the_next_wave():
    calls = [{"id": "create"}, {"id": "write"}, {"id": "weather"}]
    assert parallel_batches(calls, {"write": ["create"]}) == [
        ["create", "weather"],
        ["write"],
    ]


def test_a_chain_of_three_gives_three_waves():
    calls = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert parallel_batches(calls, {"b": ["a"], "c": ["b"]}) == [["a"], ["b"], ["c"]]


def test_every_call_appears_exactly_once_across_the_waves():
    calls = [{"id": x} for x in "abcde"]
    waves = parallel_batches(calls, {"c": ["a"], "d": ["a", "b"], "e": ["d"]})
    flat = [x for wave in waves for x in wave]
    assert sorted(flat) == list("abcde")


def test_dependency_cycle_is_refused():
    calls = [{"id": "a"}, {"id": "b"}]
    with pytest.raises(ValueError):
        parallel_batches(calls, {"a": ["b"], "b": ["a"]})


def test_dependency_on_an_unknown_call_is_refused():
    with pytest.raises(ValueError):
        parallel_batches([{"id": "a"}], {"a": ["ghost"]})
