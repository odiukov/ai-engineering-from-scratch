"""Тесты к уроку «Кэширование, rate limiting и стоимость». Правь exercise.py."""

import pytest

from exercise import (
    MODEL_PRICING,
    cache_key,
    cache_lookup,
    cache_store,
    calculate_cost,
    route_model,
    serve_query,
    summarize_usage,
    token_bucket_take,
)

APPROX = lambda x: pytest.approx(x, abs=1e-12)

MSGS = [{"role": "user", "content": "What is the return policy?"}]
SAME_MSGS = [{"role": "USER", "content": "  what is   the RETURN POLICY?  "}]
OTHER_MSGS = [{"role": "user", "content": "What are your store hours?"}]


def new_cache(ttl=60.0, max_size=8):
    """Пустой кэш нужной формы — тестам он нужен и на незаполненной заготовке."""
    return {"entries": {}, "hits": 0, "misses": 0, "ttl": ttl, "max_size": max_size}


def bucket(tokens=100.0, capacity=100.0, refill_rate=10.0, last_refill=0.0):
    return {
        "tokens": tokens,
        "capacity": capacity,
        "refill_rate": refill_rate,
        "last_refill": last_refill,
    }


# ------------------------------------------------------------ calculate_cost
def test_cost_splits_input_and_output():
    cost = calculate_cost("gpt-4o", 1000, 500)
    assert cost["input_cost"] == APPROX(0.0025)
    assert cost["output_cost"] == APPROX(0.005)
    assert cost["total_cost"] == APPROX(0.0075)


def test_output_tokens_cost_more_than_input_tokens():
    """Выход дороже входа у всех моделей — отсюда и экономия на длине ответа."""
    for model in MODEL_PRICING:
        long_answer = calculate_cost(model, 100, 1000)["total_cost"]
        long_prompt = calculate_cost(model, 1000, 100)["total_cost"]
        assert long_answer > long_prompt, model


def test_prompt_cache_makes_the_input_cheaper():
    plain = calculate_cost("claude-sonnet-4", 2000, 500)["total_cost"]
    cached = calculate_cost("claude-sonnet-4", 2000, 500, cached_input_tokens=1500)["total_cost"]
    assert cached < plain


def test_cached_tokens_are_part_of_the_input_not_extra():
    """Если считать их сверх input, счёт вырастет ровно на сэкономленное."""
    cost = calculate_cost("gpt-4o", 1000, 500, cached_input_tokens=800)
    assert cost["total_cost"] == APPROX(0.0005 + 0.001 + 0.005)


def test_more_cached_tokens_than_input_is_an_error():
    with pytest.raises(ValueError):
        calculate_cost("gpt-4o", 100, 50, cached_input_tokens=500)


def test_unknown_model_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        calculate_cost("gpt-9-ultra", 100, 50)


def test_cheap_model_really_is_cheaper_on_the_same_traffic():
    expensive = calculate_cost("claude-opus-4", 1000, 500)["total_cost"]
    cheap = calculate_cost("gpt-4o-mini", 1000, 500)["total_cost"]
    assert cheap * 20 < expensive


# ---------------------------------------------------------------- cache_key
def test_key_ignores_case_and_extra_whitespace():
    assert cache_key("gpt-4o", MSGS, 0.0) == cache_key("gpt-4o", SAME_MSGS, 0.0)


def test_key_depends_on_the_model():
    """Иначе ответ дешёвой модели отдастся вместо дорогой."""
    assert cache_key("gpt-4o", MSGS, 0.0) != cache_key("gpt-4o-mini", MSGS, 0.0)


def test_key_depends_on_the_temperature():
    assert cache_key("gpt-4o", MSGS, 0.0) != cache_key("gpt-4o", MSGS, 0.7)


def test_key_depends_on_the_message_text():
    assert cache_key("gpt-4o", MSGS, 0.0) != cache_key("gpt-4o", OTHER_MSGS, 0.0)


def test_key_is_stable_across_calls():
    assert cache_key("gpt-4o", MSGS, 0.0) == cache_key("gpt-4o", MSGS, 0.0)


# ------------------------------------------------------ cache_lookup / store
def test_first_lookup_is_a_miss():
    cache = new_cache()
    assert cache_lookup(cache, "gpt-4o", MSGS) is None
    assert (cache["hits"], cache["misses"]) == (0, 1)


def test_stored_answer_comes_back_on_the_next_lookup():
    cache = new_cache()
    cache_store(cache, "gpt-4o", MSGS, 0.0, "30 days.")
    assert cache_lookup(cache, "gpt-4o", MSGS) == "30 days."
    assert cache["hits"] == 1


def test_a_paraphrase_in_another_case_hits_the_same_entry():
    cache = new_cache()
    cache_store(cache, "gpt-4o", MSGS, 0.0, "30 days.")
    assert cache_lookup(cache, "gpt-4o", SAME_MSGS) == "30 days."


def test_non_deterministic_calls_never_touch_the_cache():
    """При temperature > 0 кэш выдавал бы старый сэмпл вместо нового."""
    cache = new_cache()
    cache_store(cache, "gpt-4o", MSGS, 0.7, "30 days.")
    assert cache["entries"] == {}
    assert cache_lookup(cache, "gpt-4o", MSGS, temperature=0.7) is None
    assert cache["misses"] == 1


def test_expired_entry_is_a_miss_and_is_dropped():
    cache = new_cache(ttl=10.0)
    cache_store(cache, "gpt-4o", MSGS, 0.0, "30 days.", now=0.0)
    assert cache_lookup(cache, "gpt-4o", MSGS, now=5.0) == "30 days."
    assert cache_lookup(cache, "gpt-4o", MSGS, now=50.0) is None
    assert cache["entries"] == {}


def test_full_cache_evicts_the_oldest_entry():
    cache = new_cache(max_size=1)
    cache_store(cache, "gpt-4o", MSGS, 0.0, "old", now=0.0)
    cache_store(cache, "gpt-4o", OTHER_MSGS, 0.0, "new", now=1.0)
    assert len(cache["entries"]) == 1
    assert cache_lookup(cache, "gpt-4o", MSGS, now=2.0) is None
    assert cache_lookup(cache, "gpt-4o", OTHER_MSGS, now=2.0) == "new"


def test_overwriting_an_entry_at_capacity_does_not_evict_another_entry():
    cache = new_cache(max_size=2)
    cache_store(cache, "gpt-4o", MSGS, 0.0, "first", now=0.0)
    cache_store(cache, "gpt-4o", OTHER_MSGS, 0.0, "keep", now=1.0)
    cache_store(cache, "gpt-4o", MSGS, 0.0, "updated", now=2.0)

    assert len(cache["entries"]) == 2
    assert cache_lookup(cache, "gpt-4o", MSGS, now=3.0) == "updated"
    assert cache_lookup(cache, "gpt-4o", OTHER_MSGS, now=3.0) == "keep"


def test_hit_counter_of_an_entry_grows():
    cache = new_cache()
    cache_store(cache, "gpt-4o", MSGS, 0.0, "30 days.")
    cache_lookup(cache, "gpt-4o", MSGS)
    cache_lookup(cache, "gpt-4o", MSGS)
    assert next(iter(cache["entries"].values()))["hits"] == 2


# --------------------------------------------------------------- route_model
def test_short_query_goes_to_the_cheap_model():
    assert route_model("Hello")["complexity"] == "simple"


def test_analysis_query_goes_to_the_expensive_model():
    route = route_model("Analyze the trade-offs between two storage engines")
    assert route["complexity"] == "complex"
    assert route["model"] == "gpt-4o"


def test_ordinary_long_query_lands_in_the_middle():
    route = route_model("Summarize this quarterly earnings report for the board")
    assert route["complexity"] == "medium"
    assert route["model"] == "claude-sonnet-4"


def test_single_word_keyword_is_not_matched_inside_another_word():
    """"monoliths" содержит "no", "this" содержит "hi" — подстрочный поиск врёт."""
    route = route_model("Analyze the trade-offs between microservices and monoliths")
    assert route["complexity"] == "complex"


def test_hyphenated_keyword_matches_its_plural():
    assert route_model("Compare the trade-offs of these two designs")["complexity"] == "complex"


def test_tier_changes_the_model_not_the_complexity():
    free = route_model("Analyze the trade-offs of event sourcing", tier="free")
    enterprise = route_model("Analyze the trade-offs of event sourcing", tier="enterprise")
    assert free["complexity"] == enterprise["complexity"] == "complex"
    assert free["model"] != enterprise["model"]


def test_keyword_router_misroutes_a_complex_query_with_a_simple_word():
    """Честное ограничение классификатора на ключевых словах."""
    assert route_model("Explain why the price of GPUs drives our margins")["complexity"] == "simple"


# --------------------------------------------------------- token_bucket_take
def test_a_burst_up_to_the_capacity_is_allowed():
    b = bucket()
    assert token_bucket_take(b, 100, now=0.0)["allowed"] is True
    assert b["tokens"] == APPROX(0.0)


def test_an_empty_bucket_refuses_and_takes_nothing():
    b = bucket(tokens=40.0)
    result = token_bucket_take(b, 60, now=0.0)
    assert result["allowed"] is False
    assert b["tokens"] == APPROX(40.0)


def test_refusal_says_how_long_to_wait():
    b = bucket(tokens=40.0)
    assert token_bucket_take(b, 60, now=0.0)["retry_after"] == APPROX(2.0)


def test_waiting_refills_the_bucket():
    b = bucket(tokens=40.0)
    assert token_bucket_take(b, 60, now=0.0)["allowed"] is False
    assert token_bucket_take(b, 60, now=2.0)["allowed"] is True


def test_bucket_never_overflows_its_capacity():
    """Долив непрерывный, но потолок есть — иначе всплеск был бы неограничен."""
    b = bucket(tokens=0.0)
    token_bucket_take(b, 0, now=10_000.0)
    assert b["tokens"] == APPROX(100.0)


def test_average_rate_is_capped_even_with_bursts():
    b = bucket(tokens=100.0)
    allowed = 0
    for step in range(20):
        if token_bucket_take(b, 50, now=float(step))["allowed"]:
            allowed += 1
    # ведро 100 + долив 10/с за 19 с = 290 токенов, по 50 за запрос
    assert allowed == 5


# ---------------------------------------------------------------- serve_query
def test_first_query_is_a_miss_and_costs_money():
    log = serve_query(new_cache(), "What is the return policy?")
    assert log["cache_status"] == "miss"
    assert log["cost"] > 0
    assert log["saved_cost"] == APPROX(0.0)


def test_repeat_query_returns_the_same_answer_for_free():
    """Главное свойство кэша: ответ тот же, цена другая."""
    cache = new_cache()
    miss = serve_query(cache, "What is the return policy?")
    hit = serve_query(cache, "What is the return policy?")
    assert hit["response"] == miss["response"]
    assert hit["cache_status"] == "hit"
    assert hit["cost"] == APPROX(0.0)
    assert hit["saved_cost"] == APPROX(miss["cost"])


def test_a_differently_cased_repeat_also_hits():
    cache = new_cache()
    first = serve_query(cache, "What is the return policy?")
    second = serve_query(cache, "  WHAT IS   the return policy?  ")
    assert second["cache_status"] == "hit"
    assert second["response"] == first["response"]


def test_cache_hit_reports_zero_tokens():
    cache = new_cache()
    serve_query(cache, "What is the return policy?")
    hit = serve_query(cache, "What is the return policy?")
    assert (hit["input_tokens"], hit["output_tokens"]) == (0, 0)


def test_routing_makes_a_simple_query_cheaper_than_a_complex_one():
    simple = serve_query(new_cache(), "What time do you close?")
    complex_ = serve_query(new_cache(), "Analyze the trade-offs of a sharded write path")
    assert simple["cost"] < complex_["cost"]


def test_different_queries_do_not_share_a_cache_entry():
    cache = new_cache()
    serve_query(cache, "What is the return policy?")
    other = serve_query(cache, "What are your store hours?")
    assert other["cache_status"] == "miss"


# ------------------------------------------------------------ summarize_usage
QUERIES = [
    "What is the return policy?",
    "What is the return policy?",
    "What are your store hours?",
    "what is THE return policy?",
    "Analyze the trade-offs of a sharded write path",
]


def test_summary_of_no_calls_is_all_zeros():
    summary = summarize_usage([])
    assert summary["calls"] == 0
    assert summary["total_cost"] == APPROX(0.0)
    assert summary["hit_rate"] == APPROX(0.0)


def test_summary_counts_hits_and_the_hit_rate():
    cache = new_cache()
    logs = [serve_query(cache, q) for q in QUERIES]
    summary = summarize_usage(logs)
    assert summary["calls"] == 5
    assert summary["cache_hits"] == 2
    assert summary["hit_rate"] == APPROX(0.4)


def test_spent_plus_saved_is_what_it_would_have_cost_without_a_cache():
    """Тождество, которое и отвечает на вопрос «кэш окупился?»."""
    cache = new_cache()
    logs = [serve_query(cache, q) for q in QUERIES]
    no_cache = [serve_query(new_cache(), q)["cost"] for q in QUERIES]
    summary = summarize_usage(logs)
    assert summary["total_cost"] + summary["saved_cost"] == pytest.approx(sum(no_cache))
    assert summary["total_cost"] < sum(no_cache)


def test_summary_breaks_the_bill_down_by_model():
    cache = new_cache()
    logs = [serve_query(cache, q) for q in QUERIES]
    by_model = summarize_usage(logs)["cost_by_model"]
    assert set(by_model) == {"gpt-4o-mini", "gpt-4o"}
    assert by_model["gpt-4o"]["cost"] > by_model["gpt-4o-mini"]["cost"]


def test_average_cost_per_call_counts_hits_too():
    cache = new_cache()
    logs = [serve_query(cache, q) for q in QUERIES]
    summary = summarize_usage(logs)
    assert summary["avg_cost_per_call"] == APPROX(summary["total_cost"] / 5)
