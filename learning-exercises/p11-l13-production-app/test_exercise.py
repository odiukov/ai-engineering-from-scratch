"""Тесты к уроку «Продакшен-приложение на LLM». Правь exercise.py."""

import random

import pytest

from exercise import (
    DEGRADED_TEXT,
    FALLBACK_CHAIN,
    MODEL_PRICING,
    ProviderError,
    ab_bucket,
    backoff_delay,
    call_with_fallback,
    estimate_tokens,
    percentiles,
    request_cost,
    retry_with_backoff,
    summarize_requests,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flaky(fail_until, text="ok"):
    """call(attempt), падающий на попытках 0..fail_until-1."""

    def call(attempt):
        if attempt < fail_until:
            raise ProviderError(f"500 on attempt {attempt}")
        return text

    return call


# --------------------------------------------------------- estimate_tokens
def test_estimate_tokens_counts_words_at_four_thirds():
    assert estimate_tokens("a b c d e f") == 8


def test_estimate_tokens_of_empty_text_is_one_not_zero():
    """Ноль токенов уронил бы любую метрику, где на них делят."""
    assert estimate_tokens("") == 1
    assert estimate_tokens("   ") == 1


def test_estimate_tokens_grows_with_text_length():
    short = estimate_tokens("one two three")
    long = estimate_tokens("one two three " * 100)
    assert long > short


# ------------------------------------------------------------ request_cost
def test_request_cost_sums_input_and_output_at_their_own_prices():
    assert request_cost("gpt-4o", 1500, 400, MODEL_PRICING) == APPROX(0.00775)


def test_request_cost_output_tokens_are_the_expensive_half():
    """У всех моделей выход дороже входа — тысяча токенов ответа бьёт по счёту сильнее."""
    same_in = request_cost("gpt-4o", 1000, 0, MODEL_PRICING)
    same_out = request_cost("gpt-4o", 0, 1000, MODEL_PRICING)
    assert same_out > same_in


def test_request_cost_mini_model_is_an_order_cheaper():
    big = request_cost("claude-sonnet-5", 1500, 400, MODEL_PRICING)
    small = request_cost("gpt-4o-mini", 1500, 400, MODEL_PRICING)
    assert big > 10 * small


def test_request_cost_refuses_unknown_model_instead_of_guessing():
    """Молчаливый фолбэк на чужой прайс — как рождаются счета-сюрпризы."""
    with pytest.raises(ValueError):
        request_cost("gpt-9-ultra", 100, 100, MODEL_PRICING)


# ----------------------------------------------------------- backoff_delay
def test_backoff_delay_first_attempt_goes_immediately():
    assert backoff_delay(0) == APPROX(0.0)


def test_backoff_delay_doubles_each_attempt():
    assert [backoff_delay(a) for a in (1, 2, 3, 4)] == APPROX([1.0, 2.0, 4.0, 8.0])


def test_backoff_delay_is_capped():
    """Без потолка десятый ретрай ждал бы восемь с половиной минут."""
    assert backoff_delay(9, cap=10.0) == APPROX(10.0)
    assert backoff_delay(20, cap=10.0) == APPROX(10.0)


def test_backoff_delay_jitter_stays_within_half_of_the_delay():
    rng = random.Random(7)
    for _ in range(50):
        d = backoff_delay(3, rng=rng)
        assert 4.0 <= d < 6.0


def test_backoff_delay_is_reproducible_for_one_seed():
    a = [backoff_delay(2, rng=random.Random(123)) for _ in range(3)]
    assert a[0] == APPROX(a[1]) == APPROX(a[2])


def test_backoff_delay_differs_between_seeds():
    """В этом весь смысл jitter: тысяча клиентов не ретраится в одну секунду."""
    assert backoff_delay(2, rng=random.Random(1)) != backoff_delay(2, rng=random.Random(2))


def test_backoff_delay_without_rng_is_deterministic():
    assert backoff_delay(3) == APPROX(backoff_delay(3))


# ------------------------------------------------------ retry_with_backoff
def test_retry_returns_on_first_success_without_waiting():
    assert retry_with_backoff(flaky(0)) == ("ok", 1, APPROX(0.0))


def test_retry_survives_two_failures_and_reports_the_waiting_time():
    text, attempts, delay = retry_with_backoff(flaky(2))
    assert (text, attempts) == ("ok", 3)
    assert delay == APPROX(1.0 + 2.0)


def test_retry_raises_provider_error_when_attempts_run_out():
    with pytest.raises(ProviderError):
        retry_with_backoff(flaky(99), max_retries=2)


def test_retry_makes_exactly_max_retries_plus_one_attempts():
    seen = []

    def call(attempt):
        seen.append(attempt)
        raise ProviderError("always down")

    with pytest.raises(ProviderError):
        retry_with_backoff(call, max_retries=3)
    assert seen == [0, 1, 2, 3]


def test_retry_passes_the_rng_through_to_the_jitter():
    _, _, a = retry_with_backoff(flaky(2), rng=random.Random(5))
    _, _, b = retry_with_backoff(flaky(2), rng=random.Random(5))
    _, _, c = retry_with_backoff(flaky(2), rng=random.Random(6))
    assert a == APPROX(b)
    assert a != c


# ----------------------------------------------------- call_with_fallback
def test_fallback_stops_at_the_first_healthy_model():
    result = call_with_fallback(FALLBACK_CHAIN, lambda model, attempt: f"from {model}")
    assert result["model"] == "claude-sonnet-5"
    assert result["degraded"] is False
    assert result["models_tried"] == ["claude-sonnet-5"]


def test_fallback_walks_down_the_chain_when_the_primary_is_down():
    def call(model, attempt):
        if model == "claude-sonnet-5":
            raise ProviderError("500")
        return f"from {model}"

    result = call_with_fallback(FALLBACK_CHAIN, call)
    assert result["model"] == "gpt-4o"
    assert result["models_tried"] == ["claude-sonnet-5", "gpt-4o"]


def test_fallback_degrades_instead_of_raising_when_everything_is_down():
    """Вторичный сбой не имеет права уронить основной поток."""

    def call(model, attempt):
        raise ProviderError("500")

    result = call_with_fallback(FALLBACK_CHAIN, call)
    assert result["degraded"] is True
    assert result["model"] is None
    assert result["text"] == DEGRADED_TEXT
    assert result["models_tried"] == list(FALLBACK_CHAIN)


def test_fallback_gives_each_model_its_own_retries():
    calls = []

    def call(model, attempt):
        calls.append((model, attempt))
        if model != "gpt-4o-mini":
            raise ProviderError("500")
        return "ok"

    call_with_fallback(FALLBACK_CHAIN, call, max_retries=1)
    assert calls == [
        ("claude-sonnet-5", 0),
        ("claude-sonnet-5", 1),
        ("gpt-4o", 0),
        ("gpt-4o", 1),
        ("gpt-4o-mini", 0),
    ]


# --------------------------------------------------------------- ab_bucket
def test_ab_bucket_is_stable_for_one_user():
    """Пользователь обязан видеть одну и ту же ветку на каждом запросе."""
    assert {ab_bucket("user_001", "chat_v2", 10) for _ in range(20)} == {"control"}


def test_ab_bucket_worked_examples():
    assert ab_bucket("user_001", "other_exp", 10) == "variant"
    assert ab_bucket("bob", "chat_v2", 20) == "variant"


def test_ab_bucket_at_zero_percent_nobody_is_in_the_variant():
    assert all(ab_bucket(f"u{i}", "exp", 0) == "control" for i in range(200))


def test_ab_bucket_at_hundred_percent_everybody_is():
    assert all(ab_bucket(f"u{i}", "exp", 100) == "variant" for i in range(200))


def test_ab_bucket_hits_the_requested_share_of_traffic():
    n = 2000
    variant = sum(1 for i in range(n) if ab_bucket(f"u{i}", "chat_v2", 10) == "variant")
    assert 0.07 < variant / n < 0.13


def test_ab_bucket_reshuffles_users_between_experiments():
    """Иначе один и тот же несчастный попадал бы в вариант всех тестов подряд."""
    first = [ab_bucket(f"u{i}", "exp_a", 50) for i in range(200)]
    second = [ab_bucket(f"u{i}", "exp_b", 50) for i in range(200)]
    assert first != second


# ------------------------------------------------------------- percentiles
def test_percentiles_nearest_rank():
    assert percentiles([1, 2, 3, 4], (50, 100)) == {50: 2, 100: 4}


def test_percentiles_p99_of_a_hundred_values():
    assert percentiles(list(range(1, 101)), (99,)) == {99: 99}


def test_percentiles_do_not_depend_on_input_order():
    values = [5, 1, 9, 3, 7]
    assert percentiles(values, (50,)) == percentiles(sorted(values, reverse=True), (50,))


def test_percentiles_do_not_mutate_the_journal():
    values = [5, 1, 9, 3, 7]
    percentiles(values, (50, 90))
    assert values == [5, 1, 9, 3, 7]


def test_percentiles_see_the_tail_that_the_mean_hides():
    """Девяносто девять быстрых ответов и один восьмисекундный."""
    values = [100] * 99 + [8000]
    got = percentiles(values, (50, 100))
    assert got[50] == 100
    assert got[100] == 8000


def test_percentiles_reject_an_empty_journal():
    with pytest.raises(ValueError):
        percentiles([], (50,))


def test_percentiles_reject_a_percentile_out_of_range():
    with pytest.raises(ValueError):
        percentiles([1, 2, 3], (0,))


# ------------------------------------------------------- summarize_requests
def _log(model="gpt-4o", inp=1500, out=400, latency=120.0, cache_hit=False, error=None):
    return {
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "latency_ms": latency,
        "cache_hit": cache_hit,
        "error": error,
    }


def test_summary_totals_match_the_per_request_prices():
    logs = [_log(), _log(model="gpt-4o-mini")]
    got = summarize_requests(logs, MODEL_PRICING)
    expected = request_cost("gpt-4o", 1500, 400, MODEL_PRICING) + request_cost(
        "gpt-4o-mini", 1500, 400, MODEL_PRICING
    )
    assert got["total_cost_usd"] == APPROX(expected)
    assert got["avg_cost_usd"] == APPROX(expected / 2)
    assert got["requests"] == 2


def test_summary_counts_a_cache_hit_as_free_but_keeps_it_in_the_denominator():
    logs = [_log(), _log(model="cache", cache_hit=True)]
    got = summarize_requests(logs, MODEL_PRICING)
    assert got["cache_hit_rate_pct"] == APPROX(50.0)
    assert got["total_cost_usd"] == APPROX(request_cost("gpt-4o", 1500, 400, MODEL_PRICING))


def test_summary_never_prices_a_cache_hit_through_the_pricing_table():
    """Модель "cache" в прайс-листе не значится — и не должна там искаться."""
    got = summarize_requests([_log(model="cache", cache_hit=True)], MODEL_PRICING)
    assert got["cost_by_model"]["cache"] == APPROX(0.0)


def test_summary_splits_cost_by_model():
    logs = [_log(), _log(), _log(model="gpt-4o-mini")]
    got = summarize_requests(logs, MODEL_PRICING)
    assert set(got["cost_by_model"]) == {"gpt-4o", "gpt-4o-mini"}
    assert got["cost_by_model"]["gpt-4o"] == APPROX(
        2 * request_cost("gpt-4o", 1500, 400, MODEL_PRICING)
    )


def test_summary_reports_latency_percentiles_not_the_mean():
    """P50 хвоста не видит, P99 видит — ради этого перцентили и считают."""
    logs = [_log(latency=100.0) for _ in range(90)] + [_log(latency=8000.0) for _ in range(10)]
    got = summarize_requests(logs, MODEL_PRICING)
    assert got["p50_latency_ms"] == 100.0
    assert got["p99_latency_ms"] == 8000.0
    assert got["requests"] == 100


def test_summary_counts_the_error_rate():
    logs = [_log(), _log(error="timeout"), _log(), _log()]
    got = summarize_requests(logs, MODEL_PRICING)
    assert got["error_rate_pct"] == APPROX(25.0)


def test_summary_refuses_an_empty_journal():
    """«Ноль запросов, всё хорошо» — худший вид зелёного дашборда."""
    with pytest.raises(ValueError):
        summarize_requests([], MODEL_PRICING)
