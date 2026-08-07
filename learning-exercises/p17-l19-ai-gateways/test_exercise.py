"""Тесты к уроку «AI-шлюзы: ключи, лимиты, фолбэк». Правь exercise.py."""

import random

import pytest

from exercise import (
    GATEWAYS,
    TIERS,
    GatewayError,
    backoff_delays,
    call_with_fallback,
    handle_request,
    latency_budget,
    pick_gateway,
    resolve_key,
    sliding_window,
    token_bucket,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

KEYRING = {
    "sk-app-1": {"tenant": "search", "tier": "paid", "secret_ref": "vault:openai_prod", "active": True},
    "sk-app-2": {"tenant": "demo", "tier": "free", "secret_ref": "vault:openai_prod", "active": True},
    "sk-old":   {"tenant": "search", "tier": "paid", "secret_ref": "vault:openai_prod", "active": False},
    "sk-typo":  {"tenant": "search", "tier": "platinum", "secret_ref": "vault:openai_prod", "active": True},
    "sk-leak":  {"tenant": "search", "tier": "paid", "secret_ref": "sk-openai-real", "active": True},
    "sk-gone":  {"tenant": "search", "tier": "paid", "secret_ref": "vault:deleted", "active": True},
}
VAULT = {"openai_prod": "sk-openai-real"}


def make_state():
    return {
        "keyring": {k: dict(v) for k, v in KEYRING.items()},
        "vault": dict(VAULT),
        "buckets": {},
        "quota_used": {},
    }


def ok_provider(usage=(10, 20)):
    def provider(model, request):
        return {"status": 200, "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}
    return provider


def scripted_provider(script):
    """script: {model: [статус, статус, ...]} — по одному на обращение."""
    calls = {}

    def provider(model, request):
        i = calls.get(model, 0)
        calls[model] = i + 1
        codes = script.get(model, [200])
        status = codes[min(i, len(codes) - 1)]
        if status == 200:
            return {"status": 200, "usage": {"input_tokens": 10, "output_tokens": 20}}
        return {"status": status}

    return provider


# ------------------------------------------------------------- resolve_key
def test_valid_key_returns_tenant_tier_and_secret():
    assert resolve_key("sk-app-1", KEYRING, VAULT) == {
        "tenant": "search", "tier": "paid", "provider_key": "sk-openai-real",
    }


def test_unknown_key_is_rejected():
    with pytest.raises(GatewayError):
        resolve_key("sk-nope", KEYRING, VAULT)


def test_revoked_key_is_rejected():
    with pytest.raises(GatewayError):
        resolve_key("sk-old", KEYRING, VAULT)


def test_unknown_tier_does_not_silently_become_unlimited():
    with pytest.raises(GatewayError):
        resolve_key("sk-typo", KEYRING, VAULT)


def test_plaintext_secret_in_config_is_refused():
    """Сырой sk-... в конфиге — это утечка, а не рабочая конфигурация."""
    with pytest.raises(GatewayError):
        resolve_key("sk-leak", KEYRING, VAULT)


def test_dangling_vault_reference_is_refused():
    with pytest.raises(GatewayError):
        resolve_key("sk-gone", KEYRING, VAULT)


# ------------------------------------------------------------ token_bucket
def test_bucket_refills_with_elapsed_time():
    allowed, state = token_bucket({"tokens": 0.0, "at": 0.0}, 2.0, 1, 5, 1.0)
    assert allowed is True
    assert state["tokens"] == APPROX(1.0)


def test_bucket_refill_stops_at_capacity():
    """Час простоя не даёт права на час запросов — потолок это burst."""
    _, state = token_bucket({"tokens": 0.0, "at": 0.0}, 3600.0, 1, 5, 1.0)
    assert state["tokens"] == APPROX(4.0)


def test_rejected_request_does_not_consume_tokens():
    allowed, state = token_bucket({"tokens": 0.5, "at": 0.0}, 0.0, 1, 5, 1.0)
    assert allowed is False
    assert state["tokens"] == APPROX(0.5)


def test_bucket_allows_a_burst_up_to_capacity():
    state, passed = None, 0
    for _ in range(10):
        allowed, state = token_bucket(state, 0.0, 1, 5, 1.0)
        passed += allowed
    assert passed == 5


def test_bucket_does_not_mutate_the_state_it_was_given():
    given = {"tokens": 3.0, "at": 0.0}
    token_bucket(given, 1.0, 1, 5, 1.0)
    assert given == {"tokens": 3.0, "at": 0.0}


def test_clock_going_backwards_is_an_error():
    with pytest.raises(GatewayError):
        token_bucket({"tokens": 3.0, "at": 10.0}, 9.0, 1, 5, 1.0)


# ---------------------------------------------------------- sliding_window
def test_window_rejects_over_the_limit():
    allowed, kept = sliding_window((0.0, 1.0), 2.0, 60.0, 2)
    assert (allowed, kept) == (False, (0.0, 1.0))


def test_window_forgets_events_older_than_the_window():
    allowed, kept = sliding_window((0.0, 1.0), 60.5, 60.0, 2)
    assert (allowed, kept) == (True, (1.0, 60.5))


def test_window_refuses_the_burst_that_the_bucket_allows():
    """Ключевая разница: ведро копит право на залп, окно — нет.

    Один и тот же номинальный темп «5 запросов за 5 секунд». После простоя
    ведро пропускает все пять мгновенно, окно — только один.
    """
    state, bucket_passed = {"tokens": 0.0, "at": 0.0}, 0
    for _ in range(5):
        allowed, state = token_bucket(state, 100.0, 1, 5, 1.0)
        bucket_passed += allowed

    events, window_passed = (), 0
    for _ in range(5):
        allowed, events = sliding_window(events, 100.0, 5.0, 1)
        window_passed += allowed

    assert bucket_passed == 5
    assert window_passed == 1


# ---------------------------------------------------------- backoff_delays
def test_backoff_doubles_each_attempt():
    assert backoff_delays(4) == APPROX((100.0, 200.0, 400.0, 800.0))


def test_backoff_is_capped():
    assert backoff_delays(5, cap_ms=300.0) == APPROX((100.0, 200.0, 300.0, 300.0, 300.0))


def test_jitter_never_exceeds_the_deterministic_delay():
    """Full jitter только уменьшает задержку — потолок остаётся потолком."""
    plain = backoff_delays(6)
    jittered = backoff_delays(6, rng=random.Random(1))
    assert all(0.0 <= j <= p for j, p in zip(jittered, plain))


def test_jitter_is_reproducible_for_the_same_seed():
    assert backoff_delays(6, rng=random.Random(7)) == APPROX(
        backoff_delays(6, rng=random.Random(7))
    )


def test_jitter_actually_spreads_the_retries():
    """Без джиттера все клиенты вернутся в одну миллисекунду и уронят провайдера."""
    a = backoff_delays(6, rng=random.Random(1))
    b = backoff_delays(6, rng=random.Random(2))
    assert a != b


def test_negative_attempt_count_is_an_error():
    with pytest.raises(GatewayError):
        backoff_delays(-1)


# ------------------------------------------------------ call_with_fallback
def test_first_healthy_provider_wins_without_fallback():
    record = call_with_fallback(("gpt-4o", "claude"), {}, ok_provider())
    assert record["model"] == "gpt-4o"
    assert record["attempts"] == (("gpt-4o", 200),)


def test_429_moves_the_request_to_the_next_provider():
    """Канонический рецепт из урока: 429 у OpenAI — идём в Anthropic."""
    provider = scripted_provider({"gpt-4o": [429], "claude": [200]})
    record = call_with_fallback(("gpt-4o", "claude"), {}, provider)
    assert record["model"] == "claude"
    assert record["attempts"] == (("gpt-4o", 429), ("claude", 200))


def test_non_retryable_status_stops_the_chain_immediately():
    """400 будет 400 и у следующего провайдера — цепочка лишь утроит счёт."""
    provider = scripted_provider({"gpt-4o": [400], "claude": [200]})
    record = call_with_fallback(("gpt-4o", "claude"), {}, provider)
    assert record["model"] is None
    assert record["attempts"] == (("gpt-4o", 400),)


def test_all_providers_down_reports_the_whole_trail():
    provider = scripted_provider({"gpt-4o": [503], "claude": [503]})
    record = call_with_fallback(("gpt-4o", "claude"), {}, provider)
    assert record["model"] is None
    assert record["error"] == "all providers failed"
    assert record["attempts"] == (("gpt-4o", 503), ("claude", 503))


def test_retries_hit_the_same_provider_before_moving_on():
    provider = scripted_provider({"gpt-4o": [503, 503, 200]})
    record = call_with_fallback(("gpt-4o", "claude"), {}, provider, retries_per_provider=2)
    assert record["model"] == "gpt-4o"
    assert record["retries"] == 2
    assert record["waited_ms"] == APPROX(300.0)


def test_empty_chain_is_a_config_error_not_an_outage():
    with pytest.raises(GatewayError):
        call_with_fallback((), {}, ok_provider())


# --------------------------------------------------------- handle_request
def test_healthy_request_returns_200_and_charges_the_quota():
    state = make_state()
    record = handle_request("sk-app-1", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))
    assert record["status"] == 200
    assert record["model"] == "gpt-4o"
    assert state["quota_used"]["search"] == 30


def test_bad_key_becomes_401_not_an_exception():
    """Наружу шлюз отдаёт HTTP-семантику, а не своё внутреннее исключение."""
    state = make_state()
    record = handle_request("sk-nope", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))
    assert record["status"] == 401
    assert record["model"] is None


def test_free_tier_runs_out_of_burst_before_paid_tier():
    state = make_state()
    free = sum(
        handle_request("sk-app-2", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))["status"] == 200
        for _ in range(30)
    )
    paid = sum(
        handle_request("sk-app-1", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))["status"] == 200
        for _ in range(30)
    )
    assert free == TIERS["free"]["burst"]
    assert paid == 30


def test_rate_limited_request_does_not_touch_the_quota():
    state = make_state()
    for _ in range(TIERS["free"]["burst"]):
        handle_request("sk-app-2", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))
    spent_before = state["quota_used"]["demo"]
    record = handle_request("sk-app-2", {"estimated_tokens": 30}, 0.0, state, ok_provider(), ("gpt-4o",))
    assert record["status"] == 429
    assert state["quota_used"]["demo"] == spent_before


def test_exhausted_monthly_quota_returns_429_without_calling_the_provider():
    """Проверять квоту после вызова провайдера — платить за отвергнутый запрос."""
    state = make_state()
    state["quota_used"]["demo"] = TIERS["free"]["monthly_tokens"]
    seen = []

    def provider(model, request):
        seen.append(model)
        return {"status": 200, "usage": {"input_tokens": 1, "output_tokens": 1}}

    record = handle_request("sk-app-2", {"estimated_tokens": 1}, 0.0, state, provider, ("gpt-4o",))
    assert record["status"] == 429
    assert seen == []


def test_bucket_refills_between_requests():
    state = make_state()
    for _ in range(TIERS["free"]["burst"]):
        handle_request("sk-app-2", {"estimated_tokens": 1}, 0.0, state, ok_provider(), ("gpt-4o",))
    blocked = handle_request("sk-app-2", {"estimated_tokens": 1}, 0.5, state, ok_provider(), ("gpt-4o",))
    later = handle_request("sk-app-2", {"estimated_tokens": 1}, 2.0, state, ok_provider(), ("gpt-4o",))
    assert blocked["status"] == 429
    assert later["status"] == 200


def test_gateway_falls_back_and_still_bills_the_provider_that_answered():
    state = make_state()
    provider = scripted_provider({"gpt-4o": [500], "claude": [200]})
    record = handle_request("sk-app-1", {"estimated_tokens": 30}, 0.0, state, provider, ("gpt-4o", "claude"))
    assert record["model"] == "claude"
    assert record["tokens"] == 30
    assert state["quota_used"]["search"] == 30


# --------------------------------------------------------- latency_budget
def test_latency_budget_adds_gateway_overhead_to_ttft():
    row = latency_budget("Kong", 300, 400)
    assert (row["total_ms"], row["headroom_ms"], row["fits"]) == (305, 95, True)


def test_tight_sla_rejects_the_heavy_gateway():
    assert latency_budget("Portkey", 300, 320)["fits"] is False


def test_unknown_gateway_is_an_error_not_a_zero_overhead():
    with pytest.raises(GatewayError):
        latency_budget("Nginx", 300, 400)


# ------------------------------------------------------------ pick_gateway
def test_without_constraints_the_lowest_overhead_wins():
    assert pick_gateway(100, 300, 500) == "Cloudflare"


def test_self_host_requirement_drops_the_managed_gateway():
    assert pick_gateway(100, 300, 500, self_host=True) == "Kong"


def test_healthcare_gets_self_hosted_guardrails():
    assert pick_gateway(100, 300, 500, self_host=True, guardrails=True) == "Portkey"


def test_high_rps_is_out_of_reach_for_every_gateway_here():
    assert pick_gateway(100_000, 300, 500) is None


def test_python_gateway_loses_at_its_documented_rps_ceiling():
    """LiteLLM — «best fit <500 RPS»: за потолком его просто нет в кандидатах."""
    assert pick_gateway(2000, 300, 500, self_host=True, guardrails=True) == "Portkey"
    assert GATEWAYS["LiteLLM"]["rps_ceiling"] < 2000


def test_latency_filter_runs_before_the_cheapest_choice():
    """Сначала жёсткие требования, потом минимальный overhead."""
    assert pick_gateway(100, 94, 100, self_host=True) == "Kong"
    assert pick_gateway(100, 99, 100, self_host=True) is None
