"""Тесты к уроку «Слой маршрутизации LLM». Правь exercise.py."""

import pytest

from exercise import (
    MODELS,
    ROUTES,
    cache_key,
    charge,
    cheapest_model,
    estimate_cost,
    redact_pii,
    resolve_chain,
    route,
    spend_report,
)

APPROX = lambda x: pytest.approx(x, abs=1e-12)

MSGS = [{"role": "user", "content": "explain MCP"}]


def fake_provider(down=(), seen=None, status_for=None):
    """Провайдер-заглушка: модели из down отвечают 503, остальные 200.

    seen — список, куда складываются пары (модель, сообщения), чтобы тест
    мог увидеть, что именно ушло наружу.
    status_for — словарь модель -> статус, чтобы задать 4xx точечно.
    """
    statuses = dict(status_for or {})

    def provider(model, messages):
        if seen is not None:
            seen.append((model, [dict(m) for m in messages]))
        status = statuses.get(model, 503 if model in down else 200)
        if status != 200:
            return {"status": status, "error": f"{model} unavailable"}
        return {
            "status": 200,
            "model": model,
            "usage": {"input_tokens": 100, "output_tokens": 200},
            "content": f"[{model}] ok",
        }

    return provider


# ------------------------------------------------------------------ redact_pii
def test_ssn_is_replaced_and_reported():
    text, tags = redact_pii("call me at 123-45-6789")
    assert text == "call me at [REDACTED]"
    assert tags == ("ssn",)


def test_clean_text_reports_no_findings():
    assert redact_pii("explain MCP") == ("explain MCP", ())


def test_several_kinds_are_reported_in_a_fixed_order():
    """Отчёт guardrail-а не должен зависеть от порядка находок в тексте."""
    first = redact_pii("mail a@b.com or card 1234567890123456")[1]
    second = redact_pii("card 1234567890123456 or mail a@b.com")[1]
    assert first == second == ("credit_card", "email")


def test_every_occurrence_is_replaced_not_just_the_first():
    text, _ = redact_pii("123-45-6789 and 987-65-4321")
    assert text == "[REDACTED] and [REDACTED]"


# --------------------------------------------------------------- estimate_cost
def test_price_is_per_million_tokens():
    """Цена 5/15 за миллион даёт 0.02 за тысячу входа и тысячу выхода."""
    assert estimate_cost("openai/gpt-4o", 1000, 1000) == APPROX(0.02)


def test_zero_tokens_cost_nothing():
    assert estimate_cost("anthropic/claude-haiku", 0, 0) == APPROX(0.0)


def test_unknown_model_is_not_silently_free():
    with pytest.raises(ValueError):
        estimate_cost("meta/llama-4", 10, 10)


# -------------------------------------------------------------- cheapest_model
def test_without_gates_the_cheapest_model_wins():
    assert cheapest_model(MODELS) == "openai/gpt-4o-mini"


def test_cheap_model_is_skipped_when_it_fails_the_quality_gate():
    """Дешёвая модель берётся, только если проходит по качеству."""
    assert cheapest_model(MODELS, min_quality=0.80) == "google/gemini-pro"


def test_latency_gate_and_quality_gate_apply_together():
    """gpt-4o-mini быстрый, но не тянет по качеству — остаётся haiku."""
    assert cheapest_model(MODELS, min_quality=0.76, max_latency_ms=400) == (
        "anthropic/claude-haiku"
    )


def test_no_model_passes_an_impossible_gate():
    assert cheapest_model(MODELS, min_quality=0.99) is None


def test_choice_does_not_depend_on_candidate_order():
    forward = cheapest_model(sorted(MODELS), min_quality=0.80)
    backward = cheapest_model(sorted(MODELS, reverse=True), min_quality=0.80)
    assert forward == backward == "google/gemini-pro"


# ------------------------------------------------------------------ cache_key
def test_whitespace_and_case_do_not_change_the_key():
    a = cache_key("fast", [{"role": "user", "content": "Explain   MCP"}])
    b = cache_key("fast", [{"role": "user", "content": "explain mcp"}])
    assert a == b


def test_different_prompts_get_different_keys():
    a = cache_key("fast", [{"role": "user", "content": "explain MCP"}])
    b = cache_key("fast", [{"role": "user", "content": "explain A2A"}])
    assert a != b


def test_alias_is_part_of_the_key():
    """Ответ дешёвой модели нельзя отдавать тому, кто просил умную."""
    assert cache_key("fast", MSGS) != cache_key("smart", MSGS)


def test_role_matters_as_much_as_content():
    a = cache_key("fast", [{"role": "user", "content": "hi"}])
    b = cache_key("fast", [{"role": "system", "content": "hi"}])
    assert a != b


# -------------------------------------------------------------- resolve_chain
def test_alias_expands_to_its_priority_chain():
    assert resolve_chain("smart") == ROUTES["smart"]


def test_a_concrete_model_resolves_to_a_chain_of_one():
    assert resolve_chain("openai/gpt-4o") == ("openai/gpt-4o",)


def test_unknown_alias_is_rejected():
    with pytest.raises(ValueError):
        resolve_chain("genius")


def test_a_chain_naming_an_unpriced_model_is_rejected_up_front():
    """Иначе сюрприз вылезет на третьем фолбэке в три часа ночи."""
    with pytest.raises(ValueError):
        resolve_chain("weird", {"weird": ("openai/gpt-4o", "meta/llama-4")})


# ---------------------------------------------------------------------- route
def test_healthy_primary_is_used_without_fallback():
    inv = route("smart", MSGS, fake_provider())
    assert inv["model"] == "openai/gpt-4o"
    assert inv["attempts"] == ["openai/gpt-4o"]
    assert inv["error"] is None


def test_outage_on_the_primary_does_not_lose_the_request():
    """Главное свойство фолбэка: ответ есть, ошибки нет, попыток две."""
    inv = route("smart", MSGS, fake_provider(down=("openai/gpt-4o",)))
    assert inv["model"] == "anthropic/claude-sonnet"
    assert inv["attempts"] == ["openai/gpt-4o", "anthropic/claude-sonnet"]
    assert inv["error"] is None
    assert inv["response"]["content"] == "[anthropic/claude-sonnet] ok"


def test_whole_chain_down_reports_an_error_after_trying_everyone():
    inv = route("fast", MSGS, fake_provider(down=ROUTES["fast"]))
    assert inv["attempts"] == list(ROUTES["fast"])
    assert inv["model"] is None
    assert inv["error"] == "all providers failed"


def test_client_error_stops_the_chain_instead_of_multiplying_the_bill():
    """400 у следующего провайдера будет тем же 400 — фолбэк только тратит деньги."""
    provider = fake_provider(status_for={"openai/gpt-4o": 400})
    inv = route("smart", MSGS, provider)
    assert inv["attempts"] == ["openai/gpt-4o"]
    assert inv["status"] == 400
    assert inv["error"] is not None


def test_rate_limit_does_fall_back():
    """429 — свойство одного провайдера, у соседа свой лимит."""
    provider = fake_provider(status_for={"openai/gpt-4o-mini": 429})
    inv = route("fast", MSGS, provider)
    assert inv["model"] == "anthropic/claude-haiku"


def test_provider_never_sees_the_raw_pii():
    seen = []
    route("fast", [{"role": "user", "content": "my ssn is 123-45-6789"}],
          fake_provider(seen=seen))
    assert "123-45-6789" not in seen[0][1][0]["content"]
    assert "[REDACTED]" in seen[0][1][0]["content"]


def test_cache_hit_skips_the_provider_and_costs_nothing():
    cache = {}
    seen = []
    provider = fake_provider(seen=seen)
    first = route("fast", MSGS, provider, cache=cache)
    second = route("fast", [{"role": "user", "content": "explain   mcp"}],
                   provider, cache=cache)
    assert first["cached"] is False and first["cost_usd"] > 0
    assert second["cached"] is True
    assert second["attempts"] == []
    assert second["cost_usd"] == APPROX(0.0)
    assert len(seen) == 1


def test_cost_is_attributed_to_the_model_that_actually_answered():
    inv = route("smart", MSGS, fake_provider(down=("openai/gpt-4o",)))
    assert inv["cost_usd"] == APPROX(estimate_cost("anthropic/claude-sonnet", 100, 200))


# --------------------------------------------------------------------- charge
def test_charge_within_the_cap_records_the_spend():
    ledger = {}
    assert charge(ledger, "search", 0.5, 1.0) is True
    assert ledger == {"search": 0.5}


def test_charge_over_the_cap_is_refused():
    ledger = {"search": 0.9}
    assert charge(ledger, "search", 0.5, 1.0) is False


def test_refused_charge_leaves_the_ledger_untouched():
    """Иначе после отказа лимит съедет и команда потеряет доступ навсегда."""
    ledger = {"search": 0.9}
    charge(ledger, "search", 0.5, 1.0)
    assert ledger == {"search": 0.9}


def test_refused_charge_does_not_create_a_row_for_a_new_team():
    ledger = {}
    charge(ledger, "newbie", 5.0, 1.0)
    assert ledger == {}


# ---------------------------------------------------------------- spend_report
def test_report_sums_cost_per_model():
    invocations = [
        route("smart", MSGS, fake_provider()),
        route("smart", MSGS, fake_provider()),
        route("fast", MSGS, fake_provider()),
    ]
    report = spend_report(invocations)
    assert report["openai/gpt-4o"]["calls"] == 2
    assert report["openai/gpt-4o"]["cost_usd"] == APPROX(
        2 * estimate_cost("openai/gpt-4o", 100, 200)
    )


def test_failed_invocations_do_not_appear_in_the_report():
    dead = route("fast", MSGS, fake_provider(down=ROUTES["fast"]))
    assert spend_report([dead]) == {}


def test_cached_calls_count_as_load_but_not_as_spend():
    cache = {}
    provider = fake_provider()
    first = route("fast", MSGS, provider, cache=cache)
    second = route("fast", MSGS, provider, cache=cache)
    row = spend_report([first, second])["openai/gpt-4o-mini"]
    assert (row["calls"], row["cached"]) == (2, 1)
    assert row["cost_usd"] == APPROX(first["cost_usd"])
