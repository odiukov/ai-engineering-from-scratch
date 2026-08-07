"""Тесты к уроку «MCP-гейтвеи и реестры». Правь exercise.py."""

import hashlib

import pytest

from exercise import (
    audit_event,
    choose_server,
    handle_call,
    merge_tool_namespaces,
    pin_filter,
    rbac_allows,
    registry_rank,
    token_bucket_take,
)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_gateway():
    """Маленький гейтвей: два бэкенда, два пользователя, бакет на 2 вызова."""
    return {
        "routes": {"search": "notes::search", "open_pr": "github::open_pr"},
        "policy": {"alice": ["notes::*", "github::open_pr"], "bob": ["notes::search"]},
        "sessions": {"tok_alice": "alice", "tok_bob": "bob"},
        "buckets": {"alice": {"tokens": 2.0, "updated": 0}, "bob": {"tokens": 2.0, "updated": 0}},
        "limit": {"capacity": 2, "refill_per_second": 0.0},
        "audit": [],
    }


# ------------------------------------------------------- merge_tool_namespaces
def test_unique_names_stay_short():
    got = merge_tool_namespaces({"notes": ["search"], "github": ["open_pr"]})
    assert got == {"search": "notes::search", "open_pr": "github::open_pr"}


def test_collision_prefixes_both_sides():
    """Если префикс получит только один, порядок обхода начнёт решать судьбу."""
    got = merge_tool_namespaces({"notes": ["search"], "archive": ["search"]})
    assert got == {"archive.search": "archive::search", "notes.search": "notes::search"}


def test_one_server_can_mix_colliding_and_unique_tools():
    got = merge_tool_namespaces({"notes": ["search", "add"], "archive": ["search"]})
    assert got["add"] == "notes::add"
    assert got["notes.search"] == "notes::search"


# ----------------------------------------------------------------- rbac_allows
def test_exact_grant_allows_the_tool():
    assert rbac_allows({"alice": ["github::open_pr"]}, "alice", "github", "open_pr") is True


def test_server_wildcard_covers_every_tool_of_that_server():
    assert rbac_allows({"alice": ["notes::*"]}, "alice", "notes", "whatever") is True


def test_wildcard_does_not_leak_to_another_server():
    assert rbac_allows({"alice": ["notes::*"]}, "alice", "github", "open_pr") is False


def test_unknown_user_is_denied_by_default():
    """Гейтвей, отвечающий неизвестному «ну ладно», — не гейтвей."""
    assert rbac_allows({}, "bob", "notes", "search") is False


# ------------------------------------------------------------------ pin_filter
def test_pinned_tool_survives():
    tools = [{"server": "n", "name": "s", "description": "Find"}]
    assert pin_filter(tools, {"n::s": sha("Find")}) == tools


def test_mutated_description_is_dropped():
    tools = [{"server": "n", "name": "s", "description": "Find and exfiltrate"}]
    assert pin_filter(tools, {"n::s": sha("Find")}) == []


def test_tool_absent_from_the_manifest_is_dropped():
    tools = [{"server": "n", "name": "s", "description": "Find"}]
    assert pin_filter(tools, {}) == []


def test_order_of_survivors_is_preserved():
    tools = [
        {"server": "n", "name": "a", "description": "A"},
        {"server": "n", "name": "b", "description": "MUTATED"},
        {"server": "n", "name": "c", "description": "C"},
    ]
    manifest = {"n::a": sha("A"), "n::b": sha("B"), "n::c": sha("C")}
    assert [t["name"] for t in pin_filter(tools, manifest)] == ["a", "c"]


# ------------------------------------------------------------ token_bucket_take
def test_full_bucket_lets_the_call_through():
    allowed, bucket = token_bucket_take({"tokens": 1.0, "updated": 0}, 0, 3, 0.5)
    assert allowed is True and bucket["tokens"] == 0.0


def test_empty_bucket_blocks():
    allowed, _ = token_bucket_take({"tokens": 0.0, "updated": 0}, 0, 3, 0.5)
    assert allowed is False


def test_tokens_refill_over_time():
    allowed, bucket = token_bucket_take({"tokens": 0.0, "updated": 0}, 10, 3, 0.5)
    assert allowed is True and bucket["tokens"] == pytest.approx(4.0 - 1.0 - 1.0)


def test_refill_is_capped_by_capacity():
    """Иначе простоявший сутки пользователь выпустит залп из тысячи вызовов."""
    _, bucket = token_bucket_take({"tokens": 0.0, "updated": 0}, 100000, 3, 0.5)
    assert bucket["tokens"] == pytest.approx(2.0)


def test_clock_going_backwards_does_not_mint_tokens():
    allowed, _ = token_bucket_take({"tokens": 0.0, "updated": 100}, 50, 3, 0.5)
    assert allowed is False


def test_bucket_argument_is_not_mutated():
    bucket = {"tokens": 1.0, "updated": 0}
    token_bucket_take(bucket, 0, 3, 0.5)
    assert bucket == {"tokens": 1.0, "updated": 0}


# ----------------------------------------------------------------- audit_event
def test_audit_event_records_who_what_when_and_result():
    assert audit_event("alice", "notes.search", "ok", 1000) == {
        "at": 1000,
        "user": "alice",
        "tool": "notes.search",
        "verdict": "ok",
    }


def test_anonymous_attempt_is_still_recorded():
    """Журнал без отказов бесполезен для расследования."""
    assert audit_event(None, "notes.search", "unauthenticated", 5)["user"] is None


def test_verdict_travels_verbatim():
    assert audit_event("bob", "x", "forbidden", 1)["verdict"] == "forbidden"


# --------------------------------------------------------------- registry_rank
def test_official_registry_outranks_a_metaregistry():
    assert registry_rank("official") < registry_rank("metaregistry")


def test_unlisted_is_the_weakest_source():
    assert registry_rank("unlisted") > registry_rank("metaregistry")


def test_unknown_source_is_an_error_not_a_default_rank():
    with pytest.raises(ValueError):
        registry_rank("подсунули-по-ссылке-из-чата")


# ---------------------------------------------------------------- choose_server
def test_official_wins_over_metaregistry():
    got = choose_server(
        [
            {"name": "io.github.a/notes", "source": "metaregistry", "verified": True},
            {"name": "io.github.a/notes", "source": "official", "verified": True},
        ]
    )
    assert got == "io.github.a/notes"


def test_unverified_candidate_is_dropped_even_if_official():
    got = choose_server([{"name": "postmark-mcp", "source": "official", "verified": False}])
    assert got is None


def test_nothing_installable_returns_none():
    """«Ну хоть что-нибудь поставить» — это и есть история поддельного Postmark."""
    assert choose_server([]) is None


def test_tie_is_broken_alphabetically_for_reproducibility():
    got = choose_server(
        [
            {"name": "io.github.b/notes", "source": "official", "verified": True},
            {"name": "io.github.a/notes", "source": "official", "verified": True},
        ]
    )
    assert got == "io.github.a/notes"


def test_unknown_source_among_candidates_is_an_error():
    with pytest.raises(ValueError):
        choose_server([{"name": "io.github.a/notes", "source": "чат", "verified": True}])


# ----------------------------------------------------------------- handle_call
def test_allowed_call_is_routed_to_its_backend():
    gw = make_gateway()
    got = handle_call(gw, "tok_alice", "open_pr", 0)
    assert (got["status"], got["route"]) == (200, "github::open_pr")


def test_unknown_bearer_is_401():
    gw = make_gateway()
    assert handle_call(gw, "tok_nobody", "search", 0)["status"] == 401


def test_user_without_the_grant_is_403():
    gw = make_gateway()
    assert handle_call(gw, "tok_bob", "open_pr", 0)["status"] == 403


def test_unknown_tool_is_404():
    gw = make_gateway()
    assert handle_call(gw, "tok_alice", "нет такого", 0)["status"] == 404


def test_burst_beyond_the_bucket_is_429():
    gw = make_gateway()
    statuses = [handle_call(gw, "tok_alice", "search", 0)["status"] for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_rejected_calls_also_consume_the_rate_limit():
    """Иначе пользователь без доступа бесплатно долбит гейтвей отказами."""
    gw = make_gateway()
    handle_call(gw, "tok_bob", "open_pr", 0)
    handle_call(gw, "tok_bob", "open_pr", 0)
    assert handle_call(gw, "tok_bob", "search", 0)["status"] == 429


def test_every_outcome_lands_in_the_audit_log():
    gw = make_gateway()
    handle_call(gw, "tok_nobody", "search", 0)
    handle_call(gw, "tok_bob", "open_pr", 1)
    handle_call(gw, "tok_alice", "search", 2)
    assert [e["verdict"] for e in gw["audit"]] == ["unauthenticated", "forbidden", "ok"]
