"""Тесты к уроку «MCP Security II — OAuth 2.1, PKCE, resource indicators».

Правь exercise.py.
"""

import pytest

from exercise import (
    code_challenge,
    issue_token,
    protected_resource_metadata,
    scopes_satisfied,
    step_up_scope,
    validate_token,
    verify_pkce,
    www_authenticate,
)

NOTES = "https://notes.example.com"
TASKS = "https://tasks.example.com"
PRM = "https://notes.example.com/.well-known/oauth-protected-resource"
VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
OTHER_VERIFIER = "aBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"


# -------------------------------------------------------------- code_challenge
def test_matches_the_rfc_7636_test_vector():
    assert code_challenge(VERIFIER) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_challenge_has_no_base64_padding():
    """Лишние "=" не совпадут с тем, что посчитает authorization server."""
    assert not code_challenge(VERIFIER).endswith("=")


def test_challenge_uses_the_url_safe_alphabet():
    assert "+" not in code_challenge(VERIFIER) and "/" not in code_challenge(VERIFIER)


def test_short_verifier_is_rejected():
    """43 символа — минимум RFC 7636; короткий verifier перебирается."""
    with pytest.raises(ValueError):
        code_challenge("too-short")


# ------------------------------------------------------------------ verify_pkce
def test_matching_verifier_passes():
    assert verify_pkce(code_challenge(VERIFIER), VERIFIER) is True


def test_substituted_verifier_fails():
    """Перехватчик кода не знает verifier — ради этого PKCE и существует."""
    assert verify_pkce(code_challenge(VERIFIER), OTHER_VERIFIER) is False


def test_plain_method_is_forbidden_even_when_the_strings_match():
    with pytest.raises(ValueError):
        verify_pkce(VERIFIER, VERIFIER, method="plain")


def test_unknown_pkce_method_is_rejected():
    with pytest.raises(ValueError):
        verify_pkce(code_challenge(VERIFIER), VERIFIER, method="S512")


def test_challenge_from_another_verifier_does_not_match():
    assert verify_pkce(code_challenge(OTHER_VERIFIER), VERIFIER) is False


# ----------------------------------------------- protected_resource_metadata
def test_metadata_names_the_resource_and_its_authorization_servers():
    got = protected_resource_metadata(NOTES, ["https://auth.example.com"], ["notes:read"])
    assert got["resource"] == NOTES
    assert got["authorization_servers"] == ["https://auth.example.com"]


def test_metadata_allows_header_only_bearer():
    """Токен в query-строке утекает в логи и в Referer."""
    got = protected_resource_metadata(NOTES, ["https://auth.example.com"], [])
    assert got["bearer_methods_supported"] == ["header"]


def test_metadata_without_authorization_servers_is_rejected():
    with pytest.raises(ValueError):
        protected_resource_metadata(NOTES, [], ["notes:read"])


def test_trailing_slash_breaks_the_canonical_resource_uri():
    """aud сравнивается побайтово — лишний слэш сделает валидный токен чужим."""
    with pytest.raises(ValueError):
        protected_resource_metadata(NOTES + "/", ["https://auth.example.com"], [])


# --------------------------------------------------------------- scopes_satisfied
def test_wider_grant_covers_a_narrower_requirement():
    assert scopes_satisfied("notes:read notes:write", "notes:write") is True


def test_missing_scope_is_not_satisfied():
    assert scopes_satisfied("notes:read", "notes:delete") is False


def test_empty_requirement_is_always_satisfied():
    assert scopes_satisfied("notes:read", "") is True


def test_scope_order_does_not_matter():
    assert scopes_satisfied("b:write a:read", "a:read b:write") is True


def test_admin_wildcard_does_not_expand_into_other_scopes():
    """«admin:*» — обычная строка, а не подстановка привилегий."""
    assert scopes_satisfied("admin:*", "notes:delete") is False


# ------------------------------------------------------------------ issue_token
def test_token_is_pinned_to_the_requested_resource():
    token = issue_token("c_1", NOTES, "notes:read", 1000, 3600)
    assert token["aud"] == NOTES


def test_token_expiry_is_issued_time_plus_ttl():
    token = issue_token("c_1", NOTES, "notes:read", 1000, 3600)
    assert (token["iat"], token["exp"]) == (1000, 4600)


def test_token_without_a_resource_cannot_be_issued():
    """Токен без aud подошёл бы любому серверу — выпускать его нельзя."""
    with pytest.raises(ValueError):
        issue_token("c_1", "", "notes:read", 1000)


def test_non_positive_ttl_is_rejected():
    with pytest.raises(ValueError):
        issue_token("c_1", NOTES, "notes:read", 1000, 0)


# ------------------------------------------------------------- www_authenticate
def test_challenge_points_at_the_metadata_document_not_the_resource():
    got = www_authenticate("invalid_token", PRM)
    assert got == f'Bearer error="invalid_token", resource_metadata="{PRM}"'


def test_insufficient_scope_challenge_names_the_missing_scope():
    got = www_authenticate("insufficient_scope", PRM, scope="notes:delete")
    assert 'scope="notes:delete"' in got and got.startswith("Bearer ")


def test_scope_is_omitted_when_not_given():
    assert "scope=" not in www_authenticate("invalid_token", PRM)


# -------------------------------------------------------------- validate_token
def test_valid_token_passes():
    token = issue_token("c_1", NOTES, "notes:read", 1000)
    got = validate_token(token, NOTES, "notes:read", 1500, PRM)
    assert (got["ok"], got["status"], got["www_authenticate"]) == (True, 200, None)


def test_token_for_another_server_is_rejected():
    """Confused deputy: токен для notes предъявили серверу tasks."""
    token = issue_token("c_1", NOTES, "notes:read", 1000)
    got = validate_token(token, TASKS, "notes:read", 1500, PRM)
    assert (got["ok"], got["status"], got["error"]) == (False, 401, "invalid_token")


def test_expired_token_is_rejected():
    token = issue_token("c_1", NOTES, "notes:read", 1000, 3600)
    assert validate_token(token, NOTES, "notes:read", 4600, PRM)["status"] == 401


def test_missing_audience_is_never_treated_as_a_wildcard():
    token = {"client_id": "c_1", "scope": "notes:read", "iat": 1000, "exp": 9999}
    assert validate_token(token, NOTES, "notes:read", 1500, PRM)["ok"] is False


def test_insufficient_scope_is_403_with_a_step_up_hint():
    token = issue_token("c_1", NOTES, "notes:read", 1000)
    got = validate_token(token, NOTES, "notes:delete", 1500, PRM)
    assert got["status"] == 403 and 'scope="notes:delete"' in got["www_authenticate"]


def test_audience_is_checked_before_scope():
    """Чужому токену не сообщаем, какого scope не хватает: это 401, не 403."""
    token = issue_token("c_1", NOTES, "notes:read", 1000)
    assert validate_token(token, TASKS, "notes:delete", 1500, PRM)["status"] == 401


# --------------------------------------------------------------- step_up_scope
def test_step_up_asks_for_old_and_new_scopes_together():
    assert step_up_scope("notes:read", "notes:delete") == "notes:delete notes:read"


def test_no_step_up_when_the_grant_already_covers_it():
    assert step_up_scope("notes:read notes:write", "notes:read") is None


def test_step_up_result_is_reproducible():
    assert step_up_scope("b:write a:read", "c:delete") == step_up_scope(
        "a:read b:write", "c:delete"
    )


def test_stepped_up_token_passes_the_check_that_failed_before():
    """Замыкаем цикл: 403 -> step-up -> новый токен -> 200."""
    token = issue_token("c_1", NOTES, "notes:read", 1000)
    denied = validate_token(token, NOTES, "notes:delete", 1500, PRM)
    wider = step_up_scope(token["scope"], "notes:delete")
    fresh = issue_token("c_1", NOTES, wider, 1500)
    after = validate_token(fresh, NOTES, "notes:delete", 1600, PRM)
    assert denied["status"] == 403 and after["ok"] is True
    assert validate_token(fresh, NOTES, "notes:read", 1600, PRM)["ok"] is True
