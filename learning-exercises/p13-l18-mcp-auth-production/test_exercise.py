"""Тесты к уроку «MCP-авторизация в проде». Правь exercise.py."""

import hashlib
import random

import pytest

from exercise import (
    as_metadata_errors,
    check_issuer_match,
    cimd_errors,
    may_deploy,
    refresh_jwks,
    register_client,
    resolve_signing_key,
    validate_token,
)

AUTH = "https://auth.example.com"
NOTES = "https://notes.example.com"
TASKS = "https://tasks.example.com"
PRM = "https://notes.example.com/.well-known/oauth-protected-resource"

GOOD_METADATA = {
    "issuer": AUTH,
    "authorization_endpoint": AUTH + "/authorize",
    "token_endpoint": AUTH + "/token",
    "jwks_uri": AUTH + "/.well-known/jwks.json",
    "registration_endpoint": AUTH + "/register",
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "code_challenge_methods_supported": ["S256"],
}


def make_server(keys=("k_2026_03",)):
    """Resource server с прогретым кэшем JWKS и одним доверенным IdP."""
    published = [{"kid": k, "alg": "RS256"} for k in keys]
    return {
        "resource": NOTES,
        "issuers": [AUTH],
        "prm_url": PRM,
        "jwks": {AUTH: {"keys": list(published), "fetched_at": 0}},
        "published": {AUTH: list(published)},
    }


def make_token(**over):
    token = {
        "iss": AUTH,
        "kid": "k_2026_03",
        "aud": NOTES,
        "scope": "mcp:tools.invoke",
        "exp": 10000,
    }
    token.update(over)
    return token


# ------------------------------------------------------------ as_metadata_errors
def test_conforming_metadata_has_no_errors():
    assert as_metadata_errors(GOOD_METADATA) == []


def test_missing_pkce_field_means_no_pkce_not_maybe_pkce():
    """Спека читает отсутствие поля как «PKCE нет», а не «наверное, есть»."""
    metadata = dict(GOOD_METADATA)
    del metadata["code_challenge_methods_supported"]
    assert "no_pkce" in as_metadata_errors(metadata)


def test_password_grant_is_forbidden():
    metadata = dict(GOOD_METADATA, grant_types_supported=["authorization_code", "password"])
    assert "forbidden_grant" in as_metadata_errors(metadata)


def test_implicit_response_type_is_rejected():
    metadata = dict(GOOD_METADATA, response_types_supported=["code", "token"])
    assert "bad_response_types" in as_metadata_errors(metadata)


def test_cimd_support_alone_satisfies_the_enrollment_requirement():
    metadata = dict(GOOD_METADATA, client_id_metadata_document_supported=True)
    del metadata["registration_endpoint"]
    assert as_metadata_errors(metadata) == []


# ------------------------------------------------------------------- may_deploy
def test_conforming_idp_may_be_deployed_against():
    assert may_deploy(GOOD_METADATA) is True


def test_missing_pkce_blocks_deployment_unconditionally():
    """У PKCE нет режима «пока без него»."""
    metadata = dict(GOOD_METADATA, code_challenge_methods_supported=["plain"])
    assert may_deploy(metadata, preregistered_client_id="c_1") is False


def test_preregistered_client_id_covers_a_missing_enrollment_path():
    metadata = dict(GOOD_METADATA)
    del metadata["registration_endpoint"]
    assert may_deploy(metadata) is False
    assert may_deploy(metadata, preregistered_client_id="c_1") is True


def test_forbidden_grant_is_not_forgiven_by_preregistration():
    metadata = dict(GOOD_METADATA, grant_types_supported=["implicit"])
    assert may_deploy(metadata, preregistered_client_id="c_1") is False


# ---------------------------------------------------------------- register_client
def test_registration_returns_a_client_id():
    registrar = {"clients": {}}
    got = register_client(
        registrar,
        {"redirect_uris": ["http://127.0.0.1:7333/callback"],
         "token_endpoint_auth_method": "none"},
        random.Random(0),
        1000,
    )
    assert got["client_id"] in registrar["clients"]
    assert got["client_id_issued_at"] == 1000


def test_registration_is_reproducible_for_a_seeded_rng():
    request = {"redirect_uris": ["https://app.example.com/cb"],
               "token_endpoint_auth_method": "none"}
    a = register_client({"clients": {}}, request, random.Random(7), 0)
    b = register_client({"clients": {}}, request, random.Random(7), 0)
    assert a["client_id"] == b["client_id"]


def test_registration_access_token_is_stored_hashed_not_in_the_clear():
    """Кража этого токена позволяет переписать redirect_uris клиента."""
    registrar = {"clients": {}}
    got = register_client(
        registrar,
        {"redirect_uris": ["https://app.example.com/cb"],
         "token_endpoint_auth_method": "none"},
        random.Random(0),
        0,
    )
    stored = registrar["clients"][got["client_id"]]
    secret = got["registration_access_token"]
    assert secret not in str(stored)
    assert stored["registration_access_token_hash"] == hashlib.sha256(
        secret.encode("ascii")
    ).hexdigest()


def test_plain_http_redirect_to_a_public_host_is_rejected():
    with pytest.raises(ValueError):
        register_client(
            {"clients": {}},
            {"redirect_uris": ["http://app.example.com/cb"],
             "token_endpoint_auth_method": "none"},
            random.Random(0),
            0,
        )


def test_client_secret_auth_method_is_rejected_for_public_clients():
    with pytest.raises(ValueError):
        register_client(
            {"clients": {}},
            {"redirect_uris": ["https://app.example.com/cb"],
             "token_endpoint_auth_method": "client_secret_post"},
            random.Random(0),
            0,
        )


# -------------------------------------------------------------------- cimd_errors
def test_matching_client_id_and_url_is_accepted():
    url = "https://app.example.com/client.json"
    doc = {"client_id": url, "redirect_uris": ["https://app.example.com/cb"]}
    assert cimd_errors(url, doc) == []


def test_client_id_that_differs_from_its_url_is_the_core_cimd_failure():
    """Без этой проверки кто угодно выдаёт себя за чужой клиент."""
    url = "https://app.example.com/client.json"
    doc = {"client_id": "https://evil.example.com/c.json", "redirect_uris": ["https://a/cb"]}
    assert "client_id_mismatch" in cimd_errors(url, doc)


def test_http_client_id_url_is_insecure():
    url = "http://app.example.com/client.json"
    doc = {"client_id": url, "redirect_uris": ["https://app.example.com/cb"]}
    assert "insecure_url" in cimd_errors(url, doc)


def test_localhost_only_redirects_are_flagged():
    """Локальный злоумышленник может присвоить чужой метадокумент."""
    url = "https://app.example.com/client.json"
    doc = {"client_id": url, "redirect_uris": ["http://127.0.0.1:7333/cb"]}
    assert cimd_errors(url, doc) == ["localhost_only_redirects"]


# ------------------------------------------------------------------ refresh_jwks
def test_refresh_writes_keys_and_fetch_time():
    got = refresh_jwks({}, AUTH, [{"kid": "k1"}], 100)
    assert got[AUTH] == {"keys": [{"kid": "k1"}], "fetched_at": 100}


def test_refresh_replaces_the_set_it_does_not_merge():
    """Снятый с публикации ключ обязан исчезнуть и у нас."""
    cache = refresh_jwks({}, AUTH, [{"kid": "k1"}], 100)
    cache = refresh_jwks(cache, AUTH, [{"kid": "k2"}], 200)
    assert [k["kid"] for k in cache[AUTH]["keys"]] == ["k2"]


def test_refresh_does_not_mutate_the_old_cache():
    cache = refresh_jwks({}, AUTH, [{"kid": "k1"}], 100)
    refresh_jwks(cache, AUTH, [{"kid": "k2"}], 200)
    assert [k["kid"] for k in cache[AUTH]["keys"]] == ["k1"]


# ----------------------------------------------------------- resolve_signing_key
def test_cached_key_is_found_without_any_refetch():
    calls = []

    def refresh(cache, issuer, now):
        calls.append(now)
        return cache

    cache = {AUTH: {"keys": [{"kid": "k1"}], "fetched_at": 0}}
    key, _ = resolve_signing_key(cache, AUTH, "k1", refresh, 100)
    assert key == {"kid": "k1"} and calls == []


def test_cache_miss_triggers_exactly_one_refetch():
    calls = []

    def refresh(cache, issuer, now):
        calls.append(now)
        return refresh_jwks(cache, issuer, [{"kid": "k1"}, {"kid": "k2"}], now)

    cache = {AUTH: {"keys": [{"kid": "k1"}], "fetched_at": 0}}
    key, cache = resolve_signing_key(cache, AUTH, "k2", refresh, 100)
    assert key == {"kid": "k2"} and len(calls) == 1


def test_bogus_kid_costs_at_most_one_refetch():
    """Иначе поток токенов со случайными kid превращается в DoS."""
    calls = []

    def refresh(cache, issuer, now):
        calls.append(now)
        return refresh_jwks(cache, issuer, [{"kid": "k1"}], now)

    cache = {AUTH: {"keys": [{"kid": "k1"}], "fetched_at": 0}}
    key, _ = resolve_signing_key(cache, AUTH, "k_случайный", refresh, 100)
    assert key is None and len(calls) == 1


def test_refetch_does_not_grow_the_published_key_set():
    """Запасной путь — перечитать, а не выпустить новый ключ."""
    published = [{"kid": "k1"}]

    def refresh(cache, issuer, now):
        return refresh_jwks(cache, issuer, published, now)

    cache = {AUTH: {"keys": list(published), "fetched_at": 0}}
    resolve_signing_key(cache, AUTH, "k_нет", refresh, 100)
    assert len(published) == 1


# ----------------------------------------------------------------- validate_token
def test_valid_token_passes():
    got = validate_token(make_server(), make_token(), "mcp:tools.invoke", 500)
    assert (got["valid"], got["status"], got["www_authenticate"]) == (True, 200, None)


def test_wrong_or_missing_audience_is_rejected():
    """Access-token privilege restriction: токен notes предъявили серверу tasks.

    Отсутствие aud тоже отказ — «нет audience» никогда не значит «подходит всем».
    """
    server = dict(make_server(), resource=TASKS)
    got = validate_token(server, make_token(), "mcp:tools.invoke", 500)
    assert got["status"] == 401 and "audience mismatch" in got["www_authenticate"]

    no_aud = make_token()
    del no_aud["aud"]
    assert validate_token(make_server(), no_aud, "mcp:tools.invoke", 500)["valid"] is False


def test_token_from_an_unlisted_issuer_is_rejected():
    """Иначе злоумышленник поднимает свой AS и выписывает себе что угодно."""
    got = validate_token(make_server(), make_token(iss="https://evil"), "", 500)
    assert "iss not allowed" in got["www_authenticate"]


def test_expired_token_is_rejected():
    assert validate_token(make_server(), make_token(exp=100), "", 500)["status"] == 401


def test_insufficient_scope_is_403_and_names_the_missing_scope():
    got = validate_token(make_server(), make_token(), "mcp:admin", 500)
    assert got["status"] == 403 and 'scope="mcp:admin"' in got["www_authenticate"]


def test_rotated_key_validates_after_the_cache_refresh():
    """Ключ, которого нет в кэше, доезжает одним перечитыванием, без рестарта."""
    server = make_server()
    server["published"][AUTH] = [{"kid": "k_2026_03"}, {"kid": "k_2026_04"}]
    got = validate_token(server, make_token(kid="k_2026_04"), "mcp:tools.invoke", 500)
    assert got["valid"] is True
    assert {k["kid"] for k in server["jwks"][AUTH]["keys"]} == {"k_2026_03", "k_2026_04"}


# -------------------------------------------------------------- check_issuer_match
def test_matching_issuer_is_accepted():
    assert check_issuer_match(AUTH, {"code": "c", "iss": AUTH}, True) == (True, "ok")


def test_foreign_issuer_is_a_mixup_attempt():
    assert check_issuer_match(AUTH, {"code": "c", "iss": "https://evil"}, True) == (
        False,
        "issuer_mismatch",
    )


def test_advertised_iss_missing_from_the_response_is_rejected():
    """Пропустить такой ответ — вернуть себе ровно ту дыру, что закрывает RFC 9207."""
    assert check_issuer_match(AUTH, {"code": "c"}, True) == (False, "missing_iss")


def test_legacy_as_without_iss_support_still_works():
    assert check_issuer_match(AUTH, {"code": "c"}, False) == (True, "ok")
