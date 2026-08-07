"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_AUTH = "https://auth.example.com"
_NOTES = "https://notes.example.com"

_metadata = {
    "issuer": _AUTH,
    "authorization_endpoint": _AUTH + "/authorize",
    "token_endpoint": _AUTH + "/token",
    "jwks_uri": _AUTH + "/.well-known/jwks.json",
    "registration_endpoint": _AUTH + "/register",
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "code_challenge_methods_supported": ["S256"],
}

# Большой JWKS: поиск ключа по kid идёт линейно, и это видно на замере.
_keys = [{"kid": f"k_{i:04d}", "alg": "RS256"} for i in range(2000)]

_cache = {_AUTH: {"keys": _keys, "fetched_at": 0}}


def _refresh(cache, issuer, now):
    updated = dict(cache)
    updated[issuer] = {"keys": _keys, "fetched_at": now}
    return updated


_wide_scope = " ".join(f"mcp:op{i}" for i in range(500))

_server = {
    "resource": _NOTES,
    "issuers": [_AUTH],
    "prm_url": _NOTES + "/.well-known/oauth-protected-resource",
    "jwks": {_AUTH: {"keys": _keys, "fetched_at": 0}},
    "published": {_AUTH: _keys},
}

_token = {
    "iss": _AUTH,
    "kid": "k_1999",
    "aud": _NOTES,
    "scope": _wide_scope,
    "exp": 100000,
}

_cimd_url = "https://app.example.com/client.json"
_cimd_doc = {
    "client_id": _cimd_url,
    "redirect_uris": [f"https://app.example.com/cb{i}" for i in range(200)],
}

BENCH = {
    "as_metadata_errors": (_metadata,),
    "may_deploy": (_metadata,),
    "register_client": (
        {"clients": {}},
        {"redirect_uris": ["https://app.example.com/cb"],
         "token_endpoint_auth_method": "none"},
        random.Random(0),
        1000,
    ),
    "cimd_errors": (_cimd_url, _cimd_doc),
    "refresh_jwks": (_cache, _AUTH, _keys, 100),
    "resolve_signing_key": (_cache, _AUTH, "k_1999", _refresh, 100),
    "validate_token": (_server, _token, "mcp:op499", 500),
    "check_issuer_match": (_AUTH, {"code": "c", "iss": _AUTH}, True),
}
