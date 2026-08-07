"""Входные данные для замера скорости."""

import random
import string

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ALPHABET = string.ascii_letters + string.digits + "-._~"
_verifier = "".join(random.choice(_ALPHABET) for _ in range(128))

# Заранее посчитанный challenge для этого verifier — чтобы verify_pkce
# в замере шёл по «совпало», а не по «не совпало».
_challenge = "y8R5p_fci2SCIyopRs1OcL7PgEVFkkw8BgdZqhgYPbU"

_NOTES = "https://notes.example.com"
_PRM = "https://notes.example.com/.well-known/oauth-protected-resource"

_wide_scope = " ".join(f"notes:op{i}" for i in range(500))
_token = {
    "client_id": "c_bench",
    "aud": _NOTES,
    "scope": _wide_scope,
    "iat": 1000,
    "exp": 100000,
}

BENCH = {
    "code_challenge": (_verifier,),
    "verify_pkce": (_challenge, _verifier),
    "protected_resource_metadata": (_NOTES, ["https://auth.example.com"], ["notes:read"]),
    "scopes_satisfied": (_wide_scope, "notes:op499"),
    "issue_token": ("c_bench", _NOTES, _wide_scope, 1000, 3600),
    "www_authenticate": ("insufficient_scope", _PRM, "notes:delete"),
    "validate_token": (_token, _NOTES, "notes:op499", 1500, _PRM),
    "step_up_scope": (_wide_scope, "notes:delete"),
}
