"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ("hello", "please", "config", "weather", "report", "summary", "deploy")
_ZWSP = "​"


def _turn(i):
    body = " ".join(_rng.choice(_WORDS) for _ in range(12))
    if i % 7 == 0:
        body += " weapon"
    if i % 11 == 0:
        body = body.replace("e", "е", 1)  # кириллическая 'е'
    if i % 13 == 0:
        body = body.replace("o", "o" + _ZWSP, 1)
    return body


_TURNS = [_turn(i) for i in range(4000)]
_CASES = [(t, "S1") for t in _TURNS]

_LONG_OUTPUT = " ".join(_TURNS[:400]) + " aws_secret_access_key=zzz"

_RAIL = {"topic": "weapon", "markers": ("weapon", "violence"), "max_mentions": 50}

_CONFIG = {
    "model": "llama-guard-4",
    "taxonomy": "S1-S14",
    "input_rail": True,
    "output_rail": True,
    "dialog_rail": False,
    "normalization": False,
}

BENCH = {
    "normalize_text": (_LONG_OUTPUT,),
    "classify": (_LONG_OUTPUT, None, True),
    "verdict": (_LONG_OUTPUT, None, True),
    "route": (["S1", "S2", "S8", "S11", "S14", "S99"] * 500,),
    "output_rail": (_LONG_OUTPUT,),
    "attack_success_rate": (_CASES, None, True),
    "dialog_rail_report": (_TURNS, _RAIL),
    "audit_stack": (_CONFIG,),
}
