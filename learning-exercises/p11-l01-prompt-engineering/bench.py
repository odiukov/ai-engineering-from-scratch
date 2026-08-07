"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_words = ["rate", "limit", "api", "requests", "token", "budget", "model", "prompt"]
_text = " ".join(random.choice(_words) for _ in range(4000))

_template = "\n".join(f"line {i}: {{var_{i % 20}}} and {{shared}}" for i in range(200))
_variables = {f"var_{i}": f"value {i}" for i in range(20)}
_variables["shared"] = "S"

_criteria = {
    "max_words": 100,
    "required_keywords": ["rate limit", "token", "budget", "nothing here"],
    "forbidden_phrases": ["in conclusion", "as an AI"],
    "expected_format": "numbered_list",
}

_responses = {f"model_{i}": _text[: 500 * (i + 1)] for i in range(8)}

BENCH = {
    "template_variables": (_template,),
    "render_template": (_template, _variables),
    "build_prompt": ("chain_of_thought", {"problem": _text[:2000]}),
    "wrap_user_input": (_text,),
    "detect_injection": (_text,),
    "score_response": (_text, _criteria),
    "composite_score": ({"a": True, "b": 0.5, "c": 12, "d": False},),
    "rank_models": (_responses, _criteria),
}
