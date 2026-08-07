"""Входные данные для замера скорости."""

_BAD = "- Paris is the capital of Germany\n- Mt Everest is in Europe\n- Water boils at 100C"
_GOOD = "- Paris is the capital of France\n- Mt Everest is in Asia\n- Water boils at 100C"

_WRONG_FACTS = tuple((f"fact{i}", f"part{i}") for i in range(200)) + (("paris", "germany"),)

_HISTORY = [
    {"iteration": i, "output": _BAD, "critique": f"critique {i}"} for i in range(1, 201)
]

# Скрипт длинный специально: наивный поиск ключа пройдёт по всему списку.
_SCRIPT = tuple((f"nomatch{i}", f"draft{i}") for i in range(300)) + (("", _BAD),)


def _verify(output):
    """Всегда отвергает — петля упирается в бюджет, замер воспроизводим."""
    return ("verifier: nope", False)


BENCH = {
    "format_issues": (_BAD,),
    "self_feedback": (_GOOD,),
    "external_verify": (_BAD, _WRONG_FACTS),
    "refine_prompt": ("world facts", _HISTORY),
    "scripted_generate": ("TASK: world facts", _SCRIPT),
    "should_stop": (2, False, False, 4),
    "refine_loop": ("world facts", _SCRIPT, _verify, 30),
    "loop_report": ([dict(a, verified=False, stop_reason="budget") for a in _HISTORY],),
}
