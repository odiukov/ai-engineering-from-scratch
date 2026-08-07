"""Входные данные для замера скорости."""

_ANSWERS = ("yes", "no", "maybe")
_counter = {"i": 0}


def _llm(prompt):
    """Дешёвая «модель»: детерминированный ответ по счётчику вызовов."""
    _counter["i"] += 1
    return _ANSWERS[_counter["i"] % len(_ANSWERS)] + ":" + prompt[:8]


_steps = tuple((f"s{i}", "step %d: {text}" % i) for i in range(200))
_sections = tuple((f"sec{i}", f"summarize chunk {i}") for i in range(200))
_workers = tuple(
    {"name": f"w{i}", "handles": (lambda t, i=i: i % 3 == 0),
     "run": (lambda t, i=i: f"w{i} ok")}
    for i in range(200)
)

_handlers = {f"cat{i}": (lambda t, i=i: f"handled by {i}") for i in range(50)}
_handlers["default"] = lambda t: "generic"

BENCH = {
    "prompt_chain": ("raw", _llm, _steps),
    "route": ("some ticket text", lambda t: ("cat17", 0.9), _handlers),
    "parallel_vote": ("safe to ship?", _llm, 200),
    "parallel_sections": (_sections, _llm, dict),
    "orchestrator_workers": ("review this change", _workers, list),
    "evaluator_optimizer": ("summarize ReAct", lambda task, fb: "draft",
                            lambda t, c: (False, "FAIL"), 200),
    "pick_pattern": ({"steps_known": True, "categories": 1, "parallel_units": 4},),
}
