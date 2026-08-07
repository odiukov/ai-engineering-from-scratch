"""Входные данные для замера скорости."""

import hashlib

_OBJECTIVE = "canonicalize whitespace and title-case the input"
_DIGEST = hashlib.sha256(_OBJECTIVE.encode("utf-8")).hexdigest()[:16]
_APPROVED = {"trim", "collapse", "lower", "upper", "title", "reverse"}

_FROZEN = [f"eval/checker_{i}.py" for i in range(200)]
_EDITED = [f"src/module_{i}.py" for i in range(200)]

_HISTORY = [{"perf": 0.5 + i / 1000, "safety": 1.0} for i in range(400)]

_POLICY = {
    "frozen": _FROZEN,
    "approved_manifest": _APPROVED,
    "approved_digest": _DIGEST,
    "minimums": {"perf": 0.5, "safety": 1.0},
    "tol": 0.2,
}
_EDIT = {
    "files": ["src/agent.py"],
    "manifest": {"trim", "title"},
    "objective": _OBJECTIVE,
    "scores": {"perf": 0.9, "safety": 1.0},
}
_PROPOSALS = [
    {**_EDIT, "scores": {"perf": 0.5 + i / 1000, "safety": 1.0}} for i in range(400)
]

BENCH = {
    "anchor_digest": (_OBJECTIVE * 50,),
    "gate_frozen": (_EDITED, _FROZEN),
    "gate_invariant": ({"trim", "title"}, _APPROVED),
    "gate_anchor": (_OBJECTIVE, _DIGEST),
    "gate_multi": ({"perf": 0.9, "safety": 1.0}, {"perf": 0.5, "safety": 1.0}),
    "gate_regression": (_HISTORY, {"perf": 0.9, "safety": 1.0}, 0.2),
    "review_edit": (_EDIT, _POLICY, _HISTORY),
    "bounded_loop": (_PROPOSALS, _POLICY, 400),
}
