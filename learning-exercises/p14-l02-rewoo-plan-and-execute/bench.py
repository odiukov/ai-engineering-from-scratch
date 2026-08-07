"""Входные данные для замера скорости."""

_N = 200

# план из 200 шагов, где каждый следующий ссылается на предыдущий
_plan_text = "\n".join(
    [f"#E1 = up[seed]"]
    + [f"#E{i} = up[#E{i - 1} tail]" for i in range(2, _N + 1)]
)

# тот же план, но записанный задом наперёд — на нём видно цену топологической
# сортировки, реализованной наивно
_reversed_plan_text = "\n".join(reversed(_plan_text.splitlines()))

_steps = [
    {"id": f"E{i}", "tool": "up", "arg": ("seed" if i == 1 else f"#E{i - 1} tail")}
    for i in range(1, _N + 1)
]
_bench_steps = [{"tool": "up", "arg": "a" * 30, "evidence": "e" * 30} for _ in range(_N)]
_tools = {"up": str.upper}
_evidence = {f"E{i}": "x" for i in range(1, _N + 1)}
_long_arg = " ".join(f"#E{i}" for i in range(1, _N + 1))

BENCH = {
    "parse_plan": (_plan_text,),
    "find_references": (_long_arg,),
    "substitute_references": (_long_arg, _evidence),
    "validate_plan": (_steps, {"up"}),
    "topological_order": (list(reversed(_steps)),),
    "run_workers": (_steps, _tools),
    "run_rewoo": ("вопрос", lambda q: _plan_text, _tools, lambda q, e: "ok"),
    "prompt_sizes": ("вопрос", _bench_steps, "react"),
}
