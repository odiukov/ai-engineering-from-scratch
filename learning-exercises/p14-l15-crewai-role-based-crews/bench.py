"""Входные данные для замера скорости."""

_N = 120


def _search(query):
    """Return top results for the query."""
    return f"src1, src2, src3 for {query}"


_AGENT = {
    "role": "writer",
    "goal": "turn sources into a draft",
    "backstory": " ".join(["terse"] * 150),
    "tools": tuple(),
}

_MANAGER = {
    "role": "manager",
    "goal": "pick the next specialist",
    "backstory": "PM background",
    "tools": tuple(),
}


def _agent(role):
    return {"role": role, "goal": f"do {role} work", "backstory": "terse", "tools": (_search,)}


def _task(role):
    return {
        "description": f"{role} step",
        "expected_output": f"{role} contract",
        "agent": _agent(role),
        "context": (),
    }


_MANAGER_TASK = {
    "description": "route the crew",
    "expected_output": "a role name or done",
    "agent": _MANAGER,
    "context": (),
}

_TASK = {
    "description": "write a draft",
    "expected_output": "3 paragraphs",
    "agent": _AGENT,
    "context": (),
}

_TASKS = [_task(f"role{i}") for i in range(_N)]
_CONTEXT = tuple(f"upstream output {i}" for i in range(_N))

_ROLES = tuple(f"role{i}" for i in range(_N))


def _run_agent(prompt):
    """Детерминированная заглушка: менеджер идёт по ролям, специалист отвечает шаблоном."""
    first = prompt.split("\n", 1)[0]
    if first == "ROLE: manager":
        done = ""
        for line in prompt.splitlines():
            if "DONE:" in line:
                done = line
                break
        for role in _ROLES:
            if f" {role}," not in done and not done.endswith(f" {role}"):
                return role
        return "done"
    return f"output of {first}"


def _start(payload):
    return ("topic0", payload)


def _listener(index):
    def step(value):
        return (f"topic{index + 1}", f"{value}+{index}")

    step.__name__ = f"on_topic{index}"
    return step


_LISTENERS = {f"topic{i}": _listener(i) for i in range(_N)}

_MEMORY = {
    "short_term": [f"this run: step {i}" for i in range(_N)],
    "long_term": [f"brief number {i} about agent engineering" for i in range(_N)],
    "entity": {"customer-7": ["on the enterprise plan"]},
}

BENCH = {
    "make_agent": ("writer", "turn sources into a draft", " ".join(["terse"] * 150), (_search,)),
    "make_task": ("write a draft", "3 paragraphs", _AGENT, ()),
    "crew_prompt": (_AGENT, _TASK, _CONTEXT),
    "run_sequential": (_TASKS, "agent engineering 2026", _run_agent),
    "run_hierarchical": (_MANAGER_TASK, _TASKS, "agent engineering 2026", _run_agent, _N + 1),
    "run_flow": (_start, _LISTENERS, "agent engineering 2026", _N + 2),
    "remember": ({"long_term": []}, "long_term", "crew shipped the brief"),
    "recall_context": (_MEMORY, "brief about agent engineering", "customer-7", 3),
}
