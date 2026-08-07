"""Тесты к уроку «ReWOO: план отдельно, исполнение отдельно». Правь exercise.py."""

import pytest

from exercise import (
    find_references,
    parse_plan,
    prompt_sizes,
    run_rewoo,
    run_workers,
    substitute_references,
    topological_order,
    validate_plan,
)

PLAN_TEXT = (
    "Plan: узнать столицу\n"
    "#E1 = search[capital of France]\n"
    "Plan: узнать население столицы\n"
    "#E2 = search[population of #E1]\n"
    "Plan: округлить\n"
    "#E3 = round_million[#E2]\n"
)


def _search(query):
    table = {
        "capital of France": "Paris",
        "population of Paris": "11.2 million",
    }
    return table.get(query, f"no result for {query!r}")


def _round_million(text):
    return f"{round(float(text.split()[0]))} million"


def _boom(_arg):
    raise RuntimeError("сеть недоступна")


TOOLS = {"search": _search, "round_million": _round_million}


class _CountingPlanner:
    """Планировщик, который считает, сколько раз его дёрнули."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def __call__(self, question):
        self.calls += 1
        return self.text


# ----------------------------------------------------------------- parse_plan
def test_parse_plan_extracts_id_tool_and_argument():
    assert parse_plan("#E1 = search[capital of France]") == [
        {"id": "E1", "tool": "search", "arg": "capital of France"}
    ]


def test_parse_plan_skips_planner_commentary():
    assert len(parse_plan(PLAN_TEXT)) == 3


def test_parse_plan_keeps_the_written_order():
    assert [s["id"] for s in parse_plan(PLAN_TEXT)] == ["E1", "E2", "E3"]


def test_parse_plan_rejects_a_malformed_step_line():
    with pytest.raises(ValueError):
        parse_plan("#E1 = search(capital of France)")


def test_parse_plan_on_empty_text_gives_an_empty_plan():
    assert parse_plan("") == []


# ------------------------------------------------------------ find_references
def test_find_references_finds_a_single_reference():
    assert find_references("population of #E1") == ["E1"]


def test_find_references_deduplicates_and_keeps_first_seen_order():
    assert find_references("#E2 minus #E1 plus #E2") == ["E2", "E1"]


def test_find_references_on_a_plain_argument_is_empty():
    assert find_references("capital of France") == []


# ------------------------------------------------------- substitute_references
def test_substitute_references_inserts_collected_evidence():
    assert substitute_references("population of #E1", {"E1": "Paris"}) == "population of Paris"


def test_substitute_references_leaves_an_unknown_reference_visible():
    assert substitute_references("population of #E9", {"E1": "Paris"}) == "population of #E9"


def test_substitute_references_replaces_every_occurrence():
    assert substitute_references("#E1 and #E1", {"E1": "x"}) == "x and x"


def test_substitute_references_does_not_touch_a_plain_string():
    assert substitute_references("capital of France", {"E1": "Paris"}) == "capital of France"


# --------------------------------------------------------------- validate_plan
def test_validate_plan_accepts_a_well_formed_plan():
    assert validate_plan(parse_plan(PLAN_TEXT), set(TOOLS)) == []


def test_validate_plan_reports_an_unknown_tool():
    errors = validate_plan(parse_plan("#E1 = browse[x]"), set(TOOLS))
    assert len(errors) == 1
    assert "browse" in errors[0]


def test_validate_plan_reports_a_duplicate_step_id():
    errors = validate_plan(parse_plan("#E1 = search[a]\n#E1 = search[b]"), set(TOOLS))
    assert any("повтор" in e for e in errors)


def test_validate_plan_reports_a_forward_reference():
    """Ссылка на шаг, который ещё не посчитан, — это цикл или опечатка."""
    errors = validate_plan(parse_plan("#E1 = search[#E2]\n#E2 = search[a]"), set(TOOLS))
    assert any("E2" in e for e in errors)


def test_validate_plan_collects_every_complaint_at_once():
    errors = validate_plan(parse_plan("#E1 = browse[#E7]"), set(TOOLS))
    assert len(errors) == 2


# ------------------------------------------------------------ topological_order
def test_topological_order_puts_a_dependency_before_its_dependent():
    steps = parse_plan("#E2 = search[population of #E1]\n#E1 = search[capital of France]")
    assert [s["id"] for s in topological_order(steps)] == ["E1", "E2"]


def test_topological_order_does_not_depend_on_the_input_order():
    """DAG есть DAG: перетасовка шагов не меняет порядок исполнения."""
    forward = parse_plan(PLAN_TEXT)
    shuffled = [forward[2], forward[0], forward[1]]
    assert [s["id"] for s in topological_order(shuffled)] == \
           [s["id"] for s in topological_order(forward)]


def test_topological_order_keeps_independent_steps_in_their_original_order():
    steps = parse_plan("#E1 = search[a]\n#E2 = search[b]")
    assert [s["id"] for s in topological_order(steps)] == ["E1", "E2"]


def test_topological_order_rejects_a_cycle():
    steps = [{"id": "E1", "tool": "search", "arg": "#E2"},
             {"id": "E2", "tool": "search", "arg": "#E1"}]
    with pytest.raises(ValueError):
        topological_order(steps)


# ---------------------------------------------------------------- run_workers
def test_run_workers_collects_evidence_for_every_step():
    evidence = run_workers(parse_plan(PLAN_TEXT), TOOLS)
    assert set(evidence) == {"E1", "E2", "E3"}


def test_run_workers_feeds_earlier_evidence_into_later_arguments():
    evidence = run_workers(parse_plan(PLAN_TEXT), TOOLS)
    assert evidence["E1"] == "Paris"
    assert evidence["E2"] == "11.2 million"
    assert evidence["E3"] == "11 million"


def test_run_workers_turns_a_failing_tool_into_an_evidence_string():
    """Отказ локализуется в узле: остальные шаги всё равно отрабатывают."""
    evidence = run_workers(parse_plan("#E1 = boom[a]\n#E2 = search[capital of France]"),
                           {"boom": _boom, "search": _search})
    assert evidence["E1"].startswith("error:")
    assert evidence["E2"] == "Paris"


def test_run_workers_reports_an_unknown_tool_without_raising():
    evidence = run_workers(parse_plan("#E1 = browse[a]"), TOOLS)
    assert evidence["E1"].startswith("error: unknown tool")


# ------------------------------------------------------------------ run_rewoo
def test_run_rewoo_answers_the_question_from_the_evidence():
    planner = _CountingPlanner(PLAN_TEXT)
    run = run_rewoo("сколько людей в столице Франции?", planner, TOOLS,
                    lambda q, e: f"{e['E1']}: {e['E3']}")
    assert run["answer"] == "Paris: 11 million"


def test_run_rewoo_calls_the_planner_exactly_once_for_a_three_step_plan():
    """Вся суть ReWOO: модель не вызывается на каждый шаг исполнения."""
    planner = _CountingPlanner(PLAN_TEXT)
    run = run_rewoo("вопрос", planner, TOOLS, lambda q, e: "ok")
    assert planner.calls == 1
    assert len(run["plan"]) == 3
    assert run["llm_calls"] == 2


def test_run_rewoo_llm_calls_do_not_grow_with_the_plan_length():
    short = run_rewoo("q", _CountingPlanner("#E1 = search[capital of France]"),
                      TOOLS, lambda q, e: "ok")
    long = run_rewoo("q", _CountingPlanner(PLAN_TEXT), TOOLS, lambda q, e: "ok")
    assert short["llm_calls"] == long["llm_calls"] == 2


def test_run_rewoo_calls_the_solver_once_with_the_whole_evidence():
    seen = []

    def solver(question, evidence):
        seen.append(dict(evidence))
        return "ok"

    run_rewoo("вопрос", _CountingPlanner(PLAN_TEXT), TOOLS, solver)
    assert len(seen) == 1
    assert set(seen[0]) == {"E1", "E2", "E3"}


def test_run_rewoo_rejects_an_invalid_plan_before_touching_any_tool():
    calls = []

    def counting_search(query):
        calls.append(query)
        return "Paris"

    with pytest.raises(ValueError):
        run_rewoo("вопрос", _CountingPlanner("#E1 = search[a]\n#E2 = browse[b]"),
                  {"search": counting_search}, lambda q, e: "ok")
    assert calls == []


# --------------------------------------------------------------- prompt_sizes
def test_prompt_sizes_react_makes_one_call_per_step_plus_the_final_one():
    steps = [{"tool": "search", "arg": "a", "evidence": "e"} for _ in range(4)]
    assert len(prompt_sizes("вопрос", steps, "react")) == 5


def test_prompt_sizes_react_prompt_grows_with_every_step():
    steps = [{"tool": "search", "arg": "aaa", "evidence": "eee"} for _ in range(4)]
    sizes = prompt_sizes("вопрос", steps, "react")
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


def test_prompt_sizes_rewoo_always_makes_exactly_two_calls():
    for n in (1, 3, 10):
        steps = [{"tool": "search", "arg": "aaa", "evidence": "eee"} for _ in range(n)]
        assert len(prompt_sizes("вопрос", steps, "rewoo")) == 2


def test_prompt_sizes_rewoo_is_cheaper_than_react_on_a_long_plan():
    steps = [{"tool": "search", "arg": "a" * 40, "evidence": "e" * 40} for _ in range(8)]
    assert sum(prompt_sizes("вопрос", steps, "rewoo")) < \
           sum(prompt_sizes("вопрос", steps, "react"))


def test_prompt_sizes_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        prompt_sizes("вопрос", [], "plan-and-act")
