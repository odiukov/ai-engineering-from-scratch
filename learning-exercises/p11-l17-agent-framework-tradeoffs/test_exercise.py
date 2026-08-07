"""Тесты к уроку «Выбор агентного фреймворка». Правь exercise.py."""

import pytest

from exercise import (
    DEFAULT_PROBLEM,
    FRAMEWORKS,
    ProblemError,
    compare_run_cost,
    hard_filter,
    normalize_problem,
    pick_framework,
    routing_cost_per_run,
    score,
    shape_of,
)

APPROX = lambda x: pytest.approx(x, rel=1e-9)


# ------------------------------------------------------- normalize_problem
def test_defaults_fill_in_every_missing_field():
    assert normalize_problem({}) == DEFAULT_PROBLEM


def test_given_fields_win_over_the_defaults():
    got = normalize_problem({"llm_calls": 5, "needs_resume": True})
    assert (got["llm_calls"], got["needs_resume"]) == (5, True)
    assert got["has_roles"] is False


def test_an_unknown_key_is_rejected_instead_of_ignored():
    """Опечатка в имени флага молча выключила бы требование."""
    with pytest.raises(ProblemError):
        normalize_problem({"need_resume": True})


def test_a_flag_that_is_not_a_bool_is_rejected():
    with pytest.raises(ProblemError):
        normalize_problem({"needs_resume": "yes"})


def test_llm_calls_must_be_a_positive_integer():
    with pytest.raises(ProblemError):
        normalize_problem({"llm_calls": 0})
    with pytest.raises(ProblemError):
        normalize_problem({"llm_calls": "many"})


def test_true_does_not_sneak_through_as_one_llm_call():
    """В Python bool — подкласс int, и llm_calls=True прошло бы как единица."""
    with pytest.raises(ProblemError):
        normalize_problem({"llm_calls": True})


def test_normalize_does_not_mutate_the_input():
    problem = {"llm_calls": 3}
    normalize_problem(problem)
    assert problem == {"llm_calls": 3}


# ---------------------------------------------------------------- shape_of
def test_two_calls_and_nothing_else_is_a_trivial_shape():
    assert shape_of({"llm_calls": 2}) == "trivial"


def test_typed_state_makes_it_a_graph():
    assert shape_of({"llm_calls": 6, "has_typed_state": True}) == "graph"


def test_dialogue_beats_roles_when_both_are_declared():
    assert shape_of({"llm_calls": 6, "has_dialogue": True, "has_roles": True}) == "dialogue"


def test_roles_without_dialogue_is_a_role_play():
    assert shape_of({"llm_calls": 6, "has_roles": True}) == "roles"


def test_many_calls_and_no_structure_is_a_single_agent():
    assert shape_of({"llm_calls": 6}) == "agent"


def test_a_runtime_requirement_outranks_a_short_pipeline():
    """Две LLM-ки, но с согласованием человеком — уже не «тридцать строк»."""
    assert shape_of({"llm_calls": 2, "needs_human_approval": True}) == "graph"


# ------------------------------------------------------------- hard_filter
def test_without_hard_requirements_everybody_passes():
    assert hard_filter({}) == list(FRAMEWORKS)


def test_resume_after_a_restart_needs_a_real_checkpointer():
    """Журнал сессии Agno и стенограмма AutoGen — это не чекпоинтер."""
    assert hard_filter({"needs_resume": True}) == ["langgraph"]


def test_human_approval_needs_interrupts():
    assert hard_filter({"needs_human_approval": True}) == ["langgraph"]


def test_parallel_fanout_needs_a_dispatch_api():
    assert hard_filter({"has_parallel_fanout": True}) == ["langgraph"]


def test_hard_filter_keeps_the_declared_order():
    assert hard_filter({"llm_calls": 4}) == list(FRAMEWORKS)


# -------------------------------------------------------------------- score
def test_matching_abstraction_is_worth_more_than_the_setup_penalty():
    graph_problem = {"llm_calls": 6, "has_typed_state": True}
    assert score("langgraph", graph_problem) > score("agno", graph_problem)


def test_the_setup_penalty_only_breaks_ties():
    """Все четверо мимо формы — тогда решает объём обвязки."""
    dialogue = {"llm_calls": 6, "has_dialogue": True}
    assert score("agno", dialogue) > score("langgraph", dialogue)
    assert score("autogen", dialogue) > score("agno", dialogue)


def test_closing_a_hard_requirement_adds_to_the_score():
    plain = {"llm_calls": 6, "has_typed_state": True}
    with_resume = dict(plain, needs_resume=True)
    assert score("langgraph", with_resume) == APPROX(score("langgraph", plain) + 1.0)


def test_score_rejects_an_unknown_framework():
    with pytest.raises(ValueError):
        score("crew-ai-2", {})


# ----------------------------------------------------------- pick_framework
def test_a_short_pipeline_gets_no_framework_at_all():
    """Ни один фреймворк не дешевле отсутствия фреймворка."""
    assert pick_framework({"llm_calls": 2})["framework"] == "plain-python"


def test_resume_forces_langgraph_even_on_a_tiny_task():
    got = pick_framework({"llm_calls": 1, "needs_resume": True})
    assert got["framework"] == "langgraph"
    assert got["shape"] == "graph"


def test_a_dialogue_goes_to_autogen():
    assert pick_framework({"llm_calls": 6, "has_dialogue": True})["framework"] == "autogen"


def test_a_role_play_goes_to_crewai():
    assert pick_framework({"llm_calls": 6, "has_roles": True})["framework"] == "crewai"


def test_a_single_agent_with_tools_goes_to_agno():
    assert pick_framework({"llm_calls": 6})["framework"] == "agno"


def test_the_runners_up_are_ranked_and_exclude_the_winner():
    got = pick_framework({"llm_calls": 6, "has_dialogue": True})
    assert got["framework"] not in got["runners_up"]
    assert len(got["runners_up"]) == len(FRAMEWORKS) - 1
    ranked = [score(name, {"llm_calls": 6, "has_dialogue": True}) for name in got["runners_up"]]
    assert ranked == sorted(ranked, reverse=True)


def test_the_recommendation_carries_a_reason():
    got = pick_framework({"llm_calls": 6, "has_typed_state": True})
    assert "graph" in got["reason"]
    assert got["framework"] in got["reason"]


def test_the_answer_does_not_depend_on_the_order_of_the_keys():
    a = pick_framework({"llm_calls": 6, "has_roles": True, "needs_resume": False})
    b = pick_framework({"needs_resume": False, "has_roles": True, "llm_calls": 6})
    assert a == b


def test_the_answer_is_the_same_on_every_call():
    problem = {"llm_calls": 6, "has_dialogue": True}
    assert pick_framework(problem) == pick_framework(problem)


# ------------------------------------------------------ routing_cost_per_run
def test_explicit_routing_costs_no_tokens():
    assert routing_cost_per_run("langgraph", 10, 5.0, 15.0) == APPROX(0.0)


def test_a_manager_agent_pays_for_every_turn():
    assert routing_cost_per_run("crewai", 10, 5.0, 15.0) == APPROX(0.018)


def test_routing_cost_grows_linearly_with_the_turns():
    one = routing_cost_per_run("autogen", 1, 5.0, 15.0)
    fifty = routing_cost_per_run("autogen", 50, 5.0, 15.0)
    assert fifty == APPROX(50 * one)


def test_the_chattier_router_costs_more():
    assert routing_cost_per_run("autogen", 10, 5.0, 15.0) > routing_cost_per_run(
        "crewai", 10, 5.0, 15.0
    )


def test_routing_cost_rejects_an_unknown_framework():
    with pytest.raises(ValueError):
        routing_cost_per_run("langchain", 10, 5.0, 15.0)


# -------------------------------------------------------- compare_run_cost
def test_comparison_is_sorted_from_cheap_to_expensive():
    costs = [cost for _, cost in compare_run_cost({"llm_calls": 6}, 10, 5.0, 15.0)]
    assert costs == sorted(costs)


def test_comparison_only_lists_frameworks_that_can_do_the_job():
    """Сравнивать по цене то, что не выполнит задачу, бессмысленно."""
    assert compare_run_cost({"needs_resume": True}, 10, 5.0, 15.0) == [("langgraph", 0.0)]


def test_ties_are_broken_by_name_so_the_report_is_reproducible():
    free = [name for name, cost in compare_run_cost({"llm_calls": 6}, 10, 5.0, 15.0) if cost == 0]
    assert free == sorted(free)


def test_explicit_routing_wins_the_comparison_outright():
    ordered = compare_run_cost({"llm_calls": 6}, 10, 5.0, 15.0)
    assert ordered[0][1] == APPROX(0.0)
    assert ordered[-1][0] == "autogen"
