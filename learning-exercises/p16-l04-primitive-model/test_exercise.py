"""Тесты к уроку «Примитивная модель мультиагента». Правь exercise.py."""

import pytest

from exercise import (
    DONE,
    agents_are_interchangeable,
    make_agent,
    post,
    project,
    round_robin_selector,
    run_handoff,
    run_selector,
    run_static,
)


# ------------------------------------------------------------- политики агентов
def researcher_policy(pool):
    return {"content": "note: FIPA ratified 2000", "handoff": "writer"}


def writer_policy(pool):
    notes = [m["content"] for m in pool if m["from"] == "researcher"]
    return {"content": "draft: " + " | ".join(notes), "handoff": "reviewer"}


def reviewer_policy(pool):
    return {"content": "verdict: approved", "handoff": DONE}


def ping_policy(pool):
    return {"content": "ping", "handoff": "pong"}


def pong_policy(pool):
    return {"content": "pong", "handoff": "ping"}


def lost_policy(pool):
    return {"content": "who is next?", "handoff": "nobody-here"}


# Пул собирается литералами, чтобы модуль не звал непройденные функции на
# импорте: иначе весь файл упал бы на коллекции и ничего бы не проверил.
POOL_OF_THREE = [
    {"from": "researcher", "content": "note", "handoff": "writer"},
    {"from": "writer", "content": "draft", "handoff": "reviewer"},
    {"from": "reviewer", "content": "verdict", "handoff": DONE},
]


def build_team():
    return {
        "researcher": make_agent("researcher", "Gather facts.", ["search"],
                                 researcher_policy),
        "writer": make_agent("writer", "Draft from research.", ["editor"],
                             writer_policy),
        "reviewer": make_agent("reviewer", "Critique the draft.", [], reviewer_policy),
    }


def speakers(pool):
    return [m["from"] for m in pool]


# --------------------------------------------------------------- make_agent
def test_make_agent_keeps_prompt_and_tools():
    a = make_agent("researcher", "Gather facts.", ["search"], researcher_policy)
    assert a["system_prompt"] == "Gather facts." and a["tools"] == ["search"]


def test_make_agent_stores_a_callable_policy():
    a = make_agent("researcher", "P", [], researcher_policy)
    assert a["policy"]([])["content"].startswith("note")


def test_make_agent_does_not_alias_the_tool_list():
    tools = ["search"]
    a = make_agent("researcher", "P", tools, researcher_policy)
    tools.append("shell")
    assert a["tools"] == ["search"]


# ------------------------------------------------- agents_are_interchangeable
def test_same_prompt_and_tools_means_interchangeable():
    """Имя — ярлык оркестратора, а не часть агента."""
    a = make_agent("a", "P", ["t"], researcher_policy)
    b = make_agent("b", "P", ["t"], writer_policy)
    assert agents_are_interchangeable(a, b) is True


def test_different_prompt_breaks_interchangeability():
    a = make_agent("a", "P", ["t"], researcher_policy)
    b = make_agent("a", "Q", ["t"], researcher_policy)
    assert agents_are_interchangeable(a, b) is False


def test_different_tools_break_interchangeability():
    a = make_agent("a", "P", ["t"], researcher_policy)
    b = make_agent("a", "P", ["t", "shell"], researcher_policy)
    assert agents_are_interchangeable(a, b) is False


def test_tool_order_does_not_matter():
    a = make_agent("a", "P", ["x", "y"], researcher_policy)
    b = make_agent("a", "P", ["y", "x"], researcher_policy)
    assert agents_are_interchangeable(a, b) is True


# --------------------------------------------------------------------- post
def test_post_appends_to_the_pool_in_place():
    pool = []
    post(pool, "researcher", "note", "writer")
    assert pool == [{"from": "researcher", "content": "note", "handoff": "writer"}]


def test_post_returns_the_message_it_wrote():
    pool = []
    assert post(pool, "a", "x") is pool[-1]


def test_post_without_a_target_addresses_nobody():
    pool = []
    assert post(pool, "a", "x")["handoff"] is None


def test_pool_is_the_only_stateful_thing():
    """Два вызова подряд копятся в одном списке — состояние живёт тут."""
    pool = []
    post(pool, "a", "x")
    post(pool, "b", "y")
    assert speakers(pool) == ["a", "b"]


# ------------------------------------------------------------------ project
def test_projection_shows_what_was_addressed_to_the_agent():
    got = project(POOL_OF_THREE, "writer")
    assert [m["from"] for m in got] == ["researcher", "writer"]


def test_projection_hides_conversations_between_others():
    """reviewer не должен видеть переписку researcher -> writer."""
    got = project(POOL_OF_THREE, "reviewer")
    assert all(m["from"] != "researcher" for m in got)


def test_projection_is_smaller_than_the_full_pool():
    assert len(project(POOL_OF_THREE, "reviewer")) < len(POOL_OF_THREE)


def test_projection_of_a_stranger_is_empty():
    assert project(POOL_OF_THREE, "auditor") == []


# --------------------------------------------------------------- run_static
def test_static_orchestrator_follows_the_declared_order():
    pool = []
    run_static(build_team(), pool, ["researcher", "writer", "reviewer"])
    assert speakers(pool) == ["researcher", "writer", "reviewer"]


def test_static_orchestrator_ignores_the_agents_own_handoff():
    """reviewer говорит DONE, но маршрут прибит в коде — researcher всё равно бежит."""
    pool = []
    run_static(build_team(), pool, ["reviewer", "researcher"])
    assert speakers(pool) == ["reviewer", "researcher"]


def test_static_orchestrator_with_one_agent_is_just_that_agent():
    pool = []
    run_static(build_team(), pool, ["writer"])
    assert speakers(pool) == ["writer"]


def test_static_orchestrator_respects_max_steps():
    pool = []
    run_static(build_team(), pool, ["researcher", "writer", "reviewer"], max_steps=2)
    assert speakers(pool) == ["researcher", "writer"]


def test_static_orchestrator_lets_the_writer_read_the_research():
    """Общее состояние работает: writer видит заметку researcher'а."""
    pool = []
    run_static(build_team(), pool, ["researcher", "writer"])
    assert "FIPA ratified 2000" in pool[1]["content"]


# -------------------------------------------------------------- run_handoff
def test_handoff_orchestrator_follows_the_chain():
    pool = []
    run_handoff(build_team(), pool, "researcher")
    assert speakers(pool) == ["researcher", "writer", "reviewer"]


def test_handoff_orchestrator_stops_on_done():
    pool = []
    run_handoff(build_team(), pool, "reviewer")
    assert speakers(pool) == ["reviewer"]


def test_handoff_to_an_unknown_agent_stops_without_crashing():
    team = {"lost": make_agent("lost", "P", [], lost_policy)}
    pool = []
    run_handoff(team, pool, "lost")
    assert speakers(pool) == ["lost"]


def test_max_steps_breaks_a_handoff_loop():
    """Ловушка LLM-маршрутизации: два агента могут пасовать друг другу вечно."""
    team = {"ping": make_agent("ping", "P", [], ping_policy),
            "pong": make_agent("pong", "P", [], pong_policy)}
    pool = []
    run_handoff(team, pool, "ping", max_steps=4)
    assert speakers(pool) == ["ping", "pong", "ping", "pong"]


def test_unknown_start_produces_nothing():
    pool = []
    run_handoff(build_team(), pool, "auditor")
    assert pool == []


# ------------------------------------------------------ round_robin_selector
def test_round_robin_picks_the_next_name():
    assert round_robin_selector(POOL_OF_THREE[:1], ["researcher", "writer"]) == "writer"


def test_round_robin_wraps_around():
    assert round_robin_selector(POOL_OF_THREE, ["researcher", "writer", "reviewer"]) == \
        "researcher"


def test_round_robin_on_an_empty_pool_selects_nobody():
    assert round_robin_selector([], ["a", "b"]) is None


def test_round_robin_does_not_guess_after_a_stranger():
    pool = [{"from": "auditor", "content": "x", "handoff": None}]
    assert round_robin_selector(pool, ["a", "b"]) is None


# ------------------------------------------------------------- run_selector
def test_selector_orchestrator_goes_round_the_team():
    pool = []
    run_selector(build_team(), pool, "researcher", round_robin_selector, max_steps=3)
    assert speakers(pool) == ["researcher", "writer", "reviewer"]


def test_selector_returning_none_stops_the_run():
    pool = []
    run_selector(build_team(), pool, "researcher", lambda pool_, names: None)
    assert speakers(pool) == ["researcher"]


def test_selector_overrides_the_agents_own_handoff():
    """Разница между оркестраторами ровно одна: КТО решает, кто следующий."""
    pool = []
    run_selector(build_team(), pool, "researcher",
                 lambda pool_, names: "reviewer" if len(pool_) == 1 else None)
    assert speakers(pool) == ["researcher", "reviewer"]


def test_all_three_orchestrators_start_the_same_way():
    """Агенты и общее состояние одинаковы; расходятся только маршруты."""
    pools = []
    for run in (
        lambda p: run_static(build_team(), p, ["researcher"]),
        lambda p: run_handoff(build_team(), p, "researcher", max_steps=1),
        lambda p: run_selector(build_team(), p, "researcher",
                               lambda pool_, names: None),
    ):
        pool = []
        run(pool)
        pools.append(pool[0]["content"])
    assert len(set(pools)) == 1
