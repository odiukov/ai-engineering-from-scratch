"""Тесты к уроку «Хендоффы и рутины». Правь exercise.py."""

import pytest

from exercise import (
    CONTEXT_POLICIES,
    HandoffLoopError,
    can_handoff,
    context_transfer,
    handoff_stats,
    is_ping_pong,
    make_agent,
    resolve_target,
    route,
    run_conversation,
)


def support_desk():
    """Триаж и три специалиста — тот же расклад, что в code/main.py урока."""
    return {
        "triage": make_agent(
            "triage", "Route the user.", ["refund", "sales", "support", "ghost"]
        ),
        "refund": make_agent("refund", "Handle refunds.", ["triage"]),
        "sales": make_agent("sales", "Handle sales.", []),
        "support": make_agent("support", "Handle tickets.", []),
        "human": make_agent("human", "Escalate to a person.", []),
    }


DESK_RULES = {
    "triage": [
        ("refund", "refund"),
        ("buy", "sales"),
        ("broken", "support"),
        ("archive", "ghost"),
    ],
}

PING_PONG_RULES = {
    "triage": [("loop", "refund")],
    "refund": [("loop", "triage")],
}


def ring_of_three():
    """Три агента, замкнутые в кольцо: пинг-понга нет, а петля есть."""
    agents = {
        "a": make_agent("a", "", ["b"]),
        "b": make_agent("b", "", ["c"]),
        "c": make_agent("c", "", ["a"]),
    }
    rules = {"a": [("go", "b")], "b": [("go", "c")], "c": [("go", "a")]}
    return agents, rules


# ---------------------------------------------------------------- make_agent
def test_make_agent_freezes_the_handoff_list():
    agent = make_agent("triage", "Route the user.", ["refund"])
    assert agent["handoffs"] == ("refund",)


def test_make_agent_rejects_a_self_handoff():
    with pytest.raises(ValueError):
        make_agent("triage", "Route the user.", ["triage", "refund"])


# --------------------------------------------------------------- can_handoff
def test_can_handoff_allows_a_declared_target():
    assert can_handoff(make_agent("triage", "", ["refund"]), "refund") is True


def test_can_handoff_blocks_an_undeclared_target():
    assert can_handoff(make_agent("triage", "", ["refund"]), "billing") is False


# ------------------------------------------------------------ resolve_target
def test_resolve_target_returns_an_existing_agent():
    assert resolve_target(support_desk(), "refund") == "refund"


def test_resolve_target_falls_back_when_the_model_invents_a_name():
    assert resolve_target(support_desk(), "archivist", "human") == "human"


def test_resolve_target_without_a_usable_fallback_is_an_error():
    with pytest.raises(ValueError):
        resolve_target(support_desk(), "archivist")


# --------------------------------------------------------- context_transfer
def test_full_policy_carries_the_whole_history():
    history = [("user", "a"), ("bot", "b")]
    assert context_transfer(history, "full") == history


def test_last_n_policy_keeps_only_the_tail():
    assert context_transfer([("user", "a"), ("bot", "b")], "last_n", 1) == [("bot", "b")]


def test_last_n_policy_with_keep_larger_than_history_keeps_everything():
    history = [("user", "a")]
    assert context_transfer(history, "last_n", 5) == history


def test_summary_policy_collapses_the_history_to_one_entry():
    result = context_transfer([("user", "a"), ("bot", "b")], "summary")
    assert len(result) == 1
    assert result[0][0] == "summary"


def test_context_transfer_returns_a_new_list_not_the_original():
    """Иначе старый и новый владелец делят одну изменяемую историю."""
    history = [("user", "a")]
    moved = context_transfer(history, "full")
    moved.append(("bot", "b"))
    assert history == [("user", "a")]


def test_every_declared_policy_transfers_and_nothing_else_is_accepted():
    """Кортеж политик — контракт: что объявлено, то работает; лишнего нет."""
    history = [("user", "a"), ("bot", "b")]
    for policy in CONTEXT_POLICIES:
        moved = context_transfer(history, policy)
        assert moved, policy
        assert moved is not history, policy
    assert "compress" not in CONTEXT_POLICIES
    with pytest.raises(ValueError):
        context_transfer(history, "compress")


# ------------------------------------------------------------- is_ping_pong
def test_ping_pong_is_detected_on_a_full_ring():
    assert is_ping_pong(["a", "b", "a", "b"]) is True


def test_a_short_trace_is_not_yet_a_ping_pong():
    assert is_ping_pong(["a", "b", "a"]) is False


def test_three_agents_in_the_window_are_not_a_ping_pong():
    assert is_ping_pong(["a", "b", "c", "a"]) is False


def test_two_names_without_alternation_are_not_a_ping_pong():
    """a,a,b,b — два имени в окне, но никто не ходит туда-обратно."""
    assert is_ping_pong(["a", "a", "b", "b"]) is False


def test_ping_pong_looks_only_at_the_tail():
    """Легальный путь в начале не мешает поймать петлю в конце."""
    assert is_ping_pong(["x", "y", "a", "b", "a", "b"]) is True


def test_ping_pong_rejects_a_ring_shorter_than_two():
    with pytest.raises(ValueError):
        is_ping_pong(["a", "b"], ring=1)


# -------------------------------------------------------------------- route
def test_route_picks_the_target_matching_the_message():
    desk = support_desk()
    assert route(desk["triage"], "I need a refund on order 77", DESK_RULES) == "refund"


def test_route_returns_none_when_the_agent_answers_itself():
    desk = support_desk()
    assert route(desk["triage"], "hello", DESK_RULES) is None


def test_route_refuses_a_handoff_the_agent_is_not_allowed_to_make():
    """Белый список охраняет права: правило есть, разрешения нет."""
    rogue = make_agent("triage", "Route the user.", ["sales"])
    with pytest.raises(ValueError):
        route(rogue, "I need a refund", DESK_RULES)


# ---------------------------------------------------------- run_conversation
def test_handoff_moves_ownership_of_the_conversation():
    state = run_conversation(support_desk(), DESK_RULES, "triage", ["I need a refund"])
    assert state["active"] == "refund"
    assert state["trace"] == ["triage", "refund"]


def test_conversation_without_a_matching_rule_keeps_the_owner():
    state = run_conversation(support_desk(), DESK_RULES, "triage", ["hello"])
    assert state["active"] == "triage"
    assert state["trace"] == ["triage"]


def test_state_survives_the_handoff_under_the_full_policy():
    """Новый владелец видит всё, что было сказано до передачи."""
    state = run_conversation(support_desk(), DESK_RULES, "triage", ["I need a refund"])
    assert ("user", "I need a refund") in state["history"]
    assert state["history"][-1] == ("refund", "handled: I need a refund")


def test_summary_policy_shrinks_what_the_new_owner_inherits():
    chat = ["hello there", "I need a refund"]
    full = run_conversation(support_desk(), DESK_RULES, "triage", chat)
    short = run_conversation(support_desk(), DESK_RULES, "triage", chat, policy="summary")
    assert len(short["history"]) < len(full["history"])
    assert short["history"][0][0] == "summary"


def test_conversation_keeps_routing_across_several_messages():
    state = run_conversation(
        support_desk(), DESK_RULES, "triage", ["I want to buy it", "anything else"]
    )
    assert state["active"] == "sales"
    assert state["trace"] == ["triage", "sales"]


def test_invented_target_lands_on_the_fallback_agent():
    state = run_conversation(
        support_desk(), DESK_RULES, "triage", ["archive this"], fallback="human"
    )
    assert state["active"] == "human"


def test_ping_pong_handoff_raises_instead_of_hanging():
    """Два агента гоняют диалог туда-обратно — ловим, а не висим."""
    agents = support_desk()
    with pytest.raises(HandoffLoopError):
        run_conversation(agents, PING_PONG_RULES, "triage", ["loop please"])


def test_a_three_agent_ring_is_caught_by_the_hop_budget():
    """Пинг-понга здесь нет, кольцо длиннее — спасает только счётчик переходов."""
    agents, rules = ring_of_three()
    with pytest.raises(HandoffLoopError):
        run_conversation(agents, rules, "a", ["go"], max_hops=5)


def test_the_loop_error_is_not_a_plain_runtime_error():
    """Свой класс исключения, а не RuntimeError: иначе заготовка «пройдёт»."""
    assert issubclass(HandoffLoopError, Exception)
    assert not issubclass(HandoffLoopError, RuntimeError)
    with pytest.raises(HandoffLoopError):
        run_conversation(support_desk(), PING_PONG_RULES, "triage", ["loop please"])


def test_run_conversation_rejects_an_unknown_start_agent():
    with pytest.raises(ValueError):
        run_conversation(support_desk(), DESK_RULES, "archivist", ["hello"])


# ------------------------------------------------------------ handoff_stats
def test_handoff_stats_counts_hops_not_owners():
    stats = handoff_stats({"trace": ["triage", "refund", "triage"]})
    assert stats == {"hops": 2, "distinct": 2, "turns": {"triage": 2, "refund": 1}}


def test_a_conversation_without_handoffs_has_zero_hops():
    state = run_conversation(support_desk(), DESK_RULES, "triage", ["hello"])
    assert handoff_stats(state)["hops"] == 0


def test_handoff_stats_of_a_real_run():
    state = run_conversation(support_desk(), DESK_RULES, "triage", ["my app is broken"])
    assert handoff_stats(state) == {
        "hops": 1,
        "distinct": 2,
        "turns": {"triage": 1, "support": 1},
    }
