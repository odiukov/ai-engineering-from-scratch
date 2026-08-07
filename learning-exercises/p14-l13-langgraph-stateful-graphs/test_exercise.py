"""Тесты к уроку «Граф состояний с чекпоинтами». Правь exercise.py."""

import pytest

from exercise import (
    END,
    START,
    load_checkpoint,
    merge_update,
    missing_from_checkpoint,
    next_node,
    resume,
    run_graph,
    save_checkpoint,
    validate_graph,
)


# ------------------------------------------------------------ вспомогательное
def _classify(state):
    text = state["input"].lower()
    if "refund" in text:
        route = "refund"
    elif "crash" in text:
        route = "bug"
    else:
        route = "sales"
    return {"route": route, "messages": [f"classified as {route}"]}


def _ticket(prefix):
    def node(state):
        return {"ticket": f"{prefix}-1", "messages": [f"{prefix} opened"]}

    return node


def _gate(state):
    if not state.get("human_approval"):
        return {"__pause__": "awaiting human approval"}
    return {"messages": ["approved"]}


def _send(state):
    return {"output": f"sent {state['ticket']}", "messages": ["sent"]}


def _support_graph():
    """classify -> (refund|bug|sales) -> gate -> send -> END."""
    return {
        "entry": "classify",
        "nodes": {
            "classify": _classify,
            "refund": _ticket("REF"),
            "bug": _ticket("BUG"),
            "sales": _ticket("SAL"),
            "gate": _gate,
            "send": _send,
        },
        "edges": {
            "classify": [
                ("refund", lambda s: s["route"] == "refund"),
                ("bug", lambda s: s["route"] == "bug"),
                ("sales", None),
            ],
            "refund": [("gate", None)],
            "bug": [("gate", None)],
            "sales": [("gate", None)],
            "gate": [("send", None)],
            "send": [(END, None)],
        },
    }


def _counting_graph(log):
    """a -> gate -> b -> END; каждый узел отмечается в log."""

    def a(state):
        log.append("a")
        return {"seen": ["a"]}

    def gate(state):
        log.append("gate")
        if not state.get("ok"):
            return {"__pause__": "need approval"}
        return {"seen": ["gate"]}

    def b(state):
        log.append("b")
        return {"seen": ["b"], "output": "final"}

    return {
        "entry": "a",
        "nodes": {"a": a, "gate": gate, "b": b},
        "edges": {"a": [("gate", None)], "gate": [("b", None)], "b": [(END, None)]},
    }


# ------------------------------------------------------------- merge_update
def test_merge_keeps_keys_the_update_does_not_mention():
    """Узел возвращает апдейт, а не состояние целиком: это слияние, не замена."""
    assert merge_update({"step": 1, "input": "x"}, {"route": "bug"}) == {
        "step": 1,
        "input": "x",
        "route": "bug",
    }


def test_merge_overwrites_scalars():
    assert merge_update({"step": 1}, {"step": 2}) == {"step": 2}


def test_merge_concatenates_lists_instead_of_replacing_them():
    """Reducer истории сообщений: список растёт, иначе диалог теряется."""
    assert merge_update({"messages": ["a"]}, {"messages": ["b"]}) == {
        "messages": ["a", "b"]
    }


def test_merge_does_not_mutate_the_old_state():
    """Чекпоинт смотрит на старое состояние — портить его нельзя."""
    old = {"messages": ["a"], "step": 1}
    merge_update(old, {"messages": ["b"], "step": 2})
    assert old == {"messages": ["a"], "step": 1}


def test_merge_of_none_update_changes_nothing():
    assert merge_update({"step": 1}, None) == {"step": 1}


# ----------------------------------------------------------------- next_node
def test_conditional_edge_picks_the_branch_that_matches_state():
    graph = _support_graph()
    assert next_node(graph, "classify", {"route": "bug"}) == "bug"


def test_unconditional_edge_is_the_fallback_branch():
    graph = _support_graph()
    assert next_node(graph, "classify", {"route": "whatever"}) == "sales"


def test_first_matching_edge_wins():
    """Порядок объявления решает: перебор идёт сверху вниз."""
    graph = {
        "entry": "a",
        "nodes": {"a": _send, "b": _send, "c": _send},
        "edges": {"a": [("b", lambda s: True), ("c", lambda s: True)]},
    }
    assert next_node(graph, "a", {}) == "b"


def test_node_without_outgoing_edges_goes_to_end():
    graph = _support_graph()
    assert next_node(graph, "nowhere", {}) == END


# ------------------------------------------------------------ validate_graph
def test_valid_graph_has_no_problems():
    assert validate_graph(_support_graph()) == []


def test_missing_entry_node_is_reported():
    graph = _support_graph()
    graph["entry"] = "typo"
    assert "no entry node: 'typo'" in validate_graph(graph)


def test_edge_into_a_node_that_does_not_exist_is_reported():
    graph = _support_graph()
    graph["edges"]["send"] = [("archive", None)]
    assert "edge to unknown node: 'archive'" in validate_graph(graph)


def test_unreachable_node_is_reported():
    """Узел есть, а дойти до него нельзя — почти всегда забытое ребро."""
    graph = _support_graph()
    graph["nodes"]["escalate"] = _send
    assert "unreachable node: 'escalate'" in validate_graph(graph)


def test_cycle_without_a_conditional_exit_is_rejected():
    """Все рёбра цикла безусловны — рантайм будет крутиться вечно."""
    graph = {
        "entry": "think",
        "nodes": {"think": _send, "act": _send},
        "edges": {"think": [("act", None)], "act": [("think", None)]},
    }
    assert "unconditional cycle: act, think" in validate_graph(graph)


def test_cycle_with_a_conditional_exit_is_allowed():
    """Цикл агента с условием выхода — нормальный паттерн, не ошибка."""
    graph = {
        "entry": "think",
        "nodes": {"think": _send, "act": _send},
        "edges": {
            "think": [(END, lambda s: s.get("done")), ("act", None)],
            "act": [("think", lambda s: True)],
        },
    }
    assert validate_graph(graph) == []


# ------------------------------------------------- save/load_checkpoint
def test_checkpoint_is_appended_to_the_session_history():
    store = {}
    assert save_checkpoint(store, "s1", "classify", {"step": 1}) == 0
    assert save_checkpoint(store, "s1", "send", {"step": 2}) == 1
    assert [node for node, _ in store["s1"]] == ["classify", "send"]


def test_checkpoint_is_a_deep_copy_of_the_state():
    """Иначе следующий узел задним числом перепишет уже сохранённый снимок."""
    store = {}
    state = {"messages": ["a"]}
    save_checkpoint(store, "s1", "classify", state)
    state["messages"].append("b")
    assert store["s1"][0][1] == {"messages": ["a"]}


def test_sessions_do_not_share_history():
    store = {}
    save_checkpoint(store, "s1", "a", {"x": 1})
    save_checkpoint(store, "s2", "a", {"x": 2})
    assert len(store["s1"]) == 1 and len(store["s2"]) == 1


def test_load_returns_the_latest_checkpoint_by_default():
    store = {}
    save_checkpoint(store, "s1", "classify", {"step": 1})
    save_checkpoint(store, "s1", "send", {"step": 2})
    assert load_checkpoint(store, "s1") == ("send", {"step": 2})


def test_load_can_reach_an_earlier_checkpoint_by_index():
    store = {}
    save_checkpoint(store, "s1", "classify", {"step": 1})
    save_checkpoint(store, "s1", "send", {"step": 2})
    assert load_checkpoint(store, "s1", 0) == ("classify", {"step": 1})


def test_loading_an_unknown_session_raises_key_error():
    with pytest.raises(KeyError):
        load_checkpoint({}, "nope")


def test_editing_a_loaded_state_does_not_rewrite_history():
    store = {}
    save_checkpoint(store, "s1", "gate", {"messages": ["a"]})
    _, state = load_checkpoint(store, "s1")
    state["messages"].append("hacked")
    assert store["s1"][0][1] == {"messages": ["a"]}


# ----------------------------------------------------------------- run_graph
def test_run_routes_through_the_branch_the_classifier_chose():
    store = {}
    result = run_graph(_support_graph(),
                       {"input": "please refund me", "human_approval": True},
                       store, "s1")
    assert result["status"] == "done"
    assert result["state"]["ticket"] == "REF-1"


def test_run_checkpoints_after_every_node():
    """Четыре узла на пути — четыре снимка, иначе возобновляться некуда."""
    store = {}
    run_graph(_support_graph(),
              {"input": "the CLI crashes", "human_approval": True},
              store, "s1")
    assert [node for node, _ in store["s1"]] == [
        "classify", "bug", "gate", "send",
    ]


def test_run_pauses_at_the_human_gate_and_saves_that_state():
    store = {}
    result = run_graph(_support_graph(), {"input": "please refund me"}, store, "s1")
    assert result["status"] == "paused"
    assert result["node"] == "gate"
    assert result["reason"] == "awaiting human approval"
    assert "__pause__" not in result["state"]


def test_start_sentinel_means_the_entry_node():
    store = {}
    from_entry = run_graph(_support_graph(),
                           {"input": "pricing", "human_approval": True},
                           store, "s1")
    from_sentinel = run_graph(_support_graph(),
                              {"input": "pricing", "human_approval": True},
                              {}, "s2", start_at=START)
    assert from_entry["state"] == from_sentinel["state"]


def test_unknown_node_raises_key_error_instead_of_stopping_silently():
    graph = _support_graph()
    graph["edges"]["classify"] = [("typo", None)]
    with pytest.raises(KeyError):
        run_graph(graph, {"input": "pricing"}, {}, "s1")


def test_max_steps_stops_a_runaway_loop():
    """Защита от зацикливания: рантайм обязан вернуться, а не висеть."""
    graph = {
        "entry": "spin",
        "nodes": {"spin": lambda s: {"n": s.get("n", 0) + 1}},
        "edges": {"spin": [("spin", None)]},
    }
    result = run_graph(graph, {"n": 0}, {}, "s1", max_steps=7)
    assert result["status"] == "max_steps"
    assert result["state"]["n"] == 7


# -------------------------------------------------------------------- resume
def test_resume_does_not_re_run_the_node_it_paused_on():
    """Узел уже отработал и записан в чекпоинт — повторять его нельзя."""
    log = []
    store = {}
    run_graph(_counting_graph(log), {}, store, "s1")
    assert log == ["a", "gate"]
    resume(_counting_graph(log), store, "s1", {"ok": True})
    assert log == ["a", "gate", "b"]


def test_resume_carries_the_state_accumulated_before_the_pause():
    store = {}
    run_graph(_support_graph(), {"input": "please refund me"}, store, "s1")
    result = resume(_support_graph(), store, "s1", {"human_approval": True})
    assert result["status"] == "done"
    assert result["state"]["output"] == "sent REF-1"
    assert result["state"]["input"] == "please refund me"


def test_human_patch_can_redirect_the_branch():
    """Правка состояния человеком должна влиять на выбор следующего ребра."""
    graph = {
        "entry": "gate",
        "nodes": {"gate": _gate, "refund": _ticket("REF"), "bug": _ticket("BUG")},
        "edges": {
            "gate": [("refund", lambda s: s.get("route") == "refund"), ("bug", None)],
        },
    }
    store = {}
    run_graph(graph, {"route": "bug"}, store, "s1")
    result = resume(graph, store, "s1", {"route": "refund"})
    assert result["state"]["ticket"] == "REF-1"


def test_resuming_an_unknown_session_raises_key_error():
    with pytest.raises(KeyError):
        resume(_support_graph(), {}, "never-started")


# --------------------------------------------------- missing_from_checkpoint
def test_key_absent_from_the_checkpoint_is_reported():
    assert missing_from_checkpoint({"a": 1, "b": 2}, {"a": 1}) == ["b"]


def test_key_whose_value_drifted_is_reported():
    assert missing_from_checkpoint({"a": 1}, {"a": 9}) == ["a"]


def test_checkpointing_only_messages_loses_the_rest_of_the_state():
    """Ровно та ошибка из урока: сохранили диалог, потеряли всё остальное."""
    state = {"messages": ["hi"], "tool_state": {"cart": 3}, "route": "refund"}
    assert missing_from_checkpoint(state, {"messages": ["hi"]}) == [
        "route", "tool_state",
    ]
