"""Тесты к уроку «Агент как машина состояний». Правь exercise.py."""

import pytest

from exercise import (
    END,
    GraphError,
    RecursionLimit,
    add_messages,
    build_react_graph,
    compile_graph,
    merge_state,
    resume,
    route,
    run_graph,
)


def inc(state):
    return {"n": state["n"] + 1}


def counter_graph():
    """Один узел, который считает до трёх, и условное ребро назад в себя."""
    return compile_graph(
        {"inc": inc},
        {"inc": (lambda s: "again" if s["n"] < 3 else "done", {"again": "inc", "done": END})},
        "inc",
    )


def scripted_model(script):
    """Модель, отдающая заранее записанные ответы по очереди."""
    turns = list(script)

    def model(messages):
        return turns.pop(0)

    return model


def weather_agent():
    """Агент, который один раз зовёт инструмент, а потом отвечает текстом."""
    script = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "temp", "args": {"city": "Paris"}}],
        },
        {"role": "assistant", "content": "In Paris it is 21C."},
    ]
    return build_react_graph(scripted_model(script), {"temp": lambda city: f"21C in {city}"})


# ------------------------------------------------------------- add_messages
def test_add_messages_appends_instead_of_replacing():
    assert add_messages([{"role": "user"}], [{"role": "ai"}]) == [
        {"role": "user"},
        {"role": "ai"},
    ]


def test_add_messages_treats_missing_history_as_empty():
    assert add_messages(None, [{"role": "user"}]) == [{"role": "user"}]


def test_add_messages_accepts_a_single_message_without_a_list():
    assert add_messages([], {"role": "ai"}) == [{"role": "ai"}]


def test_add_messages_does_not_touch_the_old_list():
    """На старый список смотрят уже снятые чекпоинты — править его нельзя."""
    old = [{"role": "user"}]
    add_messages(old, [{"role": "ai"}])
    assert old == [{"role": "user"}]


# --------------------------------------------------------------- merge_state
def test_merge_overwrites_by_default():
    assert merge_state({"n": 1}, {"n": 2}) == {"n": 2}


def test_merge_keeps_the_fields_the_node_did_not_touch():
    assert merge_state({"n": 1, "plan": "x"}, {"n": 2}) == {"n": 2, "plan": "x"}


def test_merge_uses_the_reducer_where_one_is_declared():
    got = merge_state({"m": [1]}, {"m": [2]}, {"m": add_messages})
    assert got["m"] == [1, 2]


def test_a_forgotten_reducer_silently_loses_half_the_turn():
    """Самая частая ошибка в LangGraph — и она не падает, а тихо теряет данные."""
    without = merge_state({"m": [1]}, {"m": [2]})
    assert without["m"] == [2]


def test_merge_returns_a_new_dict():
    state = {"n": 1}
    merge_state(state, {"n": 2})
    assert state == {"n": 1}


def test_merge_allows_a_node_to_introduce_a_new_field():
    assert merge_state({"n": 1}, {"plan": ["step"]}, {"plan": add_messages}) == {
        "n": 1,
        "plan": ["step"],
    }


# ------------------------------------------------------------ compile_graph
def test_compile_returns_the_topology_and_the_reducers():
    graph = compile_graph({"a": inc}, {"a": END}, "a", {"n": add_messages})
    assert graph["entry"] == "a"
    assert set(graph["nodes"]) == {"a"}
    assert graph["reducers"] == {"n": add_messages}


def test_compile_rejects_an_unknown_entry_point():
    with pytest.raises(GraphError):
        compile_graph({"a": inc}, {"a": END}, "b")


def test_compile_rejects_an_edge_into_a_node_that_does_not_exist():
    with pytest.raises(GraphError):
        compile_graph({"a": inc}, {"a": "ghost"}, "a")


def test_compile_rejects_an_edge_out_of_a_node_that_does_not_exist():
    with pytest.raises(GraphError):
        compile_graph({"a": inc}, {"a": END, "ghost": END}, "a")


def test_compile_rejects_a_node_with_nowhere_to_go():
    with pytest.raises(GraphError):
        compile_graph({"a": inc, "b": inc}, {"a": "b"}, "a")


def test_compile_catches_a_static_cycle_instead_of_hanging_on_it():
    """Два статических ребра по кругу не кончатся ни при каком состоянии."""
    with pytest.raises(GraphError):
        compile_graph({"a": inc, "b": inc}, {"a": "b", "b": "a"}, "a")


def test_compile_allows_a_cycle_that_goes_through_a_conditional_edge():
    """ReAct — это цикл. Запрещать циклы нельзя, запрещать безвыходные — нужно."""
    graph = counter_graph()
    assert run_graph(graph, {"n": 0})["state"] == {"n": 3}


def test_compile_detaches_the_graph_from_the_dicts_it_was_given():
    nodes = {"a": inc}
    graph = compile_graph(nodes, {"a": END}, "a")
    nodes["b"] = inc
    assert set(graph["nodes"]) == {"a"}


# --------------------------------------------------------------------- route
def test_route_follows_a_static_edge_regardless_of_state():
    graph = compile_graph({"a": inc}, {"a": END}, "a")
    assert route(graph, "a", {"n": 0}) == END
    assert route(graph, "a", {"n": 99}) == END


def test_route_asks_the_router_on_a_conditional_edge():
    graph = counter_graph()
    assert route(graph, "inc", {"n": 1}) == "inc"
    assert route(graph, "inc", {"n": 3}) == END


def test_route_rejects_a_branch_that_is_not_on_the_map():
    """Молча уйти в END на опечатке хуже, чем упасть: агент «отработает» вхолостую."""
    graph = compile_graph({"a": inc}, {"a": (lambda s: "typo", {"go": END})}, "a")
    with pytest.raises(GraphError):
        route(graph, "a", {"n": 0})


# ----------------------------------------------------------------- run_graph
def test_run_returns_the_final_state():
    assert run_graph(counter_graph(), {"n": 0})["state"] == {"n": 3}


def test_run_writes_one_checkpoint_per_transition_plus_the_initial_one():
    run = run_graph(counter_graph(), {"n": 0})
    assert [c["id"] for c in run["checkpoints"]] == [0, 1, 2, 3]
    assert run["checkpoints"][0]["state"] == {"n": 0}
    assert run["checkpoints"][-1]["next"] == END


def test_checkpoints_are_snapshots_and_do_not_move_with_the_state():
    run = run_graph(counter_graph(), {"n": 0})
    assert [c["state"]["n"] for c in run["checkpoints"]] == [0, 1, 2, 3]


def test_run_does_not_mutate_the_initial_state():
    start = {"n": 0}
    run_graph(counter_graph(), start)
    assert start == {"n": 0}


def test_run_stops_a_runaway_loop_instead_of_spinning_forever():
    graph = compile_graph({"a": inc}, {"a": (lambda s: "again", {"again": "a"})}, "a")
    with pytest.raises(RecursionLimit):
        run_graph(graph, {"n": 0}, max_steps=10)


def test_the_recursion_limit_counts_node_runs():
    calls = []

    def node(state):
        calls.append(state["n"])
        return {"n": state["n"] + 1}

    graph = compile_graph({"a": node}, {"a": (lambda s: "again", {"again": "a"})}, "a")
    with pytest.raises(RecursionLimit):
        run_graph(graph, {"n": 0}, max_steps=5)
    assert calls == [0, 1, 2, 3, 4]


def test_interrupt_stops_before_the_node_runs_not_after():
    """Согласовывать удаление базы после удаления уже нечего."""
    run = run_graph(counter_graph(), {"n": 0}, interrupt_before=["inc"])
    assert run["interrupted"] == "inc"
    assert run["state"] == {"n": 0}


# -------------------------------------------------------------------- resume
def test_resume_continues_from_the_last_checkpoint_of_an_interrupt():
    graph = counter_graph()
    paused = run_graph(graph, {"n": 0}, interrupt_before=["inc"])
    finished = resume(graph, paused["checkpoints"], paused["checkpoints"][-1]["id"])
    assert finished["state"] == {"n": 3}
    assert finished["interrupted"] is None


def test_resume_forks_from_a_middle_checkpoint_without_replaying_the_start():
    graph = counter_graph()
    run = run_graph(graph, {"n": 0})
    forked = resume(graph, run["checkpoints"], 2)
    assert forked["state"] == {"n": 3}
    assert len(forked["checkpoints"]) < len(run["checkpoints"])


def test_resume_can_edit_the_state_before_continuing():
    graph = counter_graph()
    run = run_graph(graph, {"n": 0})
    forked = resume(graph, run["checkpoints"], 1, {"n": -2})
    assert forked["state"] == {"n": 3}
    assert len(forked["checkpoints"]) == 6


def test_resume_puts_the_update_through_the_reducers():
    graph = compile_graph(
        {"a": lambda s: {"m": ["from node"]}}, {"a": END}, "a", {"m": add_messages}
    )
    run = run_graph(graph, {"m": ["start"]})
    forked = resume(graph, run["checkpoints"], 0, {"m": ["injected"]})
    assert forked["state"]["m"] == ["start", "injected", "from node"]


def test_resume_leaves_the_original_run_untouched():
    graph = counter_graph()
    run = run_graph(graph, {"n": 0})
    before = [c["state"]["n"] for c in run["checkpoints"]]
    resume(graph, run["checkpoints"], 1, {"n": 100})
    assert [c["state"]["n"] for c in run["checkpoints"]] == before


def test_resume_rejects_an_unknown_checkpoint():
    graph = counter_graph()
    run = run_graph(graph, {"n": 0})
    with pytest.raises(GraphError):
        resume(graph, run["checkpoints"], 99)


# -------------------------------------------------------- build_react_graph
def test_react_runs_the_tool_and_then_answers():
    run = run_graph(weather_agent(), {"messages": [{"role": "user", "content": "weather?"}]})
    roles = [m["role"] for m in run["state"]["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert run["state"]["messages"][-1]["content"] == "In Paris it is 21C."


def test_react_feeds_the_tool_result_back_into_the_history():
    run = run_graph(weather_agent(), {"messages": [{"role": "user", "content": "weather?"}]})
    tool_message = run["state"]["messages"][2]
    assert tool_message == {"role": "tool", "name": "temp", "content": "21C in Paris"}


def test_react_ends_as_soon_as_the_model_stops_calling_tools():
    script = [{"role": "assistant", "content": "done"}]
    graph = build_react_graph(scripted_model(script), {})
    run = run_graph(graph, {"messages": []})
    assert run["state"]["messages"] == [{"role": "assistant", "content": "done"}]
    assert len(run["checkpoints"]) == 2


def test_react_interrupt_pauses_with_the_tool_call_proposed_but_not_executed():
    """Вызов инструмента уже виден, побочного эффекта ещё нет."""
    graph = weather_agent()
    paused = run_graph(
        graph, {"messages": [{"role": "user", "content": "weather?"}]}, interrupt_before=["tools"]
    )
    assert paused["interrupted"] == "tools"
    assert [m["role"] for m in paused["state"]["messages"]] == ["user", "assistant"]


def test_react_resumes_after_the_human_approves():
    graph = weather_agent()
    paused = run_graph(
        graph, {"messages": [{"role": "user", "content": "weather?"}]}, interrupt_before=["tools"]
    )
    done = resume(graph, paused["checkpoints"], paused["checkpoints"][-1]["id"])
    assert [m["role"] for m in done["state"]["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_react_without_the_add_messages_reducer_would_lose_the_history():
    """Контраст: тот же граф с перезаписью вместо накопления теряет всё, кроме последнего."""
    graph = weather_agent()
    broken = dict(graph)
    broken["reducers"] = {}
    run = run_graph(broken, {"messages": [{"role": "user", "content": "weather?"}]})
    assert len(run["state"]["messages"]) == 1
