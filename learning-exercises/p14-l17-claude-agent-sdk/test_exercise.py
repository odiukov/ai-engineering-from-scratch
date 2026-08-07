"""Тесты к уроку «Харнесс как библиотека: подагенты и хранилище сессий». Правь exercise.py."""

import pytest

from exercise import (
    call_tool,
    run_agent,
    run_hooks,
    select_tools,
    session_delete,
    session_subkeys,
    spawn_subagents,
    stub_model,
)


# ------------------------------------------------------------ инструменты
def make_tools():
    """Реестр из двух инструментов и журнал их реальных срабатываний.

    written — побочный эффект: если в нём что-то есть, обработчик правда
    выполнился, сколько бы ни говорили хуки.
    """
    written = []

    def read_file(args):
        return f"content-of:{args['path']}"

    def write_file(args):
        written.append(args["path"])
        return f"written:{args['path']}"

    return {"read_file": read_file, "write_file": write_file}, written


def deny_hook(reason):
    return lambda payload: {"decision": "deny", "reason": reason}


# ------------------------------------------------------------- stub_model
def test_stub_model_is_deterministic():
    assert stub_model("plan the trip") == stub_model("plan the trip")


def test_stub_model_reacts_to_the_prompt():
    assert stub_model("a") != stub_model("b")


def test_stub_model_encodes_prompt_length():
    assert stub_model("hi") == "answer[2:209]"


def test_stub_model_sum_ignores_letter_order():
    """Подпись «по сумме кодов» — не хеш: перестановка её не меняет."""
    assert stub_model("ih") == stub_model("hi")
    assert stub_model("x" * 100).startswith("answer[100:")


# ------------------------------------------------------------ select_tools
def test_select_tools_gives_only_what_was_allowed():
    registry, _ = make_tools()
    assert sorted(select_tools(registry, ["read_file"])) == ["read_file"]


def test_select_tools_with_none_gives_everything():
    registry, _ = make_tools()
    assert sorted(select_tools(registry, None)) == ["read_file", "write_file"]


def test_select_tools_rejects_unknown_name():
    registry, _ = make_tools()
    with pytest.raises(KeyError):
        select_tools(registry, ["shell"])


def test_select_tools_returns_an_independent_copy():
    """Иначе подагент допишет себе инструмент, которого ему не давали."""
    registry, _ = make_tools()
    subset = select_tools(registry, None)
    subset["shell"] = lambda args: "pwned"
    assert "shell" not in registry


# --------------------------------------------------------------- run_hooks
def test_run_hooks_without_hooks_allows():
    assert run_hooks({}, "PreToolUse", {"tool": "read_file"}) is None


def test_run_hooks_returns_reason_of_the_denying_hook():
    verdict = run_hooks({"PreToolUse": [deny_hook("no writes")]}, "PreToolUse", {})
    assert verdict == {"reason": "no writes"}


def test_run_hooks_stops_after_the_first_deny():
    """Решение принято — остальные хуки уже не зовут."""
    seen = []
    hooks = {
        "PreToolUse": [
            lambda p: seen.append("first") or {"decision": "deny", "reason": "stop"},
            lambda p: seen.append("second"),
        ]
    }
    run_hooks(hooks, "PreToolUse", {})
    assert seen == ["first"]


def test_run_hooks_runs_every_allowing_hook_in_order():
    seen = []
    hooks = {"SessionStart": [lambda p: seen.append(1), lambda p: seen.append(2)]}
    assert run_hooks(hooks, "SessionStart", {}) is None
    assert seen == [1, 2]


def test_run_hooks_rejects_a_misspelled_event():
    """Опечатка в имени события иначе даёт хук, который молча не работает."""
    with pytest.raises(ValueError):
        run_hooks({}, "PreToolCall", {})


# --------------------------------------------------------------- call_tool
def test_call_tool_returns_the_handler_result():
    registry, _ = make_tools()
    out = call_tool(registry, {}, "read_file", {"path": "a.txt"}, [])
    assert out == {"ok": True, "tool": "read_file", "result": "content-of:a.txt"}


def test_pre_tool_use_deny_prevents_the_side_effect():
    """Ворота ДО вызова: обработчик не должен выполниться ни разу."""
    registry, written = make_tools()
    hooks = {"PreToolUse": [deny_hook("read only mode")]}
    out = call_tool(registry, hooks, "write_file", {"path": "a.txt"}, [])
    assert written == []
    assert out["ok"] is False and out["error"] == "read only mode"


def test_post_tool_use_deny_cannot_undo_the_side_effect():
    """Контраст к предыдущему: PostToolUse только помечает результат."""
    registry, written = make_tools()
    hooks = {"PostToolUse": [deny_hook("secret leaked")]}
    out = call_tool(registry, hooks, "write_file", {"path": "a.txt"}, [])
    assert written == ["a.txt"]
    assert out["ok"] is True and out["blocked"] == "secret leaked"


def test_call_tool_refuses_a_tool_outside_the_registry():
    registry, _ = make_tools()
    del registry["write_file"]
    out = call_tool(registry, {}, "write_file", {"path": "a.txt"}, [])
    assert out == {"ok": False, "tool": "write_file", "error": "tool_not_allowed"}


def test_call_tool_journals_denied_and_called_differently():
    registry, _ = make_tools()
    journal = []
    call_tool(registry, {}, "read_file", {"path": "a"}, journal)
    call_tool(registry, {"PreToolUse": [deny_hook("nope")]}, "read_file", {"path": "b"}, journal)
    assert [entry["event"] for entry in journal] == ["called", "denied"]


# ---------------------------------------------------------- session store
def test_session_subkeys_lists_only_children():
    store = {"s1": [], "s1/alpha": [], "s1/beta": [], "s2": []}
    assert session_subkeys(store, "s1") == ["s1/alpha", "s1/beta"]


def test_session_subkeys_does_not_include_the_session_itself():
    store = {"s1": []}
    assert session_subkeys(store, "s1") == []


def test_session_subkeys_matches_on_the_separator_not_the_prefix():
    """"s10" — соседняя сессия, а не подагент "s1"."""
    store = {"s1": [], "s10": [], "s1/a": []}
    assert session_subkeys(store, "s1") == ["s1/a"]


def test_session_delete_cascades_to_subagent_sessions():
    store = {"s1": [], "s1/alpha": [], "s2": []}
    assert session_delete(store, "s1") == ["s1", "s1/alpha"]
    assert list(store) == ["s2"]


def test_session_delete_of_a_missing_session_is_empty():
    store = {"s2": []}
    assert session_delete(store, "s1") == []
    assert list(store) == ["s2"]


# --------------------------------------------------------------- run_agent
def test_run_agent_records_user_tools_and_assistant_turns():
    registry, _ = make_tools()
    store = {}
    plan = [("read_file", {"path": "a"}), ("read_file", {"path": "b"})]
    run_agent(store, "s1", "summarise", plan, registry, {})
    assert [t["role"] for t in store["s1"]] == ["user", "tool", "tool", "assistant"]


def test_run_agent_answer_depends_on_tool_results():
    """Иначе инструменты можно было бы вообще не звать."""
    registry, _ = make_tools()
    a = run_agent({}, "s1", "go", [("read_file", {"path": "a"})], registry, {})
    b = run_agent({}, "s1", "go", [("read_file", {"path": "zzz"})], registry, {})
    assert a["answer"] != b["answer"]


def test_run_agent_is_reproducible():
    registry, _ = make_tools()
    plan = [("read_file", {"path": "a"})]
    first = run_agent({}, "s1", "go", plan, registry, {})["answer"]
    second = run_agent({}, "s1", "go", plan, registry, {})["answer"]
    assert first == second


def test_run_agent_fires_session_start_before_any_tool():
    order = []
    registry = {"read_file": lambda args: order.append("tool") or "ok"}
    hooks = {
        "SessionStart": [lambda p: order.append("start")],
        "SessionEnd": [lambda p: order.append("end")],
    }
    run_agent({}, "s1", "go", [("read_file", {})], registry, hooks)
    assert order == ["start", "tool", "end"]


def test_run_agent_appends_to_an_existing_session():
    registry, _ = make_tools()
    store = {}
    run_agent(store, "s1", "first", [], registry, {})
    run_agent(store, "s1", "second", [], registry, {})
    assert len(store["s1"]) == 4


# ---------------------------------------------------------- spawn_subagents
def test_subagent_gets_its_own_session():
    registry, _ = make_tools()
    store = {}
    tasks = [{"name": "alpha", "prompt": "a", "plan": [], "allowed": None}]
    spawn_subagents(store, "root", tasks, registry, {})
    assert session_subkeys(store, "root") == ["root/alpha"]


def test_subagent_does_not_inherit_tools_it_was_not_given():
    registry, written = make_tools()
    store = {}
    tasks = [
        {
            "name": "alpha",
            "prompt": "a",
            "plan": [("write_file", {"path": "a.txt"})],
            "allowed": ["read_file"],
        }
    ]
    results = spawn_subagents(store, "root", tasks, registry, {})
    assert written == []
    assert results[0]["calls"][0]["error"] == "tool_not_allowed"


def test_orchestrator_context_grows_by_one_turn_per_subagent():
    """Ради этого подагентов и заводят: контекст оркестратора ограничен."""
    registry, _ = make_tools()
    store = {}
    long_plan = [("read_file", {"path": str(i)}) for i in range(20)]
    tasks = [{"name": f"w{i}", "prompt": "a", "plan": long_plan} for i in range(3)]
    spawn_subagents(store, "root", tasks, registry, {})
    assert len(store["root"]) == 3
    assert len(store["root/w0"]) == 22


def test_subagent_transcript_stays_out_of_the_parent_session():
    registry, _ = make_tools()
    store = {}
    tasks = [{"name": "alpha", "prompt": "a", "plan": [("read_file", {"path": "x"})]}]
    spawn_subagents(store, "root", tasks, registry, {})
    assert [t["role"] for t in store["root"]] == ["subagent"]


def test_subagents_keep_task_order_in_results():
    registry, _ = make_tools()
    tasks = [{"name": n, "prompt": n, "plan": []} for n in ("c", "a", "b")]
    results = spawn_subagents({}, "root", tasks, registry, {})
    assert [r["name"] for r in results] == ["c", "a", "b"]


def test_deleting_the_parent_session_removes_subagent_sessions():
    registry, _ = make_tools()
    store = {}
    tasks = [{"name": n, "prompt": n, "plan": []} for n in ("a", "b")]
    spawn_subagents(store, "root", tasks, registry, {})
    assert session_delete(store, "root") == ["root", "root/a", "root/b"]
    assert store == {}
