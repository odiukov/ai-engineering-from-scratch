"""Тесты к уроку «Акторы: почтовые ящики и порядок доставки». Правь exercise.py."""

import pytest

from exercise import (
    dead_letter_report,
    deliver_one,
    new_runtime,
    publish,
    register,
    run_round_robin,
    run_selector,
    send,
)


# ------------------------------------------------------------ вспомогательное
def _remember(state, message, runtime):
    """Обычный актор: пишет прочитанное в СВОЁ состояние."""
    state.setdefault("seen", []).append(message["body"])


def _reviewer(state, message, runtime):
    """Отвечает отправителю; на топик crash_me падает."""
    if message["topic"] == "crash_me":
        raise RuntimeError("simulated handler failure")
    issues = ["uses eval"] if "eval(" in str(message["body"]) else []
    state.setdefault("seen", []).append(message["body"])
    send(runtime, message["recipient"], message["sender"], "review_result",
         {"ok": not issues, "issues": issues})


def _pong(state, message, runtime):
    """Пинг-понг: на каждое письмо отвечает письмом. Сам не остановится."""
    other = "b" if message["recipient"] == "a" else "a"
    send(runtime, message["recipient"], other, "ping", message["body"] + 1)


def _busiest(runtime):
    """Селектор: следующим ходит тот, у кого больше непрочитанного."""
    ranked = sorted(runtime["actors"].items(),
                    key=lambda kv: (-len(kv[1]["inbox"]), kv[0]))
    name, actor = ranked[0]
    return name if actor["inbox"] else None


# --------------------------------------------------------------- new_runtime
def test_new_runtime_has_no_actors_and_no_dead_letters():
    assert new_runtime() == {"actors": {}, "dead_letters": [], "counter": 0}


def test_two_runtimes_do_not_share_actors():
    a, b = new_runtime(), new_runtime()
    register(a, "solo", _remember)
    assert b["actors"] == {}


def test_delivering_to_an_actor_that_was_never_registered_raises_key_error():
    with pytest.raises(KeyError):
        deliver_one(new_runtime(), "ghost")


# ------------------------------------------------------------------ register
def test_registered_actor_starts_with_an_empty_inbox():
    rt = register(new_runtime(), "reviewer", _reviewer)
    assert rt["actors"]["reviewer"]["inbox"] == []
    assert rt["actors"]["reviewer"]["handler"] is _reviewer


def test_default_state_is_an_empty_dict():
    rt = register(new_runtime(), "reviewer", _reviewer)
    assert rt["actors"]["reviewer"]["state"] == {}


def test_two_actors_do_not_share_their_default_state():
    """Общий словарь по умолчанию превратил бы приватное состояние в общее."""
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    rt["actors"]["a"]["state"]["secret"] = 1
    assert rt["actors"]["b"]["state"] == {}


def test_registering_the_same_name_twice_raises_value_error():
    rt = register(new_runtime(), "a", _remember)
    with pytest.raises(ValueError):
        register(rt, "a", _remember)


# ---------------------------------------------------------------------- send
def test_send_puts_the_message_in_the_inbox_without_running_the_handler():
    """Отправка и обработка разделены: send возвращается сразу."""
    rt = register(new_runtime(), "a", _remember)
    send(rt, "__user__", "a", "note", "hello")
    assert len(rt["actors"]["a"]["inbox"]) == 1
    assert rt["actors"]["a"]["state"] == {}


def test_message_ids_start_at_one_and_grow():
    rt = register(new_runtime(), "a", _remember)
    assert send(rt, "__user__", "a", "note", "x") == 1
    assert send(rt, "__user__", "a", "note", "y") == 2


def test_inbox_keeps_the_order_messages_arrived_in():
    rt = register(new_runtime(), "a", _remember)
    for body in ("first", "second", "third"):
        send(rt, "__user__", "a", "note", body)
    assert [m["body"] for m in rt["actors"]["a"]["inbox"]] == [
        "first", "second", "third",
    ]


def test_unknown_recipient_becomes_a_dead_letter_instead_of_an_exception():
    rt = new_runtime()
    send(rt, "__user__", "typo", "note", "x")
    assert dead_letter_report(rt) == {"no actor 'typo'": 1}


# ------------------------------------------------------------------- publish
def test_publish_delivers_one_copy_per_subscriber():
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    publish(rt, "__user__", "review", ["a", "b"], "code")
    assert len(rt["actors"]["a"]["inbox"]) == 1
    assert len(rt["actors"]["b"]["inbox"]) == 1


def test_each_subscriber_gets_its_own_envelope_with_its_own_id():
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    assert publish(rt, "__user__", "review", ["a", "b"], "code") == [1, 2]
    rt["actors"]["a"]["inbox"][0]["topic"] = "tampered"
    assert rt["actors"]["b"]["inbox"][0]["topic"] == "review"


def test_one_unknown_subscriber_does_not_block_the_others():
    rt = register(new_runtime(), "a", _remember)
    publish(rt, "__user__", "review", ["a", "ghost"], "code")
    assert len(rt["actors"]["a"]["inbox"]) == 1
    assert dead_letter_report(rt) == {"no actor 'ghost'": 1}


# ---------------------------------------------------------------- deliver_one
def test_actor_processes_messages_in_the_order_they_arrived():
    rt = register(new_runtime(), "a", _remember)
    for body in ("first", "second"):
        send(rt, "__user__", "a", "note", body)
    deliver_one(rt, "a")
    assert rt["actors"]["a"]["state"]["seen"] == ["first"]


def test_handler_writes_to_its_own_state_only():
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    send(rt, "__user__", "a", "note", "hello")
    assert deliver_one(rt, "a") == "handled"
    assert rt["actors"]["b"]["state"] == {}


def test_a_crashing_handler_does_not_eat_the_rest_of_the_inbox():
    """Fault isolation: упал обработчик, а не рантайм и не остальная почта."""
    rt = register(new_runtime(), "reviewer", _reviewer)
    send(rt, "__user__", "reviewer", "crash_me", None)
    send(rt, "__user__", "reviewer", "review", "def f(): eval('1')")
    assert deliver_one(rt, "reviewer") == "dead_letter"
    assert deliver_one(rt, "reviewer") == "handled"
    assert rt["actors"]["reviewer"]["state"]["seen"] == ["def f(): eval('1')"]


def test_crash_reason_records_the_exception_type_and_text():
    rt = register(new_runtime(), "reviewer", _reviewer)
    send(rt, "__user__", "reviewer", "crash_me", None)
    deliver_one(rt, "reviewer")
    assert rt["dead_letters"][0][1] == "RuntimeError: simulated handler failure"


def test_empty_inbox_reports_idle():
    rt = register(new_runtime(), "a", _remember)
    assert deliver_one(rt, "a") == "idle"


# ------------------------------------------------------------ run_round_robin
def test_round_robin_follows_the_fixed_rotation():
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    send(rt, "__user__", "a", "note", "x")
    send(rt, "__user__", "b", "note", "y")
    assert run_round_robin(rt, ["b", "a"], rounds=1) == [("b", 2), ("a", 1)]


def test_actor_with_an_empty_inbox_is_skipped_not_blocked():
    rt = new_runtime()
    register(rt, "quiet", _remember)
    register(rt, "busy", _remember)
    send(rt, "__user__", "busy", "note", "x")
    assert run_round_robin(rt, ["quiet", "busy"], rounds=1) == [("busy", 1)]


def test_a_reply_sent_while_handling_arrives_on_the_next_round():
    """Доставка отделена от обработки: ответ ждёт своего круга."""
    rt = new_runtime()
    register(rt, "asker", _remember)
    register(rt, "reviewer", _reviewer)
    send(rt, "asker", "reviewer", "review", "def f(): pass")
    assert run_round_robin(rt, ["asker", "reviewer"], rounds=1) == [("reviewer", 1)]
    assert run_round_robin(rt, ["asker", "reviewer"], rounds=1) == [("asker", 2)]


def test_ping_pong_stops_at_the_delivery_cap():
    """Два актора, перекидывающие мяч, обязаны упереться в потолок, а не висеть."""
    rt = new_runtime()
    register(rt, "a", _pong)
    register(rt, "b", _pong)
    send(rt, "__user__", "a", "ping", 0)
    log = run_round_robin(rt, ["a", "b"], rounds=1000, max_deliveries=7)
    assert len(log) == 7


# --------------------------------------------------------------- run_selector
def test_selector_decides_the_order_not_the_registration_order():
    rt = new_runtime()
    register(rt, "b", _remember)
    register(rt, "a", _remember)
    send(rt, "__user__", "a", "note", "x1")
    send(rt, "__user__", "a", "note", "x2")
    send(rt, "__user__", "b", "note", "y1")
    assert [name for name, _mid in run_selector(rt, _busiest)] == ["a", "a", "b"]


def test_selector_sees_the_effect_of_the_previous_delivery():
    """Селектор смотрит на рантайм после каждой доставки — в этом его смысл."""
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    send(rt, "__user__", "a", "note", "x")
    send(rt, "__user__", "b", "note", "y")
    log = run_selector(rt, _busiest)
    assert sorted(name for name, _mid in log) == ["a", "b"]
    assert len(log) == 2


def test_selector_returning_none_ends_the_run():
    rt = register(new_runtime(), "a", _remember)
    send(rt, "__user__", "a", "note", "x")
    assert run_selector(rt, lambda _rt: None) == []


def test_selector_pointing_at_an_empty_inbox_ends_the_run():
    rt = new_runtime()
    register(rt, "a", _remember)
    register(rt, "b", _remember)
    send(rt, "__user__", "a", "note", "x")
    assert run_selector(rt, lambda _rt: "b") == []


# --------------------------------------------------------- dead_letter_report
def test_report_of_a_clean_runtime_is_empty():
    assert dead_letter_report(new_runtime()) == {}


def test_report_groups_failures_by_reason():
    rt = register(new_runtime(), "reviewer", _reviewer)
    send(rt, "__user__", "reviewer", "crash_me", None)
    send(rt, "__user__", "ghost", "note", "x")
    deliver_one(rt, "reviewer")
    assert dead_letter_report(rt) == {
        "RuntimeError: simulated handler failure": 1,
        "no actor 'ghost'": 1,
    }


def test_repeated_failures_collapse_into_one_counted_entry():
    """Сто писем с одной причиной — одна поломка, а не сто."""
    rt = register(new_runtime(), "reviewer", _reviewer)
    for _ in range(3):
        send(rt, "__user__", "reviewer", "crash_me", None)
        deliver_one(rt, "reviewer")
    assert dead_letter_report(rt) == {"RuntimeError: simulated handler failure": 3}
