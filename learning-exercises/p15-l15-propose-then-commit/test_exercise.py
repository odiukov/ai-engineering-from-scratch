"""Тесты к уроку «Human-in-the-loop: propose-then-commit». Правь exercise.py."""

import pytest

from exercise import (
    CHECKLIST,
    TTL_SECONDS,
    approve,
    commit,
    idempotency_key,
    is_expired,
    missing_metadata,
    propose,
    verify,
)

FULL_CHECKLIST = dict.fromkeys(CHECKLIST, True)


def proposal(action="email.send", payload=None, thread_id="t-001"):
    """Полное предложение со всеми обязательными метаданными."""
    return {
        "thread_id": thread_id,
        "action": action,
        "payload": {"to": "team@example.com", "subject": "release"}
        if payload is None
        else payload,
        "intent": "Announce the v1.2 release to the team list",
        "lineage": "release notes page /releases/1.2",
        "blast_radius": "37 recipients; wrong send = external embarrassment",
        "rollback": "no in-band rollback; send a correction email",
    }


def recorder():
    """(execute, sent): целевой сервис дедуплицирует побочный эффект по key."""
    sent = []
    results = {}

    def execute(key, action, payload):
        if key not in results:
            effect = (action, tuple(sorted(payload.items())))
            sent.append(effect)
            results[key] = effect
        return results[key]

    return execute, sent


# -------------------------------------------------------- idempotency_key
def test_idempotency_key_is_sixteen_hex_chars():
    key = idempotency_key("t-1", "email.send", {"to": "a"})
    assert len(key) == 16
    assert all(char in "0123456789abcdef" for char in key)


def test_idempotency_key_ignores_payload_key_order():
    """Один и тот же перевод денег не должен получить два разных ключа."""
    assert idempotency_key("t-1", "x", {"a": 1, "b": 2}) == idempotency_key(
        "t-1", "x", {"b": 2, "a": 1}
    )


def test_idempotency_key_separates_threads():
    assert idempotency_key("t-1", "x", {"a": 1}) != idempotency_key("t-2", "x", {"a": 1})


def test_idempotency_key_separates_payloads():
    assert idempotency_key("t-1", "x", {"amount": 100}) != idempotency_key(
        "t-1", "x", {"amount": 200}
    )


def test_idempotency_key_does_not_depend_on_time():
    """Ключ, в котором есть время, превращает повтор в двойное исполнение."""
    keys = {idempotency_key("t-1", "x", {"a": 1}) for _ in range(3)}
    assert len(keys) == 1


# -------------------------------------------------------- missing_metadata
def test_missing_metadata_is_empty_for_full_proposal():
    assert missing_metadata(proposal()) == ()


def test_missing_metadata_lists_absent_fields_in_declared_order():
    assert missing_metadata({"intent": "i"}) == ("lineage", "blast_radius", "rollback")


def test_blank_rollback_counts_as_missing():
    """rollback='  ' — это галочка ради галочки, а не план откатa."""
    weak = proposal()
    weak["rollback"] = "   "
    assert missing_metadata(weak) == ("rollback",)


# --------------------------------------------------------------- is_expired
def test_is_expired_is_false_before_deadline():
    assert is_expired({"expires_at": 100.0}, now=99.0) is False


def test_is_expired_is_true_at_the_deadline():
    """Граница закрыта в сторону отказа."""
    assert is_expired({"expires_at": 100.0}, now=100.0) is True


def test_is_expired_is_true_after_deadline():
    assert is_expired({"expires_at": 100.0}, now=100_000.0) is True


# ------------------------------------------------------------------ propose
def test_propose_stores_pending_record_with_deadline():
    store = {}
    key = propose(store, proposal(), now=0.0)
    assert store[key]["status"] == "pending"
    assert store[key]["expires_at"] == TTL_SECONDS


def test_propose_is_idempotent():
    store = {}
    first = propose(store, proposal(), now=0.0)
    second = propose(store, proposal(), now=500.0)
    assert first == second
    assert len(store) == 1


def test_propose_retry_does_not_reset_an_approval():
    """Повтор после сетевого сбоя не должен отменять уже полученное одобрение."""
    store = {}
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=10.0)
    propose(store, proposal(), now=20.0)
    assert store[key]["status"] == "approved"


def test_propose_rejects_proposal_without_metadata():
    incomplete = proposal()
    del incomplete["rollback"]
    with pytest.raises(ValueError):
        propose({}, incomplete, now=0.0)


def test_propose_separates_different_actions():
    store = {}
    propose(store, proposal(action="email.send"), now=0.0)
    propose(store, proposal(action="db.drop_table"), now=0.0)
    assert len(store) == 2


# ------------------------------------------------------------------ approve
def test_full_checklist_approves():
    store = {}
    key = propose(store, proposal(), now=0.0)
    assert approve(store, key, FULL_CHECKLIST, now=1.0) is True
    assert store[key]["status"] == "approved"


def test_rubber_stamp_with_no_answers_is_rejected():
    """Нажатая кнопка Approve без ответов на вопросы — не одобрение."""
    store = {}
    key = propose(store, proposal(), now=0.0)
    assert approve(store, key, {}, now=1.0) is False
    assert store[key]["status"] == "pending"


def test_partial_checklist_is_rejected():
    store = {}
    key = propose(store, proposal(), now=0.0)
    answers = dict(FULL_CHECKLIST)
    answers[CHECKLIST[-1]] = False
    assert approve(store, key, answers, now=1.0) is False


def test_truthy_but_not_true_answers_are_rejected():
    store = {}
    key = propose(store, proposal(), now=0.0)
    answers = {question: "yes" for question in CHECKLIST}
    assert approve(store, key, answers, now=1.0) is False


def test_expired_proposal_cannot_be_approved():
    store = {}
    key = propose(store, proposal(), now=0.0)
    assert approve(store, key, FULL_CHECKLIST, now=TTL_SECONDS + 1) is False
    assert store[key]["status"] == "expired"


def test_second_approval_is_rejected():
    store = {}
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    assert approve(store, key, FULL_CHECKLIST, now=2.0) is False


# ------------------------------------------------------------------- commit
def test_commit_refuses_without_approval_and_runs_nothing():
    """Главное свойство: без подтверждения побочного эффекта не происходит."""
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    assert commit(store, key, execute, now=1.0) == "refused"
    assert sent == []


def test_commit_executes_after_approval():
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    assert commit(store, key, execute, now=2.0) == "committed"
    assert len(sent) == 1
    assert store[key]["status"] == "committed"


def test_commit_passes_the_proposal_key_to_the_executor():
    store = {}
    received = []

    def execute(key, action, payload):
        received.append((key, action, payload))

    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    commit(store, key, execute, now=2.0)
    assert received == [(key, store[key]["action"], store[key]["payload"])]


def test_retry_after_commit_does_not_call_the_executor_again():
    """Записанный commit отсекает обычные повторы локально."""
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    commit(store, key, execute, now=2.0)
    assert commit(store, key, execute, now=3.0) == "already-committed"
    assert commit(store, key, execute, now=4.0) == "already-committed"
    assert len(sent) == 1


def test_crash_after_effect_before_status_is_safe_with_idempotent_executor():
    """Crash gap: повтор вызывает executor, но тот узнаёт key и не шлёт дубль."""
    store = {}
    sent = []
    results = {}
    crash_once = [True]

    def execute(key, action, payload):
        if key not in results:
            effect = (action, tuple(sorted(payload.items())))
            sent.append(effect)
            results[key] = effect
        if crash_once[0]:
            crash_once[0] = False
            raise RuntimeError("worker crashed after the durable side effect")
        return results[key]

    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    with pytest.raises(RuntimeError):
        commit(store, key, execute, now=2.0)
    assert store[key]["status"] == "approved"
    assert commit(store, key, execute, now=3.0) == "committed"
    assert len(sent) == 1


def test_expired_approval_is_not_committed():
    """Одобрение позавчерашнего состояния мира не применяется."""
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    assert commit(store, key, execute, now=TTL_SECONDS + 1) == "expired"
    assert sent == []


def test_rejected_proposal_never_commits():
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, {}, now=1.0)
    assert commit(store, key, execute, now=2.0) == "refused"
    assert sent == []


# ------------------------------------------------------------------- verify
def test_verify_marks_record_verified_when_side_effect_is_visible():
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    commit(store, key, execute, now=2.0)
    assert verify(store, key, lambda action, payload: len(sent) == 1) is True
    assert store[key]["status"] == "verified"


def test_verify_marks_known_bad_state_when_side_effect_is_absent():
    """'Инструмент вернул 200' — не проверка. Проверка — это чтение цели."""
    store = {}
    execute, _ = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    commit(store, key, execute, now=2.0)
    assert verify(store, key, lambda action, payload: False) is False
    assert store[key]["status"] == "verify-failed"


def test_verify_before_commit_changes_nothing():
    store = {}
    key = propose(store, proposal(), now=0.0)
    assert verify(store, key, lambda action, payload: True) is False
    assert store[key]["status"] == "pending"


def test_verified_record_is_not_committed_again():
    store = {}
    execute, sent = recorder()
    key = propose(store, proposal(), now=0.0)
    approve(store, key, FULL_CHECKLIST, now=1.0)
    commit(store, key, execute, now=2.0)
    verify(store, key, lambda action, payload: True)
    assert commit(store, key, execute, now=3.0) == "already-committed"
    assert len(sent) == 1
