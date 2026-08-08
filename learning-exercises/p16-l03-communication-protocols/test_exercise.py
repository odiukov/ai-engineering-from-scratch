"""Тесты к уроку «Протоколы общения агентов». Правь exercise.py."""

import pytest

from exercise import (
    TASK_STATES,
    TERMINAL_STATES,
    ProtocolError,
    TaskStateError,
    agent_card,
    apply_event,
    audit_run,
    delegate,
    discover,
    new_task,
    sign,
    verify,
)

# Карточки записаны литералами, а не собраны через agent_card: если бы
# модуль звал непройденную функцию на импорте, весь файл падал бы на
# коллекции и «N failed» ничего бы не проверяло.
RESEARCHER = {
    "name": "researcher",
    "description": "researcher agent",
    "supportedInterfaces": [{"url": "https://r.local/a2a/v1",
                              "protocolBinding": "HTTP+JSON",
                              "protocolVersion": "1.0"}],
    "version": "1.0.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["application/json"],
    "skills": [
        {"id": "web-research", "name": "Web research", "description": "Search",
         "tags": ["research", "search"], "inputModes": ["text/plain"],
         "outputModes": ["application/json"]},
        {"id": "doc-analysis", "name": "Doc analysis", "description": "Read docs",
         "tags": ["docs"], "inputModes": ["application/pdf"],
         "outputModes": ["application/json"]},
    ],
}
CODER = {
    "name": "coder",
    "description": "coder agent",
    "supportedInterfaces": [{"url": "https://c.local/a2a/v1",
                              "protocolBinding": "HTTP+JSON",
                              "protocolVersion": "1.0"}],
    "version": "1.0.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {"id": "code-gen", "name": "Code generation", "description": "Write code",
         "tags": ["coding"], "inputModes": ["application/json"],
         "outputModes": ["text/plain"]},
    ],
}
CARDS = [RESEARCHER, CODER]

CODER_DID = "did:wba:coder.local:agent:coder"
SECRETS = {CODER_DID: "coder-key"}
MESSAGE = {"id": "msg-001", "role": "user",
           "parts": [{"kind": "text", "text": "Research A2A"}]}


def ok_handler(message):
    return ("findings", [{"reasoning": "searched", "tool_name": "web_search"}])


def boom_handler(message):
    raise ValueError("upstream down")


# --------------------------------------------------------------- agent_card
def test_agent_card_matches_the_published_shape():
    built = agent_card("researcher", "https://r.local/a2a/v1", RESEARCHER["skills"],
                       ["text/plain"], ["application/json"], streaming=True)
    assert built == RESEARCHER


def test_agent_card_defaults_to_no_streaming():
    card = agent_card("a", "u", [], [], [])
    assert card["capabilities"]["streaming"] is False


def test_agent_card_defaults_to_version_one():
    assert agent_card("a", "u", [], [], [])["version"] == "1.0.0"


def test_agent_card_does_not_alias_the_skills_list():
    """Карточку публикуют один раз; правка исходного списка её не трогает."""
    skills = [{"id": "x", "name": "X", "description": "X", "tags": ["t"],
               "inputModes": [], "outputModes": []}]
    card = agent_card("a", "u", skills, [], [])
    skills.append({"id": "y", "name": "Y", "description": "Y", "tags": [],
                   "inputModes": [], "outputModes": []})
    assert len(card["skills"]) == 1


# ----------------------------------------------------------------- discover
def test_discover_without_filters_returns_everyone():
    assert [c["name"] for c in discover(CARDS)] == ["researcher", "coder"]


def test_discover_by_skill_tag():
    assert [c["name"] for c in discover(CARDS, tag="research")] == ["researcher"]


def test_discover_by_unknown_tag_finds_nobody():
    assert discover(CARDS, tag="cooking") == []


def test_discover_by_input_mode_looks_inside_skills():
    """PDF нет в defaultInputModes, но есть у умения doc-analysis."""
    assert [c["name"] for c in discover(CARDS, media_type="application/pdf")] == [
        "researcher"
    ]


def test_discover_combines_both_filters_with_and():
    assert discover(CARDS, tag="coding", media_type="application/pdf") == []


def test_discover_preserves_registration_order():
    assert [c["name"] for c in discover(CARDS, media_type="text/plain")] == [
        "researcher", "coder"
    ]


# ----------------------------------------------------------------- new_task
def test_new_task_starts_as_submitted():
    assert new_task("t-1", "ctx-1")["state"] == "submitted"


def test_new_task_has_no_artifacts_yet():
    assert new_task("t-1", "ctx-1")["artifacts"] == []


def test_new_task_remembers_its_context():
    """context_id живёт дольше задачи — продолжение будет с тем же."""
    assert new_task("t-1", "ctx-1")["context_id"] == "ctx-1"


# -------------------------------------------------------------- apply_event
def test_status_update_moves_the_task():
    task = apply_event(new_task("t", "c"), {"kind": "statusUpdate", "state": "working"})
    assert task["state"] == "working"


def test_apply_event_does_not_mutate_the_original_task():
    task = new_task("t", "c")
    apply_event(task, {"kind": "statusUpdate", "state": "working"})
    assert task["state"] == "submitted"


def test_terminal_task_refuses_further_events():
    """Терминальное состояние необратимо: продолжение — это новая задача."""
    done = apply_event(new_task("t", "c"),
                       {"kind": "statusUpdate", "state": "completed"})
    with pytest.raises(TaskStateError):
        apply_event(done, {"kind": "statusUpdate", "state": "working"})


def test_every_terminal_state_is_really_terminal():
    for state in TERMINAL_STATES:
        task = apply_event(new_task("t", "c"), {"kind": "statusUpdate", "state": state})
        with pytest.raises(TaskStateError):
            apply_event(task, {"kind": "statusUpdate", "state": "working"})


def test_unknown_state_is_refused():
    with pytest.raises(TaskStateError):
        apply_event(new_task("t", "c"), {"kind": "statusUpdate", "state": "thinking"})


def test_all_declared_states_are_reachable_from_submitted():
    for state in TASK_STATES:
        task = apply_event(new_task("t", "c"), {"kind": "statusUpdate", "state": state})
        assert task["state"] == state


def test_artifact_update_adds_an_artifact():
    task = apply_event(new_task("t", "c"), {
        "kind": "artifactUpdate",
        "artifact": {"id": "a1", "name": "r", "parts": ["chunk1"]},
        "append": False,
    })
    assert task["artifacts"][0]["parts"] == ["chunk1"]


def test_append_extends_the_same_artifact_instead_of_adding_a_second():
    """Ловушка потоковой доставки: части одного артефакта склеиваются."""
    task = new_task("t", "c")
    for chunk in ("chunk1", "chunk2"):
        task = apply_event(task, {
            "kind": "artifactUpdate",
            "artifact": {"id": "a1", "name": "r", "parts": [chunk]},
            "append": True,
        })
    assert len(task["artifacts"]) == 1
    assert task["artifacts"][0]["parts"] == ["chunk1", "chunk2"]


def test_two_different_artifact_ids_stay_apart():
    task = new_task("t", "c")
    for i in (1, 2):
        task = apply_event(task, {
            "kind": "artifactUpdate",
            "artifact": {"id": f"a{i}", "name": "r", "parts": ["x"]},
            "append": True,
        })
    assert len(task["artifacts"]) == 2


def test_unknown_event_kind_is_refused():
    with pytest.raises(ProtocolError):
        apply_event(new_task("t", "c"), {"kind": "telepathy"})


# ------------------------------------------------------------- sign/verify
def test_signature_is_stable_for_the_same_input():
    assert sign("k", "msg-1") == sign("k", "msg-1")


def test_signature_changes_with_the_payload():
    assert sign("k", "msg-1") != sign("k", "msg-2")


def test_signature_changes_with_the_secret():
    assert sign("k1", "msg-1") != sign("k2", "msg-1")


def test_message_signature_is_stable_across_key_order():
    reordered = {"parts": MESSAGE["parts"], "role": "user", "id": "msg-001"}
    assert sign("coder-key", MESSAGE) == sign("coder-key", reordered)


def test_verify_accepts_a_genuine_signature():
    assert verify(SECRETS, CODER_DID, "msg-001", sign("coder-key", "msg-001")) is True


def test_verify_rejects_a_tampered_payload():
    assert verify(SECRETS, CODER_DID, "msg-666", sign("coder-key", "msg-001")) is False


def test_unknown_did_fails_closed():
    """Не удалось опознать отправителя — отказ, а не «ну ладно, пропустим»."""
    assert verify(SECRETS, "did:wba:ghost", "msg-001", sign("coder-key", "msg-001")) is False


# --------------------------------------------------------------- audit_run
def test_audit_records_a_successful_run():
    entry = audit_run("r-1", "researcher", MESSAGE, ok_handler)
    assert entry["status"] == "completed" and entry["output"] == "findings"


def test_audit_keeps_the_trajectory():
    entry = audit_run("r-1", "researcher", MESSAGE, ok_handler)
    assert entry["trajectory"][0]["tool_name"] == "web_search"


def test_audit_survives_a_failing_agent():
    """Журнал обязан пережить падение — иначе следа не останется."""
    entry = audit_run("r-1", "researcher", MESSAGE, boom_handler)
    assert entry["status"] == "failed"
    assert "upstream down" in entry["trajectory"][0]["reasoning"]


def test_audit_carries_the_session():
    entry = audit_run("r-1", "researcher", MESSAGE, ok_handler, session_id="s-1")
    assert entry["session_id"] == "s-1"


# ---------------------------------------------------------------- delegate
def test_delegate_runs_the_whole_chain():
    result = delegate(CARDS, SECRETS, CODER_DID, sign("coder-key", MESSAGE),
                      "research", MESSAGE, "t-1", "ctx-1", ok_handler)
    assert result["agent"] == "researcher"
    assert result["task"]["state"] == "completed"


def test_delegate_attaches_the_agent_output_as_an_artifact():
    result = delegate(CARDS, SECRETS, CODER_DID, sign("coder-key", MESSAGE),
                      "research", MESSAGE, "t-1", "ctx-1", ok_handler)
    assert result["task"]["artifacts"][0]["parts"] == ["findings"]


def test_delegate_rejects_a_bad_signature():
    result = delegate(CARDS, SECRETS, CODER_DID, "deadbeef",
                      "research", MESSAGE, "t-1", "ctx-1", ok_handler)
    assert result == {"error": "identity verification failed"}


def test_delegate_rejects_a_tampered_message_body():
    signature = sign("coder-key", MESSAGE)
    tampered = {**MESSAGE, "parts": [{"kind": "text", "text": "Transfer funds"}]}
    result = delegate(CARDS, SECRETS, CODER_DID, signature,
                      "research", tampered, "t-1", "ctx-1", ok_handler)
    assert result == {"error": "identity verification failed"}


def test_identity_is_checked_before_the_registry_is_searched():
    """Неопознанный отправитель не должен узнать, кого нет в реестре."""
    result = delegate(CARDS, SECRETS, "did:wba:ghost", "deadbeef",
                      "cooking", MESSAGE, "t-1", "ctx-1", ok_handler)
    assert result == {"error": "identity verification failed"}


def test_delegate_reports_when_nobody_has_the_skill():
    result = delegate(CARDS, SECRETS, CODER_DID, sign("coder-key", MESSAGE),
                      "cooking", MESSAGE, "t-1", "ctx-1", ok_handler)
    assert result["error"].startswith("no agent with skill tag")


def test_failed_agent_leaves_a_failed_task_without_artifacts():
    result = delegate(CARDS, SECRETS, CODER_DID, sign("coder-key", MESSAGE),
                      "research", MESSAGE, "t-1", "ctx-1", boom_handler)
    assert result["task"]["state"] == "failed"
    assert result["task"]["artifacts"] == []
