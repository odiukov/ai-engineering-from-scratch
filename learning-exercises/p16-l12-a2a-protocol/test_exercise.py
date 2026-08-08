"""Тесты к уроку «Протокол A2A v1». Правь exercise.py."""

import json

import pytest

from exercise import (
    CARD_REQUIRED,
    MEDIA_TYPES,
    MESSAGE_SEND_PATH,
    TERMINAL_STATES,
    TRANSITIONS,
    WELL_KNOWN_PATH,
    A2AProtocolError,
    advance_task,
    decode_card,
    encode_card,
    make_agent_card,
    make_artifact,
    make_message,
    make_task,
    run_task,
    supports_skill,
)


SKILLS = [
    {"id": "review-python", "name": "Review Python",
     "description": "Reviews Python code", "tags": ["review", "python"]},
    {"id": "summarize", "name": "Summarize",
     "description": "Summarizes text", "tags": ["summary"]},
]


def card():
    return make_agent_card(
        "code-review-agent", "Reviews Python code", "0.1.0", SKILLS,
        "http://localhost:8765/", input_modes=("application/json",),
        output_modes=("text/plain", "application/json"), streaming=True,
    )


def message(skill="review-python", payload=None):
    return make_message("msg-1", skill, payload or {"code": "return 1\n"})


def reviewer(payload):
    code = payload.get("code", "")
    issues = [] if "return" in code else ["no return statement"]
    return make_artifact(
        "artifact-review", "application/json",
        {"issues": issues, "lines": code.count("\n") + 1},
    )


# ------------------------------------------------------------ Agent Card v1
def test_current_well_known_and_send_paths():
    card()  # заготовка обязана оставаться красной и на тесте констант
    assert WELL_KNOWN_PATH == "/.well-known/agent-card.json"
    assert MESSAGE_SEND_PATH == "/message:send"


def test_card_uses_v1_supported_interfaces():
    interface = card()["supportedInterfaces"][0]
    assert interface == {
        "url": "http://localhost:8765",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    }


def test_card_has_no_removed_v03_root_fields():
    built = card()
    assert "url" not in built and "protocolVersion" not in built
    assert "endpoints" not in built and "modalities" not in built


def test_card_carries_every_required_field():
    assert all(key in card() for key in CARD_REQUIRED)


def test_card_uses_camel_case_capability_and_mode_fields():
    built = card()
    assert built["capabilities"]["pushNotifications"] is False
    assert built["defaultInputModes"] == ["application/json"]
    assert built["defaultOutputModes"] == ["text/plain", "application/json"]


def test_card_does_not_alias_skills():
    skills = [dict(SKILLS[0])]
    built = make_agent_card("a", "A", "1", skills, "http://x.test")
    skills[0]["id"] = "tampered"
    assert built["skills"][0]["id"] == "review-python"


def test_a_card_without_skills_is_rejected():
    with pytest.raises(A2AProtocolError):
        make_agent_card("a", "A", "1", [], "http://x.test")


def test_a_card_with_an_unknown_media_type_is_rejected():
    assert "model/hologram" not in MEDIA_TYPES
    with pytest.raises(A2AProtocolError):
        make_agent_card("a", "A", "1", [SKILLS[0]], "http://x.test",
                        input_modes=("model/hologram",))


# --------------------------------------------------------- encode/decode card
def test_card_survives_the_json_round_trip():
    assert decode_card(encode_card(card())) == card()


def test_encoding_is_byte_stable_so_discovery_can_be_cached():
    assert encode_card(card()) == encode_card(card())


def test_encoding_a_card_without_description_is_rejected():
    broken = card()
    del broken["description"]
    with pytest.raises(A2AProtocolError):
        encode_card(broken)


def test_decoding_broken_json_is_a_protocol_error_not_a_json_error():
    with pytest.raises(A2AProtocolError):
        decode_card("{not json at all")


def test_decoding_a_valid_json_without_required_keys_is_rejected():
    with pytest.raises(A2AProtocolError):
        decode_card(json.dumps({"name": "a"}))


# ------------------------------------------------------ skill and message v1
def test_supports_skill_finds_a_declared_skill_id():
    assert supports_skill(card(), "review-python") is True


def test_supports_skill_rejects_an_undeclared_skill():
    assert supports_skill(card(), "translate") is False


def test_message_has_current_member_based_data_part():
    built = message()
    assert built["messageId"] == "msg-1" and built["role"] == "ROLE_USER"
    assert built["parts"][0]["mediaType"] == "application/json"
    assert built["parts"][0]["data"]["skill"] == "review-python"
    assert "kind" not in built["parts"][0]


def test_new_client_message_does_not_contain_a_task_id():
    assert "taskId" not in message()


# ---------------------------------------------------------- server-made Task
def test_server_assigns_the_task_id():
    task = make_task(message(), lambda: "task-server-1")
    assert task["id"] == "task-server-1"


def test_client_cannot_assign_an_id_for_a_new_task():
    client_message = {**message(), "taskId": "task-client-picked"}
    with pytest.raises(A2AProtocolError):
        make_task(client_message, lambda: "task-server-1")


def test_a_new_task_uses_current_status_and_context_fields():
    task = make_task(message(), lambda: "task-1")
    assert task["contextId"] == "ctx-task-1"
    assert task["status"] == {"state": "TASK_STATE_SUBMITTED"}
    assert task["artifacts"] == []


# -------------------------------------------------------------- Artifact v1
def test_text_artifact_uses_text_member_and_media_type():
    artifact = make_artifact("a-1", "text/plain", "looks fine")
    assert artifact == {
        "artifactId": "a-1",
        "parts": [{"mediaType": "text/plain", "text": "looks fine"}],
    }


def test_structured_artifact_uses_data_member():
    artifact = make_artifact("a-1", "application/json", {"issues": []})
    assert artifact["parts"][0]["data"] == {"issues": []}


def test_an_artifact_with_a_made_up_media_type_is_rejected():
    with pytest.raises(A2AProtocolError):
        make_artifact("a-1", "model/hologram", {})


# --------------------------------------------------------------- lifecycle
def test_submitted_may_start_working():
    task = make_task(message(), lambda: "t")
    assert advance_task(task, "TASK_STATE_WORKING")["status"]["state"] == "TASK_STATE_WORKING"


def test_working_may_complete_with_an_artifact():
    task = advance_task(make_task(message(), lambda: "t"), "TASK_STATE_WORKING")
    done = advance_task(task, "TASK_STATE_COMPLETED", reviewer({"code": "return 1"}))
    assert done["artifacts"][0]["artifactId"] == "artifact-review"


def test_submitted_may_not_jump_straight_to_completed():
    with pytest.raises(A2AProtocolError):
        advance_task(make_task(message(), lambda: "t"), "TASK_STATE_COMPLETED")


def test_every_terminal_state_is_a_dead_end():
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == ()
        stuck = make_task(message(), lambda: "t")
        stuck["status"] = {"state": state}
        with pytest.raises(A2AProtocolError):
            advance_task(stuck, "TASK_STATE_WORKING")


def test_advance_task_returns_a_snapshot():
    task = make_task(message(), lambda: "t")
    advance_task(task, "TASK_STATE_WORKING")
    assert task["status"]["state"] == "TASK_STATE_SUBMITTED"


# ---------------------------------------------------------------- run_task
def test_happy_path_uses_server_id_and_current_states():
    trace = run_task(card(), message(), reviewer, lambda: "task-server")
    assert trace[0]["id"] == "task-server"
    assert [t["status"]["state"] for t in trace] == [
        "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_COMPLETED",
    ]


def test_artifact_appears_only_in_final_snapshot():
    trace = run_task(card(), message(), reviewer, lambda: "t")
    assert [t["artifacts"] for t in trace[:-1]] == [[], []]
    assert trace[-1]["artifacts"][0]["parts"][0]["mediaType"] == "application/json"


def test_unsupported_skill_fails_without_ever_working():
    trace = run_task(card(), message("translate"), reviewer, lambda: "t")
    assert [t["status"]["state"] for t in trace] == [
        "TASK_STATE_SUBMITTED", "TASK_STATE_FAILED",
    ]


def test_failed_task_explains_itself_in_a_text_artifact():
    trace = run_task(card(), message("translate"), reviewer, lambda: "t")
    part = trace[-1]["artifacts"][0]["parts"][0]
    assert part["mediaType"] == "text/plain" and "translate" in part["text"]


def test_worker_is_never_called_for_an_unsupported_skill():
    calls = []
    run_task(card(), message("translate"), lambda payload: calls.append(payload), lambda: "t")
    assert calls == []
