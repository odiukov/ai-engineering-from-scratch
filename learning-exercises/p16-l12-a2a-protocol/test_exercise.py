"""Тесты к уроку «Протокол A2A». Правь exercise.py."""

import json

import pytest

from exercise import (
    CARD_REQUIRED,
    MODALITIES,
    TERMINAL_STATES,
    TRANSITIONS,
    WELL_KNOWN_PATH,
    A2AProtocolError,
    advance_task,
    decode_card,
    encode_card,
    make_agent_card,
    make_artifact,
    make_task,
    run_task,
    supports_skill,
)


def card():
    return make_agent_card(
        "code-review-agent",
        "0.1.0",
        ["review-python", "summarize"],
        "http://localhost:8765",
        auth="bearer",
        modalities=("text", "structured"),
    )


def reviewer(payload):
    """Worker: считает строки и ищет отсутствующий return."""
    code = payload.get("code", "")
    issues = [] if "return" in code else ["no return statement"]
    return make_artifact("structured", {"issues": issues, "lines": code.count("\n") + 1})


# ------------------------------------------------------------ make_agent_card
def test_card_declares_the_well_known_discovery_endpoint():
    assert card()["endpoints"]["card"].endswith(WELL_KNOWN_PATH)


def test_card_builds_the_tasks_endpoint_from_the_base_url():
    assert card()["endpoints"]["tasks"] == "http://localhost:8765/tasks"


def test_card_does_not_double_the_slash_on_a_trailing_base_url():
    built = make_agent_card("a", "1", ["s"], "http://x.test/")
    assert built["endpoints"]["tasks"] == "http://x.test/tasks"


def test_card_carries_every_required_field():
    assert all(key in card() for key in CARD_REQUIRED)


def test_a_card_without_skills_is_rejected():
    """Агент без навыков нечего искать через discovery."""
    with pytest.raises(A2AProtocolError):
        make_agent_card("a", "1", [], "http://x.test")


def test_a_card_with_an_unknown_modality_is_rejected():
    assert "hologram" not in MODALITIES
    with pytest.raises(A2AProtocolError):
        make_agent_card("a", "1", ["s"], "http://x.test", modalities=("hologram",))


# --------------------------------------------------------- encode/decode card
def test_card_survives_the_json_round_trip():
    restored = decode_card(encode_card(card()))
    assert restored["name"] == "code-review-agent"
    assert restored["auth"] == {"type": "bearer"}


def test_tuples_come_back_as_lists_after_the_round_trip():
    """JSON не знает кортежей — сравнивать «до» и «после» надо с оглядкой."""
    restored = decode_card(encode_card(card()))
    assert restored["skills"] == ["review-python", "summarize"]


def test_encoding_is_byte_stable_so_discovery_can_be_cached():
    assert encode_card(card()) == encode_card(card())


def test_encoding_a_card_without_auth_is_rejected():
    broken = card()
    del broken["auth"]
    with pytest.raises(A2AProtocolError):
        encode_card(broken)


def test_decoding_broken_json_is_a_protocol_error_not_a_json_error():
    with pytest.raises(A2AProtocolError):
        decode_card("{not json at all")


def test_decoding_a_valid_json_without_required_keys_is_rejected():
    with pytest.raises(A2AProtocolError):
        decode_card(json.dumps({"name": "a"}))


# ------------------------------------------------------------ supports_skill
def test_supports_skill_finds_a_declared_skill():
    assert supports_skill(card(), "review-python") is True


def test_supports_skill_rejects_an_undeclared_skill():
    assert supports_skill(card(), "translate") is False


def test_supports_skill_works_on_a_card_that_came_over_the_wire():
    assert supports_skill(decode_card(encode_card(card())), "review-python") is True


# ------------------------------------------------------------------ make_task
def test_a_new_task_starts_as_submitted():
    assert make_task("t-1", "review-python", {})["state"] == "submitted"


def test_a_new_task_has_no_artifact_yet():
    assert make_task("t-1", "review-python", {})["artifact"] is None


def test_a_task_without_an_id_is_rejected():
    """Без id повторную отправку после ретрая нечем узнать."""
    with pytest.raises(A2AProtocolError):
        make_task("", "review-python", {})


# -------------------------------------------------------------- make_artifact
def test_artifact_carries_its_modality():
    assert make_artifact("text", "looks fine") == {"type": "text", "data": "looks fine"}


def test_structured_artifact_keeps_the_payload_shape():
    art = make_artifact("structured", {"issues": ["no return statement"]})
    assert art["data"]["issues"] == ["no return statement"]


def test_an_artifact_with_a_made_up_modality_is_rejected():
    with pytest.raises(A2AProtocolError):
        make_artifact("hologram", b"...")


# --------------------------------------------------------------- advance_task
def test_submitted_may_start_working():
    assert advance_task(make_task("t", "s", {}), "working")["state"] == "working"


def test_working_may_complete_with_an_artifact():
    working = advance_task(make_task("t", "s", {}), "working")
    done = advance_task(working, "completed", make_artifact("text", "ok"))
    assert done["state"] == "completed"
    assert done["artifact"]["data"] == "ok"


def test_submitted_may_not_jump_straight_to_completed():
    with pytest.raises(A2AProtocolError):
        advance_task(make_task("t", "s", {}), "completed")


def test_a_terminal_task_never_goes_back_to_working():
    """Клиент забрал артефакт — задача не имеет права ожить."""
    done = advance_task(advance_task(make_task("t", "s", {}), "working"), "completed")
    with pytest.raises(A2AProtocolError):
        advance_task(done, "working")


def test_every_terminal_state_is_a_dead_end():
    for state in TERMINAL_STATES:
        assert TRANSITIONS[state] == ()
        stuck = dict(make_task("t", "s", {}), state=state)
        with pytest.raises(A2AProtocolError):
            advance_task(stuck, "working")


def test_advance_task_returns_a_snapshot_and_leaves_the_original_alone():
    task = make_task("t", "s", {})
    advance_task(task, "working")
    assert task["state"] == "submitted"


# ------------------------------------------------------------------- run_task
def test_the_happy_path_walks_submitted_working_completed():
    trace = run_task(card(), make_task("t-1", "review-python", {"code": "return 1\n"}), reviewer)
    assert [t["state"] for t in trace] == ["submitted", "working", "completed"]


def test_the_artifact_appears_only_in_the_final_snapshot():
    trace = run_task(card(), make_task("t-1", "review-python", {"code": "return 1\n"}), reviewer)
    assert [t["artifact"] for t in trace[:-1]] == [None, None]
    assert trace[-1]["artifact"]["type"] == "structured"


def test_an_unsupported_skill_fails_without_ever_working():
    trace = run_task(card(), make_task("t-1", "translate", {}), reviewer)
    assert [t["state"] for t in trace] == ["submitted", "failed"]


def test_a_failed_task_still_explains_itself_in_a_text_artifact():
    trace = run_task(card(), make_task("t-1", "translate", {}), reviewer)
    assert trace[-1]["artifact"]["type"] == "text"
    assert "translate" in trace[-1]["artifact"]["data"]


def test_the_worker_is_never_called_for_an_unsupported_skill():
    calls = []

    def spy(payload):
        calls.append(payload)
        return make_artifact("text", "never")

    run_task(card(), make_task("t-1", "translate", {}), spy)
    assert calls == []


def test_the_lifecycle_is_opaque_the_client_only_sees_states_and_artifact():
    """Разные worker'ы — одна и та же последовательность состояний."""
    task = make_task("t-1", "review-python", {"code": "x = 1\n"})
    a = [t["state"] for t in run_task(card(), task, reviewer)]
    b = [t["state"] for t in run_task(card(), task, lambda p: make_artifact("text", "hi"))]
    assert a == b


def test_polling_snapshots_do_not_rewrite_each_other():
    """История опроса — это снимки: ранние состояния остаются на месте."""
    trace = run_task(card(), make_task("t-1", "review-python", {"code": "return 1\n"}), reviewer)
    assert trace[0]["state"] == "submitted"
    assert trace[1]["state"] == "working"
