"""Тесты к уроку «Roots и elicitation». Правь exercise.py."""

import pytest

from exercise import (
    ELICIT_METHOD,
    delete_note,
    disambiguate,
    elicitation_request,
    handle_elicitation_response,
    normalize_root,
    update_roots,
    within_roots,
)

ROOTS = ["file:///Users/alice/Notes"]

FLAT_SCHEMA = {
    "type": "object",
    "properties": {"note_id": {"type": "string", "enum": ["note-3", "note-14"]},
                   "confirm": {"type": "boolean"}},
    "required": ["note_id", "confirm"],
}


def make_store():
    return {
        "note-3": {"title": "TPS report", "uri": "file:///Users/alice/Notes/tps-2023.md"},
        "note-14": {"title": "TPS report", "uri": "file:///Users/alice/Notes/tps-2025.md"},
        "note-9": {"title": "Groceries", "uri": "file:///Users/alice/Notes/food.md"},
        "note-99": {"title": "Passwords", "uri": "file:///Users/alice/.ssh/keys.md"},
    }


def answering(note_id, confirm=True, action="accept"):
    """Пользователь, отвечающий заранее известным образом."""
    calls = []

    def ask(request):
        calls.append(request)
        if action != "accept":
            return {"action": action}
        return {"action": "accept", "content": {"note_id": note_id, "confirm": confirm}}

    ask.calls = calls
    return ask


# ----------------------------------------------------------- normalize_root
def test_trailing_slash_is_dropped():
    assert normalize_root("file:///Users/alice/Notes/") == "file:///Users/alice/Notes"


def test_parent_segments_collapse():
    """"Notes/../.." по префиксу строки выглядит как «внутри Notes»."""
    assert normalize_root("file:///Users/alice/../bob") == "file:///Users/bob"


def test_scheme_is_preserved():
    assert normalize_root("notes://note-1/") == "notes://note-1"


def test_bare_path_is_not_a_root():
    with pytest.raises(ValueError):
        normalize_root("/Users/alice/Notes")


def test_scheme_without_a_path_is_refused():
    with pytest.raises(ValueError):
        normalize_root("file://")


# ------------------------------------------------------------- within_roots
def test_file_inside_the_root_passes():
    assert within_roots("file:///Users/alice/Notes/a.md", ROOTS) is True


def test_the_root_itself_is_inside():
    assert within_roots("file:///Users/alice/Notes", ROOTS) is True


def test_sibling_with_the_same_prefix_is_outside():
    """Граница проходит по сегментам пути, а не по символам строки."""
    assert within_roots("file:///Users/alice/Notes-evil/a.md", ROOTS) is False


def test_parent_traversal_cannot_escape_the_root():
    assert within_roots("file:///Users/alice/Notes/../.ssh/id_rsa", ROOTS) is False


def test_no_declared_roots_means_nothing_is_allowed():
    """Клиент, не объявивший roots, согласия не давал."""
    assert within_roots("file:///Users/alice/Notes/a.md", []) is False


# ------------------------------------------------------------- update_roots
def test_new_roots_are_stored_normalized():
    state = update_roots({"roots": [], "open": []}, ["file:///Users/alice/Notes/"])
    assert state["roots"] == ["file:///Users/alice/Notes"]


def test_handles_outside_the_new_roots_are_evicted():
    """Пользователь отобрал каталог — открытые в нём хендлы обязаны закрыться."""
    before = {"roots": ["file:///Users/alice"],
              "open": ["file:///Users/alice/Notes/a.md", "file:///Users/alice/.ssh/id_rsa"]}
    after = update_roots(before, ROOTS)
    assert after["evicted"] == ["file:///Users/alice/.ssh/id_rsa"]
    assert after["open"] == ["file:///Users/alice/Notes/a.md"]


def test_empty_root_list_evicts_everything():
    after = update_roots({"roots": ROOTS, "open": ["file:///Users/alice/Notes/a.md"]}, [])
    assert after["open"] == [] and len(after["evicted"]) == 1


def test_update_does_not_mutate_the_previous_state():
    before = {"roots": ["file:///Users/alice"], "open": ["file:///Users/alice/x.md"]}
    update_roots(before, ROOTS)
    assert before == {"roots": ["file:///Users/alice"], "open": ["file:///Users/alice/x.md"]}


# ------------------------------------------------------ elicitation_request
def test_form_request_carries_the_schema_and_the_message():
    request = elicitation_request(1, "Pick one", schema=FLAT_SCHEMA)
    assert request["method"] == ELICIT_METHOD and request["id"] == 1
    assert request["params"] == {"message": "Pick one", "requestedSchema": FLAT_SCHEMA}


def test_nested_form_is_refused():
    """Форма elicitation плоская; вложенный объект v1 не умеет."""
    nested = {"type": "object",
              "properties": {"owner": {"type": "object", "properties": {}}}}
    with pytest.raises(ValueError):
        elicitation_request(1, "Pick", schema=nested)


def test_required_must_reference_existing_properties():
    broken = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]}
    with pytest.raises(ValueError):
        elicitation_request(1, "Pick", schema=broken)


def test_url_mode_replaces_the_schema():
    request = elicitation_request(2, "Sign in", url="https://github.com/login/oauth")
    assert request["params"]["url"] == "https://github.com/login/oauth"
    assert "requestedSchema" not in request["params"]


def test_plain_http_url_is_refused():
    """Иначе пользователь введёт пароль по открытому каналу."""
    with pytest.raises(ValueError):
        elicitation_request(2, "Sign in", url="http://github.example/login")


def test_schema_and_url_are_mutually_exclusive():
    with pytest.raises(ValueError):
        elicitation_request(1, "Pick", schema=FLAT_SCHEMA, url="https://example.com")
    with pytest.raises(ValueError):
        elicitation_request(1, "Pick")


# ---------------------------------------------- handle_elicitation_response
def test_accepted_form_returns_its_content():
    answer = {"action": "accept", "content": {"note_id": "note-14", "confirm": True}}
    assert handle_elicitation_response(answer, FLAT_SCHEMA) == {
        "status": "accepted", "content": {"note_id": "note-14", "confirm": True}}


def test_decline_and_cancel_are_different_outcomes():
    """decline — «не отвечу», cancel — «прекрати весь вызов»."""
    declined = handle_elicitation_response({"action": "decline"})
    cancelled = handle_elicitation_response({"action": "cancel"})
    assert declined["status"] == "declined" and cancelled["status"] == "cancelled"


def test_missing_required_field_is_refused():
    with pytest.raises(ValueError):
        handle_elicitation_response({"action": "accept", "content": {"confirm": True}}, FLAT_SCHEMA)


def test_value_outside_the_enum_is_refused():
    """Содержимое формы приходит от клиента; доверять ему нельзя."""
    answer = {"action": "accept", "content": {"note_id": "note-666", "confirm": True}}
    with pytest.raises(ValueError):
        handle_elicitation_response(answer, FLAT_SCHEMA)


def test_unknown_action_is_refused():
    with pytest.raises(ValueError):
        handle_elicitation_response({"action": "maybe"})


# ------------------------------------------------------------ disambiguate
def test_single_match_needs_no_dialog():
    """Диалог рвёт разговор; спрашивать про единственного кандидата нельзя."""
    assert disambiguate(1, ["note-14"]) is None


def test_several_matches_produce_a_form_with_all_of_them():
    request = disambiguate(1, ["note-3", "note-7", "note-14"])
    schema = request["params"]["requestedSchema"]
    assert schema["properties"]["note_id"]["enum"] == ["note-3", "note-7", "note-14"]
    assert set(schema["required"]) == {"note_id", "confirm"}


def test_zero_matches_is_a_call_error_not_a_question():
    with pytest.raises(ValueError):
        disambiguate(1, [])


def test_disambiguation_form_stays_flat():
    request = disambiguate(1, ["note-3", "note-7"])
    types = {s["type"] for s in request["params"]["requestedSchema"]["properties"].values()}
    assert types == {"string", "boolean"}


# -------------------------------------------------------------- delete_note
def test_single_match_is_deleted_without_asking():
    store, ask = make_store(), answering("note-9")
    result = delete_note(store, "Groceries", ROOTS, ask)
    assert result == {"deleted": ["note-9"], "status": "accepted"}
    assert ask.calls == [] and "note-9" not in store


def test_user_choice_decides_which_note_dies():
    store, ask = make_store(), answering("note-14")
    result = delete_note(store, "TPS report", ROOTS, ask)
    assert result["deleted"] == ["note-14"]
    assert "note-3" in store and "note-14" not in store


def test_declined_dialog_deletes_nothing():
    store, ask = make_store(), answering("note-14", action="decline")
    result = delete_note(store, "TPS report", ROOTS, ask)
    assert result == {"deleted": [], "status": "declined"}
    assert len(store) == 4


def test_unchecked_confirm_is_a_refusal():
    """Форма заполнена, галочка снята — это отказ, а не согласие."""
    store, ask = make_store(), answering("note-14", confirm=False)
    result = delete_note(store, "TPS report", ROOTS, ask)
    assert result["deleted"] == [] and "note-14" in store


def test_note_outside_the_roots_is_not_deleted():
    store, ask = make_store(), answering("note-99")
    with pytest.raises(PermissionError):
        delete_note(store, "Passwords", ROOTS, ask)
    assert "note-99" in store
