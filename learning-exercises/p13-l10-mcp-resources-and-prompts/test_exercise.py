"""Тесты к уроку «Ресурсы и промпты MCP». Правь exercise.py."""

import base64

import pytest

from exercise import (
    expand_template,
    pick_primitive,
    read_resource,
    render_prompt,
    resolve,
    resource_entry,
    subscribe,
    updated_notifications,
)

PNG = b"\x89PNG\r\n\x1a\n"


def make_store():
    return {
        "notes://note-1": {"mimeType": "text/markdown", "text": "# MCP"},
        "notes://note-2": {"text": "plain body"},
        "img://logo": {"mimeType": "image/png", "data": PNG},
    }


def make_prompts():
    return {
        "review_note": {
            "description": "Review one note.",
            "arguments": [{"name": "note_id", "required": True},
                          {"name": "tone", "required": False}],
            "messages": [
                {"role": "assistant",
                 "content": {"type": "text", "text": "You review notes. Tone: {tone}."}},
                {"role": "user",
                 "content": {"type": "text", "text": "Review {note_id} please."}},
            ],
        }
    }


# ----------------------------------------------------------- pick_primitive
def test_read_only_attachable_data_is_a_resource():
    assert pick_primitive({"attachable": True}) == "resource"


def test_mutation_is_a_tool_even_if_the_result_reads_nicely():
    """Ресурсы по определению только на чтение."""
    assert pick_primitive({"mutates": True, "attachable": True}) == "tool"


def test_reusable_multi_step_workflow_is_a_prompt():
    assert pick_primitive({"workflow": True, "mutates": True}) == "prompt"


def test_unclassified_capability_defaults_to_a_tool():
    assert pick_primitive({}) == "tool"


# ----------------------------------------------------------- resource_entry
def test_manifest_entry_carries_uri_name_and_mime():
    assert resource_entry("notes://note-1", "MCP overview") == {
        "uri": "notes://note-1", "name": "MCP overview", "mimeType": "text/plain"}


def test_description_is_added_only_when_given():
    entry = resource_entry("file:///a.md", "A", "text/markdown", "Заметка")
    assert entry["description"] == "Заметка" and entry["mimeType"] == "text/markdown"


def test_absent_description_leaves_no_null_key():
    """Ключа со значением None в JSON быть не должно."""
    assert "description" not in resource_entry("notes://n1", "A")


def test_uri_without_a_scheme_is_refused():
    with pytest.raises(ValueError):
        resource_entry("note-1", "A")


# ------------------------------------------------------------ read_resource
def test_text_resource_is_read_with_its_mime_type():
    content = read_resource(make_store(), "notes://note-1")["contents"][0]
    assert content == {"uri": "notes://note-1", "mimeType": "text/markdown", "text": "# MCP"}


def test_missing_mime_type_falls_back_to_plain_text():
    content = read_resource(make_store(), "notes://note-2")["contents"][0]
    assert content["mimeType"] == "text/plain"


def test_binary_resource_travels_as_a_base64_string():
    """JSON не умеет байты; blob — это строка, а не bytes."""
    content = read_resource(make_store(), "img://logo")["contents"][0]
    assert isinstance(content["blob"], str)
    assert base64.b64decode(content["blob"]) == PNG


def test_binary_content_never_carries_a_text_key():
    """Клиент выбирает ветку по наличию ключа; лишний ключ его ломает."""
    content = read_resource(make_store(), "img://logo")["contents"][0]
    assert "text" not in content


def test_unknown_uri_raises_key_error():
    with pytest.raises(KeyError):
        read_resource(make_store(), "notes://ghost")


# ---------------------------------------------------------- expand_template
def test_template_extracts_the_named_parameter():
    assert expand_template("notes://{id}", "notes://note-14") == {"id": "note-14"}


def test_template_with_a_different_scheme_does_not_match():
    assert expand_template("notes://{id}", "files://note-14") is None


def test_placeholder_does_not_jump_over_a_slash():
    """Иначе notes://{id} совпал бы с чем угодно и потерял смысл."""
    assert expand_template("notes://{id}", "notes://a/b") is None


def test_two_placeholders_are_extracted_separately():
    assert expand_template("db://{table}/{row}", "db://users/7") == {"table": "users", "row": "7"}


def test_literal_dots_are_matched_literally_not_as_any_char():
    """Точка в шаблоне — это точка, а не «любой символ» из регулярки."""
    assert expand_template("file:///{name}.md", "file:///aXmd") is None
    assert expand_template("file:///{name}.md", "file:///a.md") == {"name": "a"}


# ----------------------------------------------------------------- resolve
def test_static_resource_wins_over_a_matching_template():
    template = {"uriTemplate": "notes://{id}", "read": lambda p: {"text": "dynamic"}}
    contents = resolve(make_store(), [template], "notes://note-1")["contents"]
    assert contents[0]["text"] == "# MCP"


def test_template_resource_is_computed_on_read():
    template = {"uriTemplate": "notes://recent/{n}",
                "read": lambda p: {"text": f"last {p['n']}"}}
    contents = resolve({}, [template], "notes://recent/5")["contents"]
    assert contents[0] == {"uri": "notes://recent/5", "mimeType": "text/plain", "text": "last 5"}


def test_dynamic_resource_reflects_the_current_state():
    """notes://recent обязан отдавать свежее, а не снимок при старте."""
    notes = ["a"]
    template = {"uriTemplate": "notes://{name}", "read": lambda p: {"text": ",".join(notes)}}
    first = resolve({}, [template], "notes://recent")["contents"][0]["text"]
    notes.append("b")
    second = resolve({}, [template], "notes://recent")["contents"][0]["text"]
    assert (first, second) == ("a", "a,b")


def test_uri_matching_nothing_raises_key_error():
    template = {"uriTemplate": "notes://{id}", "read": lambda p: {"text": "x"}}
    with pytest.raises(KeyError):
        resolve({}, [template], "files:///etc/passwd")


# --------------------------------------------------------------- subscribe
def test_subscribe_records_the_listener():
    assert subscribe({}, "notes://n1", "s1") == {"notes://n1": ["s1"]}


def test_subscribing_twice_does_not_duplicate():
    once = subscribe({}, "notes://n1", "s1")
    assert subscribe(once, "notes://n1", "s1") == {"notes://n1": ["s1"]}


def test_subscribe_does_not_mutate_the_input():
    """Правка общего состояния на месте — источник гонок между сессиями."""
    before = {"notes://n1": ["s1"]}
    subscribe(before, "notes://n1", "s2")
    assert before == {"notes://n1": ["s1"]}


def test_last_unsubscribe_removes_the_uri_entirely():
    """Отвалившийся клиент обязан исчезать, иначе словарь растёт вечно."""
    state = subscribe({}, "notes://n1", "s1")
    assert subscribe(state, "notes://n1", "s1", on=False) == {}


# ---------------------------------------------------- updated_notifications
def test_every_subscriber_gets_one_message():
    out = updated_notifications({"notes://n1": ["s1", "s2"]}, "notes://n1")
    assert [session for session, _ in out] == ["s1", "s2"]


def test_update_message_is_a_notification_without_id():
    """Ответа на неё ждать нельзя — клиент перечитает ресурс сам."""
    _, message = updated_notifications({"notes://n1": ["s1"]}, "notes://n1")[0]
    assert "id" not in message and message["method"] == "notifications/resources/updated"


def test_nobody_subscribed_means_nothing_is_sent():
    assert updated_notifications({}, "notes://n1") == []


def test_subscribers_of_other_resources_are_not_disturbed():
    subs = {"notes://n1": ["s1"], "notes://n2": ["s2"]}
    assert [s for s, _ in updated_notifications(subs, "notes://n2")] == ["s2"]


# ------------------------------------------------------------ render_prompt
def test_arguments_are_substituted_in_every_message():
    result = render_prompt(make_prompts(), "review_note", {"note_id": "note-14", "tone": "dry"})
    texts = [m["content"]["text"] for m in result["messages"]]
    assert texts == ["You review notes. Tone: dry.", "Review note-14 please."]


def test_missing_required_argument_fails_before_substitution():
    """Пустая дырка в системном сообщении хуже честной ошибки."""
    with pytest.raises(ValueError):
        render_prompt(make_prompts(), "review_note", {"tone": "dry"})


def test_unknown_prompt_raises_key_error():
    with pytest.raises(KeyError):
        render_prompt(make_prompts(), "summarize_pr", {"note_id": "x"})


def test_rendering_does_not_consume_the_template():
    """Второй вызов обязан снова увидеть {note_id}, а не результат первого."""
    prompts = make_prompts()
    render_prompt(prompts, "review_note", {"note_id": "note-1"})
    again = render_prompt(prompts, "review_note", {"note_id": "note-2"})
    assert "note-2" in again["messages"][1]["content"]["text"]


def test_braces_in_the_prompt_body_survive():
    """str.format упал бы на JSON внутри промпта."""
    prompts = {"j": {"description": "d", "arguments": [],
                     "messages": [{"role": "user",
                                   "content": {"type": "text",
                                               "text": 'Return {"ok": true} for {who}'}}]}}
    text = render_prompt(prompts, "j", {"who": "me"})["messages"][0]["content"]["text"]
    assert text == 'Return {"ok": true} for me'
