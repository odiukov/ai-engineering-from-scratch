"""Тесты к уроку «Виртуальный контекст и подкачка памяти (MemGPT)». Правь exercise.py."""

import pytest

from exercise import (
    append_message,
    archival_insert,
    archival_search,
    conversation_search,
    core_memory_append,
    core_memory_replace,
    page_in,
    render_main_context,
)


def ctx(max_messages=3, messages=(), evicted=(), core=None):
    """Свежий main context для теста."""
    return {
        "core": dict(core or {}),
        "messages": list(messages),
        "evicted": list(evicted),
        "max_messages": max_messages,
    }


# ------------------------------------------------------- core_memory_append
def test_core_append_creates_missing_section():
    assert core_memory_append({}, "user", "name=ava") == {"user": "name=ava"}


def test_core_append_joins_with_a_single_space():
    got = core_memory_append({"user": "name=ava"}, "user", "city=Berlin")
    assert got == {"user": "name=ava city=Berlin"}


def test_core_append_does_not_touch_other_sections():
    got = core_memory_append({"persona": "terse"}, "user", "name=ava")
    assert got == {"persona": "terse", "user": "name=ava"}


def test_core_append_leaves_the_input_dict_alone():
    """core читают несколько вызовов сразу — мутация испортит их все."""
    core = {"user": "name=ava"}
    core_memory_append(core, "user", "city=Berlin")
    assert core == {"user": "name=ava"}


# ------------------------------------------------------ core_memory_replace
def test_core_replace_updates_the_fact():
    got = core_memory_replace({"user": "city=Berlin"}, "user", "Berlin", "Lisbon")
    assert got == {"user": "city=Lisbon"}


def test_core_replace_refuses_when_the_old_text_is_absent():
    """Тихий no-op оставил бы в промпте устаревший факт."""
    with pytest.raises(ValueError):
        core_memory_replace({"user": "city=Berlin"}, "user", "Paris", "Lisbon")


def test_core_replace_leaves_the_input_dict_alone():
    core = {"user": "city=Berlin"}
    core_memory_replace(core, "user", "Berlin", "Lisbon")
    assert core == {"user": "city=Berlin"}


# ---------------------------------------------------------- append_message
def test_append_message_adds_to_the_tail():
    got = append_message(ctx(), "user", "hi")
    assert got["messages"] == [("user", "hi")]


def test_append_message_below_the_cap_evicts_nothing():
    main = append_message(append_message(ctx(3), "user", "a"), "user", "b")
    assert main["evicted"] == []


def test_append_message_evicts_the_oldest_first():
    main = ctx(2)
    for text in ("a", "b", "c"):
        main = append_message(main, "user", text)
    assert main["messages"] == [("user", "b"), ("user", "c")]
    assert main["evicted"] == [("user", "a")]


def test_append_message_accumulates_evicted_across_calls():
    """evicted — это архив вытеснений, а не «последнее вытесненное»."""
    main = ctx(1)
    for text in ("a", "b", "c"):
        main = append_message(main, "user", text)
    assert main["evicted"] == [("user", "a"), ("user", "b")]


def test_append_message_leaves_the_input_context_alone():
    main = ctx(1, messages=[("user", "a")])
    append_message(main, "user", "b")
    assert main["messages"] == [("user", "a")]
    assert main["evicted"] == []


# ----------------------------------------------------- render_main_context
def test_render_puts_core_sections_in_alphabetical_order():
    """Порядок вставки дал бы разный промпт на одинаковых фактах."""
    main = ctx(core={"user": "name=ava", "persona": "terse"})
    rendered = render_main_context(main)
    assert rendered.index("persona:") < rendered.index("user:")


def test_render_has_both_headers_and_two_space_indent():
    main = ctx(core={"persona": "terse"}, messages=[("user", "hi")])
    assert render_main_context(main) == "[core]\n  persona: terse\n[messages]\n  user: hi"


def test_render_hides_evicted_messages():
    """Вытесненное больше не видно модели — в этом весь смысл вытеснения."""
    main = ctx(1)
    main = append_message(main, "user", "secret older turn")
    main = append_message(main, "user", "current turn")
    assert "secret older turn" not in render_main_context(main)


# ---------------------------------------------------------- archival_insert
def test_archival_insert_hands_out_increasing_ids():
    store, first = archival_insert([], "fact one")
    store, second = archival_insert(store, "fact two")
    assert (first, second) == ("a001", "a002")
    assert len(store) == 2


def test_archival_insert_keeps_citation_fields():
    """Без session/turn агент вспомнит факт, но не сможет показать источник."""
    store, rid = archival_insert([], "12 tools", tags=("project",),
                                 session_id="s7", turn_id=4)
    assert store[0]["rid"] == rid
    assert (store[0]["session_id"], store[0]["turn_id"]) == ("s7", 4)
    assert store[0]["tags"] == ("project",)


def test_archival_insert_leaves_the_input_store_alone():
    store = []
    archival_insert(store, "fact one")
    assert store == []


# ---------------------------------------------------------- archival_search
def test_archival_search_finds_the_matching_record():
    store, _ = archival_insert([], "tool chains drift after 20 steps")
    store, _ = archival_insert(store, "sleep-time compute consolidates memory")
    hits = archival_search(store, "tool chains drift")
    assert [h["rid"] for h in hits] == ["a001"]


def test_archival_search_returns_nothing_on_zero_overlap():
    """Пустой ответ честнее случайного факта, вставленного в промпт."""
    store, _ = archival_insert([], "tool chains drift after 20 steps")
    assert archival_search(store, "quantum chromodynamics") == []


def test_archival_search_ranks_the_closer_record_first():
    store, _ = archival_insert([], "ava ships agents for a living")
    store, _ = archival_insert(store, "ava lives in berlin")
    hits = archival_search(store, "where does ava live")
    assert hits[0]["rid"] == "a002"


def test_archival_search_does_not_reward_a_record_for_being_long():
    """Деление на объединение, а не на длину запроса: длина не даёт форы."""
    short = "ava lives in berlin"
    long_noise = short + " " + " ".join(f"noise{i}" for i in range(40))
    store, _ = archival_insert([], long_noise)
    store, _ = archival_insert(store, short)
    hits = archival_search(store, "ava lives in berlin")
    assert hits[0]["rid"] == "a002"


def test_archival_search_respects_top_k():
    store = []
    for i in range(5):
        store, _ = archival_insert(store, f"ava fact number {i}")
    assert len(archival_search(store, "ava fact", top_k=2)) == 2


# ------------------------------------------------------ conversation_search
def test_conversation_search_reaches_into_evicted_history():
    main = ctx(1)
    main = append_message(main, "user", "my retrieval bot has 12 tools")
    main = append_message(main, "assistant", "let me check archival")
    assert conversation_search(main, "retrieval bot") == (
        "user", "my retrieval bot has 12 tools")


def test_conversation_search_ignores_case():
    main = ctx(2, messages=[("user", "My Retrieval Bot")])
    assert conversation_search(main, "retrieval bot")[0] == "user"


def test_conversation_search_returns_the_most_recent_match():
    """Устаревшая версия факта лежит раньше — брать надо последнюю."""
    main = ctx(4)
    main = append_message(main, "user", "city is Berlin")
    main = append_message(main, "user", "city is Lisbon now")
    assert conversation_search(main, "city is") == ("user", "city is Lisbon now")


def test_conversation_search_returns_none_when_nothing_matches():
    main = ctx(2, messages=[("user", "hi")])
    assert conversation_search(main, "kubernetes") is None


# ------------------------------------------------------------------ page_in
def test_page_in_of_nothing_changes_nothing():
    main = ctx(3, messages=[("user", "hi")])
    assert page_in(main, [])["messages"] == [("user", "hi")]


def test_page_in_cites_the_record_id_session_and_turn():
    record = {"rid": "a001", "text": "12 tools", "session_id": "s1",
              "turn_id": 4, "tags": ()}
    main = page_in(ctx(3), [record])
    assert main["messages"][-1] == ("system", "recall: a001@s1:4 12 tools")


def test_page_in_brings_back_exactly_what_was_evicted():
    """Главное свойство подкачки: вытесненный текст снова виден в промпте."""
    main = ctx(1)
    store, _ = archival_insert([], "my retrieval bot has 12 tools",
                               session_id="s1", turn_id=0)
    main = append_message(main, "user", "my retrieval bot has 12 tools")
    main = append_message(main, "user", "what did I say about tools?")
    assert "12 tools" not in render_main_context(main)

    main = page_in(main, archival_search(store, "retrieval bot tools"))
    assert "12 tools" in render_main_context(main)


def test_page_in_can_itself_evict_an_older_message():
    """Подкачка занимает место в промпте — память конечна и после неё."""
    main = ctx(1, messages=[("user", "oldest")])
    record = {"rid": "a001", "text": "fact", "session_id": "s0",
              "turn_id": 0, "tags": ()}
    main = page_in(main, [record])
    assert main["messages"] == [("system", "recall: a001@s0:0 fact")]
    assert main["evicted"] == [("user", "oldest")]
