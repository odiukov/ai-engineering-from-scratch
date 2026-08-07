"""Тесты к уроку «Безопасность: секреты, PII и неизменяемый аудит». Правь exercise.py."""

import json

import pytest

from exercise import (
    GENESIS_HASH,
    append_audit,
    audit_llm_call,
    chain_hash,
    find_secrets,
    is_allowed,
    placeholder_for,
    redact,
    verify_chain,
)

OPENAI = "sk-proj-A1b2C3d4E5f6G7h8I9j0KLMN"
ANTHROPIC = "sk-ant-api03-AAAABBBBCCCCDDDD"
AWS = "AKIAIOSFODNN7EXAMPLE"
GITHUB = "ghp_16C7e42F292c6912E7710c838347Ae178B4a"
SSN = "123-45-6789"


def kinds(text):
    """Только виды найденного — удобно сравнивать в тестах."""
    return [f["kind"] for f in find_secrets(text)]


# ------------------------------------------------------------ find_secrets
def test_provider_key_is_found_in_the_middle_of_a_line():
    """Секрет редко стоит первым словом — детектор по префиксу бесполезен."""
    text = "log line: outbound call with key " + OPENAI + " to api.openai.com"
    found = find_secrets(text)
    assert [(f["kind"], f["value"]) for f in found] == [("OPENAI_KEY", OPENAI)]


def test_offsets_point_at_the_secret_itself():
    text = "prefix " + SSN + " suffix"
    f = find_secrets(text)[0]
    assert text[f["start"] : f["end"]] == SSN


def test_anthropic_key_is_not_mislabeled_as_openai():
    """sk-ant-... подходит и под шаблон OpenAI — приоритет решает спор."""
    assert kinds("key " + ANTHROPIC) == ["ANTHROPIC_KEY"]


def test_aws_and_github_credentials_are_detected():
    assert kinds("creds " + AWS + " and " + GITHUB) == ["AWS_KEY_ID", "GITHUB_TOKEN"]


def test_lookalike_words_are_not_flagged_as_secrets():
    """Ложное срабатывание дороже пропуска: редактор портит рабочий текст."""
    assert find_secrets("Ask-me about task-2024 in the skylight branch") == []


def test_a_too_short_key_like_string_is_not_a_secret():
    assert find_secrets("try sk-test and sk-proj-abc") == []


def test_phone_number_is_not_mistaken_for_an_ssn():
    """415-555-0199 — это 3-3-4, а SSN 3-2-4. Жадный шаблон путает их."""
    assert "SSN" not in kinds("call 415-555-0199 tomorrow")


# --------------------------------------------------------- placeholder_for
def test_the_same_value_always_gets_the_same_placeholder():
    """Consistent tokenization: иначе модель не свяжет два упоминания в одно."""
    table = {}
    first = placeholder_for("SSN", SSN, table)
    second = placeholder_for("SSN", SSN, table)
    assert first == second == "[SSN_001]"


def test_different_values_get_different_placeholders():
    table = {}
    a = placeholder_for("SSN", SSN, table)
    b = placeholder_for("SSN", "999-88-7777", table)
    assert a != b


def test_numbering_is_per_kind_not_global():
    table = {}
    placeholder_for("SSN", SSN, table)
    assert placeholder_for("EMAIL", "a@b.com", table) == "[EMAIL_001]"


# ------------------------------------------------------------------ redact
def test_redaction_keeps_the_text_around_the_secret():
    assert redact("ключ " + OPENAI + " конец", {}) == "ключ [OPENAI_KEY_001] конец"


def test_two_secrets_in_one_line_are_numbered_left_to_right():
    """Замена с конца строки не должна перевернуть порядок нумерации."""
    out = redact("first 111-22-3333 second 444-55-6666", {})
    assert out == "first [SSN_001] second [SSN_002]"


def test_the_same_value_survives_across_separate_prompts():
    table = {}
    redact("мой SSN " + SSN, table)
    assert redact("тот же SSN " + SSN + " снова", table) == "тот же SSN [SSN_001] снова"


def test_clean_text_passes_through_untouched():
    assert redact("Ask-me about task-2024", {}) == "Ask-me about task-2024"


def test_no_raw_secret_value_survives_redaction():
    out = redact("k=" + OPENAI + " s=" + SSN + " m=bob@example.com", {})
    assert OPENAI not in out and SSN not in out and "bob@example.com" not in out


# -------------------------------------------------------------- is_allowed
def test_engineer_may_call_the_model():
    assert is_allowed("engineer", "call_model") is True


def test_auditor_may_read_the_log_but_not_call_the_model():
    assert (is_allowed("auditor", "read_audit"), is_allowed("auditor", "call_model")) == (True, False)


def test_unknown_role_is_denied_and_does_not_raise():
    """Опечатка в имени роли не должна ни ронять gateway, ни открывать доступ."""
    assert is_allowed("intern", "call_model") is False


# -------------------------------------------------------------- chain_hash
def test_hash_is_stable_for_the_same_record_and_predecessor():
    assert chain_hash(GENESIS_HASH, {"a": 1}) == chain_hash(GENESIS_HASH, {"a": 1})


def test_hash_depends_on_the_previous_hash():
    """Без этой зависимости журнал — просто список, а не цепочка."""
    assert chain_hash(GENESIS_HASH, {"a": 1}) != chain_hash("1" * 64, {"a": 1})


def test_key_order_in_the_record_does_not_change_the_hash():
    """Иначе целостность краснеет от того, что словарь собрали в другом порядке."""
    assert chain_hash(GENESIS_HASH, {"a": 1, "b": 2}) == chain_hash(GENESIS_HASH, {"b": 2, "a": 1})


# ------------------------------------------------------------ append_audit
def test_first_entry_links_to_the_genesis_hash():
    chain = append_audit([], {"n": 0})
    assert chain[0]["prev"] == GENESIS_HASH


def test_each_entry_links_to_the_previous_one():
    chain = append_audit(append_audit([], {"n": 0}), {"n": 1})
    assert chain[1]["prev"] == chain[0]["hash"]


def test_appending_does_not_mutate_the_old_chain():
    """Журнал append-only: старую историю нельзя трогать даже случайно."""
    old = append_audit([], {"n": 0})
    append_audit(old, {"n": 1})
    assert len(old) == 1


# ------------------------------------------------------------ verify_chain
def build_chain(n=5):
    chain = []
    for i in range(n):
        chain = append_audit(chain, {"n": i, "user": "u%d" % i})
    return chain


def test_an_untouched_chain_verifies():
    assert verify_chain(build_chain()) == -1


def test_editing_a_record_in_the_middle_is_caught_at_that_index():
    """Подделка задним числом: правим запись 2 из 5, цепочка это видит."""
    chain = build_chain()
    chain[2] = dict(chain[2], record={"n": 2, "user": "attacker"})
    assert verify_chain(chain) == 2


def test_recomputing_the_forged_entry_hash_only_moves_detection_one_step():
    """Атакующий пересчитал свой хеш — теперь не сходится ссылка следующей записи."""
    chain = build_chain()
    forged = {"n": 2, "user": "attacker"}
    chain[2] = {"record": forged, "prev": chain[2]["prev"],
                "hash": chain_hash(chain[2]["prev"], forged)}
    assert verify_chain(chain) == 3


def test_swapping_two_entries_is_caught():
    chain = build_chain()
    chain[1], chain[2] = chain[2], chain[1]
    assert verify_chain(chain) == 1


def test_a_truncated_chain_still_verifies_which_is_why_head_is_stored_apart():
    """Честная граница метода: обрезание хвоста цепочкой не ловится."""
    assert verify_chain(build_chain()[:3]) == -1


# ---------------------------------------------------------- audit_llm_call
def test_a_denied_role_produces_a_refusal_entry_and_no_prompt():
    chain, safe = audit_llm_call([], {}, "2026-08-07T10:00:00Z", "u1", "auditor",
                                 "t1", "claude", "SSN " + SSN, "ok")
    assert safe is None and chain[0]["record"]["allowed"] is False


def test_the_raw_secret_never_reaches_the_audit_log():
    chain, _ = audit_llm_call([], {}, "2026-08-07T10:00:00Z", "u1", "engineer",
                              "t1", "claude", "SSN " + SSN, "ok")
    assert SSN not in json.dumps(chain)


def test_the_entry_counts_how_many_secrets_the_prompt_carried():
    chain, _ = audit_llm_call([], {}, "2026-08-07T10:00:00Z", "u1", "engineer",
                              "t1", "claude", OPENAI + " and " + SSN, "ok")
    assert chain[0]["record"]["secrets_found"] == 2


def test_two_calls_about_the_same_person_share_the_placeholder():
    table = {}
    chain, first = audit_llm_call([], table, "2026-08-07T10:00:00Z", "u1",
                                  "engineer", "t1", "claude", "SSN " + SSN, "ok")
    _, second = audit_llm_call(chain, table, "2026-08-07T10:05:00Z", "u2",
                               "support", "t1", "claude", "again " + SSN, "ok")
    assert first.endswith("[SSN_001]") and second.endswith("[SSN_001]")


def test_the_log_written_by_the_call_path_verifies_as_a_chain():
    chain, _ = audit_llm_call([], {}, "2026-08-07T10:00:00Z", "u1", "engineer",
                              "t1", "claude", "hello", "ok")
    chain, _ = audit_llm_call(chain, {}, "2026-08-07T10:01:00Z", "u2", "auditor",
                              "t1", "claude", "hello", "ok")
    assert verify_chain(chain) == -1


def test_the_timestamp_comes_from_the_now_argument():
    """Никакого time.time() внутри: аудит обязан воспроизводиться в тесте."""
    chain, _ = audit_llm_call([], {}, "1999-01-01T00:00:00Z", "u1", "engineer",
                              "t1", "claude", "hello", "ok")
    assert chain[0]["record"]["ts"] == "1999-01-01T00:00:00Z"
