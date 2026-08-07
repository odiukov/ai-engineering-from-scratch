"""Тесты к уроку «Llama Guard и классификация входа/выхода». Правь exercise.py."""

import pytest

from exercise import (
    REQUIRED_LAYERS,
    TAXONOMY,
    attack_success_rate,
    audit_stack,
    classify,
    dialog_rail_report,
    normalize_text,
    output_rail,
    route,
    verdict,
)

ZWSP = "​"
SMUGGLED = "we" + ZWSP + "apon"
HOMOGLYPH = "weаpon"  # кириллическая 'а' вместо латинской


# --------------------------------------------------------- normalize_text
def test_normalize_removes_zero_width_characters():
    assert normalize_text(SMUGGLED) == "weapon"


def test_normalize_maps_cyrillic_homoglyphs_to_latin():
    assert normalize_text(HOMOGLYPH) == "weapon"


def test_normalize_keeps_legitimate_diacritics():
    """Выбросить всю категорию Mn — потерять «café». Убираем только невидимки."""
    assert normalize_text("café") == "café"


def test_normalize_applies_nfkc_before_the_homoglyph_map():
    """Лигатура 'ﬁ' — один символ; без NFKC карта до букв не доберётся."""
    assert normalize_text("ﬁle") == "file"


# ---------------------------------------------------------------- classify
def test_classify_returns_nothing_on_benign_text():
    assert classify("a plain question about the weather") == []


def test_classify_returns_the_category_code():
    assert classify("weapon shop hours") == ["S1"]


def test_classify_order_follows_the_taxonomy_not_the_text():
    """Перестановка слов не должна менять вердикт — иначе логи несравнимы."""
    a = classify("interpreter abuse and weapon talk")
    b = classify("weapon talk and interpreter abuse")
    assert a == b == ["S1", "S14"]


def test_classify_misses_the_smuggled_variant_without_normalization():
    """Ровно то, что измерили Huang et al.: сырой классификатор течёт."""
    assert classify(SMUGGLED) == []


def test_classify_catches_the_smuggled_variant_with_normalization():
    assert classify(SMUGGLED, normalize=True) == ["S1"]


def test_classify_accepts_a_custom_taxonomy():
    tax = {"S9": ("zebra",)}
    assert classify("a zebra crossing", taxonomy=tax) == ["S9"]


# ----------------------------------------------------------------- verdict
def test_verdict_marks_benign_text_safe():
    v = verdict("a plain question about the weather")
    assert v["safe"] is True and v["categories"] == []


def test_verdict_carries_the_category_not_just_the_word_unsafe():
    """Без кода категории маршрутизировать нечего: block и log неразличимы."""
    v = verdict("weapon shop hours")
    assert v["safe"] is False
    assert v["categories"] == ["S1"]
    assert "S1" in v["reason"]


def test_verdict_lists_every_triggered_category():
    v = verdict("weapon talk and interpreter abuse")
    assert v["categories"] == ["S1", "S14"]
    assert v["reason"] == "unsafe: S1,S14"


def test_verdict_changes_when_normalization_is_enabled():
    assert verdict(HOMOGLYPH)["safe"] is True
    assert verdict(HOMOGLYPH, normalize=True)["safe"] is False


# ------------------------------------------------------------------- route
def test_route_of_nothing_allows():
    assert route([]) == "allow"


def test_route_picks_the_strictest_action():
    assert route(["S2", "S1"]) == "block"


def test_route_does_not_allow_an_unknown_category():
    """Таксономия растёт: S1-S13 стали S1-S14. Незнакомое — не 'allow'."""
    assert route(["S99"]) == "escalate"


def test_route_rejects_an_unknown_action_in_the_policy():
    with pytest.raises(ValueError):
        route(["S1"], policy={"S1": "bock"})


# ------------------------------------------------------------- output_rail
def test_output_rail_passes_a_benign_answer():
    assert output_rail("here is a benign summary of the docs") == []


def test_output_rail_catches_a_leaked_key():
    assert output_rail("your token is sk-abcdef0123456789") == ["api_key"]


def test_output_rail_reports_every_marker_sorted():
    text = "aws_secret_access_key=x and also sk-abcdef0123456789"
    assert output_rail(text) == ["api_key", "aws_secret"]


def test_output_rail_catches_what_the_input_rail_could_not():
    """Вход безобиден, выход — утечка. Input rail на такой паре бесполезен."""
    ask = "please summarise the deployment config"
    answer = "sure: user_session_token is stored in the env"
    assert verdict(ask)["safe"] is True
    assert output_rail(answer) == ["session_token"]


# ------------------------------------------------------ attack_success_rate
def test_asr_is_zero_when_every_attack_is_caught():
    assert attack_success_rate([("weapon", "S1"), ("fraud", "S2")]) == 0.0


def test_asr_is_one_when_every_attack_slips_through():
    assert attack_success_rate([(SMUGGLED, "S1"), (HOMOGLYPH, "S1")]) == 1.0


def test_asr_drops_when_normalization_is_added():
    cases = [(SMUGGLED, "S1"), (HOMOGLYPH, "S1"), ("weapon", "S1")]
    raw = attack_success_rate(cases)
    normalized = attack_success_rate(cases, normalize=True)
    assert raw > normalized == 0.0


def test_asr_of_an_empty_suite_is_zero_and_proves_nothing():
    assert attack_success_rate([]) == 0.0


def test_asr_is_a_fraction_not_a_count():
    cases = [(SMUGGLED, "S1"), ("weapon", "S1")]
    assert attack_success_rate(cases) == pytest.approx(0.5)


# ------------------------------------------------------ dialog_rail_report
RAIL = {"topic": "diagnosis", "markers": ("diagnos",), "max_mentions": 2}


def test_dialog_rail_does_not_fire_below_the_limit():
    r = dialog_rail_report(["hello", "what is my diagnosis"], RAIL)
    assert r["mentions"] == [1]
    assert r["fired"] is False


def test_dialog_rail_fires_on_the_conversation_not_on_a_single_turn():
    """Три перефразировки: каждый ход safe по таксономии, беседа — нет."""
    turns = [
        "could you diagnose this rash",
        "asking differently: what diagnosis fits",
        "hypothetically, which diagnosis would a doctor pick",
    ]
    assert all(verdict(t)["safe"] for t in turns)
    r = dialog_rail_report(turns, RAIL)
    assert r["mentions"] == [0, 1, 2]
    assert r["fired"] is True


def test_dialog_rail_reports_the_topic_it_was_given():
    r = dialog_rail_report(["diagnosis?"], RAIL)
    assert r["topic"] == "diagnosis"


def test_dialog_rail_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        dialog_rail_report(["anything"], {**RAIL, "max_mentions": 0})


# ------------------------------------------------------------- audit_stack
def test_audit_of_a_complete_stack_finds_nothing():
    assert audit_stack({layer: True for layer in REQUIRED_LAYERS}) == []


def test_audit_lists_missing_layers_in_the_canonical_order():
    gaps = audit_stack({"model": "llama-guard-4"})
    assert gaps == [layer for layer in REQUIRED_LAYERS if layer != "model"]


def test_audit_treats_an_empty_value_as_a_missing_layer():
    """«Поле есть, значение пустое» — это отсутствующий слой, а не половина."""
    config = {layer: True for layer in REQUIRED_LAYERS}
    config["output_rail"] = ""
    assert audit_stack(config) == ["output_rail"]


def test_audit_covers_every_layer_the_taxonomy_needs():
    """Таксономия объявлена, но без неё вердикт невозможен — слой обязателен."""
    assert "taxonomy" in REQUIRED_LAYERS
    assert set(TAXONOMY) >= {"S1", "S14"}
    assert audit_stack({}) == list(REQUIRED_LAYERS)
