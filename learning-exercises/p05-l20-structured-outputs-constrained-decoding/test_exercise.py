"""Тесты к уроку «Структурированный вывод и constrained decoding». Правь exercise.py."""

import random

import pytest

from exercise import (
    check_field_order,
    generate_constrained,
    is_accept,
    mask_logits,
    pattern_fsm,
    softmax,
    transition,
    valid_tokens,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
NEG_INF = float("-inf")

DIGITS = "0123456789"
PHONE_ALPHABET = list(DIGITS + "-")


def uniform_logits(alphabet):
    """Заглушка модели: все токены одинаково вероятны, prefix игнорируется."""
    return lambda prefix: [0.0] * len(alphabet)


def favour(alphabet, token, strength=20.0):
    """Заглушка модели, которая изо всех сил тянет ровно к одному токену."""
    return lambda prefix: [strength if c == token else 0.0 for c in alphabet]


# ------------------------------------------------------------- mask_logits
def test_mask_logits_keeps_valid_and_kills_invalid():
    assert mask_logits([1.0, 2.0, 3.0], {0, 2}) == [1.0, NEG_INF, 3.0]


def test_mask_logits_with_no_valid_ids_masks_everything():
    assert mask_logits([1.0, 2.0], set()) == [NEG_INF, NEG_INF]


def test_mask_logits_with_all_ids_valid_changes_nothing():
    assert mask_logits([1.0, 2.0], {0, 1}) == [1.0, 2.0]


def test_mask_logits_uses_minus_infinity_not_a_big_negative_number():
    """-1e9 «чтобы наверняка» оставляет невалидному токену ненулевой шанс."""
    masked = mask_logits([0.0, 0.0], {0})
    assert masked[1] == NEG_INF
    assert softmax(masked)[1] == 0.0


def test_mask_logits_does_not_mutate_the_input():
    logits = [1.0, 2.0, 3.0]
    mask_logits(logits, {1})
    assert logits == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------ softmax
def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0]) == APPROX([0.5, 0.5])


def test_softmax_is_invariant_to_a_constant_shift():
    """Сдвиг всех логитов на константу не меняет распределение."""
    base = [1.0, -2.0, 3.5]
    shifted = [x + 100.0 for x in base]
    assert softmax(shifted) == pytest.approx(softmax(base), abs=1e-12)


def test_softmax_survives_huge_logits():
    """Без вычитания максимума math.exp(1000) даёт OverflowError."""
    probs = softmax([1000.0, 999.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert probs[0] > probs[1]


def test_softmax_gives_exactly_zero_to_a_masked_token():
    assert softmax([0.0, NEG_INF]) == APPROX([1.0, 0.0])


def test_softmax_rejects_a_fully_masked_vector():
    """Все -inf — это тупик грамматики, а не распределение из nan."""
    with pytest.raises(ValueError):
        softmax([NEG_INF, NEG_INF])


# -------------------------------------------------------------- pattern_fsm
def test_pattern_fsm_accepts_only_after_the_whole_shape():
    fsm = pattern_fsm("dd")
    assert fsm["initial_state"] == 0
    assert fsm["accepts"] == {2}


def test_pattern_fsm_digit_position_allows_ten_symbols():
    fsm = pattern_fsm("d")
    assert sorted(fsm["transitions"][0]) == list(DIGITS)


def test_pattern_fsm_literal_position_allows_only_itself():
    assert pattern_fsm("d-d")["transitions"][1] == {"-": 2}


def test_pattern_fsm_accepting_state_has_an_empty_row():
    """Без пустой записи valid_tokens на финале свалился бы вместо []."""
    assert pattern_fsm("dd")["transitions"][2] == {}


# -------------------------------------------------------------- valid_tokens
def test_valid_tokens_at_a_literal_position_is_one_symbol():
    assert valid_tokens(pattern_fsm("d-d"), 1) == ["-"]


def test_valid_tokens_in_accepting_state_is_empty():
    assert valid_tokens(pattern_fsm("dd"), 2) == []


def test_valid_tokens_is_sorted_for_reproducibility():
    assert valid_tokens(pattern_fsm("dd"), 0) == list(DIGITS)


def test_valid_tokens_rejects_an_unknown_state():
    """Неизвестное состояние — баг в коде, а не «грамматика закончилась»."""
    with pytest.raises(ValueError):
        valid_tokens(pattern_fsm("dd"), 99)


# ---------------------------------------------------------------- transition
def test_transition_advances_on_an_allowed_token():
    assert transition(pattern_fsm("dd"), 0, "7") == 1


def test_transition_returns_none_on_a_forbidden_token():
    assert transition(pattern_fsm("d-d"), 1, "5") is None


def test_transition_distinguishes_state_zero_from_refusal():
    """Ловушка: состояние 0 ложно в булевом смысле, но это не отказ."""
    looping = {"initial_state": 0, "transitions": {0: {"a": 0}}, "accepts": {1}}
    assert transition(looping, 0, "a") == 0
    assert transition(looping, 0, "a") is not None


# ----------------------------------------------------------------- is_accept
def test_is_accept_is_true_in_the_final_state():
    assert is_accept(pattern_fsm("dd"), 2) is True


def test_is_accept_is_false_halfway_through():
    assert is_accept(pattern_fsm("dd"), 1) is False


def test_is_accept_supports_several_accepting_states():
    """У enum из трёх вариантов принимающих состояний тоже три."""
    enum_fsm = {"initial_state": 0, "transitions": {}, "accepts": {3, 5, 8}}
    assert is_accept(enum_fsm, 5) is True
    assert is_accept(enum_fsm, 4) is False


# ------------------------------------------------------- generate_constrained
def test_generate_constrained_ignores_a_loud_forbidden_token():
    """Модель тянет к дефису изо всех сил, а маска всё равно не пускает."""
    fsm = pattern_fsm("ddd-ddd-dddd")
    outs = [
        generate_constrained(
            fsm, PHONE_ALPHABET, favour(PHONE_ALPHABET, "-", 50.0), random.Random(s)
        )
        for s in range(30)
    ]
    assert all(len(o) == 12 for o in outs)
    assert all(o.count("-") == 2 for o in outs)
    assert all(o[3] == "-" and o[7] == "-" for o in outs)
    assert all(c in DIGITS for o in outs for c in o.replace("-", ""))


def test_generate_constrained_is_reproducible_with_the_same_seed():
    fsm = pattern_fsm("dddd")
    args = (fsm, PHONE_ALPHABET, uniform_logits(PHONE_ALPHABET))
    assert generate_constrained(*args, random.Random(7)) == generate_constrained(
        *args, random.Random(7)
    )


def test_generate_constrained_actually_samples():
    """Разные seed обязаны давать разные строки, иначе это не сэмплирование."""
    fsm = pattern_fsm("dddd")
    outs = {
        generate_constrained(
            fsm, PHONE_ALPHABET, uniform_logits(PHONE_ALPHABET), random.Random(s)
        )
        for s in range(20)
    }
    assert len(outs) > 1


def test_generate_constrained_rejects_a_token_missing_from_the_alphabet():
    fsm = pattern_fsm("d-d")
    only_digits = list(DIGITS)
    with pytest.raises(ValueError):
        generate_constrained(
            fsm, only_digits, uniform_logits(only_digits), random.Random(0)
        )


def test_generate_constrained_stops_on_a_looping_grammar():
    """Без max_steps цикл висел бы вечно."""
    looping = {"initial_state": 0, "transitions": {0: {"a": 0}}, "accepts": {1}}
    with pytest.raises(ValueError):
        generate_constrained(looping, ["a"], uniform_logits(["a"]), random.Random(0), 5)


def test_generate_constrained_reports_a_dead_end():
    """Непринимающее состояние без валидных токенов — тупик, а не пустая строка."""
    dead = {"initial_state": 0, "transitions": {0: {}}, "accepts": {1}}
    with pytest.raises(ValueError):
        generate_constrained(dead, ["a"], uniform_logits(["a"]), random.Random(0))


# --------------------------------------------------------- check_field_order
def test_reasoning_before_answer_is_accepted():
    assert check_field_order(["reasoning", "answer"]) is True


def test_answer_before_reasoning_is_rejected():
    with pytest.raises(ValueError):
        check_field_order(["answer", "reasoning"])


def test_answer_without_any_reasoning_is_rejected():
    """Ответ первым полем — решение принято до единой мысли."""
    with pytest.raises(ValueError):
        check_field_order(["answer"])


def test_schema_without_answer_fields_is_accepted():
    assert check_field_order(["vendor", "total_usd", "line_items"]) is True


def test_all_answer_synonyms_are_caught():
    for name in ("answer", "decision", "verdict", "label"):
        with pytest.raises(ValueError):
            check_field_order([name, "reasoning"])


