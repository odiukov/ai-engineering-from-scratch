"""Тесты к уроку «Распознавание именованных сущностей». Правь exercise.py."""

from itertools import product

import pytest

from exercise import (
    bio_to_spans,
    constrained_decode,
    entity_f1,
    is_valid_bio,
    rule_based_ner,
    spans_to_bio,
    token_features,
    word_shape,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

TOKENS = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]


# ---------------------------------------------------------- spans_to_bio
def test_spans_to_bio_marks_the_first_token_with_b_and_the_rest_with_i():
    assert spans_to_bio(["New", "York", "is", "big"], [(0, 2, "GPE")]) == [
        "B-GPE", "I-GPE", "O", "O",
    ]


def test_spans_to_bio_without_spans_is_all_outside():
    assert spans_to_bio(TOKENS, []) == ["O"] * 7


def test_spans_to_bio_end_is_exclusive():
    """Граница как в срезах Python: (0, 1) — это ровно один токен."""
    assert spans_to_bio(["a", "b"], [(0, 1, "ORG")]) == ["B-ORG", "O"]


def test_spans_to_bio_handles_several_entities_of_different_types():
    labels = spans_to_bio(TOKENS, [(0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "PRODUCT")])
    assert labels == ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]


def test_spans_to_bio_refuses_overlapping_entities():
    """Ловушка: вложенные сущности — то, что BIO не умеет выразить в принципе."""
    with pytest.raises(ValueError):
        spans_to_bio(["Bank", "of", "America", "Tower"], [(0, 3, "ORG"), (0, 4, "FAC")])


# ---------------------------------------------------------- bio_to_spans
def test_bio_to_spans_recovers_the_lesson_example():
    labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
    assert bio_to_spans(labels) == [(0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "PRODUCT")]


def test_bio_to_spans_joins_a_multi_token_entity():
    assert bio_to_spans(["B-GPE", "I-GPE", "I-GPE", "O"]) == [(0, 3, "GPE")]


def test_bio_to_spans_closes_an_entity_at_the_end_of_the_sentence():
    """Ловушка: последняя сущность закрывается уже после цикла."""
    assert bio_to_spans(["O", "B-ORG", "I-ORG"]) == [(1, 3, "ORG")]


def test_bio_to_spans_does_not_continue_across_a_type_change():
    """«I-GPE» после «B-ORG» — не продолжение, это два разных решения модели."""
    assert bio_to_spans(["B-ORG", "I-GPE"]) == [(0, 1, "ORG")]


def test_bio_to_spans_ignores_a_stray_i_after_o():
    assert bio_to_spans(["O", "I-ORG", "O"]) == []


def test_bio_round_trips_through_spans():
    spans = [(0, 1, "ORG"), (2, 4, "GPE")]
    assert bio_to_spans(spans_to_bio(TOKENS, spans)) == spans


# ----------------------------------------------------------- is_valid_bio
def test_valid_bio_accepts_a_well_formed_entity():
    assert is_valid_bio(["B-ORG", "I-ORG", "O"]) is True


def test_valid_bio_rejects_i_right_after_o():
    """Классический признак сломанной разметки: продолжать нечего."""
    assert is_valid_bio(["O", "I-ORG"]) is False


def test_valid_bio_rejects_i_in_the_very_first_position():
    assert is_valid_bio(["I-ORG", "O"]) is False


def test_valid_bio_rejects_a_type_switch_inside_an_entity():
    assert is_valid_bio(["B-ORG", "I-GPE"]) is False


def test_valid_bio_rejects_labels_that_are_not_bio_at_all():
    assert is_valid_bio(["ORG"]) is False


def test_valid_bio_accepts_an_empty_sequence():
    assert is_valid_bio([]) is True


# -------------------------------------------------------------- word_shape
def test_word_shape_of_a_camel_case_word():
    assert word_shape("iPhone") == "xXxxxx"


def test_word_shape_keeps_punctuation_and_digits_apart():
    assert word_shape("USA-2024") == "XXX-dddd"


def test_word_shape_separates_dotted_and_plain_acronyms():
    """Ловушка: точки остаются собой, иначе U.S.A. и USA станут одним признаком."""
    assert word_shape("U.S.A.") != word_shape("USA")


def test_word_shape_of_an_empty_word_is_empty():
    assert word_shape("") == ""


# ---------------------------------------------------------- token_features
def test_token_features_marks_the_sentence_start():
    """Ловушка: отсутствующий сосед — это <BOS>, а не пустая строка."""
    assert token_features("Apple", None, "sued")["prev_lower"] == "<BOS>"


def test_token_features_marks_the_sentence_end():
    assert token_features("sales", "iPhone", None)["next_lower"] == "<EOS>"


def test_token_features_reuses_word_shape():
    assert token_features("iPhone", "over", "sales")["shape"] == word_shape("iPhone")


def test_token_features_of_a_number():
    f = token_features("2024", "in", ".")
    assert f["has_digit"] is True
    assert f["is_title"] is False


def test_token_features_suffix_of_a_short_word_is_the_whole_word():
    assert token_features("US", None, None)["suffix_3"] == "us"


def test_token_features_distinguishes_title_case_from_all_caps():
    assert token_features("Apple", None, None)["is_title"] is True
    assert token_features("APPLE", None, None)["is_upper"] is True
    assert token_features("APPLE", None, None)["is_title"] is False


# --------------------------------------------------------- rule_based_ner
GAZ = {
    "ORG": {"Apple", "Google"},
    "GPE": {"US", "France"},
    "PRODUCT": {"iPhone"},
}


def test_rule_based_ner_labels_known_names():
    assert rule_based_ner(TOKENS, GAZ) == [
        "B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O",
    ]


def test_rule_based_ner_labels_nothing_it_has_never_seen():
    """Нулевое покрытие новых имён — главная слабость словарного подхода."""
    assert rule_based_ner(["Anthropic", "shipped"], GAZ) == ["O", "O"]


def test_rule_based_ner_cannot_disambiguate_a_word():
    """«Apple» фрукт получает B-ORG: у словаря нет контекста, и не будет."""
    assert rule_based_ner(["I", "ate", "an", "Apple"], GAZ)[-1] == "B-ORG"


def test_rule_based_ner_priority_follows_the_gazetteer_order():
    """Ловушка: слово в двух словарях. Порядок ключей и есть приоритет."""
    org_first = {"ORG": {"Amazon"}, "GPE": {"Amazon"}}
    gpe_first = {"GPE": {"Amazon"}, "ORG": {"Amazon"}}
    assert rule_based_ner(["Amazon"], org_first) == ["B-ORG"]
    assert rule_based_ner(["Amazon"], gpe_first) == ["B-GPE"]


def test_rule_based_ner_splits_a_multi_token_entity_into_two():
    """Ловушка: функция не выдаёт I-, поэтому «New York» распадается на две сущности."""
    gaz = {"GPE": {"New", "York"}}
    assert bio_to_spans(rule_based_ner(["New", "York"], gaz)) == [
        (0, 1, "GPE"), (1, 2, "GPE"),
    ]


# --------------------------------------------------------------- entity_f1
def test_entity_f1_of_a_perfect_prediction_is_one():
    labels = ["B-ORG", "I-ORG", "O"]
    assert entity_f1(labels, labels)["f1"] == APPROX(1.0)


def test_entity_f1_gives_no_partial_credit_for_a_wrong_boundary():
    """Ловушка: по токенам это 2 из 3 верных, по сущностям — ноль."""
    assert entity_f1(["B-ORG", "I-ORG", "O"], ["B-ORG", "O", "O"])["f1"] == APPROX(0.0)


def test_entity_f1_gives_no_credit_for_the_wrong_type():
    assert entity_f1(["B-ORG"], ["B-GPE"])["f1"] == APPROX(0.0)


def test_entity_f1_of_an_empty_prediction_is_zero_not_a_crash():
    m = entity_f1(["B-ORG"], ["O"])
    assert m["precision"] == APPROX(0.0)
    assert m["recall"] == APPROX(0.0)


def test_entity_f1_separates_precision_from_recall():
    """Нашли одну из двух, но всё найденное верно: recall 0.5, precision 1.0."""
    m = entity_f1(["B-ORG", "O", "B-GPE"], ["B-ORG", "O", "O"])
    assert m["precision"] == APPROX(1.0)
    assert m["recall"] == APPROX(0.5)


# -------------------------------------------------------- constrained_decode
def test_constrained_decode_takes_the_argmax_when_it_is_already_valid():
    scores = [{"O": 0.1, "B-ORG": 0.9}, {"O": 0.2, "I-ORG": 0.8}]
    assert constrained_decode(scores) == ["B-ORG", "I-ORG"]


def test_constrained_decode_refuses_i_in_the_first_position():
    """Ловушка: жадный argmax взял бы I-ORG, а такой последовательности не бывает."""
    assert constrained_decode([{"O": 1.0, "I-ORG": 5.0}]) == ["O"]


def test_constrained_decode_never_returns_an_invalid_sequence():
    scores = [
        {"O": 0.9, "B-ORG": 0.1, "I-ORG": 0.4},
        {"O": 0.1, "B-ORG": 0.2, "I-ORG": 0.9},
        {"O": 0.3, "B-GPE": 0.2, "I-ORG": 0.8},
    ]
    assert is_valid_bio(constrained_decode(scores)) is True


def test_constrained_decode_beats_greedy_on_total_score():
    """Смысл динамики: иногда выгоднее уступить на первом токене ради второго."""
    scores = [
        {"O": 0.6, "B-ORG": 0.5},
        {"O": 0.1, "I-ORG": 0.9},
    ]
    greedy = ["O", "I-ORG"]
    assert is_valid_bio(greedy) is False
    assert constrained_decode(scores) == ["B-ORG", "I-ORG"]


def test_constrained_decode_matches_brute_force_search():
    scores = [
        {"O": 0.31, "B-ORG": 0.72, "I-ORG": 0.15},
        {"O": 0.44, "B-ORG": 0.23, "I-ORG": 0.61},
        {"O": 0.52, "B-GPE": 0.37, "I-ORG": 0.48},
    ]
    labels = [list(s) for s in scores]
    valid = [seq for seq in product(*labels) if is_valid_bio(list(seq))]
    best = max(valid, key=lambda seq: sum(s[l] for s, l in zip(scores, seq)))
    assert constrained_decode(scores) == list(best)


def test_constrained_decode_of_an_empty_sentence_is_empty():
    assert constrained_decode([]) == []


def test_constrained_decode_raises_when_no_valid_path_exists():
    """Ловушка: остались одни I- в первой позиции — валидного пути нет вообще."""
    with pytest.raises(ValueError):
        constrained_decode([{"I-ORG": 1.0}])
