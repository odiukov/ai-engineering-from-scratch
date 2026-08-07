"""Тесты к уроку «Entity linking и разрешение неоднозначности». Правь exercise.py."""

import pytest

from exercise import (
    build_alias_index,
    candidates,
    disambiguate,
    evaluate_linker,
    jaccard,
    link_with_nil,
    mention_recall,
    tokenize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Мини-KB: порядок кандидатов = prior, первым идёт самый популярный.
INDEX = {
    "jordan": ["Q41421", "Q810", "Q254110"],
    "paris": ["Q90", "Q663094", "Q55411"],
    "apple": ["Q312", "Q89"],
}

DESC = {
    "Q41421": "Michael Jordan basketball player Chicago Bulls NBA champion",
    "Q810": "Jordan sovereign country Middle East Amman kingdom desert",
    "Q254110": "Michael B Jordan american actor film Creed",
    "Q90": "Paris capital city of France Eiffel tower Seine river",
    "Q663094": "Paris small city in Texas United States",
    "Q55411": "Paris Hilton socialite media personality heiress",
    "Q312": "Apple Inc technology company iPhone Cupertino",
    "Q89": "apple fruit of the tree orchard harvest",
}


# ----------------------------------------------------------------- tokenize
def test_tokenize_strips_punctuation():
    assert tokenize("Paris, France!") == ["paris", "france"]


def test_tokenize_lowercases():
    assert tokenize("NBA Finals") == ["nba", "finals"]


def test_tokenize_keeps_digits():
    assert tokenize("NBA 1998 Finals") == ["nba", "1998", "finals"]


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("") == []


# -------------------------------------------------------- build_alias_index
def test_alias_index_folds_case_of_the_key():
    index = build_alias_index([("Paris", "Q90"), ("paris", "Q663094")])
    assert index == {"paris": ["Q90", "Q663094"]}


def test_alias_index_preserves_prior_order():
    """Порядок пар — это популярность, и он обязан дожить до кандидатов."""
    index = build_alias_index([("Jordan", "Q810"), ("Jordan", "Q41421")])
    assert index["jordan"] == ["Q810", "Q41421"]


def test_alias_index_drops_duplicate_entities():
    index = build_alias_index([("Apple", "Q312"), ("APPLE", "Q312"), ("apple", "Q89")])
    assert index["apple"] == ["Q312", "Q89"]


def test_alias_index_of_nothing_is_empty():
    assert build_alias_index([]) == {}


# ------------------------------------------------------------- candidates
def test_candidates_finds_a_known_alias():
    assert candidates(INDEX, "Jordan") == ["Q41421", "Q810", "Q254110"]


def test_candidates_ignores_case_and_surrounding_spaces():
    assert candidates(INDEX, "  PARIS ") == ["Q90", "Q663094", "Q55411"]


def test_candidates_of_unknown_alias_is_empty():
    assert candidates(INDEX, "Berlin") == []


def test_candidates_does_not_expose_the_index_itself():
    """Ловушка: вернув сам список, ты даёшь вызывающему испортить KB."""
    got = candidates(INDEX, "apple")
    got.append("Q999")
    assert INDEX["apple"] == ["Q312", "Q89"]


# ---------------------------------------------------------------- jaccard
def test_jaccard_of_identical_sets_is_one():
    assert jaccard(["a", "b"], ["b", "a"]) == APPROX(1.0)


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard(["a"], ["b"]) == APPROX(0.0)


def test_jaccard_of_two_empty_sets_is_zero_not_an_error():
    """Пустое объединение — деление на ноль. Договорённость: 0.0."""
    assert jaccard([], []) == APPROX(0.0)


def test_jaccard_is_symmetric():
    assert jaccard(["a", "b"], ["b", "c"]) == APPROX(jaccard(["b", "c"], ["a", "b"]))


def test_jaccard_ignores_repeated_tokens():
    assert jaccard(["a", "a", "a"], ["a"]) == APPROX(1.0)


# ------------------------------------------------------------ disambiguate
def test_disambiguate_picks_the_country_in_a_travel_context():
    entity, _ = disambiguate("Jordan", "a kingdom in the Middle East, Amman", INDEX, DESC)
    assert entity == "Q810"


def test_disambiguate_picks_the_player_in_a_sports_context():
    entity, _ = disambiguate("Jordan", "the Chicago Bulls basketball champion", INDEX, DESC)
    assert entity == "Q41421"


def test_disambiguate_separates_the_company_from_the_fruit():
    company, _ = disambiguate("Apple", "the iPhone company in Cupertino", INDEX, DESC)
    fruit, _ = disambiguate("Apple", "picked fruit from the orchard tree", INDEX, DESC)
    assert (company, fruit) == ("Q312", "Q89")


def test_disambiguate_of_unknown_mention_is_none():
    assert disambiguate("Nobody", "any text at all", INDEX, DESC) == (None, 0.0)


def test_disambiguate_falls_back_to_the_most_popular_on_a_tie():
    """Popularity bias: бессмысленный контекст — и система уверенно врёт."""
    entity, score = disambiguate("Jordan", "zzz qqq wwww", INDEX, DESC)
    assert entity == "Q41421"
    assert score == APPROX(0.0)


def test_disambiguate_score_stays_between_zero_and_one():
    _, score = disambiguate("Paris", "the capital of France", INDEX, DESC)
    assert 0.0 < score <= 1.0


# ----------------------------------------------------------- link_with_nil
def test_link_with_nil_returns_the_entity_above_threshold():
    assert link_with_nil("Paris", "the capital of France", INDEX, DESC, 0.05) == "Q90"


def test_link_with_nil_says_nothing_when_context_is_useless():
    """Ниже порога честнее вернуть NIL, чем самого популярного кандидата."""
    assert link_with_nil("Jordan", "zzz qqq wwww", INDEX, DESC, 0.05) is None


def test_link_with_nil_returns_none_for_unknown_mention():
    assert link_with_nil("Nobody", "the capital of France", INDEX, DESC) is None


def test_link_with_nil_threshold_of_zero_never_says_nil_for_known_alias():
    assert link_with_nil("Jordan", "zzz qqq wwww", INDEX, DESC, 0.0) == "Q41421"


# ---------------------------------------------------------- mention_recall
def test_mention_recall_is_one_when_every_gold_entity_is_a_candidate():
    examples = [("Paris", "France", "Q90"), ("Apple", "iPhone", "Q312")]
    assert mention_recall(examples, INDEX) == APPROX(1.0)


def test_mention_recall_drops_when_the_alias_is_missing():
    examples = [("Paris", "France", "Q90"), ("Berlin", "Germany", "Q64")]
    assert mention_recall(examples, INDEX) == APPROX(0.5)


def test_mention_recall_drops_when_the_alias_is_there_but_the_entity_is_not():
    """Алиас нашёлся, а нужного id в списке нет — это тоже промах recall."""
    assert mention_recall([("Paris", "France", "Q99999")], INDEX) == APPROX(0.0)


def test_mention_recall_of_no_examples_is_zero():
    assert mention_recall([], INDEX) == APPROX(0.0)


# --------------------------------------------------------- evaluate_linker
def test_evaluate_linker_on_a_perfect_run():
    examples = [
        ("Paris", "the capital of France with the Eiffel tower", "Q90"),
        ("Apple", "the iPhone technology company in Cupertino", "Q312"),
    ]
    report = evaluate_linker(examples, INDEX, DESC)
    assert report["mention_recall"] == APPROX(1.0)
    assert report["disambiguation_accuracy"] == APPROX(1.0)
    assert report["pipeline_accuracy"] == APPROX(1.0)


def test_pipeline_accuracy_is_the_product_of_recall_and_disambiguation():
    """Главное свойство урока: 99% на 80% recall — это 80% системы."""
    examples = [
        ("Paris", "the capital of France with the Eiffel tower", "Q90"),
        ("Apple", "the iPhone technology company in Cupertino", "Q312"),
        ("Berlin", "the capital of Germany", "Q64"),
        ("Jordan", "zzz qqq wwww", "Q810"),
    ]
    report = evaluate_linker(examples, INDEX, DESC)
    product = report["mention_recall"] * report["disambiguation_accuracy"]
    assert report["pipeline_accuracy"] == APPROX(product)


def test_disambiguation_accuracy_ignores_what_candidate_generation_missed():
    """Disambiguator нельзя штрафовать за то, чего ему не выдали."""
    good = [("Paris", "the capital of France with the Eiffel tower", "Q90")]
    with_miss = good + [("Berlin", "the capital of Germany", "Q64")]
    assert evaluate_linker(with_miss, INDEX, DESC)["disambiguation_accuracy"] == APPROX(
        evaluate_linker(good, INDEX, DESC)["disambiguation_accuracy"]
    )
    assert evaluate_linker(with_miss, INDEX, DESC)["mention_recall"] == APPROX(0.5)


def test_evaluate_linker_of_no_examples_is_all_zeros():
    report = evaluate_linker([], INDEX, DESC)
    assert report == {
        "mention_recall": APPROX(0.0),
        "disambiguation_accuracy": APPROX(0.0),
        "pipeline_accuracy": APPROX(0.0),
    }
