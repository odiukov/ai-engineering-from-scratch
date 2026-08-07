"""Тесты к уроку «Coreference resolution». Правь exercise.py."""

import pytest

from exercise import (
    agreement_score,
    build_clusters,
    extract_mentions,
    muc_f1,
    recency_score,
    resolve_document,
    resolve_pronouns,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

INF = float("-inf")


def f(gender, number, start=0):
    """Мини-mention для тестов согласования: только то, что читают функции."""
    return {"gender": gender, "number": number, "start": start}


# -------------------------------------------------------- extract_mentions
def test_mention_spans_slice_back_to_mention_text():
    """Главный инвариант span-офсетов: text[start:end] == текст mention-а."""
    text = "Tim Cook runs the company. He said it would ship."
    for m in extract_mentions(text):
        assert text[m["start"]:m["end"]] == m["text"]


def test_named_entity_carries_inferred_gender():
    mary, john = extract_mentions("Mary called John.")
    assert (mary["text"], mary["type"], mary["gender"]) == ("Mary", "ne", "f")
    assert (john["text"], john["type"], john["gender"]) == ("John", "ne", "m")


def test_multiword_name_is_a_single_mention():
    mentions = extract_mentions("Tim Cook smiled.")
    assert [m["text"] for m in mentions] == ["Tim Cook"]


def test_names_across_a_sentence_boundary_are_not_merged():
    """«John. Steve» — два имени, а не одно: между ними точка, а не пробел."""
    mentions = extract_mentions("Mary saw John. Steve waved.")
    assert [m["text"] for m in mentions] == ["Mary", "John", "Steve"]


def test_definite_description_is_a_nominal_mention():
    (nominal,) = extract_mentions("the company grew")
    assert (nominal["text"], nominal["type"], nominal["number"]) == ("the company", "nominal", "sg")


def test_pronoun_carries_number_and_gender():
    (pronoun,) = extract_mentions("They arrived.")
    assert (pronoun["type"], pronoun["gender"], pronoun["number"]) == ("pronoun", "u", "pl")


# --------------------------------------------------------- agreement_score
def test_exact_gender_match_beats_wildcard_match():
    assert agreement_score(f("f", "sg"), f("f", "sg")) == APPROX(2.0)
    assert agreement_score(f("n", "sg"), f("u", "sg")) == APPROX(1.0)


def test_gender_clash_is_incompatible():
    assert agreement_score(f("f", "sg"), f("m", "sg")) == INF


def test_number_clash_is_incompatible():
    """"they" не может указывать на одиночную "Mary", как бы близко она ни стояла."""
    assert agreement_score(f("u", "pl"), f("f", "sg")) == INF


def test_unknown_gender_is_a_wildcard_not_a_fourth_gender():
    """"u" совместим и с "m", и с "f" — но бонуса за точное совпадение не даёт."""
    assert agreement_score(f("u", "sg"), f("m", "sg")) == APPROX(1.0)
    assert agreement_score(f("u", "sg"), f("f", "sg")) == APPROX(1.0)
    assert agreement_score(f("u", "sg"), f("u", "sg")) == APPROX(2.0)


# ----------------------------------------------------------- recency_score
def test_recency_grows_as_the_candidate_gets_closer():
    mention = f("m", "sg", start=100)
    assert recency_score(mention, f("m", "sg", 0)) < recency_score(mention, f("m", "sg", 90))


def test_recency_stays_inside_zero_one():
    mention = f("m", "sg", start=1000)
    for start in (0, 1, 500, 999):
        assert 0.0 < recency_score(mention, f("m", "sg", start)) <= 1.0


def test_recency_never_outweighs_exact_agreement():
    """Согласование — жёсткое правило, близость — только тай-брейк."""
    mention = f("f", "sg", start=1000)
    far_exact = f("f", "sg", 0)
    near_wildcard = f("u", "sg", 999)
    assert (agreement_score(mention, far_exact) + recency_score(mention, far_exact)
            > agreement_score(mention, near_wildcard) + recency_score(mention, near_wildcard))


def test_candidate_that_starts_later_raises_value_error():
    with pytest.raises(ValueError):
        recency_score(f("m", "sg", 5), f("m", "sg", 5))


# -------------------------------------------------------- resolve_pronouns
def test_pronoun_links_to_the_nearest_compatible_antecedent():
    """Оба кандидата мужского рода — побеждает ближний, а не первый."""
    mentions = extract_mentions("John called Steve. He answered.")
    assert resolve_pronouns(mentions) == [(2, 1)]
    assert mentions[1]["text"] == "Steve"


def test_antecedent_with_clashing_gender_is_skipped():
    """Ближе стоит John, но "She" женского рода — берётся дальняя Mary."""
    mentions = extract_mentions("Mary called John. She was late.")
    assert resolve_pronouns(mentions) == [(2, 0)]


def test_plural_pronoun_ignores_a_singular_name():
    mentions = extract_mentions("Mary met the engineers. They left.")
    (_, antecedent), = resolve_pronouns(mentions)
    assert mentions[antecedent]["text"] == "the engineers"


def test_cataphora_leaves_the_pronoun_unresolved():
    """«When she walked in, Mary smiled» — референт справа, антецедента нет."""
    assert resolve_pronouns(extract_mentions("When she walked in, Mary smiled.")) == [(0, None)]


def test_a_pronoun_is_never_an_antecedent_for_another_pronoun():
    mentions = extract_mentions("Mary waved. She smiled. She left.")
    assert resolve_pronouns(mentions) == [(1, 0), (2, 0)]


# ----------------------------------------------------------- build_clusters
def test_clusters_are_a_transitive_closure():
    """2~0 и 3~2 дают один кластер из трёх, а не две отдельные пары."""
    assert build_clusters(4, [(2, 0), (3, 2)]) == [[0, 2, 3], [1]]


def test_clusters_partition_every_mention_exactly_once():
    clusters = build_clusters(6, [(1, 0), (3, 2), (4, 3)])
    seen = [i for c in clusters for i in c]
    assert sorted(seen) == list(range(6))
    assert len(seen) == len(set(seen))


def test_no_links_gives_only_singletons():
    assert build_clusters(3, []) == [[0], [1], [2]]


def test_none_antecedent_creates_no_link():
    assert build_clusters(2, [(1, None)]) == [[0], [1]]


def test_index_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        build_clusters(2, [(0, 7)])


# --------------------------------------------------------- resolve_document
def test_repeated_pronoun_joins_one_cluster():
    text = "Mary called John. She was late. She apologized."
    assert resolve_document(text) == [["Mary", "She", "She"]]


def test_document_without_pronouns_has_no_clusters():
    assert resolve_document("Mary called John.") == []


def test_two_entities_stay_in_two_clusters():
    text = "Tim Cook runs the company. He said it would ship."
    assert resolve_document(text) == [["Tim Cook", "He"], ["the company", "it"]]


# ------------------------------------------------------------------ muc_f1
def test_perfect_prediction_scores_one():
    assert muc_f1([[0, 1, 2]], [[0, 1, 2]]) == pytest.approx((1.0, 1.0, 1.0))


def test_singleton_explosion_scores_zero_instead_of_dividing_by_zero():
    """Каждый mention сам себе кластер: знаменатель precision — ноль."""
    assert muc_f1([[0], [1], [2]], [[0, 1, 2]]) == pytest.approx((0.0, 0.0, 0.0))


def test_over_merging_keeps_recall_and_drops_precision():
    precision, recall, _ = muc_f1([[0, 1, 2, 3]], [[0, 1], [2, 3]])
    assert recall == APPROX(1.0)
    assert precision < 1.0


def test_swapping_the_arguments_swaps_precision_and_recall():
    p, r, f1 = muc_f1([[0, 1, 2, 3]], [[0, 1], [2, 3]])
    p2, r2, f2 = muc_f1([[0, 1], [2, 3]], [[0, 1, 2, 3]])
    assert (p2, r2, f2) == pytest.approx((r, p, f1))


def test_f1_lies_between_precision_and_recall():
    precision, recall, f1 = muc_f1([[0, 1, 2, 3]], [[0, 1], [2, 3]])
    assert min(precision, recall) <= f1 <= max(precision, recall)
