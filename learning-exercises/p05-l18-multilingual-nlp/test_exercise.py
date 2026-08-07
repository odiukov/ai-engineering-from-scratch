"""Тесты к уроку «Многоязычный NLP». Правь exercise.py."""

import pytest

from exercise import (
    cosine_similarity,
    cross_lingual_retrieve,
    language_similarity,
    per_language_accuracy,
    rank_source_languages,
    subword_segment,
    tokenization_fertility,
    zero_shot_classify,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# «The cat is sleeping» и её переводы лежат рядом, «The dog is barking» — нет
EN_CAT = [1.0, 0.1, 0.0]
FR_CHAT = [0.95, 0.2, 0.0]
ES_GATO = [0.92, 0.25, 0.0]
EN_DOG = [0.0, 0.1, 1.0]

DOCS = [("fr:chat", FR_CHAT), ("en:dog", EN_DOG), ("es:gato", ES_GATO)]

SENTIMENT = {"positive": [1.0, 0.0, 0.0], "negative": [0.0, 0.0, 1.0]}

SUBWORDS = {"anti", "bio", "tico", "body", "establish", "ment"}

LATIN = {"the", "cat", "is", "sleep", "ing"}

# признаки WALS: порядок слов, наличие рода, наличие падежей, наличие артиклей
ENGLISH = {"word_order": "SVO", "gender": "no", "case": "no", "article": "yes"}
GERMAN = {"word_order": "SVO", "gender": "yes", "case": "yes", "article": "yes"}
RUSSIAN = {"word_order": "SVO", "gender": "yes", "case": "yes", "article": "no"}
POLISH = {"word_order": "SVO", "gender": "yes", "case": "yes", "article": "no"}


# ------------------------------------------------------- cosine_similarity
def test_cosine_similarity_of_a_vector_with_itself_is_one():
    assert cosine_similarity([3.0, 4.0], [3.0, 4.0]) == APPROX(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_similarity_ignores_vector_length():
    """Нормировка на длину: важно направление, а не масштаб."""
    assert cosine_similarity([1.0, 0.0], [7.0, 0.0]) == APPROX(1.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_cosine_similarity_of_a_zero_vector_is_zero_not_a_crash():
    """Ловушка: делить на нулевую длину нельзя."""
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == APPROX(0.0)


# --------------------------------------------------- cross_lingual_retrieve
def test_cross_lingual_retrieve_finds_the_translation_not_the_other_language_word():
    """Запрос по-английски достаёт французский документ: язык не мешает."""
    assert cross_lingual_retrieve(EN_CAT, DOCS, top_k=1)[0][0] == "fr:chat"


def test_cross_lingual_retrieve_puts_the_unrelated_sentence_last():
    labels = [label for label, _ in cross_lingual_retrieve(EN_CAT, DOCS, top_k=3)]
    assert labels[-1] == "en:dog"


def test_cross_lingual_retrieve_respects_top_k():
    assert len(cross_lingual_retrieve(EN_CAT, DOCS, top_k=2)) == 2


def test_cross_lingual_retrieve_keeps_the_original_order_on_a_tie():
    same = [("first", [1.0, 0.0]), ("second", [2.0, 0.0])]
    assert [label for label, _ in cross_lingual_retrieve([1.0, 0.0], same)] == [
        "first",
        "second",
    ]


# ------------------------------------------------------- zero_shot_classify
def test_zero_shot_classify_returns_a_distribution():
    assert sum(zero_shot_classify(EN_CAT, SENTIMENT).values()) == APPROX(1.0)


def test_zero_shot_classify_puts_the_nearest_label_first():
    result = zero_shot_classify(EN_CAT, SENTIMENT)
    assert list(result)[0] == "positive"


def test_zero_shot_classify_gives_equal_mass_to_equally_close_labels():
    labels = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    result = zero_shot_classify([0.3, 0.7], labels)
    assert result["a"] == APPROX(result["b"])


def test_zero_shot_classify_agrees_across_languages():
    """Тот самый zero-shot transfer: перевод получает ту же метку без разметки."""
    english = zero_shot_classify(EN_CAT, SENTIMENT)
    french = zero_shot_classify(FR_CHAT, SENTIMENT)
    assert list(english)[0] == list(french)[0]


def test_zero_shot_classify_without_labels_is_empty():
    assert zero_shot_classify(EN_CAT, {}) == {}


# ---------------------------------------------------------- subword_segment
def test_subword_segment_takes_the_longest_match_first():
    assert subword_segment("antiestablishment", SUBWORDS) == [
        "anti",
        "establish",
        "ment",
    ]


def test_subword_segment_prefers_a_longer_piece_over_a_shorter_one():
    assert subword_segment("abc", {"a", "ab", "abc"}) == ["abc"]


def test_subword_segment_falls_back_to_single_characters():
    """Byte fallback: неизвестного слова не существует, есть только дорогое."""
    assert subword_segment("xyz", SUBWORDS) == ["x", "y", "z"]


def test_subword_segment_always_joins_back_to_the_original_word():
    for word in ["antibody", "antibiotico", "xyz", "establishment"]:
        assert "".join(subword_segment(word, SUBWORDS)) == word


def test_subword_segment_shares_a_morpheme_across_languages():
    """Общий словарь: "anti" — один и тот же токен в английском и итальянском."""
    assert subword_segment("antibody", SUBWORDS)[0] == "anti"
    assert subword_segment("antibiotico", SUBWORDS)[0] == "anti"


# ---------------------------------------------------- tokenization_fertility
def test_tokenization_fertility_counts_pieces_per_word():
    assert tokenization_fertility(["cat sleeping"], LATIN) == APPROX(1.5)


def test_tokenization_fertility_shows_the_tax_on_an_uncovered_script():
    """Налог на токенизацию: то же предложение стоит в разы больше токенов."""
    covered = tokenization_fertility(["the cat is sleeping"], LATIN)
    uncovered = tokenization_fertility(["बिल्ली सो"], LATIN)
    assert uncovered / covered >= 3.0


def test_tokenization_fertility_drops_when_the_vocabulary_covers_the_language():
    """Данными это не лечится, словарём лечится."""
    before = tokenization_fertility(["बिल्ली सो"], LATIN)
    after = tokenization_fertility(["बिल्ली सो"], LATIN | {"बिल्ली", "सो"})
    assert after == APPROX(1.0)
    assert after < before


def test_tokenization_fertility_of_no_words_is_zero():
    assert tokenization_fertility(["", "   "], LATIN) == APPROX(0.0)


# ------------------------------------------------------- language_similarity
def test_language_similarity_counts_matching_features():
    assert language_similarity(RUSSIAN, GERMAN) == APPROX(0.75)


def test_language_similarity_of_a_language_with_itself_is_one():
    assert language_similarity(RUSSIAN, RUSSIAN) == APPROX(1.0)


def test_language_similarity_uses_only_features_present_in_both():
    a = {"word_order": "SVO", "gender": "yes"}
    b = {"word_order": "SVO"}
    assert language_similarity(a, b) == APPROX(1.0)


def test_language_similarity_without_shared_features_is_zero():
    assert language_similarity(RUSSIAN, {}) == APPROX(0.0)


# ---------------------------------------------------- rank_source_languages
CANDIDATES = {
    "english": (ENGLISH, 10 ** 9),
    "german": (GERMAN, 10 ** 8),
    "russian": (RUSSIAN, 10 ** 7),
}


def test_rank_source_languages_by_typology_alone_beats_english():
    """Тезис урока: для славянской цели русский обгоняет английский."""
    ranked = rank_source_languages(POLISH, CANDIDATES, weight=1.0)
    assert ranked[0][0] == "russian"


def test_rank_source_languages_by_corpus_size_alone_picks_english():
    """Как делают по умолчанию — и почему это часто неверно."""
    ranked = rank_source_languages(POLISH, CANDIDATES, weight=0.0)
    assert ranked[0][0] == "english"


def test_rank_source_languages_returns_scores_in_descending_order():
    scores = [score for _, score in rank_source_languages(POLISH, CANDIDATES)]
    assert scores == sorted(scores, reverse=True)


def test_rank_source_languages_without_candidates_is_empty():
    assert rank_source_languages(POLISH, {}) == []


# ---------------------------------------------------- per_language_accuracy
def test_per_language_accuracy_splits_by_language():
    records = [("en", 1, 1), ("en", 0, 1), ("hi", 0, 1)]
    assert per_language_accuracy(records) == {"en": APPROX(0.5), "hi": APPROX(0.0)}


def test_per_language_accuracy_exposes_what_the_aggregate_hides():
    """19 из 20 на английском и 0 из 1 на хинди дают агрегат 0.9 и тихий провал."""
    records = [("en", 1, 1)] * 19 + [("en", 0, 1)] + [("hi", 0, 1)]
    aggregate = sum(1 for _, p, g in records if p == g) / len(records)
    by_language = per_language_accuracy(records)
    assert aggregate > 0.9
    assert by_language["hi"] == APPROX(0.0)


def test_per_language_accuracy_sorts_its_keys():
    records = [("sw", 1, 1), ("am", 1, 1), ("hi", 1, 1)]
    assert list(per_language_accuracy(records)) == ["am", "hi", "sw"]


def test_per_language_accuracy_of_no_records_is_empty():
    assert per_language_accuracy([]) == {}
