"""Тесты к уроку «Обработка текста: токенизация, стемминг, лемматизация».

Правь exercise.py.
"""

import pytest

from exercise import (
    lemmatize,
    preprocess,
    stem,
    stem_step_1a,
    stem_step_1b,
    tokenize,
    tokenize_with_urls,
)

# маленькая таблица лемм, ровно как в уроке
TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}


# --------------------------------------------------------------- tokenize
def test_tokenize_separates_words_and_punctuation():
    assert tokenize("The cats weren't running at 3pm.") == [
        "The", "cats", "weren't", "running", "at", "3", "pm", ".",
    ]


def test_tokenize_keeps_the_contraction_whole():
    """don't — один токен, а не do + ' + t."""
    assert tokenize("don't") == ["don't"]


def test_tokenize_splits_a_digit_letter_glue():
    """3pm распадается на число и слово: буквы и цифры матчатся разными шаблонами."""
    assert tokenize("3pm") == ["3", "pm"]


def test_tokenize_of_empty_text_is_empty_list():
    assert tokenize("") == []


def test_tokenize_loses_only_whitespace():
    """Свойство: склеенные токены равны исходному тексту без пробелов."""
    text = "Hello, world! It's 2026; ready?"
    assert "".join(tokenize(text)) == "".join(text.split())


def test_tokenize_gives_each_punctuation_mark_its_own_token():
    assert tokenize("wow!!") == ["wow", "!", "!"]


# ----------------------------------------------------- tokenize_with_urls
def test_url_survives_as_a_single_token():
    assert tokenize_with_urls("Visit https://example.com today.") == [
        "Visit", "https://example.com", "today", ".",
    ]


def test_sentence_final_period_does_not_stick_to_the_url():
    """Ловушка: жадный шаблон утащил бы точку внутрь ссылки."""
    tokens = tokenize_with_urls("go to http://a.io/x.")
    assert tokens[-2:] == ["http://a.io/x", "."]


def test_plain_tokenizer_shreds_the_url():
    """Контраст: без отдельного шаблона ссылка разлетается на куски."""
    assert len(tokenize("https://example.com")) > 1


def test_url_tokenizer_agrees_with_plain_one_when_there_are_no_links():
    text = "The cats weren't running at 3pm."
    assert tokenize_with_urls(text) == tokenize(text)


# ---------------------------------------------------------- stem_step_1a
def test_sses_loses_exactly_two_letters():
    assert stem_step_1a("caresses") == "caress"


def test_ies_collapses_to_i_not_to_y():
    """Документированное несовершенство: ponies -> poni."""
    assert stem_step_1a("ponies") == "poni"


def test_double_s_ending_is_left_alone():
    assert stem_step_1a("caress") == "caress"


def test_plain_plural_s_is_dropped():
    assert stem_step_1a("cats") == "cat"


def test_single_letter_word_is_not_emptied():
    """Правило s -> пусто обязано пропустить односимвольное слово."""
    assert stem_step_1a("s") == "s"


# ---------------------------------------------------------- stem_step_1b
def test_ing_is_removed():
    assert stem_step_1b("watching") == "watch"


def test_ed_is_removed():
    assert stem_step_1b("watched") == "watch"


def test_double_consonant_collapses():
    assert stem_step_1b("hopping") == "hop"


def test_double_l_is_an_exception_and_survives():
    """ll, ss, zz не схлопываются: иначе fall превратится в fal."""
    assert stem_step_1b("falling") == "fall"


def test_suffix_stays_when_the_stem_has_no_vowel():
    """bl и s — не основы. Слово остаётся нетронутым."""
    assert stem_step_1b("bled") == "bled"
    assert stem_step_1b("sing") == "sing"


def test_word_without_the_suffix_passes_through():
    assert stem_step_1b("cat") == "cat"


# ---------------------------------------------------------------- stem
def test_stem_runs_both_steps():
    assert stem("running") == "run"
    assert stem("cats") == "cat"


def test_stem_glues_two_forms_of_one_root():
    """Смысл стемминга: разные формы дают один ключ."""
    assert stem("hopped") == stem("hopping")


def test_stem_is_idempotent():
    """Повторный стемминг ничего не меняет — иначе пайплайн не воспроизводим."""
    for word in ("caresses", "running", "hopping", "cats", "falling"):
        assert stem(stem(word)) == stem(word)


# ------------------------------------------------------------- lemmatize
def test_table_lookup_wins_over_the_rules():
    assert lemmatize("running", "VERB", TABLE) == "run"


def test_irregular_form_is_impossible_without_the_table():
    """ran -> run никакими суффиксными правилами не выводится."""
    assert lemmatize("ran", "VERB", TABLE) == "run"
    assert lemmatize("ran", "VERB", {}) == "ran"


def test_pos_changes_the_answer():
    """Одно слово, два тега — две разные леммы. Вот зачем нужен POS."""
    assert lemmatize("better", "ADJ", TABLE) == "good"
    assert lemmatize("better", "ADV", TABLE) == "better"


def test_lookup_ignores_the_case_of_the_word():
    assert lemmatize("Cats", "NOUN", TABLE) == "cat"


def test_ing_fallback_needs_a_vowel_in_the_stem():
    assert lemmatize("singing", "VERB", {}) == "sing"
    assert lemmatize("bring", "VERB", {}) == "bring"


def test_past_tense_has_no_fallback_at_all():
    """Ключевой момент урока: -ed правил нет, нужен настоящий морфоанализ."""
    assert lemmatize("watched", "VERB", {}) == "watched"


# ------------------------------------------------------------- preprocess
def test_preprocess_returns_three_parallel_lists():
    out = preprocess("The cats were running.", TABLE)
    assert len(out["tokens"]) == len(out["stems"]) == len(out["lemmas"])


def test_stems_are_lowercased_but_tokens_are_not():
    out = preprocess("The Cats", TABLE)
    assert out["tokens"][0] == "The"
    assert out["stems"] == ["the", "cat"]


def test_default_tagger_calls_everything_a_noun_and_gets_verbs_wrong():
    """Без POS-тегера "were" не станет "be" — урок честно признаёт предел."""
    out = preprocess("were", TABLE)
    assert out["lemmas"] == ["were"]


def test_custom_pos_tagger_repairs_the_verbs():
    tagger = lambda tokens: [(t, "VERB") for t in tokens]
    out = preprocess("were running", TABLE, pos_tagger=tagger)
    assert out["lemmas"] == ["be", "run"]


def test_preprocess_on_empty_text_gives_empty_lists():
    assert preprocess("", TABLE) == {"tokens": [], "stems": [], "lemmas": []}
