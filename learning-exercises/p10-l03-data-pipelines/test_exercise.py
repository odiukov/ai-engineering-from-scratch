"""Тесты к уроку «Данные для предобучения». Правь exercise.py."""

import pytest

from exercise import (
    MAX_HASH,
    clean_text,
    deduplicate,
    estimate_jaccard,
    jaccard,
    minhash_signature,
    pack_sequences,
    quality_filter,
    shingles,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

GOOD_DOC = " ".join(["the model learns language from data"] * 12)


# ---------------------------------------------------------------- clean_text
def test_clean_text_strips_html_tags():
    assert clean_text("<p>Hello  world</p>") == "Hello world"


def test_clean_text_strips_urls():
    assert clean_text("see http://x.com/a now") == "see now"


def test_clean_text_collapses_blank_lines_and_runs_of_spaces():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"
    assert clean_text("   spaced   out   ") == "spaced out"


def test_clean_text_keeps_non_latin_letters():
    """Урок режет всё, кроме ASCII, — на кириллице это стирает документ целиком."""
    assert clean_text("привет 你好") == "привет 你好"


def test_clean_text_drops_control_characters():
    assert clean_text("a\x00\x07b") == "ab"


# ------------------------------------------------------------ quality_filter
def test_quality_filter_rejects_a_stub_page():
    assert quality_filter("hi", min_words=5) is False


def test_quality_filter_accepts_ordinary_prose():
    assert quality_filter("a b c d e", min_words=5) is True


def test_quality_filter_rejects_all_caps_spam():
    assert quality_filter("BUY NOW CHEAP PILLS OK", min_words=5) is False


def test_quality_filter_rejects_symbol_soup():
    assert quality_filter("a! b! c! d! e!", min_words=5) is False


def test_quality_filter_verdict_does_not_depend_on_document_length():
    """Пороги — доли, а не абсолютные числа: удвоенный документ судится так же."""
    assert quality_filter(GOOD_DOC) is quality_filter(GOOD_DOC + " " + GOOD_DOC)


# ------------------------------------------------------------------ shingles
def test_shingles_slide_by_one_word():
    assert shingles("the quick brown fox", 3) == {"the quick brown", "quick brown fox"}


def test_shingles_of_a_too_short_document_are_empty():
    assert shingles("short", 3) == set()


def test_shingles_count_is_n_minus_k_plus_one():
    text = " ".join(str(i) for i in range(20))
    assert len(shingles(text, 5)) == 16


def test_shingles_ignore_case():
    assert shingles("The Quick Brown", 2) == shingles("the quick brown", 2)


# ------------------------------------------------------------------- jaccard
def test_jaccard_of_identical_sets_is_one():
    assert jaccard({1, 2}, {1, 2}) == APPROX(1.0)


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({1, 2}, {3, 4}) == APPROX(0.0)


def test_jaccard_counts_the_union_not_the_smaller_set():
    assert jaccard({1, 2}, {2, 3}) == APPROX(1 / 3)


def test_jaccard_of_two_empty_sets_is_zero_not_one():
    """Два пустых документа несравнимы, а не одинаковы."""
    assert jaccard(set(), set()) == APPROX(0.0)


# --------------------------------------------------------- minhash_signature
def test_minhash_signature_has_one_value_per_hash_function():
    assert len(minhash_signature({"a", "b"}, num_hashes=8)) == 8


def test_minhash_signature_is_reproducible_across_calls():
    """Встроенный hash() рандомизируется от запуска к запуску — hashlib нет."""
    first = minhash_signature({"a", "b", "c"}, num_hashes=8, seed=7)
    second = minhash_signature({"c", "b", "a"}, num_hashes=8, seed=7)
    assert first == second


def test_minhash_signature_depends_on_the_seed():
    assert minhash_signature({"a"}, 8, seed=0) != minhash_signature({"a"}, 8, seed=1)


def test_minhash_signature_of_an_empty_set_is_all_max():
    assert minhash_signature(set(), num_hashes=2) == [MAX_HASH, MAX_HASH]


# ---------------------------------------------------------- estimate_jaccard
def test_estimate_jaccard_counts_matching_positions():
    assert estimate_jaccard([1, 2, 3], [1, 9, 3]) == APPROX(2 / 3)
    assert estimate_jaccard([], []) == APPROX(0.0)


def test_estimate_jaccard_approximates_the_true_jaccard():
    """Ради этого свойства всё и затевалось: 256 чисел вместо тысяч шинглов."""
    a = {f"shingle {i}" for i in range(100)}
    b = {f"shingle {i}" for i in range(50, 150)}
    true_value = jaccard(a, b)
    estimate = estimate_jaccard(
        minhash_signature(a, 256, seed=1), minhash_signature(b, 256, seed=1)
    )
    assert estimate == pytest.approx(true_value, abs=0.08)


def test_estimate_jaccard_lies_about_two_empty_documents():
    """Сигнатуры пустых множеств совпадают целиком — оценка врёт, jaccard нет."""
    sig = minhash_signature(set(), 16)
    assert estimate_jaccard(sig, sig) == APPROX(1.0)
    assert jaccard(set(), set()) == APPROX(0.0)


# ---------------------------------------------------------------- deduplicate
def test_deduplicate_removes_an_exact_copy():
    kept, removed = deduplicate(["a b c d e f", "a b c d e f"], k=3)
    assert kept == ["a b c d e f"]
    assert removed == 1
    assert deduplicate([], k=3) == ([], 0)


def test_deduplicate_keeps_the_first_copy_and_the_original_order():
    docs = ["one two three four", "five six seven eight", "one two three four"]
    kept, removed = deduplicate(docs, k=3)
    assert kept == ["one two three four", "five six seven eight"]
    assert removed == 1


def test_deduplicate_keeps_documents_that_only_look_similar():
    docs = [
        "the cat sat on the mat and slept",
        "quantum chromodynamics describes the strong nuclear force",
    ]
    kept, removed = deduplicate(docs, k=3)
    assert removed == 0
    assert kept == docs


def test_deduplicate_catches_a_near_duplicate():
    """Одна изменённая строчка не делает документ новым."""
    base = " ".join(f"line {i} of the article" for i in range(20))
    tweaked = base + " copyright notice"
    kept, removed = deduplicate([base, tweaked], threshold=0.8, k=3)
    assert removed == 1
    assert kept == [base]


def test_deduplicate_threshold_one_demands_an_exact_match():
    base = " ".join(f"line {i} of the article" for i in range(20))
    tweaked = base + " copyright notice"
    _, removed = deduplicate([base, tweaked], threshold=1.0, k=3)
    assert removed == 0


def test_deduplicate_catches_leakage_between_train_and_eval():
    """Тот же документ в обучении и в валидации завышает метрику на ровном месте."""
    train = ["alpha beta gamma delta epsilon", "one two three four five"]
    evaluation = ["alpha beta gamma delta epsilon"]
    kept, removed = deduplicate(train + evaluation, k=3)
    assert removed == 1
    assert len(kept) == 2


# -------------------------------------------------------------- pack_sequences
def test_pack_sequences_splits_evenly():
    assert pack_sequences([1, 2, 3, 4], 2) == ([[1, 2], [3, 4]], [[1, 1], [1, 1]])


def test_pack_sequences_pads_only_the_last_chunk():
    assert pack_sequences([1, 2, 3], 2) == ([[1, 2], [3, 0]], [[1, 1], [1, 0]])
    assert pack_sequences([], 4) == ([], [])


def test_pack_sequences_loses_no_tokens():
    tokens = list(range(1, 30))
    sequences, masks = pack_sequences(tokens, 8)
    real = [t for seq, mask in zip(sequences, masks) for t, m in zip(seq, mask) if m]
    assert real == tokens


def test_packing_wastes_far_less_than_padding_each_document():
    """Три документа встык занимают один кусок вместо трёх почти пустых."""
    docs = [[1] * 3, [2] * 4, [3] * 5]
    packed, masks = pack_sequences([t for doc in docs for t in doc], 16)
    utilization = sum(sum(m) for m in masks) / sum(len(m) for m in masks)
    naive = sum(len(d) for d in docs) / (16 * len(docs))
    assert utilization == APPROX(12 / 16)
    assert utilization > naive
