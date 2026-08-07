"""Тесты к уроку «Стратегии чанкинга для RAG». Правь exercise.py."""

import pytest

from exercise import (
    chunk_fixed,
    chunk_parent_child,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence_window,
    retrieve_parents,
    sentence_similarity,
    split_sentences,
)

PARAGRAPHS = [
    "Chapter one. This contract is between Acme Corp and Beta Inc.",
    "Chapter two. Acme pays Beta thirty thousand dollars every month.",
    "Chapter three. Either party may terminate with ninety days notice.",
    "Chapter four. Both parties keep every trade secret confidential.",
]
DOC = "\n\n".join(PARAGRAPHS)

SEM_TEXT = (
    "The cat sat on the warm mat. The cat ate the fish quickly. "
    "The cat slept on the warm mat again. The cat purred on the mat. "
    "Rocket engines burn cold liquid fuel. Rocket engines need liquid oxidizer. "
    "Rocket stages separate high in flight. Rocket fuel tanks stay pressurized."
)

MAPPING = [
    {"child": "late payment fee is five percent", "parent_idx": 0, "parent": "PARENT ZERO"},
    {"child": "payments are due monthly on the first", "parent_idx": 0, "parent": "PARENT ZERO"},
    {"child": "disputes are resolved by arbitration", "parent_idx": 1, "parent": "PARENT ONE"},
]


# ------------------------------------------------------------- chunk_fixed
def test_fixed_without_overlap_reassembles_into_the_original_text():
    """Baseline обязан быть обратимым: склейка чанков символ в символ даёт исходник."""
    assert "".join(chunk_fixed(DOC, 50)) == DOC


def test_fixed_never_returns_a_chunk_longer_than_size():
    assert all(len(c) <= 40 for c in chunk_fixed(DOC, 40, overlap=15))


def test_fixed_with_overlap_loses_no_character_on_the_boundary():
    """Каждый символ исходника должен попасть хотя бы в один чанк."""
    size, overlap = 40, 15
    covered = set()
    for start, chunk in zip(range(0, len(DOC), size - overlap), chunk_fixed(DOC, size, overlap)):
        covered.update(range(start, start + len(chunk)))
    assert covered == set(range(len(DOC)))


def test_fixed_neighbours_really_share_overlap_characters():
    size, overlap = 40, 15
    chunks = chunk_fixed(DOC, size, overlap)
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if len(a) == size]
    assert pairs, "нужны хотя бы две полноразмерных пары"
    assert all(a[-overlap:] == b[:overlap] for a, b in pairs)


def test_fixed_rejects_overlap_that_is_not_smaller_than_size():
    """overlap >= size даёт нулевой шаг: это ValueError, а не бесконечный цикл."""
    with pytest.raises(ValueError):
        chunk_fixed(DOC, 100, overlap=100)
    with pytest.raises(ValueError):
        chunk_fixed(DOC, 100, overlap=150)


def test_fixed_rejects_non_positive_size():
    with pytest.raises(ValueError):
        chunk_fixed(DOC, 0)


# --------------------------------------------------------- split_sentences
def test_sentences_keep_their_terminal_punctuation():
    assert split_sentences("Hi there. How are you? Fine!") == [
        "Hi there.",
        "How are you?",
        "Fine!",
    ]


def test_sentences_of_blank_text_is_an_empty_list():
    """Наивный split вернёт [''] — фантомное предложение нулевой длины."""
    assert split_sentences("   ") == []
    assert split_sentences("") == []


def test_sentences_join_back_without_losing_words():
    words_in = DOC.split()
    words_out = " ".join(split_sentences(DOC)).split()
    assert words_out == words_in


# --------------------------------------------------------- chunk_recursive
def test_recursive_returns_the_whole_text_when_it_already_fits():
    assert chunk_recursive("short text", 100) == ["short text"]


def test_recursive_cuts_on_paragraph_boundaries_not_mid_sentence():
    """Абзацы помещаются по одному — значит резать надо ровно по ним."""
    size = max(len(p) for p in PARAGRAPHS) + 5
    chunks = chunk_recursive(DOC, size)
    assert chunks == PARAGRAPHS
    assert all(c.endswith(".") for c in chunks)


def test_recursive_never_exceeds_size_even_on_a_word_without_separators():
    """Fallback обязан сработать: сепараторов нет, но предел size остаётся."""
    text = "a" * 250
    chunks = chunk_recursive(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks) == text


def test_recursive_never_exceeds_size_on_a_mixed_document():
    long_doc = DOC + "\n\n" + "x" * 300 + "\n\n" + " ".join(["word"] * 200)
    assert all(len(c) <= 60 for c in chunk_recursive(long_doc, 60))


def test_recursive_of_blank_text_is_an_empty_list():
    assert chunk_recursive("   \n\n  ", 50) == []


# ---------------------------------------------------- chunk_sentence_window
def test_window_without_stride_partitions_all_sentences_exactly_once():
    sentences = split_sentences(DOC)
    chunks = chunk_sentence_window(DOC, window=2)
    assert " ".join(chunks) == " ".join(sentences)


def test_window_with_smaller_stride_shares_sentences_between_neighbours():
    chunks = chunk_sentence_window(DOC, window=2, stride=1)
    assert all(a.split(". ")[-1][:20] in b for a, b in zip(chunks, chunks[1:]))


def test_window_covers_every_sentence_without_tail_stubs():
    """Хвостовое окно не должно порождать огрызок, уже вошедший в предыдущее."""
    sentences = split_sentences(DOC)
    chunks = chunk_sentence_window(DOC, window=3, stride=2)
    assert all(any(s in c for c in chunks) for s in sentences)
    assert all(c not in other for c, other in zip(chunks, chunks[1:]))


def test_window_rejects_non_positive_window():
    with pytest.raises(ValueError):
        chunk_sentence_window(DOC, window=0)


# ------------------------------------------------------ sentence_similarity
def test_similarity_of_identical_sentences_is_one():
    assert sentence_similarity("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_similarity_of_disjoint_sentences_is_zero():
    assert sentence_similarity("cat", "dog") == pytest.approx(0.0)


def test_similarity_ignores_case_and_punctuation():
    assert sentence_similarity("Cat!", "cat") == pytest.approx(1.0)


def test_similarity_is_symmetric_and_safe_on_empty_input():
    a, b = "the cat sat on the mat", "the dog sat"
    assert sentence_similarity(a, b) == pytest.approx(sentence_similarity(b, a))
    assert sentence_similarity("", "cat") == pytest.approx(0.0)


# ---------------------------------------------------------- chunk_semantic
def test_semantic_with_zero_threshold_never_splits():
    """Похожесть всегда >= 0, значит рвать негде — один чанк на весь текст."""
    chunks = chunk_semantic(SEM_TEXT, threshold=0.0, min_chars=0, max_chars=10_000)
    assert chunks == [" ".join(split_sentences(SEM_TEXT))]


def test_semantic_without_min_chars_splits_at_every_sentence():
    chunks = chunk_semantic(SEM_TEXT, threshold=1.1, min_chars=0, max_chars=10_000)
    assert chunks == split_sentences(SEM_TEXT)


def test_semantic_min_chars_floor_produces_no_stubs():
    """Пол min_chars — то, из-за чего semantic вообще имеет смысл."""
    chunks = chunk_semantic(SEM_TEXT, threshold=1.1, min_chars=80, max_chars=10_000)
    assert len(chunks) > 1
    assert all(len(c) >= 80 for c in chunks)


def test_semantic_respects_max_chars():
    chunks = chunk_semantic(SEM_TEXT, threshold=0.0, min_chars=0, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)


def test_semantic_of_blank_text_is_an_empty_list():
    assert chunk_semantic("   ") == []


# ------------------------------------------------------ chunk_parent_child
def test_parent_child_keeps_every_child_inside_its_own_parent():
    """Чанк не имеет права пересекать границу родителя."""
    mapping = chunk_parent_child(DOC, parent_size=140, child_size=40)
    assert mapping
    assert all(m["child"] in m["parent"] for m in mapping)


def test_parent_child_covers_all_parents_with_contiguous_indices():
    mapping = chunk_parent_child(DOC, parent_size=140, child_size=40)
    idxs = sorted({m["parent_idx"] for m in mapping})
    assert idxs == list(range(len(idxs)))
    assert len(idxs) == len(chunk_recursive(DOC, 140))


def test_parent_child_respects_both_size_limits():
    mapping = chunk_parent_child(DOC, parent_size=140, child_size=40)
    assert all(len(m["child"]) <= 40 for m in mapping)
    assert all(len(p) <= 140 for p in {m["parent"] for m in mapping})


# -------------------------------------------------------- retrieve_parents
def test_retrieve_never_returns_the_same_parent_twice():
    """Два ребёнка одного родителя — родитель в ответе один раз."""
    assert retrieve_parents("late payment fee", MAPPING, top_k=2) == ["PARENT ZERO"]


def test_retrieve_orders_parents_by_best_matching_child():
    assert retrieve_parents("late payment fee", MAPPING, top_k=3) == [
        "PARENT ZERO",
        "PARENT ONE",
    ]
    assert retrieve_parents("arbitration disputes", MAPPING, top_k=1) == ["PARENT ONE"]


def test_retrieve_on_empty_mapping_is_an_empty_list():
    assert retrieve_parents("anything", [], top_k=3) == []


def test_retrieve_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        retrieve_parents("anything", MAPPING, top_k=0)
