"""Тесты к уроку «Токенизаторы: BPE, WordPiece, SentencePiece». Правь exercise.py."""

import pytest

from exercise import (
    bpe_best_pair,
    count_pairs,
    decode,
    encode,
    merge_pair,
    tokenization_stats,
    train_bpe,
    wordpiece_best_pair,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "the dog sat on the log. the dog ate the frog."
)


# ------------------------------------------------------------- count_pairs
def test_count_pairs_counts_each_adjacent_pair():
    assert count_pairs([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}


def test_count_pairs_counts_overlapping_repeats():
    """[5,5,5] содержит ДВЕ пары (5,5), а не одну — они перекрываются."""
    assert count_pairs([5, 5, 5]) == {(5, 5): 2}


def test_count_pairs_of_single_token_is_empty():
    assert count_pairs([7]) == {}
    assert count_pairs([]) == {}


def test_count_pairs_does_not_mutate_input():
    tokens = [1, 2, 3]
    count_pairs(tokens)
    assert tokens == [1, 2, 3]


# -------------------------------------------------------------- merge_pair
def test_merge_pair_replaces_every_occurrence():
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_pair_does_not_reuse_a_token_in_two_merges():
    """Ловушка: [5,5,5] даёт [99,5], а не [99,99] и не [99,5,5]."""
    assert merge_pair([5, 5, 5], (5, 5), 99) == [99, 5]


def test_merge_pair_leaves_text_without_the_pair_untouched():
    assert merge_pair([1, 2, 3], (7, 8), 99) == [1, 2, 3]


def test_merge_pair_handles_the_pair_at_the_very_end():
    """Проверка границы: обращение к tokens[i+1] не должно вылетать за конец."""
    assert merge_pair([3, 1, 2], (1, 2), 99) == [3, 99]


def test_merge_pair_does_not_mutate_input():
    tokens = [1, 2, 3]
    merge_pair(tokens, (1, 2), 99)
    assert tokens == [1, 2, 3]


# ----------------------------------------------------------- bpe_best_pair
def test_bpe_best_pair_picks_the_most_frequent():
    assert bpe_best_pair([1, 2, 1, 2, 3, 4]) == (1, 2)


def test_bpe_best_pair_is_none_when_there_are_no_pairs():
    assert bpe_best_pair([7]) is None


def test_bpe_best_pair_breaks_ties_deterministically():
    """Все три пары встречаются по разу — обучение обязано быть воспроизводимым."""
    tokens = [3, 4, 1, 2, 5, 6]
    assert bpe_best_pair(tokens) == bpe_best_pair(list(tokens))
    assert bpe_best_pair(tokens) == (1, 2)


# --------------------------------------------------------------- train_bpe
def test_train_bpe_learns_the_repeated_pair_first():
    merges, vocab = train_bpe("aaab", 1)
    assert merges == [((97, 97), 256)]
    assert vocab[256] == b"aa"


def test_train_bpe_vocabulary_grows_by_exactly_one_per_merge():
    """Словарь монотонно растёт: 256 байтов плюс по токену на слияние."""
    sizes = [len(train_bpe(CORPUS, k)[1]) for k in (0, 1, 5, 20, 30)]
    assert sizes == [256, 257, 261, 276, 286]


def test_train_bpe_assigns_consecutive_ids_from_256():
    merges, _ = train_bpe(CORPUS, 10)
    assert [new_id for _, new_id in merges] == list(range(256, 266))


def test_train_bpe_stops_early_when_nothing_is_left_to_merge():
    """Один байт — сливать нечего, сколько слияний ни проси."""
    merges, vocab = train_bpe("a", 50)
    assert merges == []
    assert len(vocab) == 256


def test_train_bpe_merged_token_bytes_are_the_concatenation_of_parents():
    merges, vocab = train_bpe(CORPUS, 30)
    for (left, right), new_id in merges:
        assert vocab[new_id] == vocab[left] + vocab[right]


def test_train_bpe_shortens_the_training_corpus():
    merges, _ = train_bpe(CORPUS, 40)
    assert len(encode(CORPUS, merges)) < len(CORPUS.encode("utf-8"))


# ------------------------------------------------------------ encode/decode
def test_encode_without_merges_is_just_utf8_bytes():
    assert encode("ab", []) == [97, 98]


def test_encode_applies_a_learned_merge():
    assert encode("aaab", [((97, 97), 256)]) == [256, 97, 98]


def test_encode_merge_order_matters():
    """Слияния применяются в порядке обучения, иначе поздние не соберутся.

    "abab": сначала (a,b)->256 даёт [256,256], потом (256,256)->257 даёт [257].
    В обратном порядке 256 ещё не существует, и второй шаг ничего не найдёт.
    """
    merges, _ = train_bpe("abab", 2)
    assert encode("abab", merges) == [257]
    assert encode("abab", list(reversed(merges))) == [256, 256]


def test_encode_decode_roundtrip_on_training_text():
    merges, vocab = train_bpe(CORPUS, 40)
    assert decode(encode(CORPUS, merges), vocab) == CORPUS


def test_encode_decode_roundtrip_on_unseen_text():
    """Byte-level BPE не знает [UNK]: любой текст разложится хотя бы на байты."""
    merges, vocab = train_bpe(CORPUS, 40)
    unseen = "unhappiness & Geschwindigkeit"
    assert decode(encode(unseen, merges), vocab) == unseen


def test_encode_decode_roundtrip_on_non_latin_text():
    merges, vocab = train_bpe(CORPUS, 40)
    text = "привет 你好 🔥"
    assert decode(encode(text, merges), vocab) == text


def test_decode_joins_bytes_before_decoding_utf8():
    """Токен может обрываться на середине символа — декодировать надо целиком."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"\xd0"
    vocab[257] = b"\xbf"
    assert decode([256, 257], vocab) == "п"


# --------------------------------------------------------- wordpiece_best_pair
def test_wordpiece_prefers_the_surprising_pair_over_the_frequent_one():
    """BPE берёт частую (1,2), WordPiece — эксклюзивную (3,4)."""
    tokens = [1, 2, 1, 2, 3, 4]
    assert bpe_best_pair(tokens) == (1, 2)
    assert wordpiece_best_pair(tokens) == (3, 4)


def test_wordpiece_is_none_when_there_are_no_pairs():
    assert wordpiece_best_pair([7]) is None


def test_wordpiece_agrees_with_bpe_when_all_tokens_are_equally_common():
    """Если все половинки встречаются одинаково часто, знаменатель не решает."""
    tokens = [1, 2, 1, 2, 3, 4, 4, 3]
    assert wordpiece_best_pair(tokens) == bpe_best_pair(tokens)


# ---------------------------------------------------- tokenization_stats
def test_stats_without_merges_have_ratio_one():
    stats = tokenization_stats("ab", [])
    assert stats["bytes"] == 2
    assert stats["tokens"] == 2
    assert stats["compression_ratio"] == APPROX(1.0)


def test_stats_on_empty_text_do_not_divide_by_zero():
    stats = tokenization_stats("", [])
    assert stats["compression_ratio"] == APPROX(1.0)
    assert stats["fertility"] == APPROX(0.0)


def test_stats_count_words_by_whitespace():
    assert tokenization_stats("the cat sat", [])["words"] == 3


def test_compression_ratio_never_grows_with_more_merges():
    """Больше слияний — не хуже сжатие. Это и есть смысл обучения BPE."""
    ratios = [
        tokenization_stats(CORPUS, train_bpe(CORPUS, k)[0])["compression_ratio"]
        for k in (0, 5, 10, 20, 40)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < ratios[0]


def test_fertility_drops_when_the_tokenizer_learns_the_words():
    """Пока слово не выучено, оно стоит по токену на букву.

    Выученный наизусть корпус проваливается ниже одного токена на слово —
    так выглядит переобученный токенизатор, который на новом тексте
    ничего не сожмёт.
    """
    text = "token token token token"
    before = tokenization_stats(text, [])["fertility"]
    merges, _ = train_bpe(text, 20)
    after = tokenization_stats(text, merges)["fertility"]
    assert before > 5.0
    assert after < 1.0


def test_non_latin_text_costs_more_tokens_per_character():
    """Многоязычный налог: один символ кириллицы — два байта, значит два токена."""
    latin = tokenization_stats("abcdef", [])
    cyrillic = tokenization_stats("абвгде", [])
    assert cyrillic["tokens"] == 2 * latin["tokens"]
