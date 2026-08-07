"""Тесты к уроку «Subword-токенизация: BPE своими руками». Правь exercise.py."""

import pytest

from exercise import (
    decode,
    encode,
    merge_vocab,
    pair_counts,
    pre_tokenize,
    tokenizer_stats,
    train_bpe,
    word_to_symbols,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CORPUS = (
    "the quick brown fox jumps over the lazy dog "
    "language models learn from statistical patterns in text "
    "tokenization splits text into smaller units called tokens "
    "subword tokenization lets rare words decompose into known pieces "
    "byte pair encoding is the dominant tokenization algorithm today "
    "the lazy dog slept while the fox jumped again and again "
    "patterns of letters in words are learnable and reusable"
)


# ------------------------------------------------------------- pre_tokenize
def test_pre_tokenize_keeps_the_leading_space_inside_the_word():
    assert pre_tokenize("the cat") == ["the", " cat"]


def test_pre_tokenize_join_restores_the_original_text():
    """Ничего не теряем: склейка кусков обязана дать ровно исходный текст."""
    for text in ("a  b ", "\n hi\tthere", "one", "   ", "низкий  уровень"):
        assert "".join(pre_tokenize(text)) == text


def test_pre_tokenize_of_empty_text_is_empty_list():
    assert pre_tokenize("") == []


def test_pre_tokenize_separates_hello_from_space_hello():
    """Тот самый whitespace ambiguity: это два разных куска, а не один."""
    assert pre_tokenize("hello") != pre_tokenize(" hello")


# ---------------------------------------------------------- word_to_symbols
def test_word_to_symbols_splits_ascii_into_characters():
    assert word_to_symbols("hi") == ["h", "i", "</w>"]


def test_word_to_symbols_ends_with_the_end_of_word_marker():
    """Без '</w>' слово 'low' и приставка 'low' в 'lower' слились бы в один токен."""
    assert word_to_symbols("low")[-1] == "</w>"
    assert word_to_symbols("")[-1] == "</w>"


def test_word_to_symbols_keeps_the_leading_space_as_its_own_symbol():
    assert word_to_symbols(" low") == [" ", "l", "o", "w", "</w>"]


def test_word_to_symbols_goes_through_bytes_so_nothing_is_unknown():
    """Байтовый алфавит закрыт: любой символ мира раскладывается, [UNK] не нужен."""
    for word in ("é", "привет", "🎉", "漢字"):
        symbols = word_to_symbols(word)
        base = symbols[:-1]
        assert len(base) == len(word.encode("utf-8"))
        assert all(len(s) == 1 and ord(s) < 256 for s in base)


# -------------------------------------------------------------- pair_counts
def test_pair_counts_finds_every_adjacent_pair():
    counts = pair_counts({("l", "o", "w", "</w>"): 5})
    assert dict(counts) == {("l", "o"): 5, ("o", "w"): 5, ("w", "</w>"): 5}


def test_pair_counts_weights_pairs_by_word_frequency():
    """Слово, встретившееся 5 раз, даёт своим парам +5, а не +1."""
    counts = pair_counts({("a", "b"): 5, ("c", "b"): 1})
    assert counts[("a", "b")] == 5
    assert counts[("c", "b")] == 1


def test_pair_counts_sums_the_same_pair_across_words():
    counts = pair_counts({("a", "b", "x"): 2, ("y", "a", "b"): 3})
    assert counts[("a", "b")] == 5


def test_frequent_bigram_beats_the_rare_one():
    """Смысл всего обучения: побеждает частая пара, а не первая попавшаяся."""
    counts = pair_counts({("r", "q", "</w>"): 1, ("t", "h", "</w>"): 40})
    assert counts[("t", "h")] > counts[("r", "q")]
    assert max(counts, key=counts.get) in (("t", "h"), ("h", "</w>"))


# -------------------------------------------------------------- merge_vocab
def test_merge_vocab_glues_the_pair_into_one_symbol():
    assert merge_vocab({("l", "o", "w", "</w>"): 5}, ("l", "o")) == {
        ("lo", "w", "</w>"): 5
    }


def test_merge_vocab_leaves_words_without_the_pair_untouched():
    vocab = {("l", "o", "w"): 5, ("c", "a", "t"): 2}
    assert merge_vocab(vocab, ("l", "o"))[("c", "a", "t")] == 2


def test_merge_vocab_handles_overlapping_runs_left_to_right():
    """В ('a','a','a') пара ('a','a') видна дважды, но склеить можно один раз."""
    assert merge_vocab({("a", "a", "a"): 1}, ("a", "a")) == {("aa", "a"): 1}


def test_merge_vocab_sums_frequencies_when_words_collide():
    """После склейки два разных слова могут стать одним — частоты складываются."""
    vocab = {("a", "b", "</w>"): 1, ("ab", "</w>"): 2}
    assert merge_vocab(vocab, ("a", "b")) == {("ab", "</w>"): 3}


def test_merge_vocab_does_not_mutate_the_input():
    vocab = {("l", "o", "w"): 5}
    merge_vocab(vocab, ("l", "o"))
    assert vocab == {("l", "o", "w"): 5}


# ---------------------------------------------------------------- train_bpe
def test_train_bpe_learns_the_expected_merges():
    assert train_bpe("low low low", 10) == [
        ("l", "o"),
        ("lo", "w"),
        ("low", "</w>"),
        (" ", "low</w>"),
    ]


def test_train_bpe_picks_the_frequent_pair_not_the_alphabetical_one():
    """('a','b') и (' ','a') идут раньше по алфавиту, но встречаются реже."""
    assert train_bpe("xy xy xy ab", 1) == [("x", "y")]


def test_train_bpe_merge_list_is_ordered_and_grows_as_a_prefix():
    """Список merge-ов упорядочен: первые два не зависят от того, сколько всего."""
    short = train_bpe(CORPUS, 3)
    long = train_bpe(CORPUS, 40)
    assert len(short) == 3
    assert long[:3] == short


def test_train_bpe_stops_when_no_pairs_are_left():
    """Просить 100 merge-ов у крошечного корпуса можно, но их там всего 4."""
    assert len(train_bpe("low low low", 100)) == 4


def test_train_bpe_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        train_bpe("", 10)


# ------------------------------------------------------------------- encode
def test_encode_without_merges_is_pure_byte_level():
    assert encode("hi", []) == ["h", "i", "</w>"]


def test_encode_applies_merges_in_training_order():
    merges = train_bpe("low low low", 10)
    assert encode("low low low", merges) == ["low</w>", " low</w>", " low</w>"]


def test_encode_order_of_merges_matters():
    """Те же merge-и в обратном порядке дают другой результат — порядок не декор."""
    merges = train_bpe("low low low", 10)
    assert encode("low", list(reversed(merges))) != encode("low", merges)


def test_encode_never_fails_on_unseen_characters():
    """Байтовый алфавит: обученный на английском токенизатор глотает что угодно."""
    merges = train_bpe(CORPUS, 40)
    tokens = encode("привет 🎉 漢字", merges)
    assert tokens
    assert all(isinstance(t, str) and t for t in tokens)


def test_encode_of_hello_differs_from_encode_of_space_hello():
    """Whitespace ambiguity в чистом виде: пробел меняет и токены, и их число."""
    merges = train_bpe("x low low low low", 100)
    assert encode(" low", merges) == [" low</w>"]
    assert encode("low", merges) == ["l", "o", "w", "</w>"]


# ------------------------------------------------------------------- decode
def test_decode_drops_the_end_of_word_marker():
    assert decode(["h", "i", "</w>"]) == "hi"
    assert decode(["low</w>", " low</w>"]) == "low low"


def test_decode_of_no_tokens_is_an_empty_string():
    assert decode([]) == ""


def test_encode_decode_is_round_trip():
    """Первый тест, который пишут для любого токенизатора."""
    merges = train_bpe(CORPUS, 60)
    for text in ("the lazy dog", " hello", "hello", "a  b ", CORPUS):
        assert decode(encode(text, merges)) == text


def test_round_trip_survives_non_ascii_text():
    """Байты собираются обратно в буквы, даже если модель их никогда не видела."""
    merges = train_bpe(CORPUS, 60)
    text = "привет, мир 🎉"
    assert decode(encode(text, merges)) == text


# ---------------------------------------------------------- tokenizer_stats
def test_stats_without_merges_counts_one_token_per_byte_plus_marker():
    stats = tokenizer_stats("the lazy dog", [])
    assert stats["chars"] == 12
    assert stats["words"] == 3
    assert stats["tokens"] == 12 + 3
    assert stats["tokens_per_word"] == APPROX(5.0)


def test_stats_tokens_per_word_never_grows_with_more_merges():
    """Каждый merge только склеивает — токенов может стать меньше, но не больше."""
    counts = [
        tokenizer_stats(CORPUS, train_bpe(CORPUS, n))["tokens_per_word"]
        for n in (0, 10, 40, 120)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(counts, counts[1:]))


def test_stats_compression_ratio_improves_with_training():
    """Символов на токен — это деньги: чем больше, тем дешевле тот же текст."""
    untrained = tokenizer_stats(CORPUS, [])["compression_ratio"]
    trained = tokenizer_stats(CORPUS, train_bpe(CORPUS, 120))["compression_ratio"]
    assert trained > untrained


def test_stats_rejects_empty_text():
    with pytest.raises(ValueError):
        tokenizer_stats("", [])
