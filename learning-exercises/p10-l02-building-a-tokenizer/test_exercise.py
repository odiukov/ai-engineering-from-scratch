"""Тесты к уроку «Собираем токенизатор с нуля». Правь exercise.py."""

from exercise import (
    IM_END,
    IM_START,
    apply_chat_template,
    apply_merges,
    encode,
    encode_with_specials,
    normalize,
    pre_tokenize,
    split_on_specials,
    train_bpe,
)

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox runs through the forest. "
    "deep learning models process natural language."
)


def render(ids, vocab):
    """Локальный хелпер: склеить байты токенов и только потом декодировать."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- normalize
def test_normalize_splits_the_fi_ligature():
    assert normalize("ﬁle") == "file"


def test_normalize_folds_fullwidth_latin():
    assert normalize("Ａ") == "A"


def test_normalize_is_idempotent():
    """Нормализовать дважды — то же, что один раз. Иначе id поплывут."""
    text = "ﬁle Ａ ①"
    assert normalize(normalize(text)) == normalize(text)


# -------------------------------------------------------------- pre_tokenize
def test_pre_tokenize_keeps_the_leading_space_with_the_word():
    assert pre_tokenize("Hello, world!") == ["Hello", ",", " world", "!"]


def test_pre_tokenize_splits_contractions():
    assert pre_tokenize("Don't stop") == ["Don", "'t", " stop"]


def test_pre_tokenize_separates_digits_from_letters():
    assert pre_tokenize("x = 1") == ["x", " =", " 1"]


def test_pre_tokenize_loses_no_characters():
    """Склейка кусков обязана дать исходный текст обратно, символ в символ."""
    for text in ("Hello, world!", "  a", "x = 1 + 2", "a\n\nb", "snake_case  ", ""):
        assert "".join(pre_tokenize(text)) == text


def test_pre_tokenize_does_not_drop_non_latin_scripts():
    """Шаблон через [a-zA-Z] молча выбросил бы иероглифы и эмодзи."""
    text = "hello 你好 🔥"
    assert "".join(pre_tokenize(text)) == text
    assert any("你" in chunk for chunk in pre_tokenize(text))


# --------------------------------------------------------------- apply_merges
def test_apply_merges_replaces_the_pair():
    assert apply_merges([97, 97, 98], [((97, 97), 256)]) == [256, 98]


def test_apply_merges_with_empty_list_changes_nothing():
    seq = [97, 98]
    assert apply_merges(seq, []) == [97, 98]
    assert seq == [97, 98]


def test_apply_merges_does_not_reuse_a_token_twice():
    assert apply_merges([5, 5, 5], [((5, 5), 99)]) == [99, 5]


def test_apply_merges_respects_order():
    """Второе слияние собирается из результата первого — обратный порядок не сработает."""
    forward = [((97, 98), 256), ((256, 256), 257)]
    assert apply_merges([97, 98, 97, 98], forward) == [257]
    assert apply_merges([97, 98, 97, 98], list(reversed(forward))) == [256, 256]


# ------------------------------------------------------------------ train_bpe
def test_train_bpe_learns_the_most_frequent_pair_first():
    merges, vocab = train_bpe("ab ab", 1)
    assert merges == [((97, 98), 256)]
    assert vocab[256] == b"ab"


def test_train_bpe_vocabulary_grows_by_one_per_merge():
    sizes = [len(train_bpe(CORPUS, k)[1]) for k in (0, 1, 5, 20)]
    assert sizes == [256, 257, 261, 276]


def test_train_bpe_stops_when_nothing_is_left_to_merge():
    """Все куски по одному байту — сливать нечего, сколько слияний ни проси."""
    merges, vocab = train_bpe("a+b", 50)
    assert merges == []
    assert len(vocab) == 256


def test_train_bpe_never_merges_across_a_word_boundary():
    """Главное, ради чего нужен пре-токенизатор: токена "b a" быть не может."""
    merges, vocab = train_bpe("ab ab ab ab", 5)
    learned = [vocab[new_id] for _, new_id in merges]
    assert b" ab" in learned
    assert all(b" " not in piece[1:] for piece in learned)


def test_train_bpe_merged_bytes_are_the_parents_concatenated():
    merges, vocab = train_bpe(CORPUS, 25)
    for (left, right), new_id in merges:
        assert vocab[new_id] == vocab[left] + vocab[right]


# --------------------------------------------------------------------- encode
def test_encode_without_merges_is_utf8_bytes():
    assert encode("ab", []) == [97, 98]
    assert encode("", []) == []


def test_encode_applies_a_learned_merge():
    assert encode("ab", [((97, 98), 256)]) == [256]


def test_encode_cannot_apply_a_cross_boundary_merge():
    """Слияние (b, пробел) не сработает: эти байты лежат в разных кусках."""
    assert encode("ab ab", [((98, 32), 999)]) == [97, 98, 32, 97, 98]


def test_encode_roundtrips_through_the_vocabulary():
    merges, vocab = train_bpe(CORPUS, 40)
    assert render(encode(CORPUS, merges), vocab) == CORPUS


def test_encode_roundtrips_on_text_the_tokenizer_never_saw():
    """Байтовый уровень: иероглифы и эмодзи не дают [UNK], только больше токенов."""
    merges, vocab = train_bpe(CORPUS, 40)
    text = "你好世界 🔥 def foo(x): return x + 1"
    assert render(encode(text, merges), vocab) == text


def test_encode_normalizes_before_tokenizing():
    """Полноширинная "Ａ" и обычная "A" обязаны дать один и тот же id."""
    assert encode("Ａ", []) == encode("A", [])


# --------------------------------------------------------- split_on_specials
def test_split_on_specials_marks_the_special_part():
    assert split_on_specials("a<|end|>b", {"<|end|>": 300}) == [
        ("a", False),
        ("<|end|>", True),
        ("b", False),
    ]


def test_split_without_specials_returns_the_whole_text():
    assert split_on_specials("abc", {}) == [("abc", False)]
    assert split_on_specials("", {}) == []


def test_split_produces_no_empty_chunks():
    """Два служебных токена подряд не должны рождать пустой кусок между ними."""
    parts = split_on_specials("<|a|><|b|>", {"<|a|>": 300, "<|b|>": 301})
    assert parts == [("<|a|>", True), ("<|b|>", True)]


def test_split_prefers_the_longest_special_token():
    """Короткий токен — префикс длинного; без сортировки длинный не сработает."""
    specials = {"<|im_end|>": 300, "<|im_end|>!": 301}
    assert split_on_specials("<|im_end|>!", specials) == [("<|im_end|>!", True)]


# ------------------------------------------------------ encode_with_specials
def test_encode_with_specials_substitutes_the_fixed_id():
    assert encode_with_specials("a<|end|>", [], {"<|end|>": 300}) == [97, 300]


def test_encode_with_specials_never_lets_bpe_split_the_marker():
    """Даже если слияние покрывает часть маркера, маркер идёт одним id."""
    merges = [((60, 124), 256)]  # "<|" — начало любого служебного токена
    ids = encode_with_specials("<|end|>x", merges, {"<|end|>": 300})
    assert ids == [300, 120]


def test_encode_with_specials_without_markers_matches_plain_encode():
    merges, _ = train_bpe(CORPUS, 20)
    assert encode_with_specials(CORPUS, merges, {"<|end|>": 300}) == encode(CORPUS, merges)


# ------------------------------------------------------- apply_chat_template
def test_chat_template_renders_one_message():
    assert apply_chat_template([{"role": "user", "content": "Hi"}]) == (
        "<|im_start|>user\nHi<|im_end|>\n"
    )
    assert apply_chat_template([]) == ""


def test_chat_template_renders_a_full_conversation():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert apply_chat_template(messages) == (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there!<|im_end|>\n"
    )


def test_chat_template_generation_prompt_leaves_the_turn_open():
    """Открытый блок ассистента без <|im_end|> — сигнал «теперь говори ты»."""
    out = apply_chat_template([{"role": "user", "content": "Hi"}], True)
    assert out.endswith(f"{IM_START}assistant\n")
    assert not out.endswith(IM_END + "\n")


def test_chat_template_markers_survive_tokenization_as_single_ids():
    """Каждый маркер обязан стать ровно одним id, а не рассыпаться на байты."""
    specials = {IM_START: 300, IM_END: 301}
    merges, _ = train_bpe(CORPUS, 20)
    text = apply_chat_template([{"role": "user", "content": "Hello"}])
    ids = encode_with_specials(text, merges, specials)
    assert ids.count(300) == 1
    assert ids.count(301) == 1
