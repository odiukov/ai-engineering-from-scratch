"""Тесты к уроку «GloVe, FastText и subword-эмбеддинги». Правь exercise.py."""

import math

import pytest

from exercise import (
    apply_bpe,
    build_cooccurrence,
    char_ngrams,
    fasttext_vector,
    glove_step,
    glove_weight,
    learn_bpe,
    merge_pair,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Разворачивает список списков в плоский список: pytest.approx не умеет вложенные."""
    return [x for row in M for x in row]


# ------------------------------------------------------ build_cooccurrence
def test_cooccurrence_weights_neighbours_by_inverse_distance():
    vocab, counts = build_cooccurrence([["a", "b", "c"]], window=2)
    assert vocab == {"a": 0, "b": 1, "c": 2}
    assert counts[(0, 1)] == APPROX(1.0)
    assert counts[(0, 2)] == APPROX(0.5)


def test_cooccurrence_is_symmetric():
    """X[i][j] и X[j][i] должны совпасть: окно смотрит в обе стороны одинаково."""
    _, counts = build_cooccurrence([["a", "b", "c", "d"]], window=3)
    assert all(counts[(i, j)] == APPROX(counts[(j, i)]) for i, j in counts)


def test_cooccurrence_never_counts_a_position_with_itself():
    """Пропускается ПОЗИЦИЯ i: у документа из разных слов диагонали не будет."""
    _, counts = build_cooccurrence([["a", "b", "c"]], window=5)
    assert all(i != j for i, j in counts)


def test_cooccurrence_counts_a_word_against_its_own_second_occurrence():
    """А вот два вхождения одного слова — законная пара, и она попадает на диагональ."""
    _, counts = build_cooccurrence([["a", "b", "a"]], window=5)
    assert counts[(0, 0)] == APPROX(1.0)  # два раза по 1/2


def test_cooccurrence_stops_at_the_document_boundary():
    """Ловушка: последнее слово первого документа не сосед первому слову второго."""
    vocab, counts = build_cooccurrence([["a"], ["b"]], window=5)
    assert counts == {}
    assert vocab == {"a": 0, "b": 1}


def test_cooccurrence_window_limits_the_reach():
    _, counts = build_cooccurrence([["a", "b", "c", "d"]], window=1)
    assert (0, 2) not in counts
    assert (0, 1) in counts


def test_cooccurrence_accumulates_repeated_contexts():
    """Слово, встреченное рядом дважды, набирает вес дважды."""
    _, counts = build_cooccurrence([["a", "b"], ["a", "b"]], window=1)
    assert counts[(0, 1)] == APPROX(2.0)


# ----------------------------------------------------------- glove_weight
def test_glove_weight_saturates_at_x_max():
    assert glove_weight(100.0) == APPROX(1.0)
    assert glove_weight(10_000.0) == APPROX(1.0)


def test_glove_weight_of_zero_is_zero():
    assert glove_weight(0.0) == APPROX(0.0)


def test_glove_weight_grows_with_the_count():
    """Монотонность — это и есть смысл: чаще пара, больше её вклад в потери."""
    values = [glove_weight(x) for x in (1.0, 10.0, 50.0, 99.0)]
    assert values == sorted(values)


def test_glove_weight_is_sublinear():
    """alpha < 1: удвоение частоты повышает вес меньше, чем вдвое."""
    assert glove_weight(20.0) < 2 * glove_weight(10.0)


# ------------------------------------------------------------- glove_step
def _tables():
    W = [[0.2, -0.1], [0.3, 0.4]]
    W_tilde = [[-0.15, 0.25], [0.05, 0.35]]
    b = [0.1, -0.2]
    b_tilde = [0.05, 0.15]
    return W, W_tilde, b, b_tilde


def _loss(v_i, u_j, b_i, b_j, x_ij):
    """Та же функция потерь, что минимизирует glove_step — для численной проверки."""
    diff = sum(a * c for a, c in zip(v_i, u_j)) + b_i + b_j - math.log(x_ij)
    return glove_weight(x_ij) * diff * diff


def test_glove_step_returns_the_weighted_squared_error_before_the_step():
    W, W_tilde, b, b_tilde = _tables()
    expected = _loss(W[0], W_tilde[1], b[0], b_tilde[1], 4.0)
    assert glove_step(W, W_tilde, b, b_tilde, 0, 1, 4.0, lr=0.01) == APPROX(expected)


def test_glove_step_center_update_matches_the_numeric_gradient():
    W, W_tilde, b, b_tilde = _tables()
    before, u_j, b_i, b_j = list(W[0]), list(W_tilde[1]), b[0], b_tilde[1]
    lr, x = 0.01, 4.0
    glove_step(W, W_tilde, b, b_tilde, 0, 1, x, lr=lr)

    h = 1e-6
    for k in range(2):
        up, down = list(before), list(before)
        up[k] += h
        down[k] -= h
        numeric = (_loss(up, u_j, b_i, b_j, x) - _loss(down, u_j, b_i, b_j, x)) / (2 * h)
        assert (before[k] - W[0][k]) / lr == pytest.approx(numeric, abs=1e-6)


def test_glove_step_context_update_matches_the_numeric_gradient():
    """Ловушка: градиент по W_tilde[j] берётся от W[i] ДО шага, а не после."""
    W, W_tilde, b, b_tilde = _tables()
    v_i, before, b_i, b_j = list(W[0]), list(W_tilde[1]), b[0], b_tilde[1]
    lr, x = 0.01, 4.0
    glove_step(W, W_tilde, b, b_tilde, 0, 1, x, lr=lr)

    h = 1e-6
    for k in range(2):
        up, down = list(before), list(before)
        up[k] += h
        down[k] -= h
        numeric = (_loss(v_i, up, b_i, b_j, x) - _loss(v_i, down, b_i, b_j, x)) / (2 * h)
        assert (before[k] - W_tilde[1][k]) / lr == pytest.approx(numeric, abs=1e-6)


def test_glove_step_bias_update_matches_the_numeric_gradient():
    W, W_tilde, b, b_tilde = _tables()
    v_i, u_j, before, b_j = list(W[0]), list(W_tilde[1]), b[0], b_tilde[1]
    lr, x, h = 0.01, 4.0, 1e-6
    glove_step(W, W_tilde, b, b_tilde, 0, 1, x, lr=lr)
    numeric = (
        _loss(v_i, u_j, before + h, b_j, x) - _loss(v_i, u_j, before - h, b_j, x)
    ) / (2 * h)
    assert (before - b[0]) / lr == pytest.approx(numeric, abs=1e-6)


def test_glove_step_lowers_the_loss_it_just_measured():
    W, W_tilde, b, b_tilde = _tables()
    first = glove_step(W, W_tilde, b, b_tilde, 0, 1, 4.0, lr=0.01)
    second = glove_step(W, W_tilde, b, b_tilde, 0, 1, 4.0, lr=0.01)
    assert second < first


def test_glove_step_does_nothing_when_the_pair_already_fits():
    """diff = 0 — предсказание уже равно log(x). Двигать нечего, потери нулевые."""
    W, W_tilde = [[1.0, 0.0]], [[1.0, 0.0]]
    b, b_tilde = [0.0], [0.0]
    loss = glove_step(W, W_tilde, b, b_tilde, 0, 0, math.e, lr=0.5)
    assert loss == APPROX(0.0)
    assert flat(W + W_tilde) == APPROX([1.0, 0.0, 1.0, 0.0])
    assert b + b_tilde == APPROX([0.0, 0.0])


def test_glove_step_leaves_untouched_rows_alone():
    W, W_tilde, b, b_tilde = _tables()
    glove_step(W, W_tilde, b, b_tilde, 0, 1, 4.0, lr=0.5)
    assert W[1] == APPROX([0.3, 0.4])
    assert W_tilde[0] == APPROX([-0.15, 0.25])


# ------------------------------------------------------------ char_ngrams
def test_char_ngrams_wraps_the_word_in_boundary_markers():
    assert char_ngrams("cat", 3, 3) == {"<ca", "cat", "at>", "<cat>"}


def test_char_ngrams_of_where_has_all_lengths_from_three_to_six():
    """5 троек + 4 четвёрки + 3 пятёрки + 2 шестёрки + само «<where>» = 15."""
    assert len(char_ngrams("where")) == 15
    assert {"<where", "where>"} <= char_ngrams("where")


def test_char_ngrams_always_contains_the_whole_word():
    """Ловушка: «<word>» кладётся всегда, даже когда длиннее n_max."""
    assert "<encyclopedia>" in char_ngrams("encyclopedia", 3, 4)


def test_char_ngrams_of_a_short_word_is_still_non_empty():
    assert char_ngrams("a", 3, 6) == {"<a>"}


def test_char_ngrams_of_related_words_overlap():
    """Ради этого FastText и придумали: playing и played делят куски."""
    shared = char_ngrams("playing") & char_ngrams("played")
    assert {"<pl", "pla", "lay", "<pla", "play"} <= shared


# -------------------------------------------------------- fasttext_vector
def test_fasttext_vector_sums_the_known_ngrams():
    table = {"<ca": [1.0, 0.0], "cat": [0.0, 1.0]}
    assert fasttext_vector("cat", table, 3, 3) == APPROX([1.0, 1.0])


def test_fasttext_vector_skips_unknown_ngrams():
    table = {"cat": [2.0, 3.0]}
    assert fasttext_vector("cat", table, 3, 3) == APPROX([2.0, 3.0])


def test_fasttext_vector_gives_an_oov_word_a_vector():
    """Главное свойство: слово, которого не было в обучении, всё равно получает вектор."""
    table = {g: [1.0] for g in char_ngrams("where")}
    assert fasttext_vector("whereupon", table) is not None


def test_fasttext_vector_is_none_when_nothing_is_known():
    table = {g: [1.0] for g in char_ngrams("where")}
    assert fasttext_vector("zqxj", table) is None


def test_fasttext_vector_of_a_morphological_variant_is_closer_than_of_a_stranger():
    """played ближе к playing, чем banana — потому что общих n-грамм больше."""
    table = {g: [1.0] for g in char_ngrams("playing")}
    near = fasttext_vector("played", table)
    far = fasttext_vector("banana", table)
    assert near is not None and near[0] > 3
    assert far is None


# ------------------------------------------------------------- merge_pair
def test_merge_pair_glues_every_occurrence():
    assert merge_pair(["l", "o", "w", "l", "o"], ("l", "o")) == ["lo", "w", "lo"]


def test_merge_pair_advances_by_two_after_a_merge():
    """Ловушка: сдвиг на один склеит средний токен дважды."""
    assert merge_pair(["a", "a", "a"], ("a", "a")) == ["aa", "a"]


def test_merge_pair_leaves_the_list_alone_when_the_pair_is_absent():
    assert merge_pair(["c", "a", "t"], ("x", "y")) == ["c", "a", "t"]


def test_merge_pair_does_not_mutate_the_input():
    tokens = ["l", "o", "w"]
    merge_pair(tokens, ("l", "o"))
    assert tokens == ["l", "o", "w"]


# -------------------------------------------------------------- learn_bpe
def test_learn_bpe_picks_the_most_frequent_pairs_first():
    assert learn_bpe({"low": 5, "lower": 2}, 2) == [("l", "o"), ("lo", "w")]


def test_learn_bpe_returns_exactly_k_merges_when_pairs_remain():
    assert len(learn_bpe({"newest": 6, "widest": 3}, 4) ) == 4


def test_learn_bpe_stops_early_when_nothing_is_left_to_merge():
    """Ловушка: слово схлопнулось в один токен, пар больше нет — надо выйти."""
    assert len(learn_bpe({"ab": 1}, 50)) == 2


def test_learn_bpe_is_deterministic_on_ties():
    corpus = {"ab": 1, "cd": 1, "ef": 1}
    assert learn_bpe(corpus, 3) == learn_bpe(dict(reversed(list(corpus.items()))), 3)


def test_learn_bpe_marks_the_end_of_the_word():
    """"</w>" отличает "est" в конце слова от "est" в середине."""
    merges = learn_bpe({"est": 10}, 3)
    assert any("</w>" in a or "</w>" in b for a, b in merges)


# -------------------------------------------------------------- apply_bpe
def test_apply_bpe_uses_the_learned_merges():
    merges = learn_bpe({"low": 5, "lower": 2}, 2)
    assert apply_bpe("low", merges) == ["low", "</w>"]


def test_apply_bpe_falls_back_to_characters_for_unknown_words():
    """У BPE не бывает OOV: незнакомое слово просто распадается на символы."""
    merges = learn_bpe({"low": 5, "lower": 2}, 2)
    assert apply_bpe("xyz", merges) == ["x", "y", "z", "</w>"]


def test_apply_bpe_always_ends_with_the_word_boundary_marker():
    merges = learn_bpe({"newest": 6, "widest": 3}, 8)
    assert "</w>" in "".join(apply_bpe("newest", merges))


def test_apply_bpe_shortens_the_sequence_as_merges_accumulate():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    few = apply_bpe("lowest", learn_bpe(corpus, 2))
    many = apply_bpe("lowest", learn_bpe(corpus, 12))
    assert len(many) < len(few)


def test_apply_bpe_respects_the_order_of_merges():
    """Ловушка: ('lo','w') бесполезен, пока не применён ('l','o')."""
    correct = [("l", "o"), ("lo", "w")]
    reversed_order = [("lo", "w"), ("l", "o")]
    assert apply_bpe("low", correct) == ["low", "</w>"]
    assert apply_bpe("low", reversed_order) == ["lo", "w", "</w>"]
