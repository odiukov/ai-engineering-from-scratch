"""Тесты к уроку «Эмбеддинги слов: Word2Vec с нуля». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    analogy,
    build_vocab,
    cosine_similarity,
    nearest,
    negative_samples,
    sigmoid,
    skipgram_pairs,
    train_pair,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Разворачивает список списков в плоский список: pytest.approx не умеет вложенные."""
    return [x for row in M for x in row]


# ------------------------------------------------------------- build_vocab
def test_build_vocab_numbers_words_by_first_appearance():
    assert build_vocab([["the", "cat"], ["the", "dog"]]) == {"the": 0, "cat": 1, "dog": 2}


def test_build_vocab_indices_are_a_dense_range():
    """Индексы — это номера строк матрицы эмбеддингов, дыр в них быть не может."""
    vocab = build_vocab([["a", "b", "a", "c", "b", "d"]])
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_build_vocab_of_empty_corpus_is_empty():
    assert build_vocab([]) == {}
    assert build_vocab([[], []]) == {}


def test_build_vocab_does_not_care_about_document_boundaries():
    """Один документ из двух или два по одному — словарь один и тот же."""
    assert build_vocab([["a", "b"]]) == build_vocab([["a"], ["b"]])


# ---------------------------------------------------------- skipgram_pairs
def test_skipgram_pairs_takes_both_neighbours():
    assert skipgram_pairs([["a", "b", "c"]], window=1) == [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
    ]


def test_skipgram_pairs_skips_the_center_position():
    """Окно шире документа: каждое слово встречается с каждым другим, но не с собой."""
    pairs = skipgram_pairs([["a", "b", "c", "d"]], window=3)
    assert len(pairs) == 4 * 3
    assert all(x != y for x, y in pairs)


def test_skipgram_pairs_pairs_two_copies_of_the_same_word():
    """Пропускается ПОЗИЦИЯ i, а не строка: два вхождения «a» — законная пара."""
    assert ("a", "a") in skipgram_pairs([["a", "b", "a"]], window=2)


def test_skipgram_pairs_clips_the_window_at_the_edges():
    """Ловушка: у краёв окно обрезается, а не заворачивается на другой конец."""
    pairs = skipgram_pairs([["a", "b", "c", "d", "e"]], window=2)
    from_first = [p for p in pairs if p[0] == "a"]
    from_middle = [p for p in pairs if p[0] == "c"]
    assert len(from_first) == 2
    assert len(from_middle) == 4
    assert ("a", "e") not in pairs


def test_skipgram_pairs_do_not_leak_across_documents():
    pairs = skipgram_pairs([["a"], ["b"]], window=5)
    assert pairs == []


def test_skipgram_pairs_is_symmetric_as_a_set():
    """Если (x, y) — пара, то и (y, x) тоже: окно смотрит в обе стороны."""
    pairs = set(skipgram_pairs([["a", "b", "c", "d"]], window=2))
    assert all((y, x) in pairs for x, y in pairs)


# ----------------------------------------------------------------- sigmoid
def test_sigmoid_of_zero_is_a_half():
    assert sigmoid(0) == APPROX(0.5)


def test_sigmoid_is_symmetric_around_a_half():
    assert sigmoid(2.0) + sigmoid(-2.0) == APPROX(1.0)


def test_sigmoid_survives_a_huge_negative_argument():
    """Без обрезки math.exp(1000) роняет программу с OverflowError."""
    assert sigmoid(-1000.0) == APPROX(sigmoid(-20.0))
    assert sigmoid(1000.0) == APPROX(sigmoid(20.0))


def test_sigmoid_stays_strictly_inside_zero_and_one():
    assert 0.0 < sigmoid(-1000.0) < sigmoid(0.0) < sigmoid(1000.0) < 1.0


# -------------------------------------------------------- negative_samples
def test_negative_samples_returns_exactly_k_indices():
    out = negative_samples(10, {0}, 4, random.Random(0))
    assert len(out) == 4


def test_negative_samples_never_returns_an_excluded_index():
    out = negative_samples(6, {0, 1, 2}, 30, random.Random(7))
    assert all(i in (3, 4, 5) for i in out)


def test_negative_samples_is_reproducible_from_the_seed():
    """Глобальный random сделал бы обучение невоспроизводимым."""
    a = negative_samples(50, {0}, 10, random.Random(123))
    b = negative_samples(50, {0}, 10, random.Random(123))
    assert a == b


def test_negative_samples_allows_repeats():
    """Настоящий Word2Vec тянет негативы с возвращением — повтор не ошибка."""
    out = negative_samples(2, {0}, 5, random.Random(1))
    assert out == [1, 1, 1, 1, 1]


def test_negative_samples_raises_instead_of_hanging_when_nothing_is_left():
    """Ловушка: цикл «тяни, пока не подойдёт» на полном exclude не кончится никогда."""
    with pytest.raises(ValueError):
        negative_samples(3, {0, 1, 2}, 1, random.Random(0))


# -------------------------------------------------------------- train_pair
def _loss(v_c, u_pos, u_negs):
    """Та же функция потерь, что оптимизирует train_pair — для численной проверки."""
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    total = -math.log(sigmoid(dot(v_c, u_pos)))
    for u in u_negs:
        total -= math.log(sigmoid(-dot(v_c, u)))
    return total


def _tables():
    W = [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.25]]
    W_prime = [[0.15, 0.05], [-0.3, 0.2], [0.4, -0.1]]
    return W, W_prime


def test_train_pair_returns_the_logistic_loss_before_the_step():
    W, W_prime = _tables()
    expected = _loss(W[0], W_prime[1], [W_prime[2]])
    assert train_pair(W, W_prime, 0, 1, [2], lr=0.1) == APPROX(expected)


def test_train_pair_center_update_matches_the_numeric_gradient():
    """Аналитический градиент по v_c против центральной разности."""
    W, W_prime = _tables()
    before = list(W[0])
    u_pos, u_negs = list(W_prime[1]), [list(W_prime[2])]
    lr = 0.05
    train_pair(W, W_prime, 0, 1, [2], lr=lr)

    h = 1e-6
    for k in range(2):
        up, down = list(before), list(before)
        up[k] += h
        down[k] -= h
        numeric = (_loss(up, u_pos, u_negs) - _loss(down, u_pos, u_negs)) / (2 * h)
        assert (before[k] - W[0][k]) / lr == pytest.approx(numeric, abs=1e-6)


def test_train_pair_context_update_matches_the_numeric_gradient():
    """Ловушка: градиент по u_pos считается от v_c ДО шага, а не после."""
    W, W_prime = _tables()
    v_c, before = list(W[0]), list(W_prime[1])
    u_negs = [list(W_prime[2])]
    lr = 0.05
    train_pair(W, W_prime, 0, 1, [2], lr=lr)

    h = 1e-6
    for k in range(2):
        up, down = list(before), list(before)
        up[k] += h
        down[k] -= h
        numeric = (_loss(v_c, up, u_negs) - _loss(v_c, down, u_negs)) / (2 * h)
        assert (before[k] - W_prime[1][k]) / lr == pytest.approx(numeric, abs=1e-6)


def test_train_pair_pulls_the_positive_pair_together():
    """Смысл шага: скалярное произведение центра и контекста растёт."""
    W, W_prime = _tables()
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    before = dot(W[0], W_prime[1])
    train_pair(W, W_prime, 0, 1, [2], lr=0.5)
    assert dot(W[0], W_prime[1]) > before


def test_train_pair_pushes_the_negative_pair_apart():
    W, W_prime = _tables()
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    before = dot(W[0], W_prime[2])
    train_pair(W, W_prime, 0, 1, [2], lr=0.5)
    assert dot(W[0], W_prime[2]) < before


def test_train_pair_leaves_untouched_rows_alone():
    """Строки, не участвующие в паре, обновляться не должны — иначе это не SGD."""
    W, W_prime = _tables()
    train_pair(W, W_prime, 0, 1, [2], lr=0.5)
    assert flat([W[1], W[2], W_prime[0]]) == APPROX(
        flat([[0.1, 0.4], [-0.5, 0.25], [0.15, 0.05]])
    )


def test_train_pair_lowers_the_loss_it_just_measured():
    W, W_prime = _tables()
    first = train_pair(W, W_prime, 0, 1, [2], lr=0.1)
    second = train_pair(W, W_prime, 0, 1, [2], lr=0.1)
    assert second < first


# ------------------------------------------------------- cosine_similarity
def test_cosine_of_identical_directions_is_one():
    assert cosine_similarity([1, 0], [2, 0]) == APPROX(1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    """Главное свойство: длина не влияет — потому эмбеддинги и сравнивают косинусом."""
    a, b = [1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]
    scaled = [100 * x for x in a]
    assert cosine_similarity(scaled, b) == pytest.approx(cosine_similarity(a, b), abs=1e-12)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1, 2], [-1, -2]) == APPROX(-1.0)


def test_cosine_with_a_zero_vector_is_zero_not_a_crash():
    """Ловушка: у нулевого вектора нет направления, делить на его длину нельзя."""
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


# ----------------------------------------------------------------- nearest
def _toy():
    vocab = {"a": 0, "b": 1, "c": 2}
    W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    return vocab, W


def test_nearest_finds_the_exact_direction_first():
    vocab, W = _toy()
    assert nearest(vocab, W, [1.0, 0.0], topk=1) == [("a", APPROX(1.0))]


def test_nearest_returns_at_most_topk_pairs():
    vocab, W = _toy()
    assert len(nearest(vocab, W, [1.0, 1.0], topk=2)) == 2


def test_nearest_is_sorted_by_decreasing_cosine():
    vocab, W = _toy()
    sims = [s for _, s in nearest(vocab, W, [1.0, 0.2], topk=3)]
    assert sims == sorted(sims, reverse=True)


def test_nearest_skips_excluded_indices():
    vocab, W = _toy()
    words = [w for w, _ in nearest(vocab, W, [1.0, 0.0], topk=3, exclude={0})]
    assert "a" not in words


def test_nearest_breaks_ties_deterministically():
    """Ловушка: при равных косинусах порядок обязан быть один и тот же каждый раз."""
    vocab = {"x": 0, "y": 1, "z": 2}
    W = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert nearest(vocab, W, [1.0, 0.0], topk=3) == nearest(vocab, W, [1.0, 0.0], topk=3)


# ----------------------------------------------------------------- analogy
def _royal():
    vocab = {"man": 0, "king": 1, "woman": 2, "queen": 3, "cat": 4}
    W = [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, -0.2]]
    return vocab, W


def test_analogy_lands_on_the_fourth_word():
    """king - man + woman = queen, ровно как в уроке."""
    vocab, W = _royal()
    assert analogy(vocab, W, "man", "king", "woman")[0][0] == "queen"


def test_analogy_excludes_its_own_three_inputs():
    """Ловушка: без исключения ответом почти всегда становится сам c."""
    vocab, W = _royal()
    words = [w for w, _ in analogy(vocab, W, "man", "king", "woman", topk=5)]
    assert not ({"man", "king", "woman"} & set(words))


def test_analogy_respects_topk():
    vocab, W = _royal()
    assert len(analogy(vocab, W, "man", "king", "woman", topk=1)) == 1


def test_analogy_direction_matters():
    """Поменять a и b местами — другой запрос и другой ответ."""
    vocab, W = _royal()
    forward = analogy(vocab, W, "man", "king", "woman", topk=2)
    backward = analogy(vocab, W, "king", "man", "woman", topk=2)
    assert forward != backward
