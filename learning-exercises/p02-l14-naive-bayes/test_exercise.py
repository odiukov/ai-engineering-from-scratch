"""Тесты к уроку «Наивный Байес». Правь exercise.py."""

import math

import pytest

from exercise import (
    bag_of_words,
    build_vocabulary,
    class_log_priors,
    feature_log_probs,
    fit_multinomial_nb,
    log_scores,
    predict,
    predict_proba,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# данные из разбора урока: spam видел "free" 80 раз, "money" 60, "meeting" 10
LESSON_X = [[80, 60, 10], [5, 10, 100]]
LESSON_Y = ["spam", "ham"]


# ------------------------------------------------------- build_vocabulary
def test_vocabulary_is_sorted_and_unique():
    assert build_vocabulary([["free", "money"], ["money", "meeting"]]) == [
        "free",
        "meeting",
        "money",
    ]


def test_vocabulary_ignores_empty_documents():
    assert build_vocabulary([[], ["a"], []]) == ["a"]


def test_vocabulary_does_not_depend_on_document_order():
    """Индекс слова — это номер столбца, он обязан быть стабильным."""
    a = build_vocabulary([["b", "a"], ["c"]])
    b = build_vocabulary([["c"], ["a", "b"]])
    assert a == b


def test_vocabulary_of_nothing_is_empty():
    assert build_vocabulary([]) == []


# ----------------------------------------------------------- bag_of_words
def test_bag_of_words_counts_repetitions():
    assert bag_of_words(["free", "free", "money"], ["free", "meeting", "money"]) == [
        2,
        0,
        1,
    ]


def test_bag_of_words_ignores_unknown_words():
    """На проде приходят слова, которых в словаре нет. Падать нельзя."""
    assert bag_of_words(["free", "discombobulate"], ["free", "money"]) == [1, 0]


def test_bag_of_words_of_empty_document_is_all_zeros():
    assert bag_of_words([], ["a", "b", "c"]) == [0, 0, 0]


def test_bag_of_words_length_always_matches_vocabulary():
    vocab = ["a", "b", "c", "d"]
    assert len(bag_of_words(["a"], vocab)) == len(vocab)


# ------------------------------------------------------ class_log_priors
def test_class_log_priors_are_class_shares():
    priors = class_log_priors(["spam", "ham", "ham"])
    assert priors["ham"] == APPROX(math.log(2 / 3))
    assert priors["spam"] == APPROX(math.log(1 / 3))


def test_class_log_priors_of_one_class_is_log_of_one():
    assert class_log_priors(["a", "a"])["a"] == APPROX(0.0)


def test_class_priors_sum_to_one_after_exp():
    """Это распределение: сумма exp по классам обязана быть единицей."""
    priors = class_log_priors(["a", "b", "b", "c"])
    assert sum(math.exp(v) for v in priors.values()) == APPROX(1.0)


# ----------------------------------------------------- feature_log_probs
def test_feature_log_probs_match_the_lesson_arithmetic():
    probs = feature_log_probs(LESSON_X, LESSON_Y, 1.0)
    assert probs["spam"] == APPROX(
        [math.log(81 / 153), math.log(61 / 153), math.log(11 / 153)]
    )
    assert probs["ham"] == APPROX(
        [math.log(6 / 118), math.log(11 / 118), math.log(101 / 118)]
    )


def test_feature_probs_sum_to_one_over_the_vocabulary():
    """Знаменатель включает alpha * размер словаря — иначе это не распределение."""
    probs = feature_log_probs(LESSON_X, LESSON_Y, 1.0)
    assert sum(math.exp(v) for v in probs["spam"]) == APPROX(1.0)


def test_smoothing_keeps_an_unseen_word_finite():
    """Ловушка: без сглаживания log(0) = -inf и одно слово убивает класс."""
    probs = feature_log_probs([[5, 0], [0, 5]], ["a", "b"], 1.0)
    assert math.isfinite(probs["a"][1])


def test_bigger_alpha_flattens_the_distribution():
    """Сильное сглаживание сближает вероятности слов внутри класса."""
    weak = feature_log_probs(LESSON_X, LESSON_Y, 0.1)["spam"]
    strong = feature_log_probs(LESSON_X, LESSON_Y, 100.0)["spam"]
    assert max(strong) - min(strong) < max(weak) - min(weak)


def test_feature_log_probs_sum_counts_across_documents_of_one_class():
    """Два документа одного класса — это один общий счётчик, а не два."""
    two_docs = feature_log_probs([[1, 0], [1, 0]], ["a", "a"], 1.0)["a"]
    one_doc = feature_log_probs([[2, 0]], ["a"], 1.0)["a"]
    assert two_docs == APPROX(one_doc)


# --------------------------------------------------- fit_multinomial_nb
def test_fit_lists_classes_sorted():
    model = fit_multinomial_nb([[1, 0], [0, 1], [1, 1]], ["spam", "ham", "spam"])
    assert model["classes"] == ["ham", "spam"]


def test_fit_stores_one_log_prob_vector_per_class():
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    assert len(model["log_probs"]["a"]) == 2
    assert len(model["log_probs"]["b"]) == 2


def test_fit_is_a_single_pass_and_does_not_mutate_inputs():
    X = [[2, 0], [0, 2]]
    y = ["a", "b"]
    fit_multinomial_nb(X, y)
    assert X == [[2, 0], [0, 2]]
    assert y == ["a", "b"]


# ------------------------------------------------------------ log_scores
def test_log_scores_add_prior_and_weighted_log_likelihoods():
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    expected = math.log(0.5) + 1 * math.log(0.75)
    assert log_scores(model, [1, 0])["a"] == APPROX(expected)


def test_absent_words_contribute_nothing():
    """Мультиномиальный NB не наказывает за отсутствие слова: x_j = 0."""
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    only_prior = log_scores(model, [0, 0])
    assert only_prior["a"] == APPROX(model["log_priors"]["a"])


def test_repeating_a_word_doubles_its_evidence():
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    prior = model["log_priors"]["a"]
    once = log_scores(model, [1, 0])["a"] - prior
    twice = log_scores(model, [2, 0])["a"] - prior
    assert twice == APPROX(2 * once)


def test_log_scores_stay_finite_on_a_very_long_document():
    """В обычных вероятностях произведение 2000 множителей давно бы обнулилось."""
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    scores = log_scores(model, [1000, 800, 200])
    assert all(math.isfinite(v) for v in scores.values())


# --------------------------------------------------------------- predict
def test_predict_picks_the_class_with_the_matching_vocabulary():
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    assert predict(model, [[1, 0], [0, 1]]) == ["a", "b"]


def test_spam_wins_on_the_lesson_email():
    """"free" дважды и "money" один раз — письмо про заработок, не про встречу."""
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    assert predict(model, [[2, 1, 0]]) == ["spam"]


def test_prior_decides_when_the_likelihoods_tie():
    """Оба класса одинаково объясняют слово — побеждает более частый класс."""
    model = fit_multinomial_nb([[1, 1], [1, 1], [1, 1]], ["a", "a", "b"])
    assert predict(model, [[1, 0]]) == ["a"]


def test_predict_is_deterministic_on_a_full_tie():
    """Полная ничья: класс берётся первый по порядку, а не случайный."""
    model = fit_multinomial_nb([[1, 1], [1, 1]], ["a", "b"])
    assert predict(model, [[0, 0]]) == ["a"]
    assert predict(model, [[0, 0]]) == ["a"]


def test_predict_returns_one_label_per_row():
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    assert len(predict(model, [[1, 0, 0], [0, 0, 1], [3, 3, 3]])) == 3


# --------------------------------------------------------- predict_proba
def test_predict_proba_normalizes_to_one():
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    assert sum(predict_proba(model, [2, 1, 0]).values()) == APPROX(1.0)


def test_predict_proba_matches_hand_computed_values():
    model = fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])
    proba = predict_proba(model, [1, 0])
    assert proba["a"] == APPROX(0.75)
    assert proba["b"] == APPROX(0.25)


def test_predict_proba_survives_extreme_log_scores():
    """Ловушка: exp(-900) — это ноль, и нормировка превратится в 0/0."""
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    proba = predict_proba(model, [500, 400, 100])
    assert sum(proba.values()) == APPROX(1.0)
    assert all(0.0 <= v <= 1.0 for v in proba.values())


def test_predict_proba_agrees_with_predict():
    model = fit_multinomial_nb(LESSON_X, LESSON_Y)
    x = [2, 1, 0]
    proba = predict_proba(model, x)
    assert max(proba, key=proba.get) == predict(model, [x])[0]
