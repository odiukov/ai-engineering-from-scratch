"""Тесты к уроку «Ансамбли». Правь exercise.py."""

import pytest

from exercise import (
    bootstrap_indices,
    fit_adaboost,
    fit_bagging,
    fit_stump,
    majority_vote,
    predict_ensemble,
    predict_stump,
    vote_accuracy,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# «Интервал»: один пень такую разметку не проведёт, ансамбль — проведёт
INTERVAL_X = [[float(i)] for i in range(8)]
INTERVAL_Y = [-1, -1, 1, 1, 1, 1, -1, -1]


def accuracy(predictions, y_true):
    return sum(1 for p, t in zip(predictions, y_true) if p == t) / len(y_true)


# ------------------------------------------------------------ majority_vote
def test_majority_vote_picks_the_most_common_label():
    assert majority_vote([1, 1, -1]) == 1


def test_majority_vote_lets_weights_override_the_head_count():
    """Мнение одной тяжёлой модели перевешивает двух лёгких — это soft voting."""
    assert majority_vote([1, 1, -1], [0.1, 0.1, 5.0]) == -1


def test_majority_vote_breaks_a_tie_deterministically():
    """Ловушка: при ничьей ответ обязан быть один и тот же от запуска к запуску."""
    assert majority_vote([1, -1]) == majority_vote([1, -1]) == 1


def test_majority_vote_works_with_any_labels():
    assert majority_vote(["cat", "dog", "cat"]) == "cat"


def test_voting_beats_every_single_member():
    """Ключевое свойство ансамбля: каждый ошибается на своём объекте,
    большинство не ошибается нигде."""
    truth = [1, 1, 1]
    members = [
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ]
    for member in members:
        assert accuracy(member, truth) == APPROX(2 / 3)
    voted = [majority_vote([m[i] for m in members]) for i in range(3)]
    assert accuracy(voted, truth) == APPROX(1.0)


# ------------------------------------------------------------ vote_accuracy
def test_vote_accuracy_of_one_model_is_its_own_accuracy():
    assert vote_accuracy(0.6, 1) == APPROX(0.6)


def test_vote_accuracy_of_three_models_matches_the_binomial_sum():
    """3 * p^2 * (1 - p) + p^3 для p = 0.6."""
    assert vote_accuracy(0.6, 3) == APPROX(0.648)


def test_vote_accuracy_grows_with_the_crowd_when_each_model_beats_the_coin():
    assert vote_accuracy(0.6, 3) < vote_accuracy(0.6, 21) < vote_accuracy(0.6, 101)


def test_vote_accuracy_collapses_when_each_model_is_worse_than_a_coin():
    """Условие теоремы Кондорсе — p > 0.5. Ниже неё толпа делает только хуже."""
    assert vote_accuracy(0.4, 3) > vote_accuracy(0.4, 21) > vote_accuracy(0.4, 101)


def test_vote_accuracy_of_coin_flippers_stays_a_coin_flip():
    assert vote_accuracy(0.5, 21) == APPROX(0.5)


def test_vote_accuracy_is_symmetric_around_one_half():
    """Ансамбль из моделей с точностью p и 1-p зеркален: суммы дают единицу."""
    assert vote_accuracy(0.7, 15) + vote_accuracy(0.3, 15) == APPROX(1.0)


# -------------------------------------------------------- bootstrap_indices
def test_bootstrap_indices_has_the_same_size_as_the_data():
    assert len(bootstrap_indices(100, 0)) == 100


def test_bootstrap_indices_stays_inside_the_range():
    assert all(0 <= i < 50 for i in bootstrap_indices(50, 1))


def test_bootstrap_indices_is_reproducible_for_the_same_seed():
    assert bootstrap_indices(50, 3) == bootstrap_indices(50, 3)


def test_bootstrap_indices_differs_between_seeds():
    """Иначе весь ансамбль обучится на одной и той же выборке."""
    assert bootstrap_indices(50, 3) != bootstrap_indices(50, 4)


def test_bootstrap_leaves_about_a_third_of_the_data_out_of_bag():
    """Ловушка: с возвращением, а не перестановка. Уникальных примерно 63%."""
    unique = len(set(bootstrap_indices(1000, 0))) / 1000
    assert 0.58 < unique < 0.68


# ------------------------------------------------- fit_stump / predict_stump
def test_stump_separates_a_trivial_split():
    stump = fit_stump([[0.0], [1.0]], [-1, 1])
    assert predict_stump(stump, [0.0]) == -1
    assert predict_stump(stump, [1.0]) == 1


def test_stump_finds_the_polarity_when_the_labels_are_flipped():
    """Ловушка: без polarity пень умеет только «выше порога — плюс»."""
    X = [[0.0], [1.0], [2.0], [3.0]]
    stump = fit_stump(X, [1, 1, -1, -1])
    assert [predict_stump(stump, x) for x in X] == [1, 1, -1, -1]


def test_stump_picks_the_informative_feature():
    X = [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    stump = fit_stump(X, [-1, -1, 1, 1])
    assert stump["feature"] == 0


def test_stump_on_constant_labels_answers_the_same_thing_everywhere():
    X = [[5.0], [9.0]]
    stump = fit_stump(X, [1, 1])
    assert [predict_stump(stump, x) for x in X] == [1, 1]


def test_stump_follows_the_sample_weights():
    """AdaBoost держится ровно на этом: тяжёлый объект обязан быть угадан."""
    X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1, -1, -1, -1, 1]
    assert predict_stump(fit_stump(X, y), [4.0]) == -1
    heavy_last = fit_stump(X, y, [0.01, 0.01, 0.01, 0.01, 0.96])
    assert predict_stump(heavy_last, [4.0]) == 1


def test_a_single_stump_cannot_carve_out_an_interval():
    """Потолок одного пня на этих данных — 0.75. Дальше нужен ансамбль."""
    stump = fit_stump(INTERVAL_X, INTERVAL_Y)
    predictions = [predict_stump(stump, x) for x in INTERVAL_X]
    assert accuracy(predictions, INTERVAL_Y) == APPROX(0.75)


# --------------------------------------------------------- predict_ensemble
def test_predict_ensemble_with_equal_weights_is_a_plain_majority():
    up = {"feature": 0, "threshold": 0.0, "polarity": 1}
    down = {"feature": 0, "threshold": 0.0, "polarity": -1}
    assert predict_ensemble([(up, 1.0), (up, 1.0), (down, 1.0)], [5.0]) == 1


def test_predict_ensemble_respects_model_weights():
    up = {"feature": 0, "threshold": 0.0, "polarity": 1}
    down = {"feature": 0, "threshold": 0.0, "polarity": -1}
    assert predict_ensemble([(up, 0.1), (up, 0.1), (down, 5.0)], [5.0]) == -1


def test_predict_ensemble_of_one_model_equals_that_model():
    stump = fit_stump(INTERVAL_X, INTERVAL_Y)
    assert all(
        predict_ensemble([(stump, 1.0)], x) == predict_stump(stump, x)
        for x in INTERVAL_X
    )


# --------------------------------------------------------------- fit_bagging
def test_fit_bagging_returns_the_requested_number_of_equal_weight_models():
    ensemble = fit_bagging(INTERVAL_X, INTERVAL_Y, n_models=5, seed=0)
    assert len(ensemble) == 5
    assert [weight for _, weight in ensemble] == APPROX([1.0] * 5)


def test_fit_bagging_is_reproducible_for_the_same_seed():
    assert fit_bagging(INTERVAL_X, INTERVAL_Y, n_models=5, seed=1) == fit_bagging(
        INTERVAL_X, INTERVAL_Y, n_models=5, seed=1
    )


def test_fit_bagging_produces_different_models():
    """Ловушка: один seed на всех даст n одинаковых пней и нулевое разнообразие."""
    ensemble = fit_bagging(INTERVAL_X, INTERVAL_Y, n_models=9, seed=0)
    distinct = {(s["feature"], s["threshold"], s["polarity"]) for s, _ in ensemble}
    assert len(distinct) > 1


def test_fit_bagging_is_no_worse_than_a_single_stump():
    ensemble = fit_bagging(INTERVAL_X, INTERVAL_Y, n_models=9, seed=0)
    single = fit_stump(INTERVAL_X, INTERVAL_Y)
    bagged = accuracy([predict_ensemble(ensemble, x) for x in INTERVAL_X], INTERVAL_Y)
    alone = accuracy([predict_stump(single, x) for x in INTERVAL_X], INTERVAL_Y)
    assert bagged >= alone


# -------------------------------------------------------------- fit_adaboost
def test_fit_adaboost_returns_one_pair_per_round():
    assert len(fit_adaboost(INTERVAL_X, INTERVAL_Y, n_rounds=6)) == 6


def test_fit_adaboost_gives_positive_weight_to_better_than_random_stumps():
    """alpha = 0.5 * ln((1 - err) / err) положительна, пока err < 0.5."""
    ensemble = fit_adaboost(INTERVAL_X, INTERVAL_Y, n_rounds=6)
    assert all(alpha > 0 for _, alpha in ensemble)


def test_adaboost_ensemble_beats_every_one_of_its_members():
    """Требование урока целиком: каждый пень внутри не выше 0.75, ансамбль — 1.0."""
    ensemble = fit_adaboost(INTERVAL_X, INTERVAL_Y, n_rounds=6)
    for stump, _ in ensemble:
        member = accuracy([predict_stump(stump, x) for x in INTERVAL_X], INTERVAL_Y)
        assert member <= 0.75
    voted = accuracy([predict_ensemble(ensemble, x) for x in INTERVAL_X], INTERVAL_Y)
    assert voted == APPROX(1.0)


def test_adaboost_of_one_round_is_just_a_stump():
    ensemble = fit_adaboost(INTERVAL_X, INTERVAL_Y, n_rounds=1)
    assert ensemble[0][0] == fit_stump(INTERVAL_X, INTERVAL_Y)


def test_fit_adaboost_is_deterministic():
    """Случайности в бустинге нет вовсе — seed ему не нужен."""
    assert fit_adaboost(INTERVAL_X, INTERVAL_Y, n_rounds=4) == fit_adaboost(
        INTERVAL_X, INTERVAL_Y, n_rounds=4
    )
