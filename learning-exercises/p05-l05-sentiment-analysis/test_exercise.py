"""Тесты к уроку «Анализ тональности». Правь exercise.py."""

import math

import pytest

from exercise import (
    apply_negation,
    evaluate,
    macro_f1,
    predict_lr,
    predict_nb,
    sigmoid,
    train_lr,
    train_nb,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------- apply_negation
def test_apply_negation_prefixes_the_scope_and_stops_at_punctuation():
    assert apply_negation(["not", "good", "at", "all", ".", "but", "funny"]) == [
        "not", "NOT_good", "NOT_at", "NOT_all", ".", "but", "funny",
    ]


def test_apply_negation_leaves_the_negation_word_itself_alone():
    """Ловушка: префикс получают слова ПОСЛЕ «not», а не само «not»."""
    assert apply_negation(["never", "again"]) == ["never", "NOT_again"]


def test_apply_negation_is_a_no_op_without_a_negation_word():
    tokens = ["a", "great", "film", "."]
    assert apply_negation(tokens) == tokens


def test_apply_negation_makes_good_and_not_good_different_features():
    """Ради этого всё и затевалось: два разных признака вместо одного."""
    plain = set(apply_negation(["good"]))
    negated = set(apply_negation(["not", "good"]))
    assert not (plain & negated)


def test_apply_negation_reopens_the_scope_after_a_terminator():
    out = apply_negation(["not", "a", ",", "not", "b"])
    assert out == ["not", "NOT_a", ",", "not", "NOT_b"]


# ---------------------------------------------------------------- train_nb
def test_train_nb_priors_are_document_shares():
    priors, _ = train_nb({"pos": [["a"], ["b"]], "neg": [["c"]]}, ["a", "b", "c"])
    assert priors == {"pos": APPROX(2 / 3), "neg": APPROX(1 / 3)}


def test_train_nb_word_probabilities_of_a_class_sum_to_one():
    """Это распределение по словарю: суммы должны сойтись ровно в 1."""
    _, probs = train_nb({"pos": [["a", "a", "b"]], "neg": [["c"]]}, ["a", "b", "c"])
    for cls in probs:
        assert sum(probs[cls].values()) == APPROX(1.0)


def test_train_nb_smoothing_keeps_unseen_words_above_zero():
    """Без alpha слово, не встреченное в классе, обнулило бы весь счёт документа."""
    _, probs = train_nb({"pos": [["good"]], "neg": [["bad"]]}, ["good", "bad"])
    assert probs["neg"]["good"] > 0.0
    assert probs["pos"]["bad"] > 0.0


def test_train_nb_matches_the_hand_computed_laplace_numbers():
    _, probs = train_nb({"pos": [["good"]], "neg": [["bad"]]}, ["good", "bad"], alpha=1.0)
    assert probs["pos"] == {"good": APPROX(2 / 3), "bad": APPROX(1 / 3)}


def test_train_nb_ignores_tokens_outside_the_vocabulary():
    """Ловушка: OOV не должен попадать ни в счётчик, ни в знаменатель."""
    _, with_oov = train_nb({"pos": [["good", "zzz"]]}, ["good"])
    _, without = train_nb({"pos": [["good"]]}, ["good"])
    assert with_oov["pos"] == without["pos"]


def test_train_nb_smaller_alpha_sharpens_the_distribution():
    _, soft = train_nb({"pos": [["a", "a", "a"]]}, ["a", "b"], alpha=1.0)
    _, sharp = train_nb({"pos": [["a", "a", "a"]]}, ["a", "b"], alpha=0.01)
    assert sharp["pos"]["a"] > soft["pos"]["a"]


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_train_nb_rejects_non_positive_smoothing(alpha):
    with pytest.raises(ValueError, match="alpha must be positive"):
        train_nb({"pos": [["good"]]}, ["good", "bad"], alpha=alpha)


# -------------------------------------------------------------- predict_nb
_PRIORS = {"pos": 0.5, "neg": 0.5}
_PROBS = {
    "pos": {"good": 0.9, "bad": 0.1},
    "neg": {"good": 0.1, "bad": 0.9},
}


def test_predict_nb_picks_the_class_the_evidence_leans_to():
    assert predict_nb(["good"], _PRIORS, _PROBS) == "pos"
    assert predict_nb(["bad"], _PRIORS, _PROBS) == "neg"


def test_predict_nb_ignores_out_of_vocabulary_tokens():
    assert predict_nb(["good", "zzz"], _PRIORS, _PROBS) == "pos"


def test_predict_nb_lets_the_prior_decide_an_empty_document():
    priors = {"pos": 0.9, "neg": 0.1}
    assert predict_nb([], priors, _PROBS) == "pos"


def test_predict_nb_accumulates_evidence_across_repeated_words():
    """Один «bad» против двух «good» — документ всё ещё положительный."""
    assert predict_nb(["good", "good", "bad"], _PRIORS, _PROBS) == "pos"


def test_predict_nb_survives_a_long_document_without_underflow():
    """Ловушка: перемножение вероятностей вместо сложения логарифмов даёт 0.0."""
    doc = ["good"] * 500 + ["bad"] * 400
    assert predict_nb(doc, _PRIORS, _PROBS) == "pos"


# ----------------------------------------------------------------- sigmoid
def test_sigmoid_of_zero_is_a_half():
    assert sigmoid(0) == APPROX(0.5)


def test_sigmoid_is_symmetric_around_a_half():
    assert sigmoid(3.0) + sigmoid(-3.0) == APPROX(1.0)


def test_sigmoid_survives_a_huge_argument():
    """Без обрезки math.exp(1000) роняет программу с OverflowError."""
    assert sigmoid(1000.0) == APPROX(sigmoid(20.0))
    assert sigmoid(-1000.0) == APPROX(sigmoid(-20.0))


# ---------------------------------------------------------------- train_lr
_X = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.2]]
_Y = [1, 1, 0, 0]


def _lr_loss(X, y, w, b, l2):
    """Та же функция потерь, что минимизирует train_lr — для численной проверки."""
    total = 0.0
    for row, yi in zip(X, y):
        p = sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b)
        total -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
    return total / len(y) + 0.5 * l2 * sum(wi * wi for wi in w)


def test_train_lr_weight_step_matches_the_numeric_gradient():
    w0, b0, lr, l2 = [0.3, -0.2], 0.1, 0.05, 0.02
    w1, _ = train_lr(_X, _Y, epochs=1, lr=lr, l2=l2, w0=w0, b0=b0)

    h = 1e-6
    for k in range(2):
        up, down = list(w0), list(w0)
        up[k] += h
        down[k] -= h
        numeric = (_lr_loss(_X, _Y, up, b0, l2) - _lr_loss(_X, _Y, down, b0, l2)) / (2 * h)
        assert (w0[k] - w1[k]) / lr == pytest.approx(numeric, abs=1e-6)


def test_train_lr_bias_step_matches_the_numeric_gradient():
    """Ловушка: L2 штрафует веса, но не сдвиг — численный градиент это покажет."""
    w0, b0, lr, l2, h = [0.3, -0.2], 0.1, 0.05, 0.02, 1e-6
    _, b1 = train_lr(_X, _Y, epochs=1, lr=lr, l2=l2, w0=w0, b0=b0)
    numeric = (_lr_loss(_X, _Y, w0, b0 + h, l2) - _lr_loss(_X, _Y, w0, b0 - h, l2)) / (2 * h)
    assert (b0 - b1) / lr == pytest.approx(numeric, abs=1e-6)


def test_train_lr_lowers_the_loss():
    w, b = train_lr(_X, _Y, epochs=300, lr=0.5, l2=0.0)
    assert _lr_loss(_X, _Y, w, b, 0.0) < _lr_loss(_X, _Y, [0.0, 0.0], 0.0, 0.0)


def test_train_lr_separates_a_separable_dataset():
    w, b = train_lr(_X, _Y, epochs=2000, lr=0.5, l2=0.0)
    assert predict_lr(_X, w, b) == _Y


def test_train_lr_l2_shrinks_the_weights():
    """Смысл регуляризации: с ней норма весов меньше при тех же данных."""
    free, _ = train_lr(_X, _Y, epochs=500, lr=0.5, l2=0.0)
    penalized, _ = train_lr(_X, _Y, epochs=500, lr=0.5, l2=1.0)
    assert sum(v * v for v in penalized) < sum(v * v for v in free)


def test_train_lr_with_zero_epochs_returns_the_starting_point():
    w, b = train_lr(_X, _Y, epochs=0, lr=0.5, l2=0.1, w0=[7.0, -3.0], b0=2.0)
    assert w == APPROX([7.0, -3.0])
    assert b == APPROX(2.0)


def test_train_lr_step_does_not_depend_on_dataset_size():
    """Ловушка: без деления на n градиент растёт вместе с выборкой."""
    small, _ = train_lr(_X, _Y, epochs=1, lr=0.5, l2=0.0)
    big, _ = train_lr(_X * 5, _Y * 5, epochs=1, lr=0.5, l2=0.0)
    assert big == APPROX(small)


# --------------------------------------------------------------- predict_lr
def test_predict_lr_thresholds_the_logit_at_zero():
    assert predict_lr([[1.0], [-1.0]], [2.0], 0.0) == [1, 0]


def test_predict_lr_sends_an_exact_tie_to_the_positive_class():
    """Ловушка: порог >= 0.5, а не > 0.5."""
    assert predict_lr([[0.0]], [1.0], 0.0) == [1]


def test_predict_lr_bias_can_flip_every_prediction():
    X = [[0.1], [0.2]]
    assert predict_lr(X, [1.0], -10.0) == [0, 0]
    assert predict_lr(X, [1.0], 10.0) == [1, 1]


def test_predict_lr_returns_labels_not_probabilities():
    out = predict_lr([[0.3], [-0.3]], [1.0], 0.0)
    assert set(out) <= {0, 1}


# ----------------------------------------------------------------- evaluate
def test_evaluate_counts_the_confusion_matrix():
    m = evaluate([1, 1, 0, 0], [1, 0, 0, 0])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 0, 2, 1)


def test_evaluate_precision_and_recall_answer_different_questions():
    m = evaluate([1, 1, 0, 0], [1, 0, 0, 0])
    assert m["precision"] == APPROX(1.0)
    assert m["recall"] == APPROX(0.5)


def test_evaluate_f1_is_the_harmonic_mean():
    m = evaluate([1, 1, 0, 0], [1, 0, 0, 0])
    p, r = m["precision"], m["recall"]
    assert m["f1"] == APPROX(2 * p * r / (p + r))


def test_evaluate_of_a_model_that_never_says_yes_is_zero_not_a_crash():
    """Ловушка: tp + fp = 0 — это precision 0.0, а не деление на ноль."""
    m = evaluate([1, 1, 0], [0, 0, 0])
    assert m["precision"] == APPROX(0.0)
    assert m["f1"] == APPROX(0.0)


def test_evaluate_counts_cover_every_example():
    m = evaluate([1, 0, 1, 0, 1], [1, 1, 0, 0, 1])
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 5


def test_evaluate_of_a_perfect_prediction_is_all_ones():
    m = evaluate([1, 0, 1], [1, 0, 1])
    assert (m["precision"], m["recall"], m["f1"], m["accuracy"]) == (
        APPROX(1.0), APPROX(1.0), APPROX(1.0), APPROX(1.0),
    )


# ----------------------------------------------------------------- macro_f1
def test_macro_f1_of_a_perfect_prediction_is_one():
    assert macro_f1([1, 1, 0, 0], [1, 1, 0, 0]) == APPROX(1.0)


def test_macro_f1_exposes_the_majority_class_classifier():
    """Главный урок метрики: accuracy 0.9 при macro-F1 0.47."""
    y_true = [1] * 9 + [0]
    y_pred = [1] * 10
    assert evaluate(y_true, y_pred)["accuracy"] == APPROX(0.9)
    assert macro_f1(y_true, y_pred) < 0.5


def test_macro_f1_does_not_care_which_class_is_called_positive():
    """Равный вес обоим классам: инверсия меток ничего не меняет."""
    y_true, y_pred = [1, 1, 0, 1, 0], [1, 0, 0, 1, 1]
    flipped_true = [1 - y for y in y_true]
    flipped_pred = [1 - y for y in y_pred]
    assert macro_f1(y_true, y_pred) == APPROX(macro_f1(flipped_true, flipped_pred))


def test_macro_f1_is_the_average_of_both_per_class_f1():
    y_true, y_pred = [1, 1, 0, 1, 0], [1, 0, 0, 1, 1]
    pos = evaluate(y_true, y_pred)["f1"]
    neg = evaluate([1 - y for y in y_true], [1 - y for y in y_pred])["f1"]
    assert macro_f1(y_true, y_pred) == APPROX((pos + neg) / 2)
