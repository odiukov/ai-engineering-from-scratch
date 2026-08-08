"""Тесты к уроку «Логистическая регрессия». Правь exercise.py."""

import math

import pytest

from exercise import (
    binary_cross_entropy,
    fit_logistic,
    logistic_gradients,
    predict_labels,
    predict_proba,
    sigmoid,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# два линейно разделимых облака: класс 0 у начала координат, класс 1 далеко
SEP_X = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [4.0, 4.0],
    [5.0, 4.0],
    [4.0, 5.0],
    [5.0, 5.0],
]
SEP_Y = [0, 0, 0, 0, 1, 1, 1, 1]


# ----------------------------------------------------------------- sigmoid
def test_sigmoid_of_zero_is_one_half():
    assert sigmoid(0) == APPROX(0.5)


def test_sigmoid_matches_the_formula():
    assert sigmoid(2) == APPROX(1 / (1 + math.exp(-2)))


def test_sigmoid_is_symmetric_around_one_half():
    """sigmoid(-z) = 1 - sigmoid(z): классы 0 и 1 равноправны."""
    for z in (0.3, 1.0, 7.5):
        assert sigmoid(-z) == pytest.approx(1 - sigmoid(z), abs=1e-12)


def test_sigmoid_is_monotone_increasing():
    values = [sigmoid(z) for z in (-5, -1, 0, 1, 5)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_sigmoid_stays_strictly_inside_zero_and_one():
    assert 0.0 < sigmoid(-20) < sigmoid(20) < 1.0


def test_sigmoid_survives_a_huge_negative_argument():
    """Ловушка: math.exp(1000) переполняется — обрежь z до вычисления exp."""
    assert sigmoid(-1000) == pytest.approx(0.0, abs=1e-9)
    assert sigmoid(1000) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------- predict_proba
def test_predict_proba_on_the_boundary_is_one_half():
    assert predict_proba([[0.0]], [1.0], 0.0) == APPROX([0.5])


def test_predict_proba_uses_all_features_and_the_bias():
    assert predict_proba([[1.0, 2.0]], [1.0, 1.0], -3.0) == APPROX([0.5])


def test_predict_proba_grows_along_a_positive_weight():
    probs = predict_proba([[-2.0], [0.0], [2.0]], [1.0], 0.0)
    assert probs[0] < probs[1] < probs[2]


def test_predict_proba_always_returns_probabilities():
    probs = predict_proba([[-500.0], [500.0]], [10.0], 0.0)
    assert all(0.0 <= p <= 1.0 for p in probs)


# ---------------------------------------------------------- predict_labels
def test_predict_labels_splits_at_the_default_threshold():
    assert predict_labels([0.2, 0.5, 0.9]) == [0, 1, 1]


def test_predict_labels_counts_exactly_the_threshold_as_positive():
    """Ловушка: сравнение >=, а не >. Ровно 0.5 — это класс 1."""
    assert predict_labels([0.5], 0.5) == [1]


def test_raising_the_threshold_can_only_reduce_the_positives():
    probs = [0.1, 0.35, 0.5, 0.65, 0.95]
    low = sum(predict_labels(probs, 0.3))
    high = sum(predict_labels(probs, 0.7))
    assert high < low


# ---------------------------------------------------- binary_cross_entropy
def test_bce_of_confident_correct_predictions_is_near_zero():
    assert binary_cross_entropy([1, 0], [0.999999, 0.000001]) < 1e-5


def test_bce_of_a_coin_flip_is_ln_two():
    assert binary_cross_entropy([1, 0], [0.5, 0.5]) == APPROX(math.log(2))


def test_bce_of_a_confidently_wrong_prediction_is_huge_but_finite():
    """Ловушка: math.log(0) бросает ValueError — вероятность надо зажать."""
    loss = binary_cross_entropy([1], [0.0])
    assert math.isfinite(loss) and loss > 30


def test_bce_punishes_confident_mistakes_harder_than_hesitant_ones():
    assert binary_cross_entropy([1], [0.01]) > binary_cross_entropy([1], [0.4])


def test_bce_is_symmetric_between_the_two_classes():
    assert binary_cross_entropy([1], [0.3]) == APPROX(binary_cross_entropy([0], [0.7]))


def test_rank_deficient_features_allow_nonunique_parameterisations():
    """Выпуклость не обещает единственные веса при зависимых признаках."""
    X = [[-1.0, -1.0], [0.0, 0.0], [2.0, 2.0]]
    first = predict_proba(X, [1.0, 0.0], 0.0)
    second = predict_proba(X, [0.0, 1.0], 0.0)
    assert first == APPROX(second)
    assert binary_cross_entropy([0, 0, 1], first) == APPROX(
        binary_cross_entropy([0, 0, 1], second)
    )


# ------------------------------------------------------- logistic_gradients
def test_logistic_gradients_on_a_hand_checked_example():
    """При нулевых весах все вероятности равны 0.5, счёт делается устно."""
    dw, db = logistic_gradients([[1.0], [3.0]], [0, 1], [0.0], 0.0)
    assert dw == APPROX([-0.5])
    assert db == APPROX(0.0)


def test_logistic_gradients_use_one_over_n_not_two_over_n():
    """Ловушка: множитель 1/n. Квадрата в логлоссе нет, двойке взяться неоткуда."""
    dw, db = logistic_gradients([[1.0]], [0], [0.0], 0.0)
    assert dw == APPROX([0.5])
    assert db == APPROX(0.5)


def test_logistic_gradient_length_matches_the_number_of_features():
    dw, _ = logistic_gradients(SEP_X, SEP_Y, [0.0, 0.0], 0.0)
    assert len(dw) == 2


def test_logistic_gradient_pushes_the_weights_toward_separation():
    """Класс 1 сидит в больших x — спуск обязан двигать вес вверх."""
    dw, _ = logistic_gradients(SEP_X, SEP_Y, [0.0, 0.0], 0.0)
    assert all(g < 0 for g in dw)


# ------------------------------------------------------------ fit_logistic
def test_fit_separates_linearly_separable_data_perfectly():
    w, b, _ = fit_logistic(SEP_X, SEP_Y, lr=0.5, epochs=1000)
    predicted = predict_labels(predict_proba(SEP_X, w, b))
    assert predicted == SEP_Y


def test_fit_loss_decreases_every_single_epoch():
    """Логлосс выпуклый: при адекватном lr спуск не имеет права подниматься."""
    _, _, history = fit_logistic(SEP_X, SEP_Y, lr=0.1, epochs=100)
    assert all(later < earlier for earlier, later in zip(history, history[1:]))


def test_fit_history_has_one_entry_per_epoch():
    _, _, history = fit_logistic(SEP_X, SEP_Y, lr=0.1, epochs=23)
    assert len(history) == 23


def test_fit_ends_up_much_better_than_the_coin_flip_it_started_from():
    _, _, history = fit_logistic(SEP_X, SEP_Y, lr=0.5, epochs=1000)
    assert history[0] < math.log(2)
    assert history[-1] < 0.05


def test_fit_gives_the_two_classes_probabilities_on_opposite_sides_of_a_half():
    w, b, _ = fit_logistic(SEP_X, SEP_Y, lr=0.5, epochs=1000)
    probs = predict_proba(SEP_X, w, b)
    assert max(probs[:4]) < 0.5 < min(probs[4:])


def test_fit_is_reproducible_without_any_seed():
    """Старт с нулей: два одинаковых вызова совпадают до бита."""
    assert fit_logistic(SEP_X, SEP_Y, lr=0.3, epochs=50) == fit_logistic(
        SEP_X, SEP_Y, lr=0.3, epochs=50
    )


def test_fit_cannot_separate_xor_because_the_boundary_is_a_line():
    """Граница логистической регрессии всегда прямая, а XOR прямой не режется."""
    xor_x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    xor_y = [0, 1, 1, 0]
    w, b, _ = fit_logistic(xor_x, xor_y, lr=0.5, epochs=2000)
    predicted = predict_labels(predict_proba(xor_x, w, b))
    assert predicted != xor_y


# ----------------------------------------------------------------- softmax
def test_softmax_probabilities_sum_to_one():
    assert sum(softmax([1.0, 2.0, 3.0])) == APPROX(1.0)


def test_softmax_of_equal_scores_is_uniform():
    assert softmax([1.0, 1.0, 1.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_keeps_the_winner():
    probs = softmax([0.5, 4.0, -1.0])
    assert probs.index(max(probs)) == 1


def test_softmax_ignores_a_constant_shift_of_all_scores():
    """Сдвиг всех очков на константу не меняет распределение — на этом и держится защита от переполнения."""
    assert softmax([1.0, 2.0, 3.0]) == pytest.approx(
        softmax([101.0, 102.0, 103.0]), abs=1e-12
    )


def test_softmax_survives_huge_scores():
    """Ловушка: math.exp(1000) переполняется — вычти максимум перед exp."""
    assert softmax([1000.0, 1001.0]) == pytest.approx(
        [1 / (1 + math.e), math.e / (1 + math.e)], abs=1e-9
    )


def test_softmax_of_two_scores_matches_the_sigmoid():
    """Softmax на двух классах — это та же сигмоида от разности очков."""
    assert softmax([2.0, 0.0])[0] == APPROX(sigmoid(2.0))
