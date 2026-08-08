"""Тесты к уроку «Метод опорных векторов». Правь exercise.py."""

import math

import pytest

from exercise import (
    decision_function,
    find_support_vectors,
    fit_linear_svm,
    hinge_gradients,
    hinge_loss,
    polynomial_kernel,
    rbf_kernel,
    svm_predict,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# линейно разделимые облака. Метки -1 и +1, как требует SVM
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
SEP_Y = [-1, -1, -1, -1, 1, 1, 1, 1]


# ------------------------------------------------------- decision_function
def test_decision_function_is_signed():
    assert decision_function([[1.0], [-1.0]], [2.0], 0.0) == APPROX([2.0, -2.0])


def test_decision_function_returns_the_bias_at_the_origin():
    assert decision_function([[0.0, 0.0]], [1.0, 1.0], 3.0) == APPROX([3.0])


def test_decision_function_is_zero_exactly_on_the_boundary():
    assert decision_function([[1.0, -1.0]], [1.0, 1.0], 0.0) == APPROX([0.0])


def test_decision_score_is_not_geometric_distance_until_normalised():
    """Масштабирование (w, b) меняет score, но не расстояние до границы."""
    point = [[3.0, 4.0]]
    score = decision_function(point, [3.0, 4.0], -5.0)[0]
    scaled = decision_function(point, [30.0, 40.0], -50.0)[0]
    assert scaled == APPROX(10.0 * score)
    assert score / 5.0 == APPROX(scaled / 50.0)


# -------------------------------------------------------------- hinge_loss
def test_hinge_loss_is_zero_beyond_the_margin():
    assert hinge_loss([[2.0]], [1], [1.0], 0.0) == APPROX(0.0)


def test_hinge_loss_is_still_zero_exactly_on_the_margin():
    """Излом ровно в y*f(x) = 1: слева штраф начинается, справа его нет."""
    assert hinge_loss([[1.0]], [1], [1.0], 0.0) == APPROX(0.0)


def test_hinge_loss_is_one_on_the_decision_boundary():
    assert hinge_loss([[0.0]], [1], [1.0], 0.0) == APPROX(1.0)


def test_hinge_loss_grows_linearly_past_the_boundary():
    """Не квадратично: один выброс не выкручивает границу, как это делает MSE."""
    assert hinge_loss([[-1.0]], [1], [1.0], 0.0) == APPROX(2.0)
    assert hinge_loss([[-2.0]], [1], [1.0], 0.0) == APPROX(3.0)


def test_hinge_loss_averages_over_the_batch():
    assert hinge_loss([[0.0], [2.0]], [1, 1], [1.0], 0.0) == APPROX(0.5)


def test_hinge_loss_needs_minus_one_labels_not_zero_labels():
    """Ловушка: с меткой 0 штраф равен max(0, 1 - 0) = 1 при любых весах."""
    assert hinge_loss([[100.0]], [0], [1.0], 0.0) == APPROX(1.0)
    assert hinge_loss([[100.0]], [-1], [-1.0], 0.0) == APPROX(0.0)


# --------------------------------------------------------- hinge_gradients
def test_hinge_gradient_is_flat_for_a_confident_point():
    assert hinge_gradients([[1.0]], [1], [5.0], 0.0, 0.0) == (APPROX([0.0]), APPROX(0.0))


def test_hinge_gradient_pushes_a_violating_point_out():
    assert hinge_gradients([[1.0]], [1], [0.0], 0.0, 0.0) == (
        APPROX([-1.0]),
        APPROX(-1.0),
    )


def test_hinge_gradient_chooses_zero_at_exactly_margin_one():
    """В изломе допустимы разные субградиенты; здесь выбран нулевой."""
    assert hinge_gradients([[1.0]], [1], [1.0], 0.0, 0.0) == (
        APPROX([0.0]),
        APPROX(0.0),
    )


def test_hinge_gradient_ignores_points_far_outside_the_margin():
    """Разреженность решения: дальние точки в градиент не входят вообще.

    Добавили вторую точку с запасом 100 — она не добавила ни одного
    слагаемого, изменился только делитель n.
    """
    dw_one, _ = hinge_gradients([[0.0]], [1], [1.0], 0.0, 0.0)
    dw_two, _ = hinge_gradients([[0.0], [100.0]], [1, 1], [1.0], 0.0, 0.0)
    assert dw_two[0] == APPROX(dw_one[0] / 2)


def test_regularisation_is_all_that_is_left_when_nobody_violates():
    dw, db = hinge_gradients([[10.0]], [1], [1.0], 0.0, lambda_param=0.25)
    assert dw == APPROX([0.25])
    assert db == APPROX(0.0)


def test_regularisation_does_not_touch_the_bias():
    """Ловушка: lambda штрафует ||w||, потому что от него зависит ширина полосы.

    Сдвиг b полосу не расширяет и не сужает — тянуть его к нулю не за что.
    """
    _, db = hinge_gradients([[10.0]], [1], [1.0], 7.0, lambda_param=1000.0)
    assert db == APPROX(0.0)


def test_hinge_gradient_length_matches_the_number_of_features():
    dw, _ = hinge_gradients(SEP_X, SEP_Y, [0.0, 0.0], 0.0)
    assert len(dw) == 2


# ---------------------------------------------------------- fit_linear_svm
def test_fit_separates_linearly_separable_data():
    w, b, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000)
    assert svm_predict(SEP_X, w, b) == SEP_Y


def test_fit_objective_drops_a_lot_from_where_it_started():
    _, _, history = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000)
    assert history[-1] < history[0]
    assert history[-1] < 0.2


def test_fit_history_has_one_entry_per_epoch():
    _, _, history = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=31)
    assert len(history) == 31


def test_fit_is_reproducible_without_any_seed():
    """Старт с нулей: два одинаковых вызова совпадают до бита."""
    assert fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=100) == fit_linear_svm(
        SEP_X, SEP_Y, lr=0.05, epochs=100
    )


def test_stronger_regularisation_gives_a_wider_margin():
    """Ширина полосы это 2/||w||: меньше норма — шире полоса."""
    w_weak, _, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000, lambda_param=0.01)
    w_strong, _, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000, lambda_param=1.0)
    norm = lambda w: math.sqrt(sum(v * v for v in w))
    assert norm(w_strong) < norm(w_weak)


def test_fit_puts_the_boundary_between_the_two_clouds():
    w, b, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000)
    scores = decision_function(SEP_X, w, b)
    assert max(scores[:4]) < 0 < min(scores[4:])


# --------------------------------------------------------------- svm_predict
def test_svm_predict_only_ever_returns_minus_one_and_plus_one():
    w, b, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=200)
    assert set(svm_predict(SEP_X, w, b)) <= {-1, 1}


def test_svm_predict_breaks_the_tie_on_the_boundary_toward_plus_one():
    assert svm_predict([[1.0], [-1.0], [0.0]], [1.0], 0.0) == [1, -1, 1]


# -------------------------------------------------- find_support_vectors
def test_support_vectors_are_the_points_on_the_margin():
    X = [[-5.0], [-1.0], [1.0], [5.0]]
    y = [-1, -1, 1, 1]
    assert find_support_vectors(X, y, [1.0], 0.0) == [1, 2]


def test_a_confident_point_is_not_a_support_vector():
    assert find_support_vectors([[5.0]], [1], [1.0], 0.0) == []


def test_a_misclassified_point_is_always_a_support_vector():
    """Запас отрицателен, а значит заведомо меньше единицы."""
    assert find_support_vectors([[-1.0]], [1], [1.0], 0.0) == [0]


def test_support_vectors_are_a_minority_of_a_well_separated_sample():
    """Ради этого SVM и хранят: на инференсе нужны не все точки, а немногие."""
    w, b, _ = fit_linear_svm(SEP_X, SEP_Y, lr=0.05, epochs=1000)
    assert len(find_support_vectors(SEP_X, SEP_Y, w, b)) < len(SEP_X)


# ---------------------------------------------------------- polynomial_kernel
def test_polynomial_kernel_of_degree_one_is_the_dot_product():
    assert polynomial_kernel([1.0, 2.0], [3.0, 4.0], degree=1, c=0.0) == APPROX(11.0)


def test_polynomial_kernel_on_a_hand_checked_example():
    assert polynomial_kernel([1.0], [2.0], degree=2, c=1.0) == APPROX(9.0)


def test_polynomial_kernel_is_symmetric():
    a, b = [1.0, -2.0, 0.5], [3.0, 4.0, -1.0]
    assert polynomial_kernel(a, b) == APPROX(polynomial_kernel(b, a))


def test_a_higher_degree_stretches_the_differences():
    """Именно это и делает ядро: усиливает разницу между похожим и непохожим."""
    close = polynomial_kernel([1.0], [1.0], degree=5, c=0.0)
    far = polynomial_kernel([1.0], [2.0], degree=5, c=0.0)
    assert far / close == APPROX(32.0)


# ------------------------------------------------------------------- rbf_kernel
def test_rbf_kernel_of_a_point_with_itself_is_one():
    assert rbf_kernel([1.0, 2.0], [1.0, 2.0]) == APPROX(1.0)


def test_rbf_kernel_on_a_hand_checked_example():
    assert rbf_kernel([0.0], [1.0], gamma=1.0) == APPROX(1 / math.e)


def test_rbf_kernel_uses_the_squared_distance():
    """Ловушка: без корня. С корнем получится совсем другая функция."""
    assert rbf_kernel([0.0], [2.0], gamma=1.0) == APPROX(math.exp(-4.0))


def test_rbf_kernel_falls_off_with_distance():
    values = [rbf_kernel([0.0], [d]) for d in (0.0, 1.0, 3.0, 10.0)]
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


def test_rbf_kernel_stays_inside_zero_and_one():
    assert 0.0 < rbf_kernel([0.0], [8.0]) < rbf_kernel([0.0], [0.5]) <= 1.0


def test_rbf_kernel_is_symmetric():
    a, b = [1.0, -2.0], [3.0, 4.0]
    assert rbf_kernel(a, b) == APPROX(rbf_kernel(b, a))


def test_a_larger_gamma_narrows_the_bell():
    """Большая gamma — каждая точка влияет только на ближайших соседей."""
    assert rbf_kernel([0.0], [1.0], gamma=5.0) < rbf_kernel([0.0], [1.0], gamma=0.1)
