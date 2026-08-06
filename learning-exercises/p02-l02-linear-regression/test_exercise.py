"""Тесты к уроку «Линейная регрессия». Правь exercise.py."""

import pytest

from exercise import (
    fit_closed_form,
    fit_gradient_descent,
    fit_ridge,
    gradients,
    mse,
    predict,
    r_squared,
    standardize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# прямая y = 2x + 1 без шума
LINE_X = [[0.0], [1.0], [2.0], [3.0]]
LINE_Y = [1.0, 3.0, 5.0, 7.0]

# плоскость y = 2*x1 + 3*x2 + 1 без шума
PLANE_X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
PLANE_Y = [3.0, 4.0, 6.0, 8.0]


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


# ----------------------------------------------------------------- predict
def test_predict_on_one_feature():
    assert predict([[1.0], [2.0]], [3.0], 7.0) == APPROX([10.0, 13.0])


def test_predict_sums_all_features():
    assert predict([[1.0, 2.0]], [10.0, 100.0], 0.0) == APPROX([210.0])


def test_predict_adds_the_bias_once_not_per_feature():
    """Ловушка: b — одно слагаемое на объект, а не на каждый признак."""
    assert predict([[1.0, 1.0, 1.0]], [0.0, 0.0, 0.0], 5.0) == APPROX([5.0])


def test_predict_with_zero_weights_is_a_flat_line():
    assert predict([[5.0], [-100.0]], [0.0], 2.0) == APPROX([2.0, 2.0])


# --------------------------------------------------------------------- mse
def test_mse_of_perfect_predictions_is_zero():
    assert mse([1.0, 2.0], [1.0, 2.0]) == APPROX(0.0)


def test_mse_on_a_hand_checked_example():
    assert mse([0.0, 0.0], [1.0, 3.0]) == APPROX(5.0)


def test_mse_does_not_care_about_the_sign_of_the_error():
    assert mse([0.0], [2.0]) == APPROX(mse([0.0], [-2.0]))


def test_mse_punishes_one_big_error_more_than_many_small_ones():
    """Суммарная ошибка одна и та же (10), а MSE отличается в десять раз."""
    one_big = mse([0.0] * 10, [10.0] + [0.0] * 9)
    ten_small = mse([0.0] * 10, [1.0] * 10)
    assert one_big == APPROX(10.0)
    assert ten_small == APPROX(1.0)


def test_mse_on_empty_input_is_zero_not_a_crash():
    assert mse([], []) == APPROX(0.0)


# --------------------------------------------------------------- gradients
def test_gradients_are_zero_when_predictions_are_exact():
    dw, db = gradients(LINE_X, LINE_Y, [2.0], 1.0)
    assert dw == APPROX([0.0])
    assert db == APPROX(0.0)


def test_gradients_on_a_hand_checked_example():
    assert gradients([[1.0]], [0.0], [1.0], 0.0) == (APPROX([2.0]), APPROX(2.0))


def test_gradient_sign_points_uphill():
    """Ловушка: ошибка это (pred - y). Модель завышает — градиент положителен."""
    dw, db = gradients([[1.0]], [0.0], [5.0], 0.0)
    assert dw[0] > 0 and db > 0


def test_gradient_length_matches_the_number_of_features():
    dw, _ = gradients(PLANE_X, PLANE_Y, [0.0, 0.0], 0.0)
    assert len(dw) == 2


def test_gradient_ignores_a_feature_that_is_always_zero():
    """Признак-ноль ничего не объясняет — его частная производная нулевая."""
    dw, _ = gradients([[1.0, 0.0], [2.0, 0.0]], [0.0, 0.0], [1.0, 1.0], 0.0)
    assert dw[1] == APPROX(0.0)


# ------------------------------------------------------ fit_gradient_descent
def test_gradient_descent_recovers_a_noiseless_line():
    w, b, _ = fit_gradient_descent(LINE_X, LINE_Y, lr=0.05, epochs=5000)
    assert w[0] == pytest.approx(2.0, abs=1e-3)
    assert b == pytest.approx(1.0, abs=1e-3)


def test_gradient_descent_recovers_a_noiseless_plane():
    w, b, _ = fit_gradient_descent(PLANE_X, PLANE_Y, lr=0.05, epochs=5000)
    assert w == pytest.approx([2.0, 3.0], abs=1e-3)
    assert b == pytest.approx(1.0, abs=1e-3)


def test_gradient_descent_loss_decreases_every_single_epoch():
    """На выпуклой MSE с адекватным lr спуск не имеет права подниматься."""
    _, _, history = fit_gradient_descent(LINE_X, LINE_Y, lr=0.05, epochs=50)
    assert all(later < earlier for earlier, later in zip(history, history[1:]))


def test_gradient_descent_history_has_one_entry_per_epoch():
    _, _, history = fit_gradient_descent(LINE_X, LINE_Y, lr=0.01, epochs=17)
    assert len(history) == 17


def test_too_large_learning_rate_makes_the_loss_explode():
    """Слишком крупный шаг перелетает минимум и уходит в расходимость."""
    _, _, history = fit_gradient_descent(LINE_X, LINE_Y, lr=1.0, epochs=30)
    assert history[-1] > history[0]


def test_gradient_descent_starts_from_zeros_so_it_is_reproducible():
    """Никакого random: два одинаковых вызова обязаны совпасть до бита."""
    first = fit_gradient_descent(PLANE_X, PLANE_Y, lr=0.05, epochs=100)
    second = fit_gradient_descent(PLANE_X, PLANE_Y, lr=0.05, epochs=100)
    assert first == second


# ---------------------------------------------------------- fit_closed_form
def test_closed_form_recovers_the_exact_line_in_one_pass():
    assert fit_closed_form([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == (
        APPROX(2.0),
        APPROX(1.0),
    )


def test_closed_form_gives_zero_slope_for_flat_data():
    assert fit_closed_form([0.0, 1.0, 5.0], [4.0, 4.0, 4.0]) == (
        APPROX(0.0),
        APPROX(4.0),
    )


def test_closed_form_survives_a_constant_x():
    """Ловушка: все x равны — знаменатель нулевой, делить нельзя."""
    assert fit_closed_form([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) == (
        APPROX(0.0),
        APPROX(2.0),
    )


def test_closed_form_and_gradient_descent_agree():
    """Один и тот же минимум одной и той же выпуклой чаши, разными путями."""
    w_gd, b_gd, _ = fit_gradient_descent(LINE_X, LINE_Y, lr=0.05, epochs=5000)
    w_cf, b_cf = fit_closed_form([row[0] for row in LINE_X], LINE_Y)
    assert w_gd[0] == pytest.approx(w_cf, abs=1e-3)
    assert b_gd == pytest.approx(b_cf, abs=1e-3)


# --------------------------------------------------------------- r_squared
def test_r_squared_of_a_perfect_fit_is_one():
    assert r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == APPROX(1.0)


def test_r_squared_of_the_mean_predictor_is_zero():
    assert r_squared([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == APPROX(0.0)


def test_r_squared_goes_negative_for_a_model_worse_than_the_mean():
    """Отрицательный R² — не баг, а диагноз: проще было брать среднее."""
    assert r_squared([1.0, 2.0, 3.0], [10.0, 10.0, 10.0]) < 0


def test_r_squared_does_not_depend_on_the_scale_of_y():
    """MSE вырастет в 100 раз, а R² не сдвинется — в этом весь смысл метрики."""
    plain = r_squared([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    scaled = r_squared([10.0, 20.0, 30.0], [11.0, 19.0, 32.0])
    assert plain == pytest.approx(scaled, abs=1e-9)


def test_r_squared_with_constant_targets_is_zero_not_a_crash():
    assert r_squared([5.0, 5.0], [5.0, 4.0]) == APPROX(0.0)


# ------------------------------------------------------------- standardize
def test_standardize_centres_and_rescales_one_column():
    scaled, means, stds = standardize([[0.0], [2.0]])
    assert flat(scaled) == APPROX([-1.0, 1.0])
    assert means == APPROX([1.0])
    assert stds == APPROX([1.0])


def test_standardized_columns_have_zero_mean_and_unit_std():
    X = [[1.0, 100.0], [2.0, 300.0], [3.0, 200.0], [10.0, 700.0]]
    scaled, _, _ = standardize(X)
    for j in range(2):
        column = [row[j] for row in scaled]
        mean = sum(column) / len(column)
        std = (sum((v - mean) ** 2 for v in column) / len(column)) ** 0.5
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert std == pytest.approx(1.0, abs=1e-9)


def test_standardize_turns_a_constant_column_into_zeros():
    """Ловушка: std = 0 — деления быть не должно, ни NaN, ни исключения."""
    scaled, _, stds = standardize([[7.0], [7.0], [7.0]])
    assert flat(scaled) == APPROX([0.0, 0.0, 0.0])
    assert stds == APPROX([0.0])


def test_standardize_makes_wildly_scaled_features_comparable():
    """Признак в сотнях больше не забивает признак в единицах."""
    X = [[1.0, 1000.0], [2.0, 2000.0], [3.0, 3000.0]]
    scaled, _, _ = standardize(X)
    assert [row[0] for row in scaled] == APPROX([row[1] for row in scaled])


# ---------------------------------------------------------------- fit_ridge
def test_ridge_with_zero_alpha_is_plain_gradient_descent():
    w_gd, b_gd, _ = fit_gradient_descent(PLANE_X, PLANE_Y, lr=0.05, epochs=500)
    w_ridge, b_ridge = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=500, alpha=0.0)
    assert w_ridge == pytest.approx(w_gd, abs=1e-9)
    assert b_ridge == pytest.approx(b_gd, abs=1e-9)


def test_ridge_shrinks_the_weights_toward_zero():
    w_plain, _ = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=3000, alpha=0.0)
    w_ridge, _ = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=3000, alpha=1.0)
    assert sum(abs(v) for v in w_ridge) < sum(abs(v) for v in w_plain)


def test_stronger_alpha_shrinks_harder():
    w_weak, _ = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=3000, alpha=0.1)
    w_strong, _ = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=3000, alpha=2.0)
    assert sum(abs(v) for v in w_strong) < sum(abs(v) for v in w_weak)


def test_ridge_does_not_penalise_the_bias():
    """Ловушка: b не отвечает за сложность модели, к нулю его не тянут.

    При сильном штрафе веса почти обнуляются, и вся работа ложится на b —
    он обязан подтянуться к среднему y, а не сползти к нулю.
    """
    _, b = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=5000, alpha=5.0)
    y_mean = sum(PLANE_Y) / len(PLANE_Y)
    assert abs(b - y_mean) < abs(b - 0.0)


def test_ridge_still_fits_the_data_when_alpha_is_tiny():
    w, b = fit_ridge(PLANE_X, PLANE_Y, lr=0.05, epochs=5000, alpha=1e-6)
    assert r_squared(PLANE_Y, predict(PLANE_X, w, b)) > 0.99
