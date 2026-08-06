"""Тесты к уроку «Смещение и разброс». Правь exercise.py."""

import math

import pytest

from exercise import (
    best_degree,
    bias_variance_decomposition,
    diagnose,
    fit_polynomial,
    learning_curve,
    make_dataset,
    mean_squared_error,
    polyval,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# истинная функция урока: изогнутая, но гладкая — прямая её не догонит
CURVE = lambda x: math.sin(1.5 * x) + 0.5 * x
LINE = lambda x: 2 * x + 1


# ----------------------------------------------------------------- polyval
def test_polyval_evaluates_a_quadratic():
    assert polyval([1, 2, 3], 2) == APPROX(17.0)


def test_polyval_of_a_constant_ignores_x():
    assert polyval([5], 100) == APPROX(5.0)


def test_polyval_reads_coefficients_from_the_lowest_power_up():
    """Ловушка порядка: coeffs[0] — свободный член, а не старший коэффициент."""
    assert polyval([0, 1], 7) == APPROX(7.0)
    assert polyval([7, 0], 1) == APPROX(7.0)


def test_polyval_handles_a_high_degree_term():
    assert polyval([0] * 15 + [1], 2) == APPROX(32768.0)


# ---------------------------------------------------------- fit_polynomial
def test_fit_polynomial_recovers_a_line():
    assert fit_polynomial([0, 1, 2], [1, 3, 5], 1) == pytest.approx([1.0, 2.0], abs=1e-8)


def test_fit_polynomial_recovers_a_parabola():
    coeffs = fit_polynomial([0, 1, 2, 3], [0, 1, 4, 9], 2)
    assert coeffs == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_fit_polynomial_of_degree_zero_is_the_mean():
    assert fit_polynomial([1, 2, 3], [2, 4, 6], 0) == pytest.approx([4.0], abs=1e-9)


def test_fit_polynomial_works_with_more_points_than_parameters():
    """МНК на переопределённой системе: пять точек, две неизвестные."""
    xs = [-2, -1, 0, 1, 2]
    coeffs = fit_polynomial(xs, [LINE(x) for x in xs], 1)
    assert coeffs == pytest.approx([1.0, 2.0], abs=1e-8)


def test_fit_polynomial_l2_shrinks_the_slope_but_not_the_intercept():
    """Ловушка: свободный член не штрафуется, иначе модель не сможет
    предсказать даже константу."""
    xs = [-2, -1, 0, 1, 2]
    intercept, slope = fit_polynomial(xs, [LINE(x) for x in xs], 1, l2=1e6)
    assert slope == pytest.approx(0.0, abs=1e-3)
    assert intercept == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------ mean_squared_error
def test_mean_squared_error_of_a_perfect_fit_is_zero():
    assert mean_squared_error([1, 2, 3], [1, 2, 3]) == APPROX(0.0)


def test_mean_squared_error_squares_before_averaging():
    assert mean_squared_error([0, 0], [1, -1]) == APPROX(1.0)


# ------------------------------------------------------------ make_dataset
def test_make_dataset_returns_the_requested_number_of_points():
    xs, ys = make_dataset(CURVE, 17, seed=1)
    assert len(xs) == len(ys) == 17


def test_make_dataset_without_noise_lands_exactly_on_the_true_function():
    xs, ys = make_dataset(LINE, 5, noise=0.0, seed=1)
    assert ys == pytest.approx([LINE(x) for x in xs])


def test_make_dataset_is_reproducible_for_the_same_seed():
    assert make_dataset(CURVE, 30, seed=5) == make_dataset(CURVE, 30, seed=5)


def test_make_dataset_gives_different_data_for_different_seeds():
    """Разложение усредняет по независимым выборкам — они обязаны отличаться."""
    assert make_dataset(CURVE, 30, seed=5) != make_dataset(CURVE, 30, seed=6)


def test_make_dataset_keeps_x_inside_the_requested_range():
    xs, _ = make_dataset(CURVE, 50, seed=2, low=-1.0, high=1.0)
    assert all(-1.0 <= x <= 1.0 for x in xs)


# ------------------------------------------ bias_variance_decomposition
def test_decomposition_sums_exactly_to_the_total_error():
    """Главное тождество урока: total = bias^2 + variance, без остатка."""
    r = bias_variance_decomposition(CURVE, 3, n_train=25, n_sets=40, seed=0)
    assert r["total"] == pytest.approx(r["bias2"] + r["variance"], abs=1e-9)


def test_decomposition_components_are_never_negative():
    r = bias_variance_decomposition(CURVE, 5, n_train=25, n_sets=40, seed=0)
    assert min(r.values()) >= 0.0


def test_a_straight_line_on_a_curve_is_all_bias():
    """Прямая систематически мимо — смещение большое, разброс маленький."""
    r = bias_variance_decomposition(CURVE, 1, n_train=25, n_sets=40, seed=0)
    assert r["bias2"] > r["variance"]


def test_a_high_degree_polynomial_is_all_variance():
    """Степень 7 гнётся под каждую выборку — разброс взлетает, смещение падает."""
    simple = bias_variance_decomposition(CURVE, 1, n_train=25, n_sets=40, seed=0)
    complex_ = bias_variance_decomposition(CURVE, 7, n_train=25, n_sets=40, seed=0)
    assert complex_["variance"] > simple["variance"]
    assert complex_["bias2"] < simple["bias2"]


def test_the_right_model_on_clean_data_has_almost_no_error():
    r = bias_variance_decomposition(LINE, 1, n_train=25, n_sets=20, noise=0.0, seed=0)
    assert r["total"] == pytest.approx(0.0, abs=1e-12)


def test_regularization_buys_lower_variance_with_higher_bias():
    """Гребень на неустойчивой модели: разброс падает в разы."""
    plain = bias_variance_decomposition(CURVE, 9, n_train=25, n_sets=40, seed=0, l2=0.0)
    ridged = bias_variance_decomposition(CURVE, 9, n_train=25, n_sets=40, seed=0, l2=1.0)
    assert ridged["variance"] < plain["variance"]


def test_more_regularization_means_more_bias():
    """Обратная сторона сделки: чем сильнее штраф, тем жёстче модель."""
    weak = bias_variance_decomposition(CURVE, 5, n_train=30, n_sets=40, seed=0, l2=0.5)
    strong = bias_variance_decomposition(CURVE, 5, n_train=30, n_sets=40, seed=0, l2=50.0)
    assert strong["bias2"] > weak["bias2"]


def test_decomposition_is_reproducible_for_the_same_seed():
    args = (CURVE, 4)
    kwargs = {"n_train": 25, "n_sets": 20, "seed": 11}
    assert bias_variance_decomposition(*args, **kwargs) == bias_variance_decomposition(
        *args, **kwargs
    )


# ------------------------------------------------------------- best_degree
def test_best_degree_of_a_linear_truth_is_one():
    assert best_degree(LINE, [1, 2, 5], n_train=25, n_sets=30) == 1


def test_best_degree_finds_the_bottom_of_the_u_curve():
    """На изогнутой функции ни прямая, ни степень 9 не выигрывают."""
    assert best_degree(CURVE, [1, 3, 9], n_train=25, n_sets=40, seed=0) == 3


def test_best_degree_with_one_candidate_returns_it():
    assert best_degree(CURVE, [4], n_train=20, n_sets=10) == 4


# ---------------------------------------------------------- learning_curve
def test_learning_curve_returns_one_point_per_size():
    train, test = learning_curve(CURVE, 3, [10, 20, 40], n_repeats=10)
    assert len(train) == len(test) == 3


def test_high_bias_curves_converge_to_the_same_high_error():
    """Прямая на синусоиде: данные добавлять бесполезно, обе кривые упираются."""
    train, test = learning_curve(CURVE, 1, [10, 20, 40, 80, 160], n_repeats=20)
    assert abs(test[-1] - train[-1]) < 0.2
    assert test[-1] > 0.5


def test_more_data_closes_the_gap_of_a_high_variance_model():
    """Степень 8 на 20 точках почти интерполирует, на 120 — уже нет."""
    train, test = learning_curve(CURVE, 8, [20, 120], n_repeats=20)
    assert test[0] - train[0] > test[1] - train[1]
    assert test[1] < test[0]


def test_training_error_grows_as_the_training_set_grows():
    """Запоминать десять точек легко, сто — уже нет."""
    train, _ = learning_curve(CURVE, 1, [10, 160], n_repeats=20)
    assert train[1] > train[0]


# ---------------------------------------------------------------- diagnose
def test_diagnose_calls_a_big_gap_variance():
    assert diagnose(0.05, 0.90) == "variance"


def test_diagnose_calls_two_high_errors_bias():
    assert diagnose(0.80, 0.85) == "bias"


def test_diagnose_calls_two_low_errors_good():
    assert diagnose(0.10, 0.15) == "good"


def test_diagnose_checks_the_gap_before_the_error_level():
    """Ловушка порядка: train 0.8 и test 2.0 — это переобучение, не недообучение."""
    assert diagnose(0.80, 2.00) == "variance"
