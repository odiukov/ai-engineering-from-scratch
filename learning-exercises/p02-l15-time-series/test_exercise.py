"""Тесты к уроку «Временные ряды». Правь exercise.py."""

import pytest

from exercise import (
    autocorrelation,
    difference,
    is_stationary,
    make_lag_features,
    rolling_mean,
    seasonal_naive_forecast,
    time_split,
    walk_forward_splits,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in rows for v in row]


TREND = [float(i) for i in range(1, 101)]
SEASONAL = [float(i % 7) for i in range(70)]


# -------------------------------------------------------------- difference
def test_difference_of_a_linear_trend_is_constant():
    assert difference([100, 102, 104, 106]) == APPROX([2, 2, 2])


def test_difference_shortens_the_series_by_one():
    assert len(difference(TREND)) == len(TREND) - 1


def test_second_difference_flattens_a_quadratic_trend():
    """[100, 102, 106, 112, 120] — квадратичный тренд, нужны две разности."""
    once = difference([100, 102, 106, 112, 120])
    assert once == APPROX([2, 4, 6, 8])
    assert difference(once) == APPROX([2, 2, 2])


def test_difference_of_a_single_point_is_empty():
    assert difference([5]) == []


# ----------------------------------------------------------- is_stationary
def test_trending_series_is_not_stationary():
    assert is_stationary(TREND) is False


def test_differencing_makes_the_trend_stationary():
    """Ровно тот приём, ради которого разности и берут."""
    assert is_stationary(difference(TREND)) is True


def test_constant_series_is_stationary():
    assert is_stationary([7.0] * 20) is True


def test_variance_blowup_is_not_stationary():
    """Среднее на месте, а разброс вырос втрое — это тоже нестационарность."""
    series = [1.0, -1.0] * 10 + [30.0, -30.0] * 10
    assert is_stationary(series) is False


# ------------------------------------------------------- autocorrelation
def test_autocorrelation_at_lag_zero_is_one():
    assert autocorrelation(TREND, 3)[0] == APPROX(1.0)


def test_autocorrelation_returns_one_value_per_lag():
    assert len(autocorrelation(TREND, 5)) == 6


def test_seasonal_series_peaks_at_its_period():
    """У ряда с периодом 7 лаг 7 — самый похожий на сам ряд."""
    acf = autocorrelation(SEASONAL, 10)
    assert acf[7] == pytest.approx(max(acf[1:]), abs=1e-9)


def test_trend_gives_positive_autocorrelation_at_short_lags():
    assert autocorrelation(TREND, 1)[1] > 0.9


def test_constant_series_has_no_autocorrelation_to_report():
    """Ловушка: нулевая дисперсия — делить на неё нельзя."""
    assert autocorrelation([3.0] * 10, 2) == APPROX([0.0, 0.0, 0.0])


# ---------------------------------------------------------- rolling_mean
def test_rolling_mean_of_window_two():
    assert rolling_mean([1, 2, 3, 4], 2) == APPROX([1.5, 2.5, 3.5])


def test_rolling_mean_length_shrinks_by_window_minus_one():
    assert len(rolling_mean(TREND, 7)) == len(TREND) - 6


def test_rolling_mean_window_ends_on_the_current_point():
    """Окно смотрит назад, а не вперёд: первое значение — среднее первых window.

    Центрированное окно подсматривало бы в будущее, и это самая тихая
    из утечек: бэктест прекрасен, прод разваливается.
    """
    assert rolling_mean([0, 0, 0, 100], 2)[0] == APPROX(0.0)


def test_rolling_mean_of_window_one_is_the_series_itself():
    assert rolling_mean([1.0, 5.0, 2.0], 1) == APPROX([1.0, 5.0, 2.0])


def test_rolling_mean_with_window_longer_than_series_is_empty():
    assert rolling_mean([1, 2], 5) == []


# ----------------------------------------------------- make_lag_features
def test_lag_features_match_the_lesson_table():
    X, y = make_lag_features([10, 12, 14, 13, 15], 2)
    assert flat(X) == APPROX([10, 12, 12, 14, 14, 13])
    assert y == APPROX([14, 13, 15])


def test_lag_features_never_include_the_target_itself():
    """Главная ловушка урока: значение момента t не имеет права быть признаком.

    Ряд строго возрастает, поэтому цель обязана быть строго больше любого
    своего признака. Если это не так — модель получила ответ на входе.
    """
    X, y = make_lag_features(list(range(30)), 3)
    assert all(y[i] > max(X[i]) for i in range(len(y)))


def test_lag_features_are_in_chronological_order():
    """Слева самое старое значение, справа — самое свежее."""
    X, _ = make_lag_features([10, 20, 30, 40], 3)
    assert X[0] == APPROX([10, 20, 30])


def test_lag_features_drop_the_first_n_lags_rows():
    X, y = make_lag_features(list(range(20)), 4)
    assert len(X) == 16
    assert len(y) == 16


def test_changing_the_future_does_not_change_earlier_rows():
    """Строка t собрана только из прошлого, значит хвост ряда на неё не влияет."""
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    tweaked = base[:-1] + [999.0]
    X_base, _ = make_lag_features(base, 2)
    X_tweaked, _ = make_lag_features(tweaked, 2)
    assert flat(X_base[:-1]) == APPROX(flat(X_tweaked[:-1]))


# ---------------------------------------------------------- time_split
def test_time_split_puts_the_tail_into_test():
    X_train, X_test, y_train, y_test = time_split(
        [[1], [2], [3], [4]], [1, 2, 3, 4], 1
    )
    assert flat(X_train) == APPROX([1, 2, 3])
    assert flat(X_test) == APPROX([4])
    assert y_train == APPROX([1, 2, 3])
    assert y_test == APPROX([4])


def test_time_split_does_not_shuffle():
    """Порядок обязан сохраниться: train + test склеиваются обратно в исходный."""
    X = [[float(i)] for i in range(20)]
    y = [float(i) for i in range(20)]
    X_train, X_test, y_train, y_test = time_split(X, y, 5)
    assert y_train + y_test == APPROX(y)
    assert flat(X_train) + flat(X_test) == APPROX(flat(X))


def test_time_split_train_is_entirely_in_the_past():
    """Каждое значение train строго раньше каждого значения test."""
    y = [float(i) for i in range(30)]
    X = [[v] for v in y]
    _, _, y_train, y_test = time_split(X, y, 7)
    assert max(y_train) < min(y_test)


def test_time_split_keeps_features_and_targets_aligned():
    y = [float(i) for i in range(10)]
    X = [[v * 2] for v in y]
    _, X_test, _, y_test = time_split(X, y, 3)
    assert flat(X_test) == APPROX([v * 2 for v in y_test])


def test_time_split_rejects_a_test_size_that_eats_everything():
    with pytest.raises(ValueError):
        time_split([[1], [2]], [1, 2], 2)


# -------------------------------------------------- walk_forward_splits
def test_walk_forward_expands_the_training_window():
    splits = walk_forward_splits(8, 2, 4)
    assert splits[0][0] == [0, 1, 2, 3]
    assert splits[0][1] == [4, 5]
    assert splits[1][0] == [0, 1, 2, 3, 4, 5]
    assert splits[1][1] == [6, 7]


def test_walk_forward_never_trains_on_the_future():
    """Обязательный инвариант: весь train строго раньше всего test.

    Именно это отличает walk-forward от обычного k-fold, который спокойно
    кладёт июнь в обучение, а март — в проверку.
    """
    for train_idx, test_idx in walk_forward_splits(200, 5, 50):
        assert max(train_idx) < min(test_idx)


def test_walk_forward_folds_do_not_overlap_train_and_test():
    for train_idx, test_idx in walk_forward_splits(200, 5, 50):
        assert set(train_idx).isdisjoint(test_idx)


def test_walk_forward_test_folds_move_forward():
    starts = [test_idx[0] for _, test_idx in walk_forward_splits(200, 5, 50)]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_walk_forward_rejects_min_train_bigger_than_the_data():
    with pytest.raises(ValueError):
        walk_forward_splits(10, 3, 10)


# ---------------------------------------------- seasonal_naive_forecast
def test_seasonal_naive_repeats_the_last_period():
    assert seasonal_naive_forecast([1, 2, 3, 4, 5, 6], 3, 4) == APPROX([4, 5, 6, 4])


def test_period_one_is_the_persistence_baseline():
    """«Завтра будет как сегодня» — базовая линия, которую надо обыграть."""
    assert seasonal_naive_forecast([1, 2, 3], 1, 2) == APPROX([3, 3])


def test_seasonal_naive_forecast_length_equals_horizon():
    assert len(seasonal_naive_forecast(SEASONAL, 7, 20)) == 20


def test_seasonal_naive_is_exact_on_a_perfectly_periodic_series():
    """Ряд с периодом 7 продолжается сам собой без единой ошибки."""
    forecast = seasonal_naive_forecast(SEASONAL, 7, 7)
    assert forecast == APPROX([float(i % 7) for i in range(70, 77)])


def test_seasonal_naive_rejects_a_period_longer_than_the_series():
    with pytest.raises(ValueError):
        seasonal_naive_forecast([1, 2, 3], 5, 2)
