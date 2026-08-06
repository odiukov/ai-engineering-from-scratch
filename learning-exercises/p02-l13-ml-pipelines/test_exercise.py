"""Тесты к уроку «ML-пайплайны». Правь exercise.py."""

import pytest

from exercise import (
    apply_step,
    column_medians,
    dump_pipeline,
    fit_pipeline,
    fit_step,
    fit_transform_split,
    load_pipeline,
    transform_pipeline,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in rows for v in row]


# ---------------------------------------------------------- column_medians
def test_column_medians_of_clean_data():
    assert column_medians([[1, 10], [3, 20], [5, 30]]) == APPROX([3.0, 20.0])


def test_column_medians_skips_none():
    """None не участвует в подсчёте, а не считается нулём."""
    assert column_medians([[1, 10], [3, None], [5, 30]]) == APPROX([3.0, 20.0])


def test_column_medians_averages_two_middles_on_even_length():
    assert column_medians([[1], [2], [3], [4]]) == APPROX([2.5])


def test_column_medians_of_all_none_column_is_zero():
    """Ловушка: медианы пустого столбца не существует, но упасть нельзя."""
    assert column_medians([[None, 1], [None, 3]]) == APPROX([0.0, 2.0])


def test_column_medians_ignores_order():
    """Медиана — порядковая статистика, перестановка строк её не меняет."""
    a = column_medians([[5], [1], [3]])
    b = column_medians([[1], [3], [5]])
    assert a == APPROX(b)


# ------------------------------------------------------------ fit_step
def test_fit_impute_stores_medians():
    assert fit_step("impute", [[1, 10], [3, None]])["medians"] == APPROX([2.0, 10.0])


def test_fit_scale_stores_mean_and_std():
    state = fit_step("scale", [[0.0], [2.0]])
    assert state["means"] == APPROX([1.0])
    assert state["stds"] == APPROX([1.0])


def test_fit_scale_uses_population_std_not_sample_std():
    """Делим на n, а не на n-1: для [0, 2] это 1.0, а не 1.414."""
    assert fit_step("scale", [[0.0], [2.0]])["stds"] == APPROX([1.0])


def test_fit_scale_replaces_zero_std_with_one():
    """Ловушка: константный столбец даст std = 0 и деление на ноль."""
    assert fit_step("scale", [[7.0], [7.0], [7.0]])["stds"] == APPROX([1.0])


def test_fit_step_rejects_unknown_kind():
    with pytest.raises(ValueError):
        fit_step("magic", [[1.0]])


# ----------------------------------------------------------- apply_step
def test_apply_impute_fills_none_with_stored_median():
    assert flat(apply_step("impute", {"medians": [2.0]}, [[None], [5.0]])) == APPROX(
        [2.0, 5.0]
    )


def test_apply_scale_centers_and_divides():
    out = apply_step("scale", {"means": [1.0], "stds": [2.0]}, [[3.0], [1.0]])
    assert flat(out) == APPROX([1.0, 0.0])


def test_apply_step_does_not_mutate_input_rows():
    """Ловушка: правка на месте испортит исходный датасет у вызывающего."""
    rows = [[None, 1.0]]
    apply_step("impute", {"medians": [9.0, 0.0]}, rows)
    assert rows == [[None, 1.0]]


def test_apply_step_ignores_statistics_of_the_new_data():
    """Одинаковые строки после scale обязаны остаться ненулевыми."""
    out = apply_step("scale", {"means": [0.0], "stds": [1.0]}, [[5.0], [5.0]])
    assert flat(out) == APPROX([5.0, 5.0])


# --------------------------------------------------------- fit_pipeline
def test_fit_pipeline_keeps_step_order():
    fitted = fit_pipeline(["impute", "scale"], [[1.0], [None], [3.0]])
    assert [s["kind"] for s in fitted] == ["impute", "scale"]


def test_fit_pipeline_fits_each_step_on_the_previous_output():
    """scale обязан видеть уже заполненные данные [1, 2, 3], а не [1, None, 3].

    Медиана [1, 3] равна 2.0, значит среднее после импьютера тоже 2.0.
    Если scale обучить на исходных rows, он вообще упадёт на None.
    """
    fitted = fit_pipeline(["impute", "scale"], [[1.0], [None], [3.0]])
    assert fitted[1]["state"]["means"] == APPROX([2.0])


def test_fit_pipeline_of_empty_steps_is_empty():
    assert fit_pipeline([], [[1.0]]) == []


# ---------------------------------------------------- transform_pipeline
def test_transform_pipeline_applies_steps_in_order():
    fitted = fit_pipeline(["scale"], [[0.0], [2.0]])
    assert flat(transform_pipeline(fitted, [[4.0]])) == APPROX([3.0])


def test_transform_pipeline_reproduces_training_output():
    rows = [[1.0], [2.0], [6.0]]
    fitted = fit_pipeline(["scale"], rows)
    again = transform_pipeline(fitted, rows)
    assert flat(again) == APPROX(flat(transform_pipeline(fitted, rows)))


def test_transform_pipeline_does_not_refit_on_new_data():
    """Главный тест урока: transform ничего не доучивает.

    Три одинаковые строки — если реализация пересчитает среднее и std по ним,
    получится std = 0 -> 1.0 и на выходе нули. Правильная реализация возьмёт
    статистики обучения и выдаст три одинаковых НЕнулевых числа.
    """
    fitted = fit_pipeline(["scale"], [[0.0], [2.0]])
    out = transform_pipeline(fitted, [[10.0], [10.0], [10.0]])
    assert flat(out) == APPROX([9.0, 9.0, 9.0])


def test_transform_pipeline_of_unseen_extreme_row_stays_extreme():
    """Выброс на проде обязан остаться выбросом, а не «нормализоваться»."""
    fitted = fit_pipeline(["scale"], [[0.0], [1.0], [2.0]])
    out = transform_pipeline(fitted, [[100.0]])
    assert out[0][0] > 50


# -------------------------------------------------- fit_transform_split
def test_fit_transform_split_returns_both_parts():
    fitted, train_out, test_out = fit_transform_split(
        ["scale"], [[0.0], [2.0], [100.0]], 2
    )
    assert flat(train_out) == APPROX([-1.0, 1.0])
    assert flat(test_out) == APPROX([99.0])


def test_fit_transform_split_ignores_the_test_tail_when_fitting():
    """Тест на утечку: подмена тестового хвоста НЕ должна менять обучение.

    Если статистики посчитаны по всем rows, второй fitted окажется другим —
    и это ровно та ошибка, из-за которой офлайн-метрики врут.
    """
    a, _, _ = fit_transform_split(["scale"], [[0.0], [2.0], [3.0]], 2)
    b, _, _ = fit_transform_split(["scale"], [[0.0], [2.0], [9999.0]], 2)
    assert a[0]["state"]["means"] == APPROX(b[0]["state"]["means"])
    assert a[0]["state"]["stds"] == APPROX(b[0]["state"]["stds"])


def test_fit_transform_split_train_output_is_centered():
    """Обученный на train scale обязан дать по train нулевое среднее."""
    _, train_out, _ = fit_transform_split(
        ["scale"], [[1.0], [3.0], [5.0], [1000.0]], 3
    )
    column = [row[0] for row in train_out]
    assert sum(column) / len(column) == APPROX(0.0)


def test_fit_transform_split_imputes_test_with_train_median():
    """None в тесте заполняется медианой ТРЕЙНА, а не медианой теста."""
    _, _, test_out = fit_transform_split(
        ["impute"], [[1.0], [3.0], [None], [100.0]], 2
    )
    assert test_out[0][0] == APPROX(2.0)


# --------------------------------------------- dump_pipeline / load_pipeline
def test_dump_returns_a_string():
    fitted = fit_pipeline(["scale"], [[0.0], [2.0]])
    assert isinstance(dump_pipeline(fitted), str)


def test_round_trip_restores_the_pipeline():
    fitted = fit_pipeline(["impute", "scale"], [[1.0], [None], [3.0]])
    assert load_pipeline(dump_pipeline(fitted)) == fitted


def test_round_trip_preserves_predictions():
    """Смысл сериализации: восстановленный пайплайн считает те же числа."""
    fitted = fit_pipeline(["impute", "scale"], [[1.0], [None], [3.0], [8.0]])
    restored = load_pipeline(dump_pipeline(fitted))
    new_rows = [[None], [5.0]]
    assert flat(transform_pipeline(restored, new_rows)) == APPROX(
        flat(transform_pipeline(fitted, new_rows))
    )


def test_dump_is_deterministic():
    """Один и тот же пайплайн — одна и та же строка, иначе версии не сравнить."""
    fitted = fit_pipeline(["scale"], [[0.0], [2.0], [4.0]])
    assert dump_pipeline(fitted) == dump_pipeline(fitted)


def test_round_trip_keeps_step_order():
    fitted = fit_pipeline(["impute", "scale"], [[1.0], [None], [3.0]])
    restored = load_pipeline(dump_pipeline(fitted))
    assert [s["kind"] for s in restored] == ["impute", "scale"]
