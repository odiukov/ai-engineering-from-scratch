"""Тесты к уроку «Признаки: конструирование и отбор». Правь exercise.py."""

import math

import pytest

from exercise import (
    bin_values,
    impute_median,
    min_max_scale,
    mutual_information,
    one_hot_encode,
    standardize,
    target_encode,
    tfidf,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


# ------------------------------------------------------------ min_max_scale
def test_min_max_scale_maps_the_range_onto_zero_one():
    assert min_max_scale([10, 20, 30]) == APPROX([0.0, 0.5, 1.0])


def test_min_max_scale_puts_the_extremes_exactly_at_the_borders():
    scaled = min_max_scale([-7.0, 0.0, 3.0, 100.0])
    assert min(scaled) == APPROX(0.0)
    assert max(scaled) == APPROX(1.0)


def test_min_max_scale_of_a_constant_column_does_not_divide_by_zero():
    """Ловушка: max - min = 0. Ответ должен быть нулями, а не ZeroDivisionError."""
    assert min_max_scale([5, 5, 5]) == APPROX([0.0, 0.0, 0.0])


def test_min_max_scale_keeps_the_order_of_values():
    scaled = min_max_scale([3.0, 1.0, 2.0])
    assert scaled[1] < scaled[2] < scaled[0]


# -------------------------------------------------------------- standardize
def test_standardize_known_values():
    assert standardize([1, 2, 3]) == pytest.approx([-1.224744871, 0.0, 1.224744871])


def test_standardize_gives_mean_zero_and_unit_std():
    values = standardize([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])
    n = len(values)
    assert sum(values) / n == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(sum(v * v for v in values) / n) == pytest.approx(1.0)


def test_standardize_is_invariant_to_shifting_the_whole_column():
    """Сдвиг всех значений на константу не меняет z-оценки — вычитается среднее."""
    assert standardize([1, 2, 3]) == pytest.approx(standardize([101, 102, 103]))


def test_standardize_of_a_constant_column_is_zeros():
    assert standardize([5, 5, 5]) == APPROX([0.0, 0.0, 0.0])


# --------------------------------------------------------------- bin_values
def test_bin_values_spreads_a_ramp_across_all_bins():
    assert bin_values([1, 2, 3, 4], n_bins=4) == [0, 1, 2, 3]


def test_bin_values_puts_the_maximum_into_the_last_bin():
    """Ловушка: масштабированный максимум равен 1.0 и метит в корзину n_bins."""
    assert bin_values([0, 1, 2, 3, 4, 5], n_bins=3)[-1] == 2


def test_bin_values_of_a_constant_column_is_a_single_bin():
    assert bin_values([7, 7, 7], n_bins=5) == [0, 0, 0]


# ------------------------------------------------------------ impute_median
def test_impute_median_fills_the_hole():
    filled, median = impute_median([1.0, None, 3.0])
    assert filled == APPROX([1.0, 2.0, 3.0])
    assert median == APPROX(2.0)


def test_impute_median_of_an_even_count_averages_the_middle_pair():
    _, median = impute_median([1.0, 2.0, 3.0, 4.0])
    assert median == APPROX(2.5)


def test_impute_median_ignores_an_extreme_outlier():
    """Ради этого медиана и берётся: среднее ушло бы за миллион, медиана — нет."""
    _, median = impute_median([1.0, 2.0, None, 3.0, 10_000_000.0])
    assert median == APPROX(2.5)


def test_impute_median_returns_the_fill_value_for_reuse_on_test_data():
    """Значение возвращается, чтобы применить его к тесту, а не пересчитать —
    пересчёт по тесту и есть утечка."""
    _, median = impute_median([10.0, 20.0, 30.0])
    assert median == APPROX(20.0)


def test_impute_median_of_an_all_missing_column_does_not_crash():
    filled, median = impute_median([None, None])
    assert filled == APPROX([0.0, 0.0])
    assert median == APPROX(0.0)


# ----------------------------------------------------------- one_hot_encode
def test_one_hot_encode_makes_one_column_per_category():
    rows, categories = one_hot_encode(["a", "b", "a"])
    assert categories == ["a", "b"]
    assert flat(rows) == [1, 0, 0, 1, 1, 0]


def test_one_hot_encode_lights_exactly_one_column_per_row():
    rows, _ = one_hot_encode(["red", "green", "blue", "green"])
    assert all(sum(row) == 1 for row in rows)
    assert all(len(row) == 3 for row in rows)


def test_one_hot_encode_does_not_depend_on_the_order_of_the_input():
    """Столбцы отсортированы, поэтому перестановка строк не переименует признаки."""
    _, first = one_hot_encode(["z", "a", "m"])
    _, second = one_hot_encode(["m", "z", "a"])
    assert first == second == ["a", "m", "z"]


# ----------------------------------------------------------- target_encode
def test_target_encode_without_smoothing_is_the_plain_category_mean():
    encoded, mapping = target_encode(["a", "a", "b"], [10, 20, 100], smoothing=0)
    assert encoded == APPROX([15.0, 15.0, 100.0])
    assert mapping["b"] == APPROX(100.0)


def test_target_encode_pulls_a_rare_category_toward_the_global_mean():
    """У «b» одно наблюдение — сглаживание не даёт поверить ему на слово."""
    _, mapping = target_encode(["a", "a", "b"], [10, 20, 100], smoothing=1)
    global_mean = (10 + 20 + 100) / 3
    assert abs(mapping["b"] - global_mean) < abs(100.0 - global_mean)


def test_target_encode_with_huge_smoothing_collapses_to_the_global_mean():
    _, mapping = target_encode(["a", "a", "b"], [10, 20, 100], smoothing=10_000)
    global_mean = (10 + 20 + 100) / 3
    assert mapping["a"] == pytest.approx(global_mean, abs=0.05)
    assert mapping["b"] == pytest.approx(global_mean, abs=0.05)


def test_target_encode_without_smoothing_leaks_the_answer():
    """Ловушка урока: одиночная категория кодируется собственным таргетом."""
    _, mapping = target_encode(["rare", "x", "x"], [777.0, 1.0, 2.0], smoothing=0)
    assert mapping["rare"] == APPROX(777.0)


def test_target_encode_covers_every_category():
    _, mapping = target_encode(["a", "b", "c", "a"], [1.0, 2.0, 3.0, 4.0])
    assert set(mapping) == {"a", "b", "c"}


# ------------------------------------------------------------------- tfidf
def test_tfidf_zeroes_out_a_word_present_in_every_document():
    """IDF = log(1) = 0. Слово-паразит просто исчезает из представления."""
    vectors, vocab = tfidf(["a b", "a c"])
    assert vectors[0][vocab["a"]] == APPROX(0.0)
    assert vectors[1][vocab["a"]] == APPROX(0.0)


def test_tfidf_gives_a_distinctive_word_a_positive_weight():
    vectors, vocab = tfidf(["a b", "a c"])
    assert vectors[0][vocab["b"]] > 0
    assert vectors[1][vocab["c"]] > 0


def test_tfidf_vector_length_equals_vocabulary_size():
    vectors, vocab = tfidf(["the cat sat", "the mat", "a dog"])
    assert len(vocab) == 6
    assert all(len(v) == 6 for v in vectors)


def test_tfidf_counts_a_repeated_word_more_than_a_single_one():
    """TF — доля слова в документе, повтор обязан поднять вес."""
    vectors, vocab = tfidf(["cat cat dog", "bird"])
    assert vectors[0][vocab["cat"]] > vectors[0][vocab["dog"]]


def test_tfidf_is_case_insensitive():
    _, vocab = tfidf(["Cat CAT cat"])
    assert list(vocab) == ["cat"]


# ------------------------------------------------------ mutual_information
def test_mutual_information_of_a_perfect_predictor_is_log_two():
    assert mutual_information([0, 0, 1, 1], [0, 0, 1, 1], n_bins=2) == APPROX(math.log(2))


def test_mutual_information_of_a_constant_feature_is_zero():
    """Признак-константа не снижает неопределённость ни на нат."""
    assert mutual_information([5, 5, 5, 5], [0, 0, 1, 1], n_bins=2) == APPROX(0.0)


def test_mutual_information_is_never_negative():
    feature = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    target = [0, 1, 0, 1, 1, 0, 0, 1]
    assert mutual_information(feature, target, n_bins=4) >= -1e-12


def test_mutual_information_ranks_a_signal_above_noise():
    """Отбор признаков и держится на этом: у полезного столбца MI больше."""
    target = [0, 0, 0, 0, 1, 1, 1, 1]
    signal = [0.0, 0.1, 0.2, 0.3, 9.0, 9.1, 9.2, 9.3]
    noise = [1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0]
    assert mutual_information(signal, target, n_bins=2) > mutual_information(
        noise, target, n_bins=2
    )
