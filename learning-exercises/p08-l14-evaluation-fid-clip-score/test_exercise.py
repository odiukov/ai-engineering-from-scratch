"""Тесты к уроку «Оценка генеративных моделей: FID, CLIP score, Elo». Правь exercise.py."""

import random

import pytest

from exercise import (
    clip_score,
    covariance,
    elo_update,
    fid,
    matmul,
    matrix_inverse,
    matrix_sqrt,
    mean_vector,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def features(n, d, center, rng, scale=1.0):
    """Синтетические «признаки Inception»: облако вокруг center."""
    return [[center + rng.gauss(0.0, scale) for _ in range(d)] for _ in range(n)]


def shifted(vectors, delta):
    return [[x + delta for x in v] for v in vectors]


IDENTITY_2 = [[1.0, 0.0], [0.0, 1.0]]


# ----------------------------------------------------------------- matmul
def test_matmul_by_identity_returns_the_matrix():
    assert flat(matmul([[1.0, 2.0], [3.0, 4.0]], IDENTITY_2)) == APPROX([1.0, 2.0, 3.0, 4.0])


def test_matmul_of_row_by_column_is_a_dot_product():
    assert flat(matmul([[1.0, 2.0]], [[3.0], [4.0]])) == APPROX([11.0])


def test_matmul_rejects_mismatched_shapes():
    """Молча обрезать по короткому нельзя: получится правдоподобный мусор."""
    with pytest.raises(ValueError):
        matmul([[1.0, 2.0, 3.0]], [[1.0], [2.0]])


# --------------------------------------------------------- matrix_inverse
def test_matrix_inverse_of_a_diagonal():
    assert flat(matrix_inverse([[2.0, 0.0], [0.0, 4.0]])) == APPROX([0.5, 0.0, 0.0, 0.25])


def test_matrix_inverse_times_the_matrix_is_identity():
    M = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    assert flat(matmul(M, matrix_inverse(M))) == pytest.approx(flat(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), abs=1e-9)


def test_matrix_inverse_rejects_a_singular_matrix():
    """Вторая строка — первая, умноженная на два. Обратной не существует."""
    with pytest.raises(ValueError):
        matrix_inverse([[1.0, 2.0], [2.0, 4.0]])


def test_matrix_inverse_survives_a_zero_leading_pivot():
    """Без выбора главного элемента первый же шаг делит на ноль."""
    M = [[0.0, 1.0], [1.0, 0.0]]
    assert flat(matrix_inverse(M)) == APPROX([0.0, 1.0, 1.0, 0.0])


# ------------------------------------------------------------ matrix_sqrt
def test_matrix_sqrt_of_a_diagonal():
    assert flat(matrix_sqrt([[4.0, 0.0], [0.0, 9.0]])) == pytest.approx(
        [2.0, 0.0, 0.0, 3.0], abs=1e-8)


def test_matrix_sqrt_squared_returns_the_original():
    """Единственная настоящая проверка корня: возведи обратно в квадрат."""
    M = [[5.0, 2.0, 1.0], [2.0, 4.0, 0.5], [1.0, 0.5, 3.0]]
    assert flat(matmul(matrix_sqrt(M), matrix_sqrt(M))) == pytest.approx(flat(M), abs=1e-7)


def test_matrix_sqrt_is_symmetric_for_a_symmetric_input():
    M = [[3.0, 1.0], [1.0, 2.0]]
    S = matrix_sqrt(M)
    assert S[0][1] == pytest.approx(S[1][0], abs=1e-9)


def test_matrix_sqrt_handles_a_singular_psd_matrix_without_inverse():
    """Нулевая дисперсия — допустимое собственное значение, не авария."""
    S = matrix_sqrt([[4.0, 0.0], [0.0, 0.0]])
    assert flat(S) == pytest.approx([2.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_matrix_sqrt_handles_a_rank_one_psd_matrix():
    M = [[1.0, 1.0], [1.0, 1.0]]
    S = matrix_sqrt(M)
    assert flat(matmul(S, S)) == pytest.approx(flat(M), abs=1e-9)


# ------------------------------------------------------------ mean_vector
def test_mean_vector_averages_each_coordinate():
    assert mean_vector([[1.0, 2.0], [3.0, 4.0]]) == APPROX([2.0, 3.0])


def test_mean_vector_of_a_single_vector_is_that_vector():
    assert mean_vector([[5.0, -5.0]]) == APPROX([5.0, -5.0])


def test_mean_vector_ignores_sample_order():
    vs = [[1.0, 7.0], [3.0, -1.0], [8.0, 0.5]]
    assert mean_vector(vs) == APPROX(mean_vector(list(reversed(vs))))


# ------------------------------------------------------------- covariance
def test_covariance_uses_the_unbiased_divisor():
    """Два числа 1 и 3: дисперсия 2.0 при делителе n-1, а не 1.0 при n."""
    assert flat(covariance([[1.0, 0.0], [3.0, 0.0]])) == APPROX([2.0, 0.0, 0.0, 0.0])


def test_covariance_of_one_sample_is_zeros_not_a_crash():
    assert flat(covariance([[1.0, 2.0]])) == APPROX([0.0, 0.0, 0.0, 0.0])


def test_covariance_is_symmetric():
    rng = random.Random(11)
    C = covariance(features(60, 4, 0.0, rng))
    for i in range(4):
        for j in range(4):
            assert C[i][j] == APPROX(C[j][i])


def test_covariance_does_not_change_under_translation():
    """Ковариация видит только разброс — сдвиг всего облака ей безразличен."""
    rng = random.Random(12)
    vs = features(50, 3, 0.0, rng)
    assert flat(covariance(vs)) == pytest.approx(flat(covariance(shifted(vs, 10.0))), abs=1e-9)


# -------------------------------------------------------------------- fid
def test_fid_is_zero_on_identical_sets():
    rng = random.Random(21)
    vs = features(200, 3, 0.0, rng)
    assert fid(vs, vs) == pytest.approx(0.0, abs=1e-6)


def test_fid_handles_identical_constant_one_sample_clouds():
    """Обе ковариации нулевые: FID равен нулю и inverse не нужен."""
    assert fid([[3.0, -2.0]], [[3.0, -2.0]]) == APPROX(0.0)


def test_fid_of_constant_one_sample_clouds_is_the_squared_mean_gap():
    """При нулевых ковариациях остаётся только квадрат расстояния средних."""
    assert fid([[1.0, 2.0]], [[4.0, 6.0]]) == APPROX(25.0)


def test_fid_handles_singular_line_clouds():
    line = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    assert fid(line, line) == pytest.approx(0.0, abs=1e-9)


def test_fid_is_never_negative():
    rng = random.Random(22)
    real = features(150, 3, 0.0, rng)
    for shift in (0.0, 0.4, 1.0):
        assert fid(real, features(150, 3, shift, rng)) >= -1e-6


def test_fid_is_symmetric():
    rng = random.Random(23)
    real = features(150, 3, 0.0, rng)
    gen = features(150, 3, 0.6, rng)
    assert fid(real, gen) == pytest.approx(fid(gen, real), rel=1e-6)


def test_fid_does_not_depend_on_sample_order():
    """FID — расстояние между РАСПРЕДЕЛЕНИЯМИ, перестановка ничего не меняет."""
    rng = random.Random(24)
    real = features(120, 3, 0.0, rng)
    gen = features(120, 3, 0.5, rng)
    mixed = list(gen)
    random.Random(99).shuffle(mixed)
    assert fid(real, mixed) == pytest.approx(fid(real, gen), abs=1e-6)


def test_fid_equals_the_squared_mean_shift_when_only_the_mean_moves():
    """Сдвинули облако целиком: ковариации совпали, остался только ||mu_r - mu_g||^2."""
    rng = random.Random(25)
    real = features(150, 3, 0.0, rng)
    assert fid(real, shifted(real, 0.7)) == pytest.approx(3 * 0.7 ** 2, abs=1e-6)


def test_fid_grows_with_the_distribution_shift():
    rng = random.Random(26)
    real = features(200, 3, 0.0, rng)
    scores = [fid(real, features(200, 3, s, rng)) for s in (0.0, 0.3, 0.8, 1.5)]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0] + 1.0


def test_fid_notices_a_change_in_spread_even_at_the_same_mean():
    """Одних средних мало: разный разброс при том же центре тоже штрафуется."""
    rng = random.Random(27)
    real = features(200, 3, 0.0, rng, scale=1.0)
    wide = features(200, 3, 0.0, rng, scale=2.5)
    assert fid(real, wide) > 1.0


def test_fid_is_biased_upward_on_a_small_sample():
    """На двенадцати сэмплах FID далёк от нуля, хотя распределение то же."""
    rng = random.Random(28)
    small = fid(features(12, 3, 0.0, rng), features(12, 3, 0.0, rng))
    big = fid(features(800, 3, 0.0, rng), features(800, 3, 0.0, rng))
    assert small > 10 * big


# ------------------------------------------------------------- clip_score
def test_clip_score_of_a_perfect_match_is_one():
    assert clip_score([1.0, 0.0], [1.0, 0.0]) == APPROX(1.0)


def test_clip_score_falls_through_zero_to_minus_one_as_the_image_diverges():
    """Ортогональные эмбеддинги дают 0, противоположные — нижнюю границу -1."""
    assert clip_score([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)
    assert clip_score([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_clip_score_ignores_vector_length():
    """Косинус меряет направление: яркость картинки на adherence не влияет."""
    assert clip_score([2.0, 4.0], [1.0, 1.0]) == APPROX(clip_score([1.0, 2.0], [5.0, 5.0]))


def test_clip_score_rises_when_the_image_matches_the_prompt():
    prompt = [1.0, 0.5, -0.2, 0.3]
    matching = [0.9, 0.6, -0.1, 0.4]
    unrelated = [-0.8, 0.1, 0.7, -0.5]
    assert clip_score(matching, prompt) > clip_score(unrelated, prompt)


def test_clip_score_of_a_zero_embedding_is_zero_not_a_crash():
    assert clip_score([0.0, 0.0], [1.0, 2.0]) == APPROX(0.0)


# ------------------------------------------------------------ elo_update
def test_elo_equal_ratings_move_by_half_the_k_factor():
    assert elo_update(1000, 1000, "a") == APPROX((1016.0, 984.0))


def test_elo_is_zero_sum():
    """Сколько победитель приобрёл, столько проигравший потерял."""
    a, b = elo_update(1234, 1088, "b")
    assert a + b == APPROX(1234 + 1088)


def test_elo_favourite_gains_little_from_an_expected_win():
    strong, _ = elo_update(1600, 1000, "a")
    even, _ = elo_update(1000, 1000, "a")
    assert 0 < strong - 1600 < even - 1000


def test_elo_underdog_gains_a_lot_from_an_upset():
    _, upset = elo_update(1600, 1000, "b")
    assert upset - 1000 > 25


def test_elo_separates_models_over_a_stream_of_matches():
    """200 сравнений, модель A выигрывает 70% — рейтинг обязан это показать."""
    rng = random.Random(29)
    r_a, r_b = 1000.0, 1000.0
    for _ in range(200):
        r_a, r_b = elo_update(r_a, r_b, "a" if rng.random() < 0.7 else "b")
    assert r_a > r_b + 100
