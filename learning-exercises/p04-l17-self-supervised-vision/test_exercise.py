"""Тесты к уроку «Self-supervised vision: SimCLR, DINO, MAE». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    center_and_sharpen,
    cosine_similarity_matrix,
    dino_loss,
    ema_update,
    info_nce,
    l2_normalize,
    masked_reconstruction_loss,
    random_mask_indices,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Развернуть матрицу в плоский список: pytest.approx не умеет вложенность."""
    return [v for row in M for v in row]


def eye(n):
    """n единичных ортов размерности n — удобный «идеально разнесённый» батч."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ------------------------------------------------------------ l2_normalize
def test_l2_normalize_gives_unit_length():
    v = l2_normalize([3.0, 4.0])
    assert math.sqrt(sum(x * x for x in v)) == APPROX(1.0)


def test_l2_normalize_keeps_the_direction():
    assert l2_normalize([3.0, 4.0]) == APPROX([0.6, 0.8])


def test_l2_normalize_rejects_the_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0, 0.0])


# ------------------------------------------------- cosine_similarity_matrix
def test_cosine_matrix_of_orthonormal_vectors_is_the_identity():
    assert flat(cosine_similarity_matrix(eye(3))) == APPROX(flat(eye(3)))


def test_cosine_matrix_ignores_vector_length():
    """Косинус меряет направление: удлинение вектора ничего не меняет."""
    short = cosine_similarity_matrix([[1.0, 0.0], [1.0, 1.0]])
    long = cosine_similarity_matrix([[7.0, 0.0], [3.0, 3.0]])
    assert flat(long) == APPROX(flat(short))


def test_cosine_matrix_is_symmetric_with_ones_on_the_diagonal():
    M = cosine_similarity_matrix([[1.0, 2.0], [-3.0, 1.0], [0.5, 0.5]])
    assert [M[i][i] for i in range(3)] == APPROX([1.0, 1.0, 1.0])
    assert flat(M) == APPROX(flat([[M[j][i] for j in range(3)] for i in range(3)]))


def test_cosine_matrix_rejects_a_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity_matrix([[1.0, 0.0], [0.0, 0.0]])


# ----------------------------------------------------------------- info_nce
def test_info_nce_with_a_single_pair_has_no_negatives_and_is_zero():
    """Главный урок про размер батча: при N=1 знаменатель пуст, лосс тождественно 0."""
    assert info_nce([[1.0, 0.0]], [[0.0, 1.0]]) == APPROX(0.0)
    assert info_nce([[1.0, 0.0]], [[1.0, 0.0]]) == APPROX(0.0)


def test_info_nce_of_mutually_orthogonal_views_equals_log_of_negative_count():
    """Все 2N-1 кандидатов равновероятны -> лосс ровно log(2N-1)."""
    n = 4
    z1 = [[1.0 if i == j else 0.0 for j in range(2 * n)] for i in range(n)]
    z2 = [[1.0 if i + n == j else 0.0 for j in range(2 * n)] for i in range(n)]
    assert info_nce(z1, z2, tau=0.1) == pytest.approx(math.log(2 * n - 1), abs=1e-9)


def test_info_nce_is_lower_when_the_positive_pair_is_aligned():
    """Смысл лосса: сближение пары видов одной картинки его уменьшает."""
    z1 = eye(3)

    def views(t):
        return [[1.0, t, 0.0], [0.0, 1.0, t], [t, 0.0, 1.0]]

    assert info_nce(z1, views(0.1)) < info_nce(z1, views(1.0))


def test_info_nce_does_not_depend_on_embedding_scale():
    z1, z2 = eye(3), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    scaled = [[5.0 * x for x in v] for v in z1]
    assert info_nce(scaled, z2) == APPROX(info_nce(z1, z2))


def test_info_nce_rejects_unpaired_views():
    with pytest.raises(ValueError):
        info_nce([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])


# --------------------------------------------------------------- ema_update
def test_ema_update_mixes_old_and_new():
    assert ema_update([1.0, 0.0], [0.0, 1.0], 0.9) == APPROX([0.9, 0.1])


def test_ema_momentum_one_freezes_the_teacher():
    assert ema_update([1.0, 2.0], [5.0, 5.0], 1.0) == APPROX([1.0, 2.0])


def test_ema_momentum_zero_copies_the_student():
    assert ema_update([1.0, 2.0], [5.0, 5.0], 0.0) == APPROX([5.0, 5.0])


def test_ema_converges_towards_the_student_over_many_steps():
    """Учитель отстаёт, но не отстаёт навсегда: 200 шагов при m=0.9 почти догоняют."""
    teacher = [0.0]
    for _ in range(200):
        teacher = ema_update(teacher, [1.0], 0.9)
    assert teacher[0] == pytest.approx(1.0, abs=1e-6)


def test_ema_rejects_momentum_outside_the_unit_interval():
    with pytest.raises(ValueError):
        ema_update([1.0], [2.0], 1.5)


# -------------------------------------------------------- center_and_sharpen
def test_teacher_output_is_a_probability_distribution():
    p = center_and_sharpen([2.0, -1.0, 0.5], [0.0, 0.0, 0.0], 0.4)
    assert sum(p) == APPROX(1.0)
    assert all(x >= 0.0 for x in p)


def test_centering_removes_a_permanently_dominant_dimension():
    """Без вычитания центра это измерение выигрывало бы всегда — коллапс DINO."""
    logits, center = [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]
    assert center_and_sharpen(logits, center, 1.0) == APPROX([1 / 3, 1 / 3, 1 / 3])
    uncentered = center_and_sharpen(logits, [0.0, 0.0, 0.0], 1.0)
    assert uncentered[0] > 0.99


def test_lower_temperature_sharpens_the_distribution():
    zero = [0.0, 0.0, 0.0]
    warm = center_and_sharpen([1.0, 0.5, 0.0], zero, 1.0)
    cold = center_and_sharpen([1.0, 0.5, 0.0], zero, 0.04)
    assert max(cold) > max(warm)
    assert max(cold) > 0.99


def test_equal_logits_give_a_uniform_distribution():
    assert center_and_sharpen([1.0, 1.0], [0.0, 0.0], 1.0) == APPROX([0.5, 0.5])


def test_center_and_sharpen_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        center_and_sharpen([1.0, 0.0], [0.0, 0.0], 0.0)


# --------------------------------------------------------------- dino_loss
def test_dino_loss_is_near_zero_when_the_student_nails_a_sharp_teacher():
    assert dino_loss([10.0, 0.0], [1.0, 0.0], 1.0) == pytest.approx(0.0, abs=1e-4)


def test_dino_loss_floor_is_the_teacher_entropy_not_zero():
    """Размытого учителя не переиграть: минимум кросс-энтропии равен H(teacher)."""
    p = [0.7, 0.3]
    entropy = -sum(x * math.log(x) for x in p)
    perfect = dino_loss([math.log(0.7), math.log(0.3)], p, 1.0)
    assert perfect == pytest.approx(entropy, abs=1e-9)
    assert dino_loss([0.0, 0.0], p, 1.0) > perfect


def test_dino_loss_drops_as_the_student_moves_towards_the_teacher():
    p = [0.9, 0.1]
    assert dino_loss([2.0, 0.0], p, 1.0) < dino_loss([0.5, 0.0], p, 1.0)


def test_dino_loss_rejects_raw_teacher_logits():
    """Подсунуть сюда логиты вместо softmax — типичная ошибка, она даёт минус вместо падения."""
    with pytest.raises(ValueError):
        dino_loss([1.0, 0.0], [3.0, -2.0], 1.0)


# ------------------------------------------------------- random_mask_indices
def test_mask_ratio_sets_how_many_patches_stay_visible():
    visible, masked = random_mask_indices(196, 0.75, random.Random(0))
    assert len(visible) == 49
    assert len(masked) == 147


def test_visible_and_masked_partition_every_patch_exactly_once():
    visible, masked = random_mask_indices(20, 0.75, random.Random(7))
    assert sorted(visible + masked) == list(range(20))
    assert set(visible) & set(masked) == set()


def test_both_index_lists_come_back_sorted():
    visible, masked = random_mask_indices(30, 0.75, random.Random(3))
    assert visible == sorted(visible)
    assert masked == sorted(masked)


def test_same_seed_reproduces_the_mask_and_a_different_seed_does_not():
    a = random_mask_indices(64, 0.75, random.Random(1))
    b = random_mask_indices(64, 0.75, random.Random(1))
    c = random_mask_indices(64, 0.75, random.Random(2))
    assert a == b
    assert a != c


def test_full_masking_is_rejected_because_the_encoder_would_see_nothing():
    with pytest.raises(ValueError):
        random_mask_indices(16, 1.0, random.Random(0))


# ------------------------------------------- masked_reconstruction_loss
def test_loss_counts_only_the_hidden_patches():
    """Видимый патч подан на вход — учитывать его ошибку значит дать списать."""
    original = [[0.0], [2.0]]
    assert masked_reconstruction_loss(original, [[9.0], [0.0]], [1]) == APPROX(4.0)
    assert masked_reconstruction_loss(original, [[0.0], [0.0]], [1]) == APPROX(4.0)


def test_perfect_reconstruction_of_hidden_patches_is_zero():
    assert masked_reconstruction_loss([[1.0], [2.0]], [[7.0], [2.0]], [1]) == APPROX(0.0)


def test_loss_averages_over_pixels_of_all_hidden_patches():
    original = [[0.0, 0.0], [0.0, 0.0]]
    assert masked_reconstruction_loss(original, [[1.0, 1.0], [3.0, 3.0]], [0, 1]) == APPROX(5.0)


def test_empty_mask_is_rejected_instead_of_reporting_a_perfect_zero():
    with pytest.raises(ValueError):
        masked_reconstruction_loss([[1.0]], [[2.0]], [])
