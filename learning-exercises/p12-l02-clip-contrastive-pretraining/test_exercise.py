"""Тесты к уроку «CLIP и контрастивное предобучение». Правь exercise.py."""

import math

import pytest

from exercise import (
    cosine_similarity,
    infonce_grad,
    infonce_loss,
    l2_normalize,
    prompt_ensemble,
    sigmoid_pairwise_loss,
    similarity_matrix,
    zero_shot_classify,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-6)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def numeric_grad(f, S, h=1e-6):
    """Численный градиент центральной разностью — эталон для сверки."""
    out = []
    for i in range(len(S)):
        line = []
        for j in range(len(S[i])):
            up = [row[:] for row in S]
            down = [row[:] for row in S]
            up[i][j] += h
            down[i][j] -= h
            line.append((f(up) - f(down)) / (2 * h))
        out.append(line)
    return out


# ------------------------------------------------------------- l2_normalize
def test_l2_normalize_makes_unit_length():
    assert l2_normalize([3.0, 4.0]) == APPROX([0.6, 0.8])


def test_l2_normalize_keeps_direction():
    """Нормировка меняет длину, но не направление: пропорции сохраняются."""
    v = l2_normalize([2.0, 4.0, 6.0])
    assert v[1] / v[0] == APPROX(2.0)
    assert v[2] / v[0] == APPROX(3.0)


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0])


# -------------------------------------------------------- cosine_similarity
def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_ignores_magnitude():
    """Главное свойство: яркость картинки или длина подписи не влияют."""
    a, b = [1.0, 2.0, -3.0], [0.5, -1.0, 2.0]
    base = cosine_similarity(a, b)
    assert cosine_similarity([100 * x for x in a], b) == APPROX(base)
    assert cosine_similarity(a, [0.001 * x for x in b]) == APPROX(base)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == APPROX(-1.0)


def test_cosine_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ------------------------------------------------------- similarity_matrix
def test_similarity_matrix_shape_and_values():
    S = similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert flat(S) == APPROX([1.0, 0.0])


def test_matching_pairs_land_on_the_diagonal():
    """Правильная пара обязана быть максимумом своей строки — иначе лосс бессмыслен."""
    imgs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    S = similarity_matrix(imgs, imgs)
    for i, row in enumerate(S):
        assert row[i] == max(row)


def test_low_temperature_sharpens_the_logits():
    S1 = similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], temperature=1.0)
    S2 = similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], temperature=0.07)
    assert S2[0][0] - S2[0][1] > S1[0][0] - S1[0][1]


def test_similarity_matrix_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        similarity_matrix([[1.0]], [[1.0]], temperature=0.0)


# ------------------------------------------------------------ infonce_loss
def test_infonce_on_a_flat_matrix_is_log_n():
    """Никакой информации — модель угадывает, лосс равен log(N)."""
    assert infonce_loss([[0.0] * 4 for _ in range(4)]) == ROUGH(math.log(4))


def test_infonce_is_near_zero_for_a_perfect_batch():
    assert infonce_loss([[10.0, 0.0], [0.0, 10.0]]) < 1e-3


def test_infonce_is_symmetric_under_transpose():
    """Лосс считается и по картинкам, и по подписям — транспонирование ничего не меняет."""
    S = [[3.0, 1.0, -2.0], [0.5, 2.0, 0.0], [-1.0, 0.25, 4.0]]
    T = [[S[j][i] for j in range(3)] for i in range(3)]
    assert infonce_loss(T) == ROUGH(infonce_loss(S))


def test_shuffling_the_captions_hurts():
    """Если подписи перепутаны местами, лосс обязан вырасти — это и есть сигнал."""
    good = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    bad = [row[1:] + row[:1] for row in good]  # подписи сдвинуты на одну
    assert infonce_loss(bad) > infonce_loss(good)


def test_infonce_survives_huge_logits():
    """Наивный exp(1000) даёт OverflowError; logsumexp со сдвигом — нет."""
    S = [[1000.0, 0.0], [0.0, 1000.0]]
    assert infonce_loss(S) < 1e-6


def test_infonce_ignores_a_global_constant():
    """Прибавление одной константы ко всей матрице не меняет softmax."""
    S = [[1.0, -0.5], [0.25, 2.0]]
    shifted = [[x + 7.0 for x in row] for row in S]
    assert infonce_loss(shifted) == ROUGH(infonce_loss(S))


# ------------------------------------------------------------ infonce_grad
def test_infonce_grad_matches_central_difference():
    S = [[1.2, -0.4, 0.3], [0.1, 0.8, -1.1], [-0.6, 0.2, 1.5]]
    assert flat(infonce_grad(S)) == pytest.approx(
        flat(numeric_grad(infonce_loss, S)), abs=1e-6
    )


def test_infonce_grad_matches_central_difference_on_a_flat_matrix():
    S = [[0.0] * 3 for _ in range(3)]
    assert flat(infonce_grad(S)) == pytest.approx(
        flat(numeric_grad(infonce_loss, S)), abs=1e-6
    )


def test_infonce_grad_sums_to_zero():
    """Следствие сдвиговой инвариантности лосса: полная сумма градиента нулевая."""
    S = [[1.2, -0.4, 0.3], [0.1, 0.8, -1.1], [-0.6, 0.2, 1.5]]
    assert sum(flat(infonce_grad(S))) == ROUGH(0.0)


def test_infonce_grad_pushes_the_diagonal_up():
    """На диагонали градиент отрицательный — шаг спуска увеличит сходство пары."""
    S = [[0.0, 0.0], [0.0, 0.0]]
    g = infonce_grad(S)
    assert g[0][0] < 0 and g[1][1] < 0
    assert g[0][1] > 0 and g[1][0] > 0


def test_infonce_grad_vanishes_on_a_solved_batch():
    S = [[30.0, 0.0], [0.0, 30.0]]
    assert max(abs(x) for x in flat(infonce_grad(S))) < 1e-9


# ---------------------------------------------------- sigmoid_pairwise_loss
def test_sigmoid_loss_on_a_flat_matrix():
    """N^2 клеток по log 2, делённые на N: для N=2 это 2*log 2."""
    assert sigmoid_pairwise_loss([[0.0, 0.0], [0.0, 0.0]]) == ROUGH(2 * math.log(2))


def test_sigmoid_loss_is_near_zero_for_a_perfect_batch():
    assert sigmoid_pairwise_loss([[20.0, -20.0], [-20.0, 20.0]]) < 1e-6


def test_sigmoid_loss_survives_huge_logits():
    """Ловушка: exp(1000) падает, log_sigmoid через min(x,0)-log1p — нет."""
    assert sigmoid_pairwise_loss([[1000.0, -1000.0], [-1000.0, 1000.0]]) < 1e-6


def test_negative_bias_helps_when_negatives_dominate():
    """Негативов N^2-N против N позитивов: отрицательный bias сдвигает порог к ним."""
    S = [[0.0] * 8 for _ in range(8)]
    assert sigmoid_pairwise_loss(S, bias=-3.0) < sigmoid_pairwise_loss(S, bias=0.0)


def test_sigmoid_loss_unlike_infonce_reacts_to_a_global_shift():
    """У softmax сдвиг всей матрицы бесплатен, у сигмоиды — нет: пары независимы."""
    S = [[2.0, -1.0], [-1.0, 2.0]]
    assert sigmoid_pairwise_loss(S, bias=5.0) != ROUGH(sigmoid_pairwise_loss(S))


# ------------------------------------------------------- zero_shot_classify
def test_zero_shot_picks_the_nearest_class():
    got = zero_shot_classify([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], ["cat", "dog"])
    assert got[0] == "cat"
    assert got[1] == APPROX(1.0)


def test_zero_shot_ignores_prompt_vector_length():
    """Косинус, а не скалярное произведение: длинный эмбеддинг класса не жульничает."""
    classes = [[0.9, 0.1], [50.0, 50.0]]
    assert zero_shot_classify([1.0, 0.0], classes, ["a", "b"])[0] == "a"


def test_zero_shot_breaks_ties_toward_the_first_class():
    same = [[1.0, 0.0], [2.0, 0.0]]
    assert zero_shot_classify([1.0, 0.0], same, ["first", "second"])[0] == "first"


def test_zero_shot_rejects_empty_class_list():
    with pytest.raises(ValueError):
        zero_shot_classify([1.0, 0.0], [], [])


# ---------------------------------------------------------- prompt_ensemble
def test_prompt_ensemble_averages_and_renormalizes():
    assert prompt_ensemble([[1.0, 0.0], [0.0, 1.0]]) == ROUGH(
        [math.sqrt(0.5), math.sqrt(0.5)]
    )


def test_prompt_ensemble_returns_a_unit_vector():
    got = prompt_ensemble([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 5.0, 0.0]])
    assert math.sqrt(sum(x * x for x in got)) == ROUGH(1.0)


def test_prompt_ensemble_ignores_template_order():
    templates = [[1.0, 2.0], [-3.0, 1.0], [0.5, 0.5]]
    assert prompt_ensemble(templates) == ROUGH(prompt_ensemble(list(reversed(templates))))


def test_prompt_ensemble_normalizes_each_template_first():
    """Иначе один длинный шаблон перевесит остальные и ансамбль выродится."""
    short = [[1.0, 0.0], [0.0, 1.0]]
    long_one = [[1.0, 0.0], [0.0, 1000.0]]
    assert prompt_ensemble(long_one) == ROUGH(prompt_ensemble(short))


