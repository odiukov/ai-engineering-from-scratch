"""Тесты к уроку «Open-vocabulary vision: CLIP». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    average_class_embeddings,
    build_prompts,
    clip_loss,
    normalize_rows,
    similarity_matrix,
    siglip_loss,
    zero_shot_classify,
    zero_shot_probabilities,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Развернуть матрицу в плоский список: pytest.approx не умеет вложенность."""
    return [v for row in M for v in row]


def eye(n):
    """n взаимно ортогональных единичных векторов — батч «ничего ни на что не похоже»."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ----------------------------------------------------------- normalize_rows
def test_normalize_rows_makes_every_row_unit_length():
    rows = normalize_rows([[3.0, 4.0], [0.0, 2.0], [-1.0, -1.0]])
    for r in rows:
        assert math.sqrt(sum(x * x for x in r)) == APPROX(1.0)


def test_normalize_rows_keeps_the_direction():
    assert flat(normalize_rows([[3.0, 4.0]])) == APPROX([0.6, 0.8])


def test_normalize_rows_rejects_a_zero_embedding():
    with pytest.raises(ValueError):
        normalize_rows([[1.0, 0.0], [0.0, 0.0]])


# -------------------------------------------------------- similarity_matrix
def test_similarity_spans_the_whole_range_from_minus_one_to_one():
    M = similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert flat(M) == APPROX([1.0, 0.0, -1.0])


def test_similarity_shape_is_images_by_texts():
    """Строки — картинки, столбцы — тексты. Перепутанный порядок не падает, а врёт."""
    M = similarity_matrix(eye(2), eye(3))
    assert len(M) == 2
    assert all(len(row) == 3 for row in M)


def test_similarity_ignores_feature_magnitude():
    small = similarity_matrix([[1.0, 1.0]], [[1.0, 0.0]])
    large = similarity_matrix([[50.0, 50.0]], [[7.0, 0.0]])
    assert flat(large) == APPROX(flat(small))


def test_similarity_of_two_independent_towers_is_zero():
    assert flat(similarity_matrix([[1.0, 0.0]], [[0.0, 5.0]])) == APPROX([0.0])


# --------------------------------------------------------------- clip_loss
def test_clip_loss_of_an_unrelated_batch_equals_log_of_batch_size():
    """Необученная модель: все похожести равны, лосс ровно log(N)."""
    n = 4
    images = [[1.0 if i == j else 0.0 for j in range(2 * n)] for i in range(n)]
    texts = [[1.0 if i + n == j else 0.0 for j in range(2 * n)] for i in range(n)]
    assert clip_loss(images, texts, 100.0) == pytest.approx(math.log(n), abs=1e-9)


def test_clip_loss_with_a_single_pair_is_zero_because_there_are_no_negatives():
    assert clip_loss([[1.0, 0.0]], [[0.0, 1.0]], 100.0) == APPROX(0.0)


def test_clip_loss_is_symmetric_in_its_two_towers():
    """image-to-text и text-to-image равноправны: обмен башен ничего не меняет."""
    images = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.3], [0.1, 0.0, 1.0]]
    texts = [[1.0, 0.0, 0.4], [0.2, 1.0, 0.0], [0.0, 0.5, 1.0]]
    assert clip_loss(texts, images, 5.0) == APPROX(clip_loss(images, texts, 5.0))


def test_clip_loss_falls_when_matching_pairs_come_closer():
    images = eye(3)
    far = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    near = [[1.0, 0.3, 0.0], [0.0, 1.0, 0.3], [0.3, 0.0, 1.0]]
    assert clip_loss(images, near, 5.0) < clip_loss(images, far, 5.0)


def test_clip_loss_rejects_a_batch_where_towers_disagree_on_size():
    with pytest.raises(ValueError):
        clip_loss(eye(3), eye(2), 10.0)


# ------------------------------------------------------------- siglip_loss
def test_siglip_still_learns_from_a_batch_of_one_where_clip_does_not():
    """Ради этого SigLIP и придуман: попарный лосс не нуждается в негативах батча."""
    image, text = [[1.0, 0.0]], [[-1.0, 0.0]]
    assert clip_loss(image, text, 10.0) == APPROX(0.0)
    assert siglip_loss(image, text, 10.0) > 1.0


def test_siglip_loss_is_near_zero_only_once_the_bias_separates_the_pairs():
    """Без bias даже идеальный батч платит softplus(0) за каждый негатив."""
    assert siglip_loss(eye(3), eye(3), 20.0, bias=-10.0) == pytest.approx(0.0, abs=1e-3)
    assert siglip_loss(eye(3), eye(3), 20.0) > 0.4


def test_siglip_loss_punishes_a_shuffled_pairing():
    shuffled = [eye(3)[1], eye(3)[2], eye(3)[0]]
    assert siglip_loss(eye(3), shuffled, 20.0) > siglip_loss(eye(3), eye(3), 20.0)


def test_siglip_bias_compensates_for_negatives_outnumbering_positives():
    """В батче N=4 негативов 12, а позитивов 4; отрицательный bias удешевляет негативы."""
    assert siglip_loss(eye(4), eye(4), 10.0, bias=-10.0) < siglip_loss(eye(4), eye(4), 10.0)


# ------------------------------------------------------------ build_prompts
def test_build_prompts_produces_one_string_per_class_and_template():
    got = build_prompts(["cat", "dog"], ["a photo of a {}", "a sketch of a {}"])
    assert got == [
        ["a photo of a cat", "a sketch of a cat"],
        ["a photo of a dog", "a sketch of a dog"],
    ]


def test_build_prompts_keeps_the_class_order():
    """Этот порядок станет порядком столбцов матрицы похожестей — перемешать нельзя."""
    names = ["zebra", "apple", "car"]
    got = build_prompts(names, ["{}"])
    assert [group[0] for group in got] == names


def test_build_prompts_rejects_a_template_without_a_placeholder():
    with pytest.raises(ValueError):
        build_prompts(["cat"], ["a photo of a dog"])


def test_build_prompts_rejects_an_empty_template_list():
    with pytest.raises(ValueError):
        build_prompts(["cat"], [])


# ------------------------------------------------ average_class_embeddings
def test_averaged_class_embedding_is_unit_length():
    """Без финальной нормализации класс с согласованными шаблонами получил бы фору."""
    got = average_class_embeddings([[[3.0, 0.0], [0.0, 3.0]]])
    assert flat(got) == APPROX([1 / math.sqrt(2), 1 / math.sqrt(2)])


def test_averaging_identical_templates_changes_nothing():
    assert flat(average_class_embeddings([[[1.0, 0.0], [1.0, 0.0]]])) == APPROX([1.0, 0.0])


def test_template_averaging_denoises_the_class_direction():
    """Ровно за этим 80 шаблонов вместо одного: шум формулировок гасится усреднением."""
    rng = random.Random(0)
    clean = [1.0] + [0.0] * 7
    noisy = [[c + rng.gauss(0.0, 0.8) for c in clean] for _ in range(40)]
    averaged = average_class_embeddings([noisy])[0]
    single = normalize_rows([noisy[0]])[0]
    cos = lambda v: sum(a * b for a, b in zip(v, normalize_rows([clean])[0]))
    assert cos(averaged) > cos(single)


def test_class_without_templates_is_rejected():
    with pytest.raises(ValueError):
        average_class_embeddings([[[1.0, 0.0]], []])


def test_templates_that_cancel_each_other_out_are_rejected():
    with pytest.raises(ValueError):
        average_class_embeddings([[[1.0, 0.0], [-1.0, 0.0]]])


# ----------------------------------------------- zero_shot_probabilities
def test_zero_shot_probabilities_rows_sum_to_one():
    probs = zero_shot_probabilities(eye(3), eye(4), 100.0)
    assert [sum(row) for row in probs] == APPROX([1.0, 1.0, 1.0])


def test_zero_shot_probabilities_match_the_softmax_of_scaled_cosines():
    probs = zero_shot_probabilities([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], 1.0)
    denom = math.e + 1.0
    assert flat(probs) == pytest.approx([math.e / denom, 1.0 / denom], abs=1e-9)


def test_higher_logit_scale_sharpens_the_prediction():
    """Косинусы живут в тесном [-1, 1]; logit_scale растягивает их в разделимые логиты."""
    warm = zero_shot_probabilities([[1.0, 0.2]], eye(2), 1.0)[0]
    cold = zero_shot_probabilities([[1.0, 0.2]], eye(2), 100.0)[0]
    assert max(cold) > max(warm)


def test_probabilities_stay_confident_even_when_no_class_fits():
    """Сумма в единицу — не уверенность: чужая картинка всё равно получит 0.99."""
    probs = zero_shot_probabilities([[1.0, 0.0]], [[1.0, 3.0], [-1.0, 3.0]], 100.0)[0]
    assert max(probs) > 0.99


# -------------------------------------------------------- zero_shot_classify
def test_zero_shot_classify_picks_the_nearest_class():
    assert zero_shot_classify(eye(2), eye(2), ["cat", "dog"]) == ["cat", "dog"]


def test_zero_shot_classify_ignores_image_feature_magnitude():
    assert zero_shot_classify([[9.0, 1.0]], eye(2), ["cat", "dog"]) == ["cat"]
    assert zero_shot_classify([[0.9, 0.1]], eye(2), ["cat", "dog"]) == ["cat"]


def test_a_new_class_needs_only_a_new_row_not_new_training():
    """Смысл слова open-vocabulary: словарь расширяется строкой текста."""
    image = [[0.1, 0.0, 1.0]]
    two = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert zero_shot_classify(image, two, ["cat", "dog"]) == ["cat"]
    assert zero_shot_classify(image, two + [[0.0, 0.0, 1.0]], ["cat", "dog", "plane"]) == ["plane"]


def test_zero_shot_classify_rejects_a_name_count_mismatch():
    with pytest.raises(ValueError):
        zero_shot_classify(eye(2), eye(3), ["cat", "dog"])
