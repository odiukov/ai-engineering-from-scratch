"""Тесты к уроку «BERT и masked language modeling». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    build_bert_input,
    classify_from_cls,
    create_mlm_batch,
    mlm_accuracy,
    mlm_loss,
    mlm_loss_grad,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CLS, SEP, MASK = 1, 2, 1000
VOCAB = 1000


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


def uniform_logits(n, vocab):
    return [[0.0] * vocab for _ in range(n)]


# ----------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([2.0, -1.0, 0.5])) == pytest.approx(1.0)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([4.0, 4.0]) == APPROX([0.5, 0.5])


def test_softmax_survives_huge_logits():
    """Без вычитания максимума math.exp(900) кидает OverflowError."""
    assert sum(softmax([900.0, 899.0])) == pytest.approx(1.0)


def test_softmax_rejects_an_empty_logit_vector():
    with pytest.raises(ValueError):
        softmax([])


# -------------------------------------------------------- build_bert_input
def test_single_sequence_gets_cls_in_front_and_sep_at_the_end():
    ids, segments = build_bert_input([7, 8], CLS, SEP)
    assert ids == [CLS, 7, 8, SEP]
    assert segments == [0, 0, 0, 0]


def test_a_pair_gets_two_separators_and_two_segment_ids():
    ids, segments = build_bert_input([7], CLS, SEP, tokens_b=[9, 10])
    assert ids == [CLS, 7, SEP, 9, 10, SEP]
    assert segments == [0, 0, 0, 1, 1, 1]


def test_cls_is_always_position_zero():
    """За нулевую позицию потом цепляется голова классификации."""
    ids, _ = build_bert_input([42, 43, 44], CLS, SEP)
    assert ids[0] == CLS


def test_the_first_separator_belongs_to_the_first_segment():
    _, segments = build_bert_input([7], CLS, SEP, tokens_b=[9])
    assert segments[2] == 0


def test_build_bert_input_does_not_mutate_the_token_lists():
    a, b = [7, 8], [9]
    build_bert_input(a, CLS, SEP, tokens_b=b)
    assert a == [7, 8] and b == [9]


# -------------------------------------------------------- create_mlm_batch
def test_mlm_batch_keeps_the_sequence_length():
    tokens = [5] * 20
    ids, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(0))
    assert len(ids) == 20 and len(labels) == 20


def test_zero_probability_selects_nothing():
    tokens = [5, 6, 7]
    ids, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(0), mask_prob=0.0)
    assert ids == tokens
    assert labels == [-100, -100, -100]


def test_full_probability_labels_every_position_with_its_original_token():
    tokens = [5, 6, 7, 8]
    _, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(1), mask_prob=1.0)
    assert labels == tokens


def test_special_tokens_are_never_selected_for_prediction():
    """[CLS], [SEP] и [MASK] остаются контекстом, а не MLM-целями."""
    tokens = [CLS, 5, SEP, MASK]
    ids, labels = create_mlm_batch(
        tokens,
        VOCAB,
        MASK,
        random.Random(1),
        mask_prob=1.0,
        special_token_ids={CLS, SEP, MASK},
    )
    assert ids[0] == CLS and ids[2] == SEP and ids[3] == MASK
    assert labels == [-100, 5, -100, -100]


def test_unselected_positions_are_never_touched():
    """Ключевой инвариант: label = -100 означает «вход тут исходный»."""
    tokens = [i % 50 + 3 for i in range(300)]
    ids, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(2))
    for i, label in enumerate(labels):
        if label == -100:
            assert ids[i] == tokens[i]


def test_the_same_seed_gives_the_same_batch():
    tokens = [5] * 50
    first = create_mlm_batch(tokens, VOCAB, MASK, random.Random(7))
    second = create_mlm_batch(tokens, VOCAB, MASK, random.Random(7))
    assert first == second


def test_different_seeds_give_different_batches():
    tokens = [5] * 50
    first = create_mlm_batch(tokens, VOCAB, MASK, random.Random(7))
    second = create_mlm_batch(tokens, VOCAB, MASK, random.Random(8))
    assert first != second


def test_create_mlm_batch_does_not_mutate_the_input():
    tokens = [5, 6, 7, 8]
    create_mlm_batch(tokens, VOCAB, MASK, random.Random(3), mask_prob=1.0)
    assert tokens == [5, 6, 7, 8]


def test_about_fifteen_percent_gets_selected_by_default():
    tokens = [5] * 4000
    _, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(4))
    selected = sum(1 for label in labels if label != -100)
    assert 0.12 < selected / len(tokens) < 0.18


def test_the_eighty_ten_ten_rule_holds():
    """Ветки считаются по решению RNG, а не по случайно совпавшим token id."""
    tokens = [5] * 4000
    decisions = []
    create_mlm_batch(
        tokens, VOCAB, MASK, random.Random(5), mask_prob=1.0, decisions=decisions
    )
    assert 0.75 < decisions.count("mask") / len(decisions) < 0.85
    assert 0.05 < decisions.count("unchanged") / len(decisions) < 0.15
    assert 0.05 < decisions.count("random") / len(decisions) < 0.15


def test_random_branch_is_recorded_when_replacement_equals_the_original():
    """По ids такой случай похож на unchanged, но фактически сработала ветка random."""

    class OriginalCollisionRng:
        rolls = iter((0.0, 0.85))

        def random(self):
            return next(self.rolls)

        def randrange(self, stop):
            # При special ids 1 и 2 список допустимых начинается [0, 3, 4, 5].
            return 3

    decisions = []
    ids, labels = create_mlm_batch(
        [5],
        VOCAB,
        MASK,
        OriginalCollisionRng(),
        mask_prob=1.0,
        special_token_ids={CLS, SEP, MASK},
        decisions=decisions,
    )
    assert ids == [5] and labels == [5]
    assert decisions == ["random"]


def test_some_selected_positions_keep_their_original_token():
    """Те самые 10% «нетронутых» — без них модель ждала бы [MASK] всегда."""
    tokens = [5] * 500
    ids, labels = create_mlm_batch(tokens, VOCAB, MASK, random.Random(6), mask_prob=1.0)
    honest = [i for i in range(len(tokens)) if labels[i] != -100 and ids[i] == tokens[i]]
    assert len(honest) > 0


def test_random_replacements_stay_inside_the_vocabulary():
    tokens = [5] * 500
    ids, _ = create_mlm_batch(tokens, VOCAB, MASK, random.Random(9), mask_prob=1.0)
    assert all(token == MASK or 0 <= token < VOCAB for token in ids)


def test_random_replacements_exclude_special_tokens():
    """Даже если special ids лежат внутри словаря, random-ветка их не выдаёт."""
    tokens = [5] * 2000
    decisions = []
    ids, _ = create_mlm_batch(
        tokens,
        VOCAB,
        MASK,
        random.Random(10),
        mask_prob=1.0,
        special_token_ids={0, CLS, SEP, MASK},
        decisions=decisions,
    )
    random_ids = [token for token, branch in zip(ids, decisions) if branch == "random"]
    assert random_ids
    assert all(token not in {0, CLS, SEP, MASK} for token in random_ids)


# ---------------------------------------------------------------- mlm_loss
def test_loss_on_uniform_logits_equals_log_of_the_vocabulary():
    """Первый ориентир при отладке обучения: старт обязан быть log(V)."""
    assert mlm_loss(uniform_logits(3, 50), [0, 1, 2]) == pytest.approx(math.log(50))


def test_loss_of_a_confident_correct_prediction_is_near_zero():
    assert mlm_loss([[20.0, 0.0, 0.0]], [0]) < 1e-6


def test_loss_of_a_confident_wrong_prediction_is_large():
    assert mlm_loss([[20.0, 0.0, 0.0]], [1]) > 15.0


def test_loss_with_nothing_to_predict_is_zero():
    assert mlm_loss(uniform_logits(3, 10), [-100, -100, -100]) == 0.0


def test_loss_ignores_the_logits_of_unpredicted_positions():
    """-100 значит «эту строку не смотреть» — даже если логиты дикие."""
    quiet = [[0.0, 0.0], [0.0, 0.0]]
    loud = [[0.0, 0.0], [500.0, -500.0]]
    assert mlm_loss(quiet, [0, -100]) == APPROX(mlm_loss(loud, [0, -100]))


def test_loss_divides_by_the_predicted_count_not_the_sequence_length():
    """Делить на длину — ошибка в 6.7 раза при mask_prob=0.15, и её не видно."""
    logits = uniform_logits(10, 4)
    labels = [-100] * 10
    labels[3] = 2
    assert mlm_loss(logits, labels) == pytest.approx(math.log(4))


# ----------------------------------------------------------- mlm_loss_grad
def test_grad_of_a_uniform_row_is_prediction_minus_truth():
    assert flat(mlm_loss_grad([[0.0, 0.0]], [0])) == APPROX([-0.5, 0.5])


def test_grad_rows_sum_to_zero():
    """Softmax суммируется в 1, one-hot тоже — разность обязана дать 0."""
    logits = [[1.0, -2.0, 0.5, 3.0], [0.0, 0.0, 1.0, 2.0]]
    grad = mlm_loss_grad(logits, [2, 0])
    assert [sum(row) for row in grad] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_grad_of_unpredicted_rows_is_all_zeros():
    grad = mlm_loss_grad([[1.0, 2.0], [3.0, 4.0]], [-100, 1])
    assert grad[0] == APPROX([0.0, 0.0])
    assert grad[1] != APPROX([0.0, 0.0])


def test_grad_pushes_the_correct_class_up():
    """У правильного класса градиент отрицательный: шаг спуска поднимет логит."""
    grad = mlm_loss_grad([[0.0, 0.0, 0.0]], [1])
    assert grad[0][1] < 0
    assert grad[0][0] > 0 and grad[0][2] > 0


def test_grad_matches_the_numeric_derivative():
    """Центральная разность по одному логиту — единственная честная проверка."""
    logits = [[0.7, -1.2, 0.3], [2.0, 0.1, -0.4], [0.0, 0.0, 0.0]]
    labels = [2, 0, -100]
    grad = mlm_loss_grad(logits, labels)
    h = 1e-6
    for i in range(len(logits)):
        for j in range(len(logits[i])):
            up = [row[:] for row in logits]
            down = [row[:] for row in logits]
            up[i][j] += h
            down[i][j] -= h
            numeric = (mlm_loss(up, labels) - mlm_loss(down, labels)) / (2 * h)
            assert grad[i][j] == pytest.approx(numeric, abs=1e-6)


def test_grad_with_nothing_to_predict_is_all_zeros():
    grad = mlm_loss_grad([[1.0, 2.0], [3.0, 4.0]], [-100, -100])
    assert flat(grad) == APPROX([0.0, 0.0, 0.0, 0.0])


# ------------------------------------------------------------ mlm_accuracy
def test_accuracy_is_one_when_every_argmax_is_right():
    assert mlm_accuracy([[9.0, 0.0], [0.0, 9.0]], [0, 1]) == pytest.approx(1.0)


def test_accuracy_counts_only_predicted_positions():
    assert mlm_accuracy([[9.0, 0.0], [0.0, 9.0]], [0, -100]) == pytest.approx(1.0)


def test_accuracy_of_half_right_is_a_half():
    assert mlm_accuracy([[9.0, 0.0], [0.0, 9.0]], [0, 0]) == pytest.approx(0.5)


def test_accuracy_with_nothing_to_predict_is_zero():
    assert mlm_accuracy([[9.0, 0.0]], [-100]) == 0.0


def test_accuracy_ignores_confidence_while_loss_does_not():
    """Уверенность растёт, argmax тот же: loss падает, accuracy стоит."""
    weak = [[0.1, 0.0]]
    strong = [[9.0, 0.0]]
    assert mlm_accuracy(weak, [0]) == mlm_accuracy(strong, [0])
    assert mlm_loss(strong, [0]) < mlm_loss(weak, [0])


# ------------------------------------------------------- classify_from_cls
def test_classification_head_returns_a_distribution():
    hidden = [[1.0, 0.0], [0.5, 0.5]]
    W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    out = classify_from_cls(hidden, W, [0.0, 0.0, 0.0])
    assert len(out) == 3
    assert sum(out) == pytest.approx(1.0)


def test_classification_head_reads_only_the_cls_position():
    """Весь «pooling» BERT — это взять нулевую строку и больше ничего."""
    W = [[1.0, 0.0], [0.0, 1.0]]
    b = [0.0, 0.0]
    first = classify_from_cls([[1.0, 0.0], [0.0, 0.0]], W, b)
    second = classify_from_cls([[1.0, 0.0], [99.0, -99.0]], W, b)
    assert first == APPROX(second)


def test_classification_head_reacts_to_the_cls_vector():
    W = [[1.0, 0.0], [0.0, 1.0]]
    b = [0.0, 0.0]
    positive = classify_from_cls([[3.0, 0.0]], W, b)
    negative = classify_from_cls([[0.0, 3.0]], W, b)
    assert positive[0] > positive[1]
    assert negative[1] > negative[0]


def test_the_bias_alone_can_flip_the_decision():
    W = [[0.0, 0.0], [0.0, 0.0]]
    out = classify_from_cls([[1.0, 1.0]], W, [0.0, 5.0])
    assert out[1] > out[0]


def test_classification_head_rejects_an_empty_encoder_output():
    with pytest.raises(ValueError):
        classify_from_cls([], [[1.0]], [0.0])
