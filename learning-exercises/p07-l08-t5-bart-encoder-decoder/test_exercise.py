"""Тесты к уроку «T5 и BART: encoder-decoder». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    corrupt_spans,
    cross_attention,
    document_rotate,
    pick_spans,
    round_trip,
    shift_right,
    softmax,
    text_infill,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def words(n):
    """n различимых токенов: удобно проверять, что именно спрятала порча."""
    return [f"t{i}" for i in range(n)]


# ---------------------------------------------------------------- softmax
def test_softmax_of_equal_scores_is_uniform():
    assert softmax([1.0, 1.0, 1.0, 1.0]) == APPROX([0.25] * 4)


def test_softmax_survives_huge_scores():
    """Без вычитания максимума exp(800) это OverflowError."""
    assert softmax([800.0, 800.0]) == APPROX([0.5, 0.5])


# -------------------------------------------------------- cross_attention
def test_cross_attention_has_one_row_per_target_token():
    out = cross_attention([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                          [[1.0, 0.0], [0.0, 1.0]],
                          [[1.0, 2.0], [3.0, 4.0]])
    assert len(out) == 3


def test_cross_attention_output_width_comes_from_the_values():
    """d_v живёт своей жизнью: ширина выхода берётся из V, не из Q."""
    out = cross_attention([[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 2.0, 3.0]])
    assert len(out[0]) == 3


def test_cross_attention_row_is_a_convex_combination_of_values():
    """Веса неотрицательны и суммируются в единицу, значит одинаковые
    значения энкодера дают ровно себя, чем бы ни был запрос."""
    out = cross_attention([[5.0, -2.0], [0.0, 0.0]],
                          [[1.0, 1.0], [2.0, -3.0]],
                          [[7.0], [7.0]])
    assert flat(out) == APPROX([7.0, 7.0])


def test_cross_attention_divides_scores_by_sqrt_of_dk():
    """Без деления на sqrt(d_k) ответ был бы 8.808 вместо 8.044."""
    out = cross_attention([[1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[10.0], [0.0]])
    w = math.exp(math.sqrt(2)) / (math.exp(math.sqrt(2)) + 1)
    assert out[0][0] == pytest.approx(10.0 * w, abs=1e-9)


def test_cross_attention_is_not_causal():
    """Ключевое отличие от self-attention декодера: нулевая позиция выхода
    видит ПОСЛЕДНИЙ токен источника, маски здесь нет."""
    Q = [[1.0, 0.0], [0.0, 1.0]]
    V = [[1.0], [5.0]]
    before = cross_attention(Q, [[1.0, 0.0], [0.0, 1.0]], V)
    after = cross_attention(Q, [[1.0, 0.0], [4.0, 4.0]], V)
    assert before[0][0] != pytest.approx(after[0][0], abs=1e-6)


def test_cross_attention_rejects_mismatched_source_length():
    with pytest.raises(ValueError):
        cross_attention([[1.0]], [[1.0], [2.0]], [[3.0]])


# ------------------------------------------------------------ shift_right
def test_shift_right_puts_the_start_token_first():
    assert shift_right([7, 8, 9], 0)[0] == 0


def test_shift_right_feeds_the_previous_target_at_every_position():
    """Teacher forcing: на позиции i декодер видит настоящий токен i-1."""
    targets = [4, 5, 6, 7]
    shifted = shift_right(targets, 99)
    assert all(shifted[i] == targets[i - 1] for i in range(1, len(targets)))


def test_shift_right_does_not_mutate_the_targets():
    targets = [1, 2, 3]
    shift_right(targets, 0)
    assert targets == [1, 2, 3]


# ------------------------------------------------------------- pick_spans
def test_picked_spans_are_sorted_and_disjoint():
    spans = pick_spans(60, random.Random(3), mask_rate=0.2)
    ends = [0]
    for start, length in spans:
        assert start >= ends[-1]
        ends.append(start + length)
    assert spans == sorted(spans)


def test_picked_spans_stay_inside_the_sequence():
    n = 40
    for seed in range(5):
        for start, length in pick_spans(n, random.Random(seed)):
            assert 0 <= start and start + length <= n


def test_picked_spans_mask_the_requested_share():
    """15% от 100 токенов — ровно 15 замаскированных, как в T5."""
    spans = pick_spans(100, random.Random(0), mask_rate=0.15)
    assert sum(length for _, length in spans) == 15


def test_span_count_follows_the_mean_span_length():
    """15 токенов средними спанами по 3 — это примерно пять спанов."""
    assert len(pick_spans(100, random.Random(0), mask_rate=0.15, mean_span=3.0)) == 5


def test_same_seed_gives_the_same_spans():
    assert pick_spans(50, random.Random(11)) == pick_spans(50, random.Random(11))


def test_different_seeds_give_different_spans():
    a = pick_spans(30, random.Random(0), mask_rate=0.2)
    b = pick_spans(30, random.Random(1), mask_rate=0.2)
    assert a != b


def test_pick_spans_rejects_an_impossible_mask_rate():
    with pytest.raises(ValueError):
        pick_spans(50, random.Random(0), mask_rate=1.5)


# ---------------------------------------------------------- corrupt_spans
def test_corrupt_spans_follows_the_t5_format():
    source, target = corrupt_spans(["a", "b", "c", "d"], [(1, 2)])
    assert source == ["a", "<extra_id_0>", "d"]
    assert target == ["<extra_id_0>", "b", "c", "<extra_id_1>"]


def test_corrupted_source_hides_every_masked_token():
    tokens = words(30)
    spans = pick_spans(30, random.Random(5), mask_rate=0.2)
    source, _ = corrupt_spans(tokens, spans)
    hidden = [tokens[i] for start, length in spans for i in range(start, start + length)]
    assert all(token not in source for token in hidden)


def test_target_ends_with_the_closing_sentinel():
    tokens = words(30)
    spans = pick_spans(30, random.Random(5), mask_rate=0.2)
    _, target = corrupt_spans(tokens, spans)
    assert target[-1] == f"<extra_id_{len(spans)}>"


def test_source_length_swaps_each_span_for_one_sentinel():
    tokens = words(40)
    spans = pick_spans(40, random.Random(2), mask_rate=0.2)
    source, _ = corrupt_spans(tokens, spans)
    masked = sum(length for _, length in spans)
    assert len(source) == len(tokens) - masked + len(spans)


def test_corrupt_spans_rejects_overlapping_spans():
    with pytest.raises(ValueError):
        corrupt_spans(words(10), [(1, 3), (2, 2)])


# ------------------------------------------------------------- round_trip
def test_round_trip_restores_the_original():
    tokens = ["the", "quick", "brown", "fox", "jumps", "over"]
    source, target = corrupt_spans(tokens, [(1, 2), (4, 1)])
    assert round_trip(source, target) == tokens


def test_round_trip_survives_a_span_at_the_very_end():
    """Закрывающий sentinel не должен утащить с собой последний спан."""
    tokens = words(6)
    source, target = corrupt_spans(tokens, [(4, 2)])
    assert round_trip(source, target) == tokens


def test_round_trip_survives_a_span_at_the_very_start():
    tokens = words(6)
    source, target = corrupt_spans(tokens, [(0, 2)])
    assert round_trip(source, target) == tokens


def test_round_trip_holds_for_many_random_seeds():
    """Проверка обратимости на десятках раскладок — ловит любую ошибку
    на единицу в бухгалтерии спанов."""
    for seed in range(20):
        tokens = words(45)
        spans = pick_spans(45, random.Random(seed), mask_rate=0.2)
        source, target = corrupt_spans(tokens, spans)
        assert round_trip(source, target) == tokens


# ------------------------------------------------------------ text_infill
def test_text_infill_replaces_a_span_with_a_single_mask():
    assert text_infill(["a", "b", "c", "d"], [(1, 2)]) == ["a", "<mask>", "d"]


def test_text_infill_forgets_the_span_length():
    """Порча T5 обратима, а эта — нет: два разных исходника дают один и тот
    же результат, поэтому декодер BART обязан угадывать длину сам."""
    assert text_infill(["a", "b", "c", "d"], [(1, 2)]) == text_infill(["a", "b", "d"], [(1, 1)])


def test_text_infill_shortens_the_sequence():
    tokens = words(20)
    spans = [(2, 3), (8, 4)]
    assert len(text_infill(tokens, spans)) == 20 - (3 + 4) + 2


def test_text_infill_rejects_out_of_range_spans():
    with pytest.raises(ValueError):
        text_infill(words(5), [(3, 9)])


# -------------------------------------------------------- document_rotate
def test_document_rotate_moves_the_head_to_the_tail():
    assert document_rotate(["a", "b", "c", "d"], 1) == ["b", "c", "d", "a"]


def test_rotation_is_invertible():
    """Прокрутка на p и потом на n - p возвращает исходный порядок."""
    tokens = words(9)
    assert document_rotate(document_rotate(tokens, 4), 9 - 4) == tokens


def test_document_rotate_keeps_every_token():
    tokens = words(7)
    assert sorted(document_rotate(tokens, 3)) == sorted(tokens)


def test_document_rotate_rejects_an_out_of_range_pivot():
    with pytest.raises(ValueError):
        document_rotate(["a", "b"], 2)
