"""Тесты к уроку «Video-language модели: временные токены и grounding».

Правь exercise.py.
"""

import pytest

from exercise import (
    dynamic_sample,
    frame_difference,
    grounding_recall,
    parse_time_tokens,
    pooled_tokens,
    position_ids,
    temporal_iou,
    uniform_sample,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# посекундное движение из демо урока: всплески на 2-4 и 7-9 секундах
MOTION = [0.1, 0.1, 0.8, 0.9, 0.9, 0.2, 0.1, 0.5, 0.9, 0.9]


def per_second(times):
    """Сколько отметок пришлось на каждую секунду."""
    counts = {}
    for t in times:
        counts[int(t)] = counts.get(int(t), 0) + 1
    return counts


# --------------------------------------------------------- frame_difference
def test_frame_difference_of_a_static_video_is_all_zeros():
    assert frame_difference([[5.0], [5.0], [5.0]]) == APPROX([0.0, 0.0, 0.0])


def test_first_frame_has_no_previous_frame_to_compare_with():
    assert frame_difference([[9.0, 9.0], [0.0, 0.0]])[0] == APPROX(0.0)


def test_frame_difference_spikes_exactly_where_the_picture_changes():
    frames = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    motion = frame_difference(frames)
    assert motion == APPROX([0.0, 0.0, 1.0, 0.0])


def test_frame_difference_rejects_frames_of_different_size():
    """zip молча обрезал бы длинный кадр и сравнил разные куски картинки."""
    with pytest.raises(ValueError):
        frame_difference([[1.0, 2.0], [1.0]])


# ------------------------------------------------------------ uniform_sample
def test_uniform_sample_takes_bin_centres():
    assert uniform_sample(10.0, 2) == APPROX([2.5, 7.5])


def test_uniform_sample_of_one_frame_lands_in_the_middle():
    assert uniform_sample(10.0, 1) == APPROX([5.0])


def test_uniform_sample_never_touches_the_very_first_or_last_moment():
    """Кадр в нуле секунд почти всегда чёрный, кадра ровно в конце может не быть."""
    times = uniform_sample(4.0, 8)
    assert all(0.0 < t < 4.0 for t in times)


def test_uniform_sample_gaps_are_all_equal():
    times = uniform_sample(9.0, 6)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps == APPROX([gaps[0]] * len(gaps))


# ------------------------------------------------------------ dynamic_sample
def test_dynamic_sample_worked_example():
    assert dynamic_sample([0.0, 1.0], 2, 4) == APPROX([0.5, 1.5])


def test_dynamic_sample_never_exceeds_the_total_budget():
    assert len(dynamic_sample([0.0, 1.0, 1.0], 2, 4)) == 2
    assert len(dynamic_sample([1.0, 1.0, 1.0], 4, 4)) == 4


def test_dynamic_sample_puts_more_frames_where_the_motion_is():
    counts = per_second(dynamic_sample(MOTION, 14, 4))
    assert counts[3] > counts[0]
    assert counts[8] > counts[6]


def test_dynamic_sample_never_leaves_a_second_unseen():
    """Секунда без единого кадра для модели просто не существует."""
    counts = per_second(dynamic_sample(MOTION, 12, 4))
    assert all(counts.get(sec, 0) >= 1 for sec in range(len(MOTION)))


def test_dynamic_sample_respects_the_fps_cap():
    times = dynamic_sample(MOTION, 200, 4)
    counts = per_second(times)
    assert max(counts.values()) <= 4
    assert len(times) == len(MOTION) * 4


def test_a_static_camera_degrades_to_uniform_sampling():
    """Нулевое движение — делить не на что, и dynamic обязан не упасть."""
    assert dynamic_sample([0.0, 0.0, 0.0], 6, 4) == APPROX(uniform_sample(3.0, 6))
    assert dynamic_sample([0.0, 0.0, 0.0], 4, 4) == APPROX(uniform_sample(3.0, 4))


# ------------------------------------------------------------- pooled_tokens
def test_pooling_three_by_three_turns_576_tokens_into_64():
    assert pooled_tokens(24, 3) == 64


def test_pooling_is_off_when_the_pool_is_one():
    assert pooled_tokens(24, 1) == 576


def test_pooling_is_what_makes_five_minutes_of_video_fit():
    """Числа урока: 300 кадров по 576 токенов против тех же 300 по 64."""
    assert 300 * pooled_tokens(24, 1) == 172800
    assert 300 * pooled_tokens(24, 3) == 19200


def test_pooled_tokens_rejects_a_pool_bigger_than_the_grid():
    with pytest.raises(ValueError):
        pooled_tokens(3, 4)


# -------------------------------------------------------------- temporal_iou
def test_identical_intervals_have_iou_one():
    assert temporal_iou(0.0, 2.0, 0.0, 2.0) == APPROX(1.0)


def test_half_overlap_worked_example():
    assert temporal_iou(0.0, 2.0, 1.0, 3.0) == APPROX(1.0 / 3.0)


def test_disjoint_intervals_have_iou_zero():
    assert temporal_iou(0.0, 1.0, 5.0, 6.0) == APPROX(0.0)


def test_temporal_iou_is_symmetric():
    assert temporal_iou(1.0, 4.0, 2.0, 9.0) == APPROX(temporal_iou(2.0, 9.0, 1.0, 4.0))


def test_temporal_iou_rejects_an_event_of_zero_length():
    """Нулевое событие обнуляет знаменатель — это не «мгновенное», а битое."""
    with pytest.raises(ValueError):
        temporal_iou(3.0, 3.0, 1.0, 4.0)


# ----------------------------------------------------------- grounding_recall
def test_grounding_recall_counts_a_close_prediction_as_a_hit():
    assert grounding_recall([("jump", 4.1, 4.7)], [("jump", 4.0, 4.5)], 0.3) == APPROX(1.0)


def test_right_time_but_wrong_event_is_not_a_hit():
    assert grounding_recall([("turn", 4.1, 4.7)], [("jump", 4.0, 4.5)], 0.3) == APPROX(0.0)


def test_raising_the_tolerance_can_only_lower_recall():
    preds = [("jump", 4.1, 4.7), ("sit", 9.2, 9.6)]
    truths = [("jump", 4.0, 4.5), ("sit", 8.5, 9.5), ("turn", 6.0, 6.5)]
    assert grounding_recall(preds, truths, 0.3) >= grounding_recall(preds, truths, 0.9)


def test_grounding_recall_refuses_an_empty_benchmark():
    """Вернуть 1.0 на пустом ground truth — самый обидный способ соврать себе."""
    with pytest.raises(ValueError):
        grounding_recall([("jump", 1.0, 2.0)], [], 0.3)


# --------------------------------------------------------------- position_ids
def test_index_mode_numbers_the_frames():
    assert position_ids([0.0, 0.5, 4.0], "index") == [0, 1, 2]


def test_time_mode_keeps_the_actual_seconds():
    assert position_ids([0.0, 0.5, 4.0], "time") == APPROX([0.0, 0.5, 4.0])


def test_frame_indices_erase_uneven_sampling_but_timestamps_keep_it():
    """Ровно то, что даёт TMRoPE: два разных сэмплирования обязаны различаться."""
    even = uniform_sample(2.0, 3)
    uneven = dynamic_sample([0.0, 3.0], 3, 4)
    assert len(even) == len(uneven)
    assert position_ids(even, "index") == position_ids(uneven, "index")
    assert position_ids(even, "time") != position_ids(uneven, "time")


def test_position_ids_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        position_ids([0.0, 1.0], "rope3d")


# ----------------------------------------------------------- parse_time_tokens
def test_parse_time_tokens_reads_a_single_mark():
    assert parse_time_tokens("The cat jumps at <time>4.2</time>") == APPROX([4.2])


def test_parse_time_tokens_keeps_the_order_of_events():
    assert parse_time_tokens("<time>1.0</time> then <time>3.5</time>") == APPROX([1.0, 3.5])


def test_free_text_without_marks_yields_nothing():
    assert parse_time_tokens("the cat jumps around the four second mark") == []


def test_parse_time_tokens_rejects_a_non_numeric_mark():
    """Модель регулярно пишет «early»; пропустить это дальше значит получить NaN."""
    with pytest.raises(ValueError):
        parse_time_tokens("jump at <time>early</time>")


def test_parse_time_tokens_rejects_negative_time():
    with pytest.raises(ValueError):
        parse_time_tokens("jump at <time>-2.0</time>")
