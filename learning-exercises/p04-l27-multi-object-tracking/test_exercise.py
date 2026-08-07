"""Тесты к уроку «Многообъектный трекинг и память видео». Правь exercise.py."""

import pytest

from exercise import (
    associate,
    count_id_switches,
    iou,
    iou_matrix,
    mota,
    optimal_assignment,
    run_tracker,
    update_tracks,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """Развернуть матрицу в плоский список — pytest.approx не умеет вложенность."""
    return [value for row in matrix for value in row]


# --------------------------------------------------------------------- iou
def test_iou_of_a_box_with_itself_is_one():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == APPROX(1.0)


def test_iou_of_far_apart_boxes_is_zero():
    """Ловушка: ширина пересечения отрицательна, и без обрезки нулём
    два минуса дадут положительную «площадь»."""
    assert iou((0, 0, 2, 2), (5, 5, 7, 7)) == APPROX(0.0)


def test_iou_of_half_overlapping_boxes():
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == APPROX(1 / 3)


def test_iou_never_exceeds_one_for_touching_boxes():
    """Боксы, соприкасающиеся ребром, не пересекаются: IoU ровно 0."""
    assert iou((0, 0, 2, 2), (2, 0, 4, 2)) == APPROX(0.0)


# -------------------------------------------------------------- iou_matrix
def test_iou_matrix_has_one_row_per_first_box():
    matrix = iou_matrix([(0, 0, 2, 2), (5, 5, 7, 7)], [(0, 0, 2, 2)])
    assert len(matrix) == 2
    assert flat(matrix) == APPROX([1.0, 0.0])


def test_iou_matrix_has_one_column_per_second_box():
    matrix = iou_matrix([(0, 0, 2, 2)], [(0, 0, 2, 2), (5, 5, 7, 7)])
    assert flat(matrix) == APPROX([1.0, 0.0])


def test_iou_matrix_without_tracks_is_empty():
    assert iou_matrix([], [(0, 0, 1, 1)]) == []


# ------------------------------------------------------- optimal_assignment
def test_assignment_beats_the_greedy_choice():
    """Жадность схватит клетку 1 и застрянет на 9. Оптимум берёт 2 + 1."""
    assert optimal_assignment([[1.0, 2.0], [1.0, 9.0]]) == [(0, 1), (1, 0)]


def test_assignment_with_more_detections_than_tracks_leaves_columns_free():
    pairs = optimal_assignment([[5.0, 1.0, 5.0]])
    assert pairs == [(0, 1)]


def test_assignment_with_more_tracks_than_detections_leaves_rows_free():
    pairs = optimal_assignment([[5.0], [1.0], [5.0]])
    assert pairs == [(1, 0)]


def test_assignment_of_an_empty_matrix_is_empty():
    assert optimal_assignment([]) == []


def test_assignment_refuses_matrices_too_big_for_brute_force():
    with pytest.raises(ValueError):
        optimal_assignment([[0.0] * 9 for _ in range(9)])


# --------------------------------------------------------------- associate
def test_associate_matches_a_track_to_the_same_box():
    assert associate([(0, 0, 2, 2)], [(0, 0, 2, 2)]) == ([(0, 0)], [], [])


def test_associate_rejects_a_pair_below_the_iou_threshold():
    """Оптимум обязан кого-то назначить, но IoU = 0 это не совпадение."""
    assert associate([(0, 0, 2, 2)], [(9, 9, 11, 11)]) == ([], [0], [0])


def test_associate_without_tracks_reports_every_detection_as_new():
    assert associate([], [(0, 0, 2, 2)]) == ([], [], [0])


def test_associate_does_not_cross_two_nearby_objects():
    """Два трека и две детекции: каждый обязан достаться своему, не соседу."""
    tracks = [(0, 0, 10, 10), (100, 0, 110, 10)]
    dets = [(101, 1, 111, 11), (1, 1, 11, 11)]
    matches, lost, new = associate(tracks, dets)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert lost == [] and new == []


def test_associate_honours_a_stricter_threshold():
    tracks = [(0, 0, 10, 10)]
    dets = [(5, 0, 15, 10)]                 # IoU = 1/3
    assert associate(tracks, dets, iou_threshold=0.3)[0] == [(0, 0)]
    assert associate(tracks, dets, iou_threshold=0.5)[0] == []


# ------------------------------------------------------------ update_tracks
def test_update_births_a_track_for_a_new_detection():
    tracks, next_id = update_tracks([], [(0, 0, 2, 2)], 0, 1)
    assert next_id == 2
    assert tracks == [{"id": 1, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]


def test_update_keeps_the_id_and_counts_the_hit():
    start = [{"id": 7, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]
    tracks, _ = update_tracks(start, [(1, 0, 3, 2)], 1, 8)
    assert tracks[0]["id"] == 7
    assert tracks[0]["hits"] == 2
    assert tracks[0]["bbox"] == (1, 0, 3, 2)


def test_update_keeps_an_unmatched_track_alive_within_max_age():
    """Пропуск кадра — не смерть трека, ради этого max_age и существует."""
    start = [{"id": 3, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]
    tracks, _ = update_tracks(start, [], 5, 4, max_age=5)
    assert [t["id"] for t in tracks] == [3]


def test_update_deletes_a_track_older_than_max_age():
    start = [{"id": 3, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]
    tracks, _ = update_tracks(start, [], 6, 4, max_age=5)
    assert tracks == []


def test_update_does_not_mutate_the_tracks_it_was_given():
    start = [{"id": 1, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]
    update_tracks(start, [(0, 0, 2, 2)], 1, 2)
    assert start == [{"id": 1, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}]


# --------------------------------------------------------------- run_tracker
def test_run_tracker_keeps_one_id_for_a_still_object():
    result = run_tracker([[(0, 0, 2, 2)], [(0, 0, 2, 2)]])
    assert result == [[(1, (0, 0, 2, 2))], [(1, (0, 0, 2, 2))]]


def test_run_tracker_keeps_ids_of_three_objects_moving_in_straight_lines():
    frames = []
    for f in range(20):
        frames.append([
            (10 + 3 * f, 10, 30 + 3 * f, 30),
            (100, 20 + 2 * f, 120, 40 + 2 * f),
            (200 - 4 * f, 200, 220 - 4 * f, 220),
        ])
    result = run_tracker(frames)
    assert all(len(frame) == 3 for frame in result)
    assert all([tid for tid, _ in frame] == [1, 2, 3] for frame in result)


def test_run_tracker_does_not_change_the_id_after_a_missed_frame():
    """Детектор моргнул на одном кадре — id обязан остаться прежним."""
    box = (0, 0, 10, 10)
    result = run_tracker([[box], [], [box]])
    assert result[0] == [(1, box)]
    assert result[2] == [(1, box)]


def test_run_tracker_starts_a_new_id_when_an_object_appears():
    frames = [[(0, 0, 10, 10)], [(0, 0, 10, 10), (100, 100, 110, 110)]]
    result = run_tracker(frames)
    assert [tid for tid, _ in result[1]] == [1, 2]


# --------------------------------------------------------- count_id_switches
def test_no_switches_when_the_id_stays_the_same():
    tracks = [[(1, (0, 0, 2, 2))], [(1, (0, 0, 2, 2))]]
    gts = [[(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))]]
    assert count_id_switches(tracks, gts) == 0


def test_one_switch_when_the_assigned_track_id_changes():
    tracks = [[(1, (0, 0, 2, 2))], [(2, (0, 0, 2, 2))]]
    gts = [[(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))]]
    assert count_id_switches(tracks, gts) == 1


def test_a_skipped_frame_is_not_a_switch():
    """Кадр без детекций сохраняет прошлое назначение, а не ломает его."""
    tracks = [[(1, (0, 0, 2, 2))], [], [(1, (0, 0, 2, 2))]]
    gts = [[(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))]]
    assert count_id_switches(tracks, gts) == 0


def test_a_track_too_far_from_the_ground_truth_is_ignored():
    tracks = [[(1, (0, 0, 2, 2))], [(2, (50, 50, 52, 52))], [(1, (0, 0, 2, 2))]]
    gts = [[(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))]]
    assert count_id_switches(tracks, gts) == 0


# ---------------------------------------------------------------------- mota
def test_mota_of_a_perfect_tracker_is_one():
    assert mota(0, 0, 0, 100) == APPROX(1.0)


def test_mota_drops_by_the_share_of_missed_objects():
    assert mota(10, 0, 0, 100) == APPROX(0.9)


def test_mota_can_go_negative():
    """Ложных срабатываний больше, чем объектов — метрика честно уходит в минус."""
    assert mota(50, 60, 10, 100) == APPROX(-0.2)


def test_mota_weighs_all_three_error_types_equally():
    assert mota(3, 0, 0, 100) == APPROX(mota(0, 0, 3, 100))


def test_mota_without_ground_truth_is_undefined():
    with pytest.raises(ValueError):
        mota(0, 0, 0, 0)
