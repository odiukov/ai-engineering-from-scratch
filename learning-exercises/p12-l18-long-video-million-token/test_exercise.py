"""Тесты к уроку «Длинное видео и контекст на миллион токенов».

Правь exercise.py.
"""

import random

import pytest

from exercise import (
    compression_gain,
    max_duration,
    needle_trial,
    pick_strategy,
    recall_at,
    ring_chunk,
    summary_token_budget,
    token_budget,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# кривая recall открытой 72B-модели на получасовом ролике из урока
CURVE = [(0.1, 0.95), (0.5, 0.85), (1.0, 0.75)]


# --------------------------------------------------------------- token_budget
def test_half_an_hour_at_one_fps_is_the_number_from_the_lesson():
    assert token_budget(1800, 1, 81) == 145800


def test_a_two_hour_film_blows_past_every_open_context():
    assert token_budget(7200, 1, 81) == 583200


def test_doubling_the_frame_rate_doubles_the_bill():
    assert token_budget(1800, 2, 81) == 2 * token_budget(1800, 1, 81)


def test_pooling_is_the_cheapest_lever_on_the_budget():
    """3x3 pooling режет 729 токенов до 81 — ровно в девять раз дешевле."""
    assert token_budget(1800, 1, 729) == 9 * token_budget(1800, 1, 81)


def test_token_budget_rejects_a_zero_frame_rate():
    with pytest.raises(ValueError):
        token_budget(1800, 0, 81)


# --------------------------------------------------------------- max_duration
def test_max_duration_inverts_token_budget():
    assert max_duration(145800, 1, 81) == APPROX(1800.0)


def test_max_duration_is_the_largest_that_still_fits():
    limit = 1000
    seconds = max_duration(limit, 1, 81)
    assert token_budget(seconds, 1, 81) <= limit
    assert token_budget(seconds + 1, 1, 81) > limit


def test_a_context_smaller_than_one_frame_holds_no_video_at_all():
    """Ноль — законный ответ: так выглядит 2 FPS в контексте на 32k."""
    assert max_duration(50, 1, 81) == APPROX(0.0)


def test_higher_frame_rate_shortens_what_fits():
    assert max_duration(145800, 2, 81) < max_duration(145800, 1, 81)


# ------------------------------------------------------- summary_token_budget
def test_summary_token_budget_worked_example():
    assert summary_token_budget(1800, 1, 16) == 113


def test_a_partial_clip_still_gets_its_summary_token():
    """Округление вниз потеряло бы хвост ролика целиком."""
    assert summary_token_budget(10, 1, 16) == 1


def test_summary_tokens_are_three_orders_of_magnitude_cheaper():
    raw = token_budget(1800, 1, 81)
    summarised = summary_token_budget(1800, 1, 16)
    assert compression_gain(raw, summarised) > 1000


def test_summary_token_budget_rejects_an_empty_clip():
    with pytest.raises(ValueError):
        summary_token_budget(1800, 1, 0)


# ----------------------------------------------------------- compression_gain
def test_compression_gain_worked_example():
    assert compression_gain(145800, 145) == pytest.approx(1005.517, abs=1e-3)


def test_no_compression_means_a_gain_of_one():
    assert compression_gain(100, 100) == APPROX(1.0)


def test_compression_gain_rejects_a_zero_denominator():
    with pytest.raises(ValueError):
        compression_gain(145800, 0)


# ----------------------------------------------------------------- ring_chunk
def test_ring_chunk_worked_example():
    assert ring_chunk(10, 4) == [3, 3, 2, 2]


def test_ring_chunk_loses_no_token():
    assert sum(ring_chunk(268000, 8)) == 268000


def test_ring_chunks_differ_by_at_most_one():
    """Пиковую память определяет самый большой кусок — перекос дорог."""
    chunks = ring_chunk(268000, 7)
    assert max(chunks) - min(chunks) <= 1


def test_adding_devices_shrinks_the_per_device_chunk():
    assert max(ring_chunk(268000, 16)) < max(ring_chunk(268000, 8))


def test_ring_chunk_needs_at_least_one_device():
    with pytest.raises(ValueError):
        ring_chunk(1000, 0)


# ------------------------------------------------------------------ recall_at
def test_recall_at_the_start_of_the_video_is_the_first_bucket():
    assert recall_at(CURVE, 0.05) == APPROX(0.95)


def test_a_position_exactly_on_a_threshold_belongs_to_that_bucket():
    assert recall_at(CURVE, 0.5) == APPROX(0.85)


def test_recall_degrades_towards_the_end_of_a_long_video():
    assert recall_at(CURVE, 0.99) < recall_at(CURVE, 0.05)


def test_recall_at_rejects_an_unsorted_curve():
    """Первый подходящий порог оказался бы не тем, и ответ был бы тихо неверным."""
    with pytest.raises(ValueError):
        recall_at([(0.5, 0.85), (0.1, 0.95)], 0.3)


def test_recall_at_rejects_a_position_measured_in_seconds():
    with pytest.raises(ValueError):
        recall_at(CURVE, 42.0)


# ---------------------------------------------------------------- needle_trial
def test_needle_trial_is_reproducible_for_a_given_seed():
    a = needle_trial(1800.0, CURVE, random.Random(7))
    b = needle_trial(1800.0, CURVE, random.Random(7))
    assert a == b


def test_different_seeds_move_the_needle():
    a = needle_trial(1800.0, CURVE, random.Random(1))
    b = needle_trial(1800.0, CURVE, random.Random(2))
    assert a["needle_time"] != b["needle_time"]


def test_needle_position_is_a_fraction_of_the_video():
    trial = needle_trial(1800.0, CURVE, random.Random(3))
    assert 0.0 <= trial["position"] <= 1.0
    assert trial["needle_time"] == APPROX(trial["position"] * 1800.0)


def test_needle_recall_comes_from_the_curve():
    trial = needle_trial(1800.0, CURVE, random.Random(4))
    assert trial["recall"] == APPROX(recall_at(CURVE, trial["position"]))


# --------------------------------------------------------------- pick_strategy
def test_a_short_clip_just_goes_into_the_context():
    assert pick_strategy(10, "general") == "brute"


def test_half_an_hour_needs_compression():
    assert pick_strategy(30, "general") == "compression"


def test_a_specific_question_about_two_hours_goes_to_retrieval():
    assert pick_strategy(120, "specific") == "agentic"


def test_retrieval_does_not_help_a_general_question():
    """«О чём ролик» требует пройти весь ролик — выдернутые клипы не спасут."""
    assert pick_strategy(120, "general") == "brute"


def test_pick_strategy_rejects_an_unknown_question_type():
    with pytest.raises(ValueError):
        pick_strategy(120, "multiple choice")
