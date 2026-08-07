"""Тесты к уроку «Семейство Qwen-VL и видео с динамическим FPS». Правь exercise.py."""

import math

import pytest

from exercise import (
    frame_timestamps,
    mrope_positions,
    mrope_rotate,
    parse_tool_call,
    pick_fps,
    rope_frequencies,
    rotate_pairs,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v))


# ------------------------------------------------------------ rope_frequencies
def test_rope_frequencies_start_at_one():
    assert rope_frequencies(4)[0] == APPROX(1.0)


def test_rope_frequencies_count_is_half_the_dim():
    assert len(rope_frequencies(96)) == 48


def test_rope_frequencies_decrease_with_index():
    """Быстрые частоты различают соседей, медленные — далёкие позиции."""
    freqs = rope_frequencies(16)
    assert freqs == sorted(freqs, reverse=True)
    assert freqs[0] > freqs[-1]


def test_rope_frequencies_rejects_an_odd_dim():
    with pytest.raises(ValueError):
        rope_frequencies(5)


# ----------------------------------------------------------------- rotate_pairs
def test_rotate_pairs_at_position_zero_changes_nothing():
    assert rotate_pairs([1.0, 2.0, 3.0, 4.0], 0, rope_frequencies(4)) == APPROX(
        [1.0, 2.0, 3.0, 4.0]
    )


def test_rotate_pairs_is_a_quarter_turn_at_the_right_angle():
    assert rotate_pairs([1.0, 0.0], math.pi / 2, [1.0]) == APPROX([0.0, 1.0])


def test_rotate_pairs_preserves_vector_length():
    """RoPE только крутит активации, не масштабирует их."""
    vec = [0.3, -1.2, 2.0, 0.5]
    turned = rotate_pairs(vec, 7.0, rope_frequencies(4))
    assert norm(turned) == pytest.approx(norm(vec), abs=1e-12)


def test_rotated_dot_product_depends_only_on_relative_position():
    """Главное свойство RoPE: сдвиг обеих позиций на одно и то же — ничего не меняет."""
    freqs = rope_frequencies(8)
    q = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3]
    k = [1.0, 0.5, -0.5, 2.0, 0.25, -1.25, 0.9, 0.4]
    near = dot(rotate_pairs(q, 3, freqs), rotate_pairs(k, 5, freqs))
    far = dot(rotate_pairs(q, 103, freqs), rotate_pairs(k, 105, freqs))
    assert near == pytest.approx(far, abs=1e-9)


def test_rotate_pairs_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        rotate_pairs([1.0, 2.0, 3.0], 1, rope_frequencies(4))


# --------------------------------------------------------------- mrope_positions
def test_mrope_positions_of_text_then_image():
    assert mrope_positions([("text", 2), ("image", 2, 2)]) == [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 0, 1),
        (2, 1, 0),
        (2, 1, 1),
    ]


def test_video_frames_advance_the_time_axis():
    assert mrope_positions([("video", 2, 1, 2)]) == [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 0, 1),
    ]


def test_all_patches_of_one_image_share_a_timestamp():
    """Картинка происходит в один момент времени — иначе строки станут «позже» друг друга."""
    times = {t for t, _, _ in mrope_positions([("image", 3, 4)])}
    assert times == {0}


def test_the_time_cursor_continues_after_an_image():
    """Текст после картинки не начинает отсчёт заново, иначе порядок теряется."""
    positions = mrope_positions([("image", 2, 2), ("text", 2)])
    assert positions[-2:] == [(1, 0, 0), (2, 0, 0)]


def test_mrope_positions_rejects_an_unknown_chunk():
    with pytest.raises(ValueError):
        mrope_positions([("audio", 5)])


# ----------------------------------------------------------------- mrope_rotate
def test_mrope_rotate_at_the_origin_changes_nothing():
    assert mrope_rotate([1.0] * 6, (0, 0, 0)) == APPROX([1.0] * 6)


def test_text_positions_leave_the_spatial_bands_alone():
    """У текста h = w = 0, значит M-RoPE вырождается в обычный одномерный RoPE."""
    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    turned = mrope_rotate(vec, (5, 0, 0))
    assert turned[2:] == APPROX(vec[2:])
    assert turned[:2] != APPROX(vec[:2])


def test_changing_the_width_touches_only_the_width_band():
    vec = [0.5] * 12
    a = mrope_rotate(vec, (0, 0, 0))
    b = mrope_rotate(vec, (0, 0, 3))
    assert a[:8] == APPROX(b[:8])
    assert a[8:] != APPROX(b[8:])


def test_mrope_rotate_preserves_vector_length():
    vec = [0.3, -1.2, 2.0, 0.5, -0.9, 1.1]
    assert norm(mrope_rotate(vec, (4, 2, 9))) == pytest.approx(norm(vec), abs=1e-12)


def test_mrope_rotate_requires_three_equal_bands():
    with pytest.raises(ValueError):
        mrope_rotate([1.0] * 8, (1, 1, 1))


# ---------------------------------------------------------------------- pick_fps
def test_pick_fps_spends_the_whole_budget_on_high_motion():
    assert pick_fps(60, 19440, 81, "high") == 4


def test_pick_fps_saves_tokens_when_nothing_moves():
    """Тот же бюджет, но запись экрана не нуждается в 4 FPS."""
    assert pick_fps(60, 19440, 81, "low") == 1


def test_pick_fps_returns_none_when_even_one_frame_per_second_overflows():
    assert pick_fps(600, 1000, 81, "high") is None


def test_pick_fps_never_exceeds_the_token_budget():
    for duration in (5, 30, 60, 300):
        for budget in (1000, 5000, 20000, 100000):
            fps = pick_fps(duration, budget, 81, "high")
            if fps is not None:
                assert fps * duration * 81 <= budget + 1e-3


def test_pick_fps_rejects_an_unknown_motion_level():
    with pytest.raises(ValueError):
        pick_fps(60, 19440, 81, "insane")


# -------------------------------------------------------------- frame_timestamps
def test_frame_timestamps_are_evenly_spaced():
    assert frame_timestamps(2.0, 2) == APPROX([0.0, 0.5, 1.0, 1.5])


def test_frame_timestamps_start_at_zero_and_stay_inside_the_clip():
    stamps = frame_timestamps(7.0, 4)
    assert stamps[0] == APPROX(0.0)
    assert max(stamps) < 7.0


def test_a_clip_shorter_than_one_frame_still_yields_a_frame():
    assert frame_timestamps(0.3, 1) == APPROX([0.0])


def test_frame_count_grows_with_fps():
    assert len(frame_timestamps(10.0, 4)) == 4 * len(frame_timestamps(10.0, 1))


def test_frame_timestamps_rejects_a_zero_fps():
    with pytest.raises(ValueError):
        frame_timestamps(10.0, 0)


# ---------------------------------------------------------------- parse_tool_call
def test_parse_tool_call_reads_a_plain_object():
    assert parse_tool_call('{"tool": "click", "coords": [380, 220]}') == {
        "tool": "click",
        "coords": [380, 220],
    }


def test_parse_tool_call_ignores_prose_and_code_fences():
    text = 'Конечно, вот действие:\n```json\n{"tool": "scroll"}\n```\nГотово.'
    assert parse_tool_call(text) == {"tool": "scroll"}


def test_parse_tool_call_ignores_braces_inside_strings():
    """Ловушка: rfind("}") и наивный счётчик скобок здесь ломаются."""
    assert parse_tool_call('{"tool": "type", "text": "{}"}') == {
        "tool": "type",
        "text": "{}",
    }


def test_parse_tool_call_rejects_malformed_json():
    with pytest.raises(ValueError):
        parse_tool_call('{"tool": "click", }')


def test_parse_tool_call_rejects_a_missing_required_key():
    with pytest.raises(ValueError):
        parse_tool_call('{"coords": [1, 2]}')


def test_parse_tool_call_rejects_coords_that_are_not_a_pair():
    with pytest.raises(ValueError):
        parse_tool_call('{"tool": "click", "coords": [1, 2, 3]}')
