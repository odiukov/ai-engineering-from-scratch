"""Тесты к уроку «Omni-модели: Qwen2.5-Omni и разделение Thinker/Talker».

Правь exercise.py.
"""

import pytest

from exercise import (
    STAGE_MS_7B,
    interleave_by_time,
    pipeline_total_ms,
    scaled_budget,
    speech_tokens_needed,
    talker_keeps_up,
    tmrope_positions,
    ttfab_ms,
    turn_end_frame,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------------ ttfab_ms
def test_ttfab_is_the_sum_of_sequential_stages():
    assert ttfab_ms({"prefill": 150.0, "rvq_decode": 30.0}) == APPROX(180.0)


def test_ttfab_of_an_empty_budget_is_zero():
    assert ttfab_ms({}) == APPROX(0.0)


def test_ttfab_at_seven_b_fits_the_conversational_window():
    """Урок обещает 320-510 мс на 7B — этот бюджет обязан попасть внутрь."""
    assert 320.0 <= ttfab_ms(STAGE_MS_7B) <= 510.0


def test_ttfab_does_not_depend_on_stage_order():
    """Стадии последовательные, сумма коммутативна — переставленный словарь даёт то же."""
    reversed_stages = dict(reversed(list(STAGE_MS_7B.items())))
    assert ttfab_ms(reversed_stages) == APPROX(ttfab_ms(STAGE_MS_7B))


# ------------------------------------------------------------- scaled_budget
def test_scaled_budget_at_the_reference_size_changes_nothing():
    assert scaled_budget(STAGE_MS_7B, 7.0) == STAGE_MS_7B


def test_scaled_budget_stretches_only_prefill():
    scaled = scaled_budget({"prefill": 150.0, "rvq_decode": 30.0}, 70.0)
    assert scaled["prefill"] == APPROX(1500.0)
    assert scaled["rvq_decode"] == APPROX(30.0)


def test_scaled_budget_pushes_seventy_b_out_of_the_conversational_window():
    """Смысловая проверка урока: на 70B TTFAB перестаёт быть разговорным."""
    assert ttfab_ms(scaled_budget(STAGE_MS_7B, 70.0)) > 500.0


def test_scaled_budget_does_not_mutate_the_input():
    """Бюджет считают в цикле по размерам — испорченный вход отравит все итерации."""
    stages = {"prefill": 150.0, "rvq_decode": 30.0}
    scaled_budget(stages, 70.0)
    assert stages == {"prefill": 150.0, "rvq_decode": 30.0}


# ------------------------------------------------------ speech_tokens_needed
def test_speech_tokens_needed_for_one_second():
    assert speech_tokens_needed(1.0) == 50


def test_speech_tokens_needed_scales_with_duration():
    assert speech_tokens_needed(2.5) == 125


def test_speech_tokens_needed_rounds_up_a_partial_token():
    """Половины токена не бывает, а обрезанный хвост слышно как обрыв."""
    assert speech_tokens_needed(0.101) == 6


def test_speech_tokens_needed_of_silence_is_zero():
    assert speech_tokens_needed(0.0) == 0


# ---------------------------------------------------------- talker_keeps_up
def test_talker_keeps_up_when_faster_than_the_speech_rate():
    assert talker_keeps_up(80) is True


def test_talker_exactly_at_the_speech_rate_still_keeps_up():
    assert talker_keeps_up(50) is True


def test_slow_talker_falls_behind():
    assert talker_keeps_up(30) is False


def test_a_seven_b_talker_cannot_keep_up_at_thirty_tokens_per_second():
    """Отсюда и берутся отдельные маленькие Talker-модели на 200-300M."""
    assert talker_keeps_up(30, token_rate_hz=50) is False
    assert talker_keeps_up(120, token_rate_hz=50) is True


# -------------------------------------------------------- pipeline_total_ms
def test_pipeline_total_is_bounded_by_the_slowest_stage():
    assert pipeline_total_ms(10, 40.0, 20.0) == APPROX(400.0)


def test_pipeline_total_adds_the_startup_that_cannot_overlap():
    assert pipeline_total_ms(10, 40.0, 20.0, 100.0) == APPROX(500.0)


def test_pipeline_with_no_text_tokens_costs_only_startup():
    assert pipeline_total_ms(0, 40.0, 20.0, 100.0) == APPROX(100.0)


def test_pipeline_never_loses_to_the_sequential_schedule():
    """Выигрыш стриминга: max(a, b) на токен вместо a + b."""
    for thinker, talker in ((40.0, 20.0), (20.0, 40.0), (30.0, 30.0)):
        parallel = pipeline_total_ms(12, thinker, talker, 100.0)
        sequential = 100.0 + 12 * (thinker + talker)
        assert parallel <= sequential


def test_pipeline_matches_sequential_when_the_talker_is_free():
    """Если одна ступень мгновенная, совмещать нечего — схемы совпадают."""
    assert pipeline_total_ms(12, 40.0, 0.0, 100.0) == APPROX(100.0 + 12 * 40.0)


# --------------------------------------------------------- tmrope_positions
def test_tmrope_positions_count_time_bins_not_list_slots():
    assert tmrope_positions([(0.0, "audio", "a"), (0.04, "vision", "v")], 25) == [0, 1]


def test_tmrope_gives_concurrent_events_the_same_position():
    """Голос на 1.00 с и жест на 1.01 с — для модели один момент."""
    events = [(1.0, "audio", "hello"), (1.01, "vision", "wave")]
    assert tmrope_positions(events, 25) == [25, 25]


def test_tmrope_positions_keep_the_input_order():
    events = [(2.0, "text", "b"), (0.0, "audio", "a")]
    assert tmrope_positions(events, 10) == [20, 0]


def test_tmrope_resolution_controls_how_much_counts_as_simultaneous():
    """Чем выше resolution_hz, тем меньше событий сливается в один момент."""
    events = [(1.0, "audio", "a"), (1.01, "vision", "v")]
    assert len(set(tmrope_positions(events, 25))) == 1
    assert len(set(tmrope_positions(events, 1000))) == 2


# -------------------------------------------------------- interleave_by_time
def test_interleave_orders_events_by_timestamp():
    events = [(2.0, "text", "t"), (1.0, "audio", "a")]
    assert interleave_by_time(events) == [(1.0, "audio", "a"), (2.0, "text", "t")]


def test_interleave_mixes_modalities_instead_of_grouping_them():
    """Наивная раскладка «все кадры, потом весь звук» рвёт временную связь."""
    events = [
        (0.0, "vision", "f0"),
        (0.5, "vision", "f1"),
        (0.2, "audio", "a0"),
        (0.7, "audio", "a1"),
    ]
    assert [e[1] for e in interleave_by_time(events)] == [
        "vision", "audio", "vision", "audio",
    ]


def test_interleave_is_stable_on_equal_timestamps():
    """Одинаковое время — сохраняем порядок поступления, иначе контекст плавает."""
    events = [(1.0, "audio", "a"), (1.0, "vision", "v"), (1.0, "text", "t")]
    assert interleave_by_time(events) == events


def test_interleave_does_not_mutate_the_input():
    events = [(2.0, "text", "t"), (1.0, "audio", "a")]
    interleave_by_time(events)
    assert events == [(2.0, "text", "t"), (1.0, "audio", "a")]


# ------------------------------------------------------------ turn_end_frame
def test_turn_end_frame_fires_after_enough_silence():
    energies = [1.0, 1.0] + [0.0] * 10
    assert turn_end_frame(energies, frame_ms=20, silence_ms=200) == 11


def test_turn_end_frame_returns_none_while_the_user_keeps_talking():
    assert turn_end_frame([1.0] * 20) is None


def test_a_short_breath_does_not_end_the_turn():
    """Три тихих кадра — это вдох, а не конец реплики: 60 мс < 200 мс."""
    energies = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert turn_end_frame(energies, frame_ms=20, silence_ms=200) is None


def test_the_silence_run_restarts_after_speech():
    """Счётчик тишины обязан обнуляться, иначе паузы «накопятся» и оборвут речь."""
    energies = [0.0] * 5 + [1.0] + [0.0] * 10
    assert turn_end_frame(energies, frame_ms=20, silence_ms=200) == 15


def test_a_longer_threshold_delays_the_turn_end():
    energies = [1.0] + [0.0] * 30
    early = turn_end_frame(energies, frame_ms=20, silence_ms=200)
    late = turn_end_frame(energies, frame_ms=20, silence_ms=400)
    assert early == 10
    assert late == 20
