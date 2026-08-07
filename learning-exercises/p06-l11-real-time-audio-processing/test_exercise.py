"""Тесты к уроку «Обработка звука в реальном времени». Правь exercise.py."""

import pytest

from exercise import (
    RingBuffer,
    barge_in,
    buffer_latency_ms,
    energy_vad,
    frame_length,
    jitter_buffer,
    keeps_up_with_realtime,
    pipeline_latency,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------ frame_length
def test_twenty_ms_at_16khz_is_320_samples():
    assert frame_length(16000, 20) == 320


def test_frame_length_scales_with_the_sample_rate():
    """Тот же кадр 20 мс на 48 кГц — втрое больше сэмплов."""
    assert frame_length(48000, 20) == 3 * frame_length(16000, 20)


def test_frame_length_scales_with_the_frame_duration():
    assert frame_length(44100, 10) == 441
    assert frame_length(44100, 20) == 882


def test_frame_length_rejects_a_fractional_frame():
    """44100 Гц и 5 мс дают 220.5 сэмпла — такой кадр не нарезать."""
    with pytest.raises(ValueError):
        frame_length(44100, 5)
    with pytest.raises(ValueError):
        frame_length(16000, 0)


# ------------------------------------------------------- buffer_latency_ms
def test_two_second_ring_at_16khz_costs_2000_ms():
    assert buffer_latency_ms(32000, 16000) == APPROX(2000.0)


def test_one_frame_of_buffer_costs_one_frame_of_latency():
    assert buffer_latency_ms(320, 16000) == APPROX(20.0)


def test_buffer_latency_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        buffer_latency_ms(-1, 16000)
    with pytest.raises(ValueError):
        buffer_latency_ms(320, 0)


# --------------------------------------------------------------- RingBuffer
def test_ring_buffer_returns_samples_in_write_order():
    rb = RingBuffer(8)
    rb.write([1, 2, 3])
    rb.write([4, 5])
    assert rb.read(5) == [1, 2, 3, 4, 5]


def test_ring_buffer_loses_nothing_below_capacity():
    """Пока места хватает, кольцо обязано вернуть ровно то, что в него писали."""
    rb = RingBuffer(1000)
    written = []
    for i in range(0, 900, 100):
        frame = list(range(i, i + 100))
        rb.write(frame)
        written.extend(frame)
    assert rb.read(len(written)) == written


def test_ring_buffer_drops_the_oldest_on_overflow():
    """Переполнение вытесняет просроченное, а не свежее: микрофон не остановить."""
    rb = RingBuffer(4)
    rb.write([1, 2, 3, 4, 5, 6])
    assert rb.level() == 4
    assert rb.read(4) == [3, 4, 5, 6]


def test_ring_buffer_underrun_returns_what_it_has():
    """Consumer пришёл раньше producer'а — это не ошибка, а обычное дело."""
    rb = RingBuffer(8)
    rb.write([1, 2])
    assert rb.read(5) == [1, 2]
    assert rb.level() == 0


def test_ring_buffer_rejects_a_capacity_it_cannot_hold():
    with pytest.raises(ValueError):
        RingBuffer(0)


# ---------------------------------------------------------------- energy_vad
def test_silence_is_not_speech():
    assert energy_vad([0.0] * 320, 0.01) is False


def test_vad_squares_the_samples_instead_of_averaging_them():
    """Ловушка: среднее у [-1, 1, -1, 1] равно нулю, а кадр громкий."""
    assert energy_vad([-1.0, 1.0, -1.0, 1.0], 0.5) is True


def test_vad_threshold_is_compared_on_the_rms_scale():
    """0.02 по амплитуде против порога 0.01 — речь; против 0.05 — уже нет."""
    frame = [0.02] * 320
    assert energy_vad(frame, 0.01) is True
    assert energy_vad(frame, 0.05) is False


def test_energy_vad_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        energy_vad([], 0.01)
    with pytest.raises(ValueError):
        energy_vad([0.1], -0.01)


# -------------------------------------------------------------- jitter_buffer
def test_jitter_buffer_reorders_a_late_packet():
    assert jitter_buffer([(0, "a"), (2, "c"), (1, "b")], depth=2) == ["a", "b", "c"]


def test_jitter_buffer_passes_an_ordered_stream_through_unchanged():
    packets = [(i, i) for i in range(6)]
    assert jitter_buffer(packets, depth=2) == list(range(6))


def test_a_shallow_buffer_drops_what_a_deep_one_saves():
    """Тот самый компромисс: глубина лечит потери и добавляет задержку."""
    packets = [(0, "a"), (2, "c"), (3, "d"), (1, "b")]
    assert jitter_buffer(packets, depth=1) == ["a", "c", "d"]
    assert jitter_buffer(packets, depth=3) == ["a", "b", "c", "d"]


def test_jitter_buffer_never_invents_or_duplicates_a_packet():
    packets = [(0, "a"), (3, "d"), (1, "b"), (2, "c")]
    out = jitter_buffer(packets, depth=4)
    assert sorted(out) == ["a", "b", "c", "d"]


def test_jitter_buffer_rejects_zero_depth():
    with pytest.raises(ValueError):
        jitter_buffer([(0, "a")], depth=0)


# ----------------------------------------------------------- pipeline_latency
def test_pipeline_latency_is_the_sum_of_the_stages():
    assert pipeline_latency({"vad": 10, "asr": 150, "llm": 100}) == 260


def test_the_2026_budget_fits_in_400_ms():
    budget = {"mic": 20, "vad": 10, "asr": 150, "llm": 100, "tts": 100, "out": 20}
    assert pipeline_latency(budget) == 400


def test_pipeline_latency_rejects_a_pipeline_it_cannot_measure():
    with pytest.raises(ValueError):
        pipeline_latency({})
    with pytest.raises(ValueError):
        pipeline_latency({"asr": -5})


# ---------------------------------------------------- keeps_up_with_realtime
def test_a_stage_slower_than_the_frame_breaks_the_stream():
    assert keeps_up_with_realtime({"vad": 10, "asr": 150}, frame_ms=20) is False


def test_every_stage_inside_the_frame_keeps_the_stream():
    assert keeps_up_with_realtime({"vad": 10, "asr": 18}, frame_ms=20) is True


def test_throughput_is_bounded_by_the_slowest_stage_not_the_sum():
    """Двадцать стадий по 15 мс — сумма 300 мс, а поток 20 мс держится."""
    stages = {f"s{i}": 15 for i in range(20)}
    assert pipeline_latency(stages) == 300
    assert keeps_up_with_realtime(stages, frame_ms=20) is True


def test_a_stage_exactly_the_size_of_the_frame_still_fits():
    assert keeps_up_with_realtime({"asr": 20}, frame_ms=20) is True


def test_keeps_up_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        keeps_up_with_realtime({}, frame_ms=20)
    with pytest.raises(ValueError):
        keeps_up_with_realtime({"asr": 10}, frame_ms=0)


# ------------------------------------------------------------------ barge_in
def _playing():
    return {"tts_playing": True, "pending_chunks": ["a", "b"]}


def test_barge_in_stops_playback_and_drops_the_queue():
    """Обе вещи сразу: иначе бот замолчит и договорит старую фразу поверх."""
    out = barge_in(_playing(), True, 40)
    assert out["tts_playing"] is False
    assert out["pending_chunks"] == []
    assert out["cancelled"] is True


def test_silence_does_not_cancel_anything():
    out = barge_in(_playing(), False, 40)
    assert out["tts_playing"] is True
    assert out["pending_chunks"] == ["a", "b"]
    assert out["cancelled"] is False


def test_speaking_while_the_bot_is_silent_is_not_an_interruption():
    quiet = {"tts_playing": False, "pending_chunks": []}
    assert barge_in(quiet, True, 40)["cancelled"] is False


def test_barge_in_does_not_mutate_the_state_of_the_audio_thread():
    state = _playing()
    barge_in(state, True, 40)
    assert state == {"tts_playing": True, "pending_chunks": ["a", "b"]}


def test_a_slow_cancellation_is_flagged_late():
    """Отменили, но человек уже решил, что ассистент глухой: порог 100 мс."""
    assert barge_in(_playing(), True, 40)["late"] is False
    assert barge_in(_playing(), True, 250)["late"] is True


def test_barge_in_rejects_a_negative_reaction_time():
    with pytest.raises(ValueError):
        barge_in(_playing(), True, -1)
