"""Тесты к уроку «Детекция речи и передача хода». Правь exercise.py."""

import random

import pytest

from exercise import (
    TurnDetector,
    dbfs,
    energy_vad,
    flush_latency_ms,
    hysteresis_flags,
    pre_roll,
    rms,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def _loud_noise(n=320, seed=0):
    """Громкий шум: энергии много, речи нет. Нужен, чтобы поймать смысл tier 1."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.2) for _ in range(n)]


# --------------------------------------------------------------------- rms
def test_rms_of_symmetric_square_wave_equals_amplitude():
    assert rms([1.0, -1.0, 1.0, -1.0]) == APPROX(1.0)


def test_rms_of_silence_is_zero():
    assert rms([0.0, 0.0]) == APPROX(0.0)


def test_rms_does_not_cancel_out_on_symmetric_signal():
    """Среднее у такого куска ноль, RMS — нет. Ради этого он и нужен."""
    assert rms([0.5, -0.5, 0.5, -0.5]) == APPROX(0.5)


def test_rms_scales_with_amplitude():
    quiet = [0.1, -0.1, 0.1]
    loud = [x * 10 for x in quiet]
    assert rms(loud) == APPROX(10 * rms(quiet))


def test_rms_of_empty_chunk_raises_value_error():
    with pytest.raises(ValueError):
        rms([])


# -------------------------------------------------------------------- dbfs
def test_dbfs_of_full_scale_signal_is_zero():
    assert dbfs([1.0, -1.0]) == APPROX(0.0)


def test_dbfs_drops_twenty_per_tenfold_attenuation():
    assert dbfs([0.1, -0.1]) == pytest.approx(-20.0, abs=1e-9)
    assert dbfs([0.01, -0.01]) == pytest.approx(-40.0, abs=1e-9)


def test_dbfs_of_digital_silence_is_floored_not_infinite():
    """Ловушка: log10(0) взорвётся. Пол на 1e-10 даёт ровно -200 dBFS."""
    assert dbfs([0.0, 0.0, 0.0]) == pytest.approx(-200.0, abs=1e-9)


# --------------------------------------------------------------- energy_vad
def test_energy_vad_fires_on_loud_chunk():
    assert energy_vad([0.3, -0.3, 0.3, -0.3]) is True


def test_energy_vad_stays_silent_on_quiet_chunk():
    assert energy_vad([0.0005, -0.0005]) is False


def test_energy_vad_follows_its_threshold():
    """Один и тот же кусок по разные стороны порога даёт разные ответы."""
    chunk = [0.02, -0.02, 0.02, -0.02]  # ровно -34 dBFS с небольшим хвостом
    assert energy_vad(chunk, threshold_dbfs=-40.0) is True
    assert energy_vad(chunk, threshold_dbfs=-20.0) is False


def test_energy_vad_cannot_tell_speech_from_noise():
    """Главная слабость первого яруса: громкий шум для него — речь."""
    assert energy_vad(_loud_noise()) is True


# ---------------------------------------------------------- hysteresis_flags
def test_hysteresis_starts_in_the_off_state():
    assert hysteresis_flags([0.4, 0.45]) == [False, False]


def test_hysteresis_turns_on_at_the_upper_threshold():
    assert hysteresis_flags([0.4, 0.6]) == [False, True]


def test_hysteresis_holds_on_between_the_thresholds():
    """0.4 ниже верхнего порога, но выше нижнего — детектор не отпускает."""
    assert hysteresis_flags([0.6, 0.4, 0.2]) == [True, True, False]


def test_hysteresis_does_not_chatter_around_the_single_threshold():
    """Вероятность Silero пляшет вокруг 0.5; переключение должно быть одно."""
    probs = [0.6, 0.49, 0.51, 0.48, 0.52, 0.47]
    flags = hysteresis_flags(probs)
    switches = sum(1 for a, b in zip(flags, flags[1:]) if a != b)
    assert flags[0] is True
    assert switches == 0


def test_hysteresis_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        hysteresis_flags([0.5], on_threshold=0.3, off_threshold=0.7)


# ---------------------------------------------------------------- pre_roll
def test_pre_roll_returns_the_tail_of_the_buffer():
    assert pre_roll(["a", "b", "c", "d"], 40) == ["c", "d"]


def test_pre_roll_rounds_the_chunk_count_up():
    """50 мс при кусках по 20 мс — это три куска, а не два: лучше лишнее."""
    assert pre_roll(["a", "b", "c", "d"], 50) == ["b", "c", "d"]


def test_pre_roll_of_zero_is_empty_not_the_whole_buffer():
    """Ловушка среза: chunks[-0:] вернул бы весь список."""
    assert pre_roll(["a", "b"], 0) == []


def test_pre_roll_longer_than_the_buffer_returns_everything():
    assert pre_roll(["a", "b"], 200) == ["a", "b"]


def test_pre_roll_rejects_negative_duration():
    with pytest.raises(ValueError):
        pre_roll(["a"], -20)


# -------------------------------------------------------- flush_latency_ms
def test_flush_latency_divides_lookahead_by_speedup():
    assert flush_latency_ms(500.0, 4.0) == APPROX(125.0)
    assert flush_latency_ms(2500.0, 4.0) == APPROX(625.0)


def test_flush_beats_waiting_in_realtime():
    """Смысл трюка Kyutai: дожать буфер быстрее, чем он звучал бы вживую."""
    assert flush_latency_ms(500.0, 4.0) < 500.0


def test_flush_latency_rejects_nonpositive_speedup():
    with pytest.raises(ValueError):
        flush_latency_ms(500.0, 0.0)


# ------------------------------------------------------------ TurnDetector
def test_turn_start_waits_for_the_minimum_speech_duration():
    td = TurnDetector()
    assert [td.update(True) for _ in range(12)] == [None] * 12  # 240 мс — мало
    assert td.update(True) == "START"  # 260 мс >= 250


def test_short_cough_does_not_start_a_turn():
    td = TurnDetector()
    events = [td.update(True) for _ in range(5)] + [td.update(False) for _ in range(5)]
    assert set(events) == {None}


def test_turn_end_waits_for_the_full_hangover():
    td = TurnDetector()
    for _ in range(13):
        td.update(True)
    assert [td.update(False) for _ in range(24)] == [None] * 24  # 480 мс
    assert td.update(False) == "END"  # 500 мс


def test_speech_resets_the_silence_counter():
    """Пауза внутри фразы не должна складываться с паузой после неё."""
    td = TurnDetector()
    for _ in range(13):
        td.update(True)
    for _ in range(24):
        td.update(False)
    td.update(True)
    assert [td.update(False) for _ in range(24)] == [None] * 24


def test_scattered_coughs_do_not_accumulate_into_a_start():
    """Ловушка: без обнуления счётчика в idle тринадцать кашлей дадут START."""
    td = TurnDetector()
    events = []
    for _ in range(30):
        events.append(td.update(True))
        events.append(td.update(False))
    assert set(events) == {None}


def test_detector_returns_to_idle_and_can_start_a_second_turn():
    td = TurnDetector()
    for _ in range(13):
        td.update(True)
    for _ in range(25):
        td.update(False)
    assert [td.update(True) for _ in range(12)] == [None] * 12
    assert td.update(True) == "START"


def test_turn_detector_rejects_nonpositive_chunk_ms():
    with pytest.raises(ValueError):
        TurnDetector(chunk_ms=0)
