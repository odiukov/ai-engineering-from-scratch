"""Тесты к уроку «Стриминговый speech-to-speech: Moshi и Hibiki». Правь exercise.py."""

import pytest

from exercise import (
    DuplexSession,
    build_frame,
    depth_decode,
    frame_ms_from_rate,
    pipeline_latency_ms,
    theoretical_latency_ms,
    tokens_per_second,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------- frame_ms_from_rate
def test_mimi_rate_gives_an_eighty_millisecond_frame():
    assert frame_ms_from_rate(12.5) == APPROX(80.0)


def test_frame_length_is_inverse_to_rate():
    """Удвоил частоту кадров — кадр стал вдвое короче."""
    assert frame_ms_from_rate(25.0) == APPROX(frame_ms_from_rate(12.5) / 2)


def test_frame_ms_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        frame_ms_from_rate(0.0)


# --------------------------------------------------------- tokens_per_second
def test_mimi_single_stream_is_one_hundred_tokens_per_second():
    assert tokens_per_second(12.5, 8) == APPROX(100.0)


def test_full_duplex_costs_exactly_two_streams():
    single = tokens_per_second(12.5, 8)
    assert tokens_per_second(12.5, 8, streams=2) == APPROX(2 * single)


def test_token_rate_grows_with_codebook_count():
    assert tokens_per_second(12.5, 16) > tokens_per_second(12.5, 8)


def test_token_rate_rejects_zero_codebooks():
    with pytest.raises(ValueError):
        tokens_per_second(12.5, 0)


# --------------------------------------------------- theoretical_latency_ms
def test_moshi_theoretical_floor_is_one_sixty():
    """80 мс кадра плюс 80 мс акустической задержки."""
    assert theoretical_latency_ms() == APPROX(160.0)


def test_latency_without_acoustic_delay_is_one_frame():
    assert theoretical_latency_ms(80.0, 0) == APPROX(80.0)


def test_latency_rejects_negative_acoustic_delay():
    with pytest.raises(ValueError):
        theoretical_latency_ms(80.0, -1)


# ------------------------------------------------------ pipeline_latency_ms
def test_pipeline_latency_is_the_sum_of_its_stages():
    assert pipeline_latency_ms([30.0, 120.0, 200.0, 90.0]) == APPROX(440.0)


def test_empty_pipeline_costs_nothing():
    assert pipeline_latency_ms([]) == APPROX(0.0)


def test_adding_a_stage_can_only_make_the_pipeline_slower():
    """Стадии не перекрываются — значит сумма, а не максимум."""
    base = [30.0, 120.0]
    assert pipeline_latency_ms(base + [200.0]) > pipeline_latency_ms(base)


def test_pipeline_rejects_a_negative_stage():
    with pytest.raises(ValueError):
        pipeline_latency_ms([30.0, -10.0])


def test_full_duplex_beats_a_realistic_pipeline():
    """Главный аргумент урока: VAD -> STT -> LLM -> TTS против одного кадра."""
    assert theoretical_latency_ms() < pipeline_latency_ms([30.0, 120.0, 200.0, 90.0])


# ---------------------------------------------------------------- build_frame
def test_frame_puts_the_inner_monologue_first():
    """Текст предсказывается ПЕРЕД аудио — иначе монолог не влияет на звук."""
    assert build_frame("да", [1, 2, 3], n_codebooks=3) == ["да", 1, 2, 3]


def test_frame_length_is_codebooks_plus_one():
    frame = build_frame("", [0] * 8)
    assert len(frame) == 9


def test_frame_rejects_the_wrong_codebook_count():
    with pytest.raises(ValueError):
        build_frame("да", [1, 2], n_codebooks=8)


def test_frame_does_not_alias_the_acoustic_buffer():
    """Кадр уезжает в историю сессии; буфер после этого переиспользуют."""
    buf = [1, 2]
    frame = build_frame("x", buf, n_codebooks=2)
    buf[0] = 99
    assert frame == ["x", 1, 2]


# --------------------------------------------------------------- depth_decode
def test_each_head_sees_all_previously_decoded_tokens():
    assert depth_decode("ctx", [lambda c, p: len(p)] * 4) == [0, 1, 2, 3]


def test_codebooks_really_depend_on_each_other():
    """Параллельное предсказание дало бы [1, 1, 1] — зависимость видна в числах."""
    assert depth_decode("ctx", [lambda c, p: sum(p) + 1] * 3) == [1, 2, 4]


def test_every_head_receives_the_shared_frame_context():
    seen = []
    depth_decode("ctx", [lambda c, p: seen.append(c) or 0] * 3)
    assert seen == ["ctx", "ctx", "ctx"]


def test_a_head_cannot_corrupt_already_decoded_tokens():
    """Голове отдаётся копия: мутация внутри головы не должна ломать кадр."""

    def greedy(c, prev):
        prev.append(999)
        return len(prev)

    assert depth_decode("ctx", [greedy] * 3) == [1, 2, 3]


def test_depth_decode_rejects_a_frame_without_codebooks():
    with pytest.raises(ValueError):
        depth_decode("ctx", [])


# -------------------------------------------------------------- DuplexSession
def test_both_streams_advance_on_the_same_step():
    """Full-duplex: слушаем и говорим в одном кадре, а не по очереди."""
    s = DuplexSession(n_codebooks=2)
    s.step([7, 7], "привет", [1, 2])
    assert len(s.user_stream) == len(s.own_stream) == 1


def test_step_returns_the_frame_it_appended():
    s = DuplexSession(n_codebooks=2)
    frame = s.step([7, 7], "привет", [1, 2])
    assert frame == ["привет", 1, 2]
    assert s.own_stream[-1] == frame


def test_elapsed_time_grows_by_one_frame_per_step():
    s = DuplexSession(frame_ms=80.0, n_codebooks=2)
    for _ in range(3):
        s.step([0, 0], "", [0, 0])
    assert s.elapsed_ms() == APPROX(240.0)


def test_transcript_drops_the_silent_frames():
    """Кадров молчания во внутреннем монологе большинство."""
    s = DuplexSession(n_codebooks=2)
    s.step([0, 0], "привет", [1, 1])
    s.step([0, 0], "", [2, 2])
    s.step([0, 0], "мир", [3, 3])
    assert s.transcript() == "привет мир"


def test_session_rejects_a_user_frame_of_the_wrong_width():
    s = DuplexSession(n_codebooks=8)
    with pytest.raises(ValueError):
        s.step([1, 2], "", [0] * 8)


def test_two_sessions_do_not_share_their_streams():
    """Ловушка изменяемого атрибута класса: у каждой сессии своя история."""
    a, b = DuplexSession(n_codebooks=2), DuplexSession(n_codebooks=2)
    a.step([0, 0], "a", [1, 1])
    assert b.own_stream == []
