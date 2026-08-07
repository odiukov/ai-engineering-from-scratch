"""Тесты к уроку «Голосовые агенты: Pipecat и LiveKit». Правь exercise.py."""

import random

import pytest

from exercise import (
    DEFAULT_REPLY,
    REPEAT_PROMPT,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    UPSTREAM_KINDS,
    frame_direction,
    gate_transcript,
    is_end_of_turn,
    latency_budget,
    play_tts,
    run_turn_script,
    turn_transition,
)

REPLIES = {"hello": "hi there friend", "refund please": "what order number"}


# ------------------------------------------------------------ frame_direction
def test_frame_direction_sends_cancel_upstream():
    assert frame_direction("cancel") == "upstream"


def test_frame_direction_sends_payload_downstream():
    assert frame_direction("transcript") == "downstream"


def test_frame_direction_treats_every_control_kind_as_upstream():
    assert all(frame_direction(kind) == "upstream" for kind in UPSTREAM_KINDS)


# ------------------------------------------------------------ latency_budget
def test_latency_budget_sums_the_chain():
    stages = [("vad", 40), ("stt", 200), ("llm", 250), ("tts", 150)]
    assert latency_budget(stages)["total_ms"] == 640


def test_latency_budget_names_the_slowest_stage():
    stages = [("vad", 40), ("stt", 200), ("llm", 250), ("tts", 150)]
    assert latency_budget(stages)["worst_stage"] == "llm"


def test_latency_budget_calls_a_fast_chain_premium():
    stages = [("vad", 30), ("stt", 120), ("llm", 200), ("tts", 120), ("rtt", 40)]
    assert latency_budget(stages)["tier"] == "premium"


def test_latency_budget_calls_a_slow_chain_broken():
    """Всё, что дольше 1500 мс, звучит для человека как сломанный бот."""
    assert latency_budget([("llm", 1800)])["tier"] == "broken"


def test_latency_budget_refuses_an_empty_pipeline():
    with pytest.raises(ValueError):
        latency_budget([])


def test_latency_budget_refuses_a_negative_stage():
    with pytest.raises(ValueError):
        latency_budget([("stt", -10)])


# ----------------------------------------------------------- gate_transcript
def test_gate_transcript_lets_a_confident_transcript_through():
    assert gate_transcript("  refund please  ", 0.9) == {
        "accepted": True, "text": "refund please"}


def test_gate_transcript_asks_to_repeat_when_confidence_is_low():
    assert gate_transcript("refund please", 0.2) == {
        "accepted": False, "text": REPEAT_PROMPT}


def test_gate_transcript_accepts_exactly_at_the_threshold():
    assert gate_transcript("hello", 0.6, 0.6)["accepted"] is True


def test_gate_transcript_never_accepts_silence_however_confident():
    """STT честно распознала тишину — уверенность 0.99 этого не меняет."""
    assert gate_transcript("   ", 0.99)["accepted"] is False


# ------------------------------------------------------------ is_end_of_turn
def test_is_end_of_turn_on_a_finished_question():
    assert is_end_of_turn("what is my balance?", 100) is True


def test_is_end_of_turn_is_false_while_the_person_is_still_talking():
    assert is_end_of_turn("I want to", 100) is False


def test_is_end_of_turn_becomes_true_after_a_long_silence():
    assert is_end_of_turn("I want to", 900) is True


def test_is_end_of_turn_ignores_silence_after_a_continuation_word():
    """Человек тянет "and" — он подбирает слово, а не закончил ход."""
    assert is_end_of_turn("I want a refund and", 5000) is False


def test_is_end_of_turn_of_an_empty_transcript_is_false():
    assert is_end_of_turn("", 9000) is False


# ----------------------------------------------------------- turn_transition
def test_turn_transition_walks_a_full_turn_back_to_idle():
    state = STATE_IDLE
    for event, expected in (("speech_start", STATE_LISTENING),
                            ("speech_end", STATE_THINKING),
                            ("llm_reply", STATE_SPEAKING),
                            ("tts_end", STATE_IDLE)):
        state, actions = turn_transition(state, event)
        assert (state, actions) == (expected, [])


def test_turn_transition_barge_in_cancels_tts_and_llm():
    """Перебивание во время речи бота обязано поднять UPSTREAM-отмену."""
    state, actions = turn_transition(STATE_SPEAKING, "speech_start")
    assert state == STATE_LISTENING
    assert set(actions) == {"cancel_tts", "cancel_llm"}


def test_turn_transition_barge_in_while_thinking_cancels_only_the_llm():
    state, actions = turn_transition(STATE_THINKING, "speech_start")
    assert (state, actions) == (STATE_LISTENING, ["cancel_llm"])


def test_turn_transition_ignores_an_event_that_makes_no_sense_here():
    """Лишний "tts_end" в тишине не должен ронять конвейер."""
    assert turn_transition(STATE_IDLE, "tts_end") == (STATE_IDLE, [])


def test_turn_transition_refuses_an_unknown_event():
    with pytest.raises(ValueError):
        turn_transition(STATE_IDLE, "hang_up")


# ------------------------------------------------------------------ play_tts
def test_play_tts_says_everything_when_nobody_interrupts():
    assert play_tts("hi there friend") == (["hi", "there", "friend"], [])


def test_play_tts_splits_the_utterance_at_the_barge_in():
    assert play_tts("hi there friend", 1) == (["hi"], ["there", "friend"])


def test_play_tts_handles_a_cancel_that_arrives_too_late():
    assert play_tts("hi there friend", 9) == (["hi", "there", "friend"], [])


def test_play_tts_never_loses_a_word():
    """Произнесённое плюс недоговорённое всегда равно исходной реплике."""
    rng = random.Random(23)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for _ in range(20):
        text = " ".join(rng.choices(words, k=rng.randint(1, 6)))
        said, left = play_tts(text, rng.randint(0, 8))
        assert said + left == text.split()


def test_play_tts_refuses_a_negative_cancel_point():
    with pytest.raises(ValueError):
        play_tts("hi there", -1)


# ------------------------------------------------------------ run_turn_script
def test_run_turn_script_speaks_a_whole_reply_and_returns_to_idle():
    script = [("speech_start", None), ("speech_end", "hello"),
              ("llm_reply", None), ("tts_end", None)]
    report = run_turn_script(script, REPLIES)
    assert report["spoken"] == ["hi there friend"]
    assert report["state"] == STATE_IDLE


def test_run_turn_script_cuts_the_reply_at_the_barge_in():
    script = [("speech_start", None), ("speech_end", "hello"),
              ("llm_reply", None), ("tts_progress", 2), ("speech_start", None)]
    report = run_turn_script(script, REPLIES)
    assert report["interrupted"] == [("hi there friend", ["hi", "there"])]
    assert report["spoken"] == []


def test_run_turn_script_keeps_what_the_user_said_over_the_bot():
    """Barge-in гасит TTS, но реплику человека терять нельзя — иначе переспрос."""
    script = [("speech_start", None), ("speech_end", "hello"),
              ("llm_reply", None), ("tts_progress", 1),
              ("speech_start", None), ("speech_end", "refund please")]
    report = run_turn_script(script, REPLIES)
    assert report["heard"] == ["hello", "refund please"]
    assert "cancel_tts" in report["actions"]


def test_run_turn_script_falls_back_when_the_phrase_is_unknown():
    script = [("speech_start", None), ("speech_end", "мяу"),
              ("llm_reply", None), ("tts_end", None)]
    assert run_turn_script(script, REPLIES)["spoken"] == [DEFAULT_REPLY]


def test_run_turn_script_treats_tts_progress_as_telemetry_not_a_transition():
    script = [("speech_start", None), ("tts_progress", 3)]
    report = run_turn_script(script, REPLIES)
    assert (report["state"], report["actions"]) == (STATE_LISTENING, [])
