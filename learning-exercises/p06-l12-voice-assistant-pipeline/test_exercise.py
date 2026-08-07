"""Тесты к уроку «Голосовой ассистент целиком». Правь exercise.py."""

import pytest

from exercise import (
    assistant_turn,
    capture_turn,
    dispatch_tool,
    filter_silence_hallucination,
    first_audio_latency,
    prune_turn_log,
    run_tool_with_retry,
    wake_word_gate,
)

DAY = 86400
BLOCKLIST = ["Thanks for watching", "Subtitles by the Amara.org community"]

# кадр 1 — речь, кадр 0 — тишина
loud_vad = lambda chunk: chunk == 1


# ------------------------------------------------------------ capture_turn
def test_capture_turn_returns_nothing_when_nobody_speaks():
    assert capture_turn([0] * 50, loud_vad) == []


def test_pre_roll_saves_the_first_word():
    """VAD срабатывает с опозданием — начало фразы приходит из pre-roll."""
    chunks = [0, 0, 0, 1, 0, 0, 0]
    with_roll = capture_turn(chunks, loud_vad, 20, pre_roll_ms=40, silence_ms=60)
    without = capture_turn(chunks, loud_vad, 20, pre_roll_ms=0, silence_ms=60)
    assert with_roll == [0, 0, 1, 0, 0, 0]
    assert without == [1, 0, 0, 0]


def test_capture_turn_ends_after_enough_silence():
    result = capture_turn([1, 1, 1] + [0] * 10, loud_vad, 20, 0, silence_ms=100)
    assert result == [1, 1, 1, 0, 0, 0, 0, 0]


def test_a_pause_inside_a_phrase_does_not_end_the_turn():
    """Между словами человек молчит — реплику это не завершает."""
    chunks = [1, 1, 0, 0, 1, 1] + [0] * 6
    result = capture_turn(chunks, loud_vad, 20, 0, silence_ms=100)
    assert result.count(1) == 4
    assert len(result) == 11


def test_capture_turn_returns_what_it_has_when_the_stream_ends():
    assert capture_turn([1, 1, 0], loud_vad, 20, 0, silence_ms=500) == [1, 1, 0]


def test_capture_turn_rejects_impossible_timing():
    with pytest.raises(ValueError):
        capture_turn([1], loud_vad, chunk_ms=0)
    with pytest.raises(ValueError):
        capture_turn([1], loud_vad, 20, pre_roll_ms=-20)


# --------------------------------------------------------- wake_word_gate
def test_wake_word_returns_the_command_after_it():
    assert wake_word_gate("Hey assistant set a timer", "hey assistant") == "set a timer"


def test_wake_word_alone_wakes_up_with_an_empty_command():
    assert wake_word_gate("hey assistant", "hey assistant") == ""


def test_without_the_wake_word_the_assistant_stays_asleep():
    assert wake_word_gate("please stop the music", "hey assistant") is None


def test_wake_word_must_start_the_phrase_and_match_whole_words():
    """Иначе пересказ «I told my hey assistant story» разбудит микрофон."""
    assert wake_word_gate("I told my hey assistant story", "hey assistant") is None
    assert wake_word_gate("assistants are useful", "assistant") is None


def test_wake_word_gate_rejects_an_empty_key():
    with pytest.raises(ValueError):
        wake_word_gate("hey assistant", "")


# ---------------------------------------------- filter_silence_hallucination
def test_silence_produces_no_transcript_at_all():
    """VAD-гейт сильнее любого списка фраз: речи не было — текста нет."""
    assert filter_silence_hallucination("set a timer", False, BLOCKLIST) == ""


def test_a_known_hallucination_is_dropped():
    assert filter_silence_hallucination("Thanks for watching", True, BLOCKLIST) == ""


def test_matching_ignores_case_and_the_trailing_period():
    assert filter_silence_hallucination("thanks for watching.", True, BLOCKLIST) == ""


def test_a_real_phrase_containing_a_blocked_one_survives():
    """Ловушка калибровки: сравниваем фразу целиком, а не по вхождению."""
    text = "thanks for watching the demo, what is next"
    assert filter_silence_hallucination(text, True, BLOCKLIST) == text


# ------------------------------------------------------------ dispatch_tool
def _boom(**kwargs):
    raise RuntimeError("weather api timeout")


TOOLS = {"add": lambda a, b: a + b, "weather": _boom}


def test_dispatch_tool_returns_the_result():
    call = {"name": "add", "args": {"a": 1, "b": 2}}
    assert dispatch_tool(call, TOOLS) == {"ok": True, "result": 3}


def test_an_unknown_tool_is_an_answer_not_an_exception():
    """LLM должен ПРОЧИТАТЬ ошибку и попробовать иначе, а не получить стек."""
    out = dispatch_tool({"name": "nope", "args": {}}, TOOLS)
    assert out["ok"] is False
    assert "nope" in out["error"]


def test_a_crashing_tool_does_not_kill_the_turn():
    out = dispatch_tool({"name": "weather", "args": {"city": "Kyiv"}}, TOOLS)
    assert out["ok"] is False
    assert "timeout" in out["error"]


def test_tool_arguments_are_passed_by_name():
    call = {"name": "add", "args": {"b": 10, "a": 1}}
    assert dispatch_tool(call, TOOLS)["result"] == 11


# ------------------------------------------------------ run_tool_with_retry
def _flaky(fails):
    """Инструмент, который падает первые fails раз, потом работает."""
    state = {"calls": 0}

    def tool():
        state["calls"] += 1
        if state["calls"] <= fails:
            raise RuntimeError("transient")
        return "ok"

    return tool, state


def test_a_transient_failure_is_retried_once():
    tool, state = _flaky(1)
    out = run_tool_with_retry({"name": "t", "args": {}}, {"t": tool})
    assert out["ok"] is True
    assert out["attempts"] == 2
    assert state["calls"] == 2


def test_a_working_tool_is_never_called_twice():
    """Повтор удавшегося вызова завёл бы второй таймер."""
    tool, state = _flaky(0)
    out = run_tool_with_retry({"name": "t", "args": {}}, {"t": tool})
    assert out["attempts"] == 1
    assert state["calls"] == 1


def test_a_dead_tool_degrades_gracefully():
    tool, state = _flaky(99)
    out = run_tool_with_retry({"name": "t", "args": {}}, {"t": tool}, max_attempts=3)
    assert out["ok"] is False
    assert out["degraded"] is True
    assert state["calls"] == 3


def test_run_tool_with_retry_rejects_zero_attempts():
    with pytest.raises(ValueError):
        run_tool_with_retry({"name": "t", "args": {}}, {}, max_attempts=0)


# ------------------------------------------------------ first_audio_latency
def test_first_audio_latency_adds_the_llm_wait_to_the_stages():
    assert first_audio_latency({"stt": 150, "tts": 100}, 20, 10) == 450


def test_waiting_for_more_tokens_costs_latency():
    """Компромисс: больше токенов до старта TTS — ровнее речь, дольше пауза."""
    stages = {"vad": 50, "stt": 150, "tts": 100}
    assert first_audio_latency(stages, 40, 10) > first_audio_latency(stages, 10, 10)


def test_a_realistic_budget_fits_the_800_ms_target():
    stages = {"endpoint": 100, "stt": 150, "tts": 100}
    assert first_audio_latency(stages, 20, 10) <= 800


def test_first_audio_latency_rejects_impossible_budgets():
    with pytest.raises(ValueError):
        first_audio_latency({}, 20, 10)
    with pytest.raises(ValueError):
        first_audio_latency({"stt": -1}, 20, 10)


# ---------------------------------------------------------- prune_turn_log
def test_expired_turns_are_dropped():
    log = [{"ts": 0, "audio": [1]}, {"ts": 40 * DAY, "audio": [2]}]
    assert prune_turn_log(log, now_ts=41 * DAY) == [{"ts": 40 * DAY, "audio": [2]}]


def test_a_turn_exactly_at_the_limit_is_still_kept():
    log = [{"ts": 0, "audio": [1]}]
    assert prune_turn_log(log, now_ts=30 * DAY) == log


def test_prune_turn_log_does_not_touch_the_shared_log():
    """Журнал общий: чистить его на месте — уронить чужой поток."""
    log = [{"ts": 0}, {"ts": 40 * DAY}]
    prune_turn_log(log, now_ts=100 * DAY)
    assert len(log) == 2


def test_prune_turn_log_rejects_a_negative_retention():
    with pytest.raises(ValueError):
        prune_turn_log([], now_ts=0, retention_days=-1)


# --------------------------------------------------------- assistant_turn
def _spies():
    calls = []
    stt = lambda audio: (calls.append("stt"), "set a timer")[1]
    llm = lambda text: (calls.append("llm"), f"ok, {text}")[1]
    tts = lambda text: (calls.append("tts"), [0.1] * len(text))[1]
    return stt, llm, tts, calls


def test_a_full_turn_walks_all_the_stages():
    stt, llm, tts, calls = _spies()
    out = assistant_turn([1, 1] + [0] * 30, loud_vad, stt, llm, tts)
    assert out["spoke"] is True
    assert out["transcript"] == "set a timer"
    assert out["reply"] == "ok, set a timer"
    assert calls == ["stt", "llm", "tts"]


def test_silence_never_reaches_the_llm():
    """Галлюцинация с тишины дошла бы до LLM, и ассистент заговорил бы сам с собой."""
    stt, llm, tts, calls = _spies()
    out = assistant_turn([0] * 50, loud_vad, stt, llm, tts)
    assert out == {"transcript": "", "reply": "", "audio": [], "spoke": False}
    assert calls == []


def test_the_turn_speaks_the_reply_not_the_transcript():
    stt, llm, tts, _ = _spies()
    out = assistant_turn([1, 1] + [0] * 30, loud_vad, stt, llm, tts)
    assert len(out["audio"]) == len(out["reply"])


def test_the_turn_hands_the_pre_roll_to_the_recognizer():
    """STT должен увидеть кадры до срабатывания VAD, иначе первое слово потеряно."""
    seen = {}
    stt = lambda audio: seen.setdefault("frames", list(audio)) and "text"
    out = assistant_turn(
        [0, 0, 1] + [0] * 30,
        loud_vad,
        stt,
        lambda t: "reply",
        lambda t: [0.0],
        20,
        pre_roll_ms=40,
        silence_ms=60,
    )
    assert out["spoke"] is True
    assert seen["frames"][:2] == [0, 0]
