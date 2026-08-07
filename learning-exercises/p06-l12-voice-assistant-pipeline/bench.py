"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# поток: тишина, длинная реплика, снова тишина
_chunks = [0] * 500 + [_rng.choice([0, 1, 1, 1]) for _ in range(4000)] + [0] * 200
_vad = lambda chunk: chunk == 1

_transcript = "hey assistant " + "set a timer for five minutes and tell me the weather " * 300
_blocklist = [f"phrase number {i}" for i in range(500)] + ["Thanks for watching"]
_long_text = "thanks for watching the demo, what is next " * 400

_tools = {"add": lambda a, b: a + b}
_call = {"name": "add", "args": {"a": 1, "b": 2}}

_stages = {f"stage_{i}": _rng.randint(1, 50) for i in range(2000)}

_DAY = 86400
_log = [{"ts": i * 3600, "audio": [0.0] * 4} for i in range(20000)]

_stt = lambda audio: "set a timer"
_llm = lambda text: "ok, timer set"
_tts = lambda text: [0.0] * len(text)

BENCH = {
    "capture_turn": (_chunks, _vad, 20, 300, 500),
    "wake_word_gate": (_transcript, "hey assistant"),
    "filter_silence_hallucination": (_long_text, True, _blocklist),
    "dispatch_tool": (_call, _tools),
    "run_tool_with_retry": (_call, _tools, 2),
    "first_audio_latency": (_stages, 20, 10),
    "prune_turn_log": (_log, 20000 * 3600, 30),
    "assistant_turn": (_chunks, _vad, _stt, _llm, _tts),
}
