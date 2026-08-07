"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_stages = [(f"stage_{i:04d}", random.randint(10, 300)) for i in range(20000)]

_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
_utterance = " ".join(random.choices(_words, k=20000))

_replies = {f"phrase {i}": f"reply number {i} for the caller" for i in range(500)}

_script = []
for i in range(3000):
    phrase = f"phrase {i % 500}"
    _script.append(("speech_start", None))
    _script.append(("speech_end", phrase))
    _script.append(("llm_reply", None))
    if i % 3 == 0:
        _script.append(("tts_progress", 2))
        _script.append(("speech_start", None))
        _script.append(("speech_end", phrase))
        _script.append(("llm_reply", None))
    _script.append(("tts_end", None))

BENCH = {
    "frame_direction": ("transcript",),
    "latency_budget": (_stages,),
    "gate_transcript": ("  refund please  ", 0.91),
    "is_end_of_turn": (_utterance, 800),
    "turn_transition": ("speaking", "speech_start"),
    "play_tts": (_utterance, 9000),
    "run_turn_script": (_script, _replies),
}
