"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_notes = [_rng.randint(21, 108) for _ in range(20000)]
_logits = [_rng.uniform(-5.0, 5.0) for _ in range(2048)]
_tokens = [_rng.randint(0, 7) for _ in range(20000)]
_clip_a = [_rng.uniform(-1.0, 1.0) for _ in range(20000)]
_clip_b = [_rng.uniform(-1.0, 1.0) for _ in range(20000)]
_real = [[_rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(300)]
_fake = [[_rng.gauss(0.3, 1.2) for _ in range(64)] for _ in range(300)]

_prompt = "warm lo-fi hip hop with rhodes keys and brushed drums at 88 bpm " * 40
_blocked = ["Taylor Swift", "The Beatles", "Queen", "Drake", "Beyonce"]

_model = lambda ctx: [((ctx[-1] * 7 + i * 13) % 11) / 3.0 for i in range(32)]

BENCH = {
    "midi_to_hz": (69,),
    "chroma_vector": (_notes,),
    "sample_token": (_logits, random.Random(1), 1.0, 64),
    "generate_tokens": ([0], _model, 200, random.Random(2), 1.0, 8),
    "repetition_rate": (_tokens, 4),
    "crossfade": (_clip_a, _clip_b, 4000),
    "fad": (_real, _fake),
    "is_prompt_blocked": (_prompt, _blocked),
}
