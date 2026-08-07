"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_words = ["hello", "world", "please", "water", "the", "plants", "at", "six", "pm"]
_text = " ".join(random.choice(_words) + random.choice([".", ",", "", "!"]) for _ in range(4000))
_abbrev = {"pm": "p m", "dr": "doctor"}

_tokens = [random.choice(_words) for _ in range(4000)]
_lexicon = {"hello": ["HH", "AH", "L", "OW"], "world": ["W", "ER", "L", "D"]}

_phonemes = [random.choice(["K", "AE", "T", "S", "AH"]) for _ in range(6000)]
_ms = {"AE": 116.0, "K": 40.0, "T": 45.0}

_vectors = [[random.random() for _ in range(8)] for _ in range(600)]
_durations = [random.randint(1, 12) for _ in range(600)]

_frames = [(random.random(), 100.0 + 400.0 * random.random()) for _ in range(400)]
_wav = [random.uniform(-1.5, 1.5) for _ in range(60000)]

_ref = "".join(random.choice("abcdefg ") for _ in range(300))
_hyp = "".join(random.choice("abcdefg ") for _ in range(300))

BENCH = {
    "normalize_text": (_text, _abbrev),
    "grapheme_to_phoneme": (_tokens, _lexicon, lambda w: list(w.upper())),
    "predict_durations": (_phonemes, _ms),
    "length_regulate": (_vectors, _durations),
    "vocode": (_frames, 256, 24000),
    "clip_waveform": (_wav,),
    "resample_linear": (_wav, 24000, 16000),
    "character_error_rate": (_ref, _hyp),
}
