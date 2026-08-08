"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_signal = [_rng.uniform(-1, 1) for _ in range(200000)]
_mel_power = [[10 ** _rng.uniform(-10, 2) for _ in range(80)] for _ in range(3000)]
_prompt = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]

_words = [f"w{i}" for i in range(4000)]
_chunks = [" ".join(_words[i : i + 60]) for i in range(0, 3900, 50)]

_shapes = [(1280, 1280)] * 64

BENCH = {
    "pad_or_trim": (_signal, 480000),
    "frame_budget": (600.0,),
    "normalize_log_mel": (_mel_power,),
    "build_prompt": ("en", "transcribe", False),
    "parse_prompt": (_prompt,),
    "chunk_schedule": (36000.0,),
    "merge_chunk_transcripts": (_chunks,),
    "lora_parameter_count": (_shapes, 16),
}
