"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_frame = [_rng.uniform(-1.0, 1.0) for _ in range(200000)]

# поток пакетов с перестановками: каждый третий приходит на два позже
_seqs = list(range(20000))
for i in range(0, len(_seqs) - 2, 3):
    _seqs[i], _seqs[i + 2] = _seqs[i + 2], _seqs[i]
_packets = [(s, s) for s in _seqs]

_stages = {f"stage_{i}": _rng.randint(1, 50) for i in range(2000)}

_state = {"tts_playing": True, "pending_chunks": [f"chunk_{i}" for i in range(20000)]}

BENCH = {
    "frame_length": (16000, 20),
    "buffer_latency_ms": (32000, 16000),
    "energy_vad": (_frame, 0.01),
    "jitter_buffer": (_packets, 8),
    "pipeline_latency": (_stages,),
    "keeps_up_with_realtime": (_stages, 20),
    "barge_in": (_state, True, 40),
}
