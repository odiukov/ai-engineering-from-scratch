"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_frames = [[_rng.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(300)]
_pool = [[_rng.uniform(-1.0, 1.0) for _ in range(16)] for _ in range(300)]
_emb_a = [_rng.uniform(-1.0, 1.0) for _ in range(16)]
_emb_b = [_rng.uniform(-1.0, 1.0) for _ in range(16)]

_wav = [_rng.uniform(-1.0, 1.0) for _ in range(20000)]
_bits = [_rng.randint(0, 1) for _ in range(32)]
_marked = [
    round((x - (0.005 if _bits[i % 32] else -0.005)) / 0.02) * 0.02
    + (0.005 if _bits[i % 32] else -0.005)
    for i, x in enumerate(_wav)
]

_sent = [_rng.randint(0, 1) for _ in range(20000)]
_received = [_rng.randint(0, 1) for _ in range(20000)]

_record = {"speaker_id": "rohit", "expires_ts": 2000, "signature": "rohit:2000"}
_sign = lambda r: f"{r['speaker_id']}:{r['expires_ts']}"

BENCH = {
    "speaker_embedding": (_frames,),
    "secs": (_emb_a, _emb_b),
    "knn_convert": (_frames[:100], _pool, 4),
    "swap_speaker": (_frames, _emb_a, _emb_b),
    "embed_watermark": (_wav, _bits),
    "detect_watermark": (_marked, 32),
    "bit_accuracy": (_sent, _received),
    "consent_gate": (_record, "rohit", 1000, _sign),
}
