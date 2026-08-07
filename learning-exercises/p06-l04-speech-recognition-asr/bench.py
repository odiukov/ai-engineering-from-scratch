"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_V = 29                          # 28 символов + blank
_ids = [_rng.randrange(_V) for _ in range(4000)]


def _frame():
    raw = [_rng.random() for _ in range(_V)]
    s = sum(raw)
    return [v / s for v in raw]


_frames = [_frame() for _ in range(120)]
_short = _frames[:25]

_ref = " ".join(f"w{_rng.randrange(60)}" for _ in range(300))
_hyp = " ".join(f"w{_rng.randrange(60)}" for _ in range(300))
_ref_tokens = _ref.split()
_hyp_tokens = _hyp.split()

BENCH = {
    "collapse_ctc": (_ids,),
    "ctc_greedy_decode": (_frames,),
    "ctc_beam_decode": (_short, 8),
    "count_ctc_alignments": ([1, 2, 3, 4, 5], 60),
    "normalize_text": ("Turn ON the kitchen LIGHTS, please!! " * 200,),
    "edit_distance": (_ref_tokens, _hyp_tokens),
    "edit_counts": (_ref_tokens, _hyp_tokens),
    "wer": (_ref, _hyp),
}
