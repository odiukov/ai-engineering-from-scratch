"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 512
_HIDDEN = 128
_DEPTH = 4


def _matrix(rows, cols):
    return [[random.gauss(0.0, 0.05) for _ in range(cols)] for _ in range(rows)]


_embedding = _matrix(_VOCAB, _HIDDEN)
_W_out = _matrix(_VOCAB, _HIDDEN)
_projections = [_matrix(_HIDDEN, 2 * _HIDDEN) for _ in range(_DEPTH - 1)]
_h0 = [random.gauss(0.0, 1.0) for _ in range(_HIDDEN)]
_targets = [random.randrange(_VOCAB) for _ in range(_DEPTH)]

_logits = [random.gauss(0.0, 1.0) for _ in range(_VOCAB)]

BENCH = {
    "matvec": (_W_out, _h0),
    "softmax": (_logits,),
    "cross_entropy": (_logits, 17),
    "rms_norm": (_h0,),
    "depth_hidden": (_h0, _embedding[3], _projections[0]),
    "mtp_depth_losses": (_h0, _targets, _projections, _embedding, _W_out),
    "joint_loss": (2.0, [1.0, 2.0, 3.0, 4.0], 0.3),
    "mtp_extra_params": (7168, 1),
}
