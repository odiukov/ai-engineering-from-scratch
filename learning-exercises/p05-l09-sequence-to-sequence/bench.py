"""Входные данные для замера скорости."""

import math
import random

random.seed(0)

_HIDDEN = 32
_EMBED = 32
_VOCAB = 64

_emb = [[random.uniform(-1, 1) for _ in range(_EMBED)] for _ in range(_VOCAB)]
_W_x = [[random.uniform(-0.3, 0.3) for _ in range(_EMBED)] for _ in range(_HIDDEN)]
_W_h = [[random.uniform(-0.3, 0.3) for _ in range(_HIDDEN)] for _ in range(_HIDDEN)]
_b = [0.0] * _HIDDEN
_W_out = [[random.uniform(-0.3, 0.3) for _ in range(_HIDDEN)] for _ in range(_VOCAB)]
_b_out = [0.0] * _VOCAB

_tokens = [random.randrange(_VOCAB) for _ in range(120)]
_x = _emb[0]
_h = [0.0] * _HIDDEN
_logits = [[random.uniform(-4, 4) for _ in range(_VOCAB)] for _ in range(200)]
_targets = [random.randrange(_VOCAB) for _ in range(200)]
_rng = random.Random(0)

_BOS, _EOS = 0, 1


def _step(token_id, hidden):
    """Обученного декодера здесь нет — берём тот же RNN-шаг, что в уроке."""
    state = []
    for i in range(_HIDDEN):
        total = _b[i]
        total += sum(w * v for w, v in zip(_W_x[i], _emb[token_id % _VOCAB]))
        total += sum(w * v for w, v in zip(_W_h[i], hidden))
        state.append(math.tanh(total))
    logits = [
        _b_out[v] + sum(w * s for w, s in zip(_W_out[v], state)) for v in range(_VOCAB)
    ]
    return logits, state


BENCH = {
    "softmax": (_logits[0],),
    "rnn_step": (_x, _h, _W_x, _W_h, _b),
    "encode": (_tokens, _emb, _W_x, _W_h, _b),
    "decode_step": (3, _h, _emb, _W_x, _W_h, _b, _W_out, _b_out),
    "teacher_forcing_input": (1, 2, 0.5, _rng),
    "sequence_cross_entropy": (_logits, _targets),
    "greedy_decode": (_step, _BOS, _EOS, _h, 25),
    "beam_search": (_step, _BOS, _EOS, _h, 3, 12),
}
