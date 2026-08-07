"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 20000  # словарь размером с настоящий: маска обязана быть линейной
_logits = [random.uniform(-5.0, 5.0) for _ in range(_VOCAB)]
_valid = list(range(0, _VOCAB, 97))

_ALPHABET = list("0123456789-")
_uniform = lambda prefix: [0.0] * len(_ALPHABET)


def _phone_fsm():
    """Тот же FSM, что и в упражнении, но собранный без импорта решения."""
    shape = "ddd-ddd-dddd"
    transitions = {}
    for pos, ch in enumerate(shape):
        tokens = "0123456789" if ch == "d" else ch
        transitions[pos] = {t: pos + 1 for t in tokens}
    transitions[len(shape)] = {}
    return {"initial_state": 0, "transitions": transitions, "accepts": {len(shape)}}


_FSM = _phone_fsm()

BENCH = {
    "mask_logits": (_logits, _valid),
    "softmax": (_logits,),
    "pattern_fsm": ("ddd-ddd-dddd",),
    "valid_tokens": (_FSM, 0),
    "transition": (_FSM, 0, "7"),
    "is_accept": (_FSM, 12),
    "generate_constrained": (_FSM, _ALPHABET, _uniform, random.Random(0)),
    "check_field_order": (["reasoning", "evidence", "answer"],),
}
