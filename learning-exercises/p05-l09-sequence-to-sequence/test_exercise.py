"""Тесты к уроку «Sequence-to-sequence модели». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    beam_search,
    decode_step,
    encode,
    greedy_decode,
    rnn_step,
    sequence_cross_entropy,
    softmax,
    teacher_forcing_input,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


# ------------------------------------------------------------ игрушечная RNN
EMB = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
W_X = [[0.5, 0.0], [0.0, 0.5]]
W_H = [[0.1, 0.0], [0.0, 0.1]]
B = [0.0, 0.0]
W_OUT = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
B_OUT = [0.0, 0.0, 0.0]


# --------------------------------------------------- игрушечный шаг декодера
EOS, A, B_TOK, C = 0, 1, 2, 3
BOS = 9
DEFAULT_LOGITS = [-5.0, 0.0, -1.0, -1.0]


def make_step(table):
    """Декодер-таблица: состояние — это префикс уже выданных токенов."""

    def step(token_id, hidden):
        prefix = hidden + (token_id,)
        return table.get(prefix, DEFAULT_LOGITS), prefix

    return step


# ловушка для жадности: локально лучший A ведёт в тупик, B — в хороший путь
TRAP = {
    (BOS,): [-5.0, 1.0, 0.9, -5.0],
    (BOS, A): [-5.0, -5.0, 0.0, 0.0],
    (BOS, B_TOK): [-5.0, -5.0, -5.0, 4.0],
    (BOS, A, B_TOK): [0.0, -5.0, -5.0, -5.0],
    (BOS, B_TOK, C): [0.0, -5.0, -5.0, -5.0],
}

# здесь жадный путь и есть оптимальный
DECISIVE = {
    (BOS,): [-5.0, 3.0, -5.0, -5.0],
    (BOS, A): [-5.0, -5.0, 3.0, -5.0],
    (BOS, A, B_TOK): [3.0, -5.0, -5.0, -5.0],
}


# ----------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([2.0, -1.0, 0.5, 3.0])) == pytest.approx(1.0)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([7.0, 7.0, 7.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_invariant_to_a_constant_shift():
    """Прибавили ко всем логитам одно и то же — распределение не изменилось."""
    base = [1.0, 2.0, 3.0]
    assert softmax([v + 100 for v in base]) == pytest.approx(softmax(base))


def test_softmax_survives_huge_logits():
    """Ловушка: math.exp(1000) переполняется, если не вычесть максимум."""
    probs = softmax([1000.0, 999.0, 998.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_softmax_rejects_an_empty_vocabulary():
    with pytest.raises(ValueError):
        softmax([])


# ---------------------------------------------------------------- rnn_step
def test_rnn_step_applies_tanh_to_the_input_contribution():
    assert rnn_step([1.0], [0.0], [[2.0]], [[0.0]], [0.0]) == APPROX([math.tanh(2.0)])


def test_rnn_step_output_stays_inside_the_tanh_range():
    huge = rnn_step([10.0, -10.0], [1.0, 1.0], W_X, W_H, [2.0, -2.0])
    assert all(-1.0 < v < 1.0 for v in huge)
    assert all(abs(v) > 0.99 for v in huge)  # состояние прижато к границе, но не за ней


def test_rnn_step_rejects_matrices_that_disagree_on_hidden_size():
    with pytest.raises(ValueError):
        rnn_step([1.0, 0.0], [0.0, 0.0], [[0.5, 0.0]], W_H, B)


# ------------------------------------------------------------------ encode
def test_encode_returns_one_state_per_token():
    states, _ = encode([1, 2, 3, 1], EMB, W_X, W_H, B)
    assert len(states) == 4


def test_encode_context_is_the_last_state():
    states, context = encode([1, 2, 3], EMB, W_X, W_H, B)
    assert context == APPROX(states[-1])


def test_encode_of_an_empty_source_gives_a_zero_context():
    states, context = encode([], EMB, W_X, W_H, B)
    assert states == []
    assert context == APPROX([0.0, 0.0])


def test_encode_depends_on_word_order():
    """Перестановка токенов меняет контекст — иначе это был бы bag of words."""
    _, first = encode([1, 2], EMB, W_X, W_H, B)
    _, second = encode([2, 1], EMB, W_X, W_H, B)
    assert flat([first]) != pytest.approx(flat([second]))


def test_encode_context_size_does_not_grow_with_the_source():
    """Тот самый bottleneck: длина контекста фиксирована при любом входе."""
    _, short = encode([1], EMB, W_X, W_H, B)
    _, long = encode([1, 2, 3, 1, 2, 3, 1], EMB, W_X, W_H, B)
    assert len(short) == len(long) == len(B)


# ------------------------------------------------------------- decode_step
def test_decode_step_returns_one_logit_per_vocabulary_word():
    logits, _ = decode_step(1, [0.0, 0.0], EMB, W_X, W_H, B, W_OUT, B_OUT)
    assert len(logits) == len(B_OUT)


def test_decode_step_advances_the_hidden_state():
    _, hidden = decode_step(1, [0.0, 0.0], EMB, W_X, W_H, B, W_OUT, B_OUT)
    assert hidden == APPROX(rnn_step(EMB[1], [0.0, 0.0], W_X, W_H, B))


def test_decode_step_logits_change_with_the_incoming_state():
    """Ловушка: если состояние не передавать дальше, выход перестаёт зависеть
    от истории и декодер печатает один и тот же токен."""
    a, _ = decode_step(1, [0.0, 0.0], EMB, W_X, W_H, B, W_OUT, B_OUT)
    b, _ = decode_step(1, [0.9, -0.9], EMB, W_X, W_H, B, W_OUT, B_OUT)
    assert flat([a]) != pytest.approx(flat([b]))


# ------------------------------------------------- teacher_forcing_input
def test_teacher_forcing_ratio_one_always_takes_the_truth():
    rng = random.Random(0)
    assert all(teacher_forcing_input(7, 3, 1.0, rng) == 7 for _ in range(200))


def test_teacher_forcing_ratio_zero_always_takes_the_prediction():
    rng = random.Random(0)
    assert all(teacher_forcing_input(7, 3, 0.0, rng) == 3 for _ in range(200))


def test_teacher_forcing_half_ratio_mixes_roughly_evenly():
    rng = random.Random(1234)
    picks = [teacher_forcing_input(1, 0, 0.5, rng) for _ in range(4000)]
    assert sum(picks) / len(picks) == pytest.approx(0.5, abs=0.03)


def test_teacher_forcing_is_reproducible_for_the_same_seed():
    """Ловушка: глобальный random.random() делает прогон невоспроизводимым."""
    rng_x, rng_y = random.Random(42), random.Random(42)
    seq_x = [teacher_forcing_input(1, 0, 0.5, rng_x) for _ in range(50)]
    seq_y = [teacher_forcing_input(1, 0, 0.5, rng_y) for _ in range(50)]
    assert seq_x == seq_y
    assert set(seq_x) == {0, 1}


# ------------------------------------------------ sequence_cross_entropy
def test_cross_entropy_of_a_uniform_prediction_is_log_vocab():
    assert sequence_cross_entropy([[0.0, 0.0, 0.0, 0.0]], [1]) == pytest.approx(
        math.log(4)
    )


def test_cross_entropy_is_near_zero_when_the_model_is_confidently_right():
    assert sequence_cross_entropy([[20.0, 0.0, 0.0]], [0], pad_id=9) < 1e-6


def test_cross_entropy_punishes_a_confidently_wrong_step():
    right = sequence_cross_entropy([[20.0, 0.0, 0.0]], [0], pad_id=9)
    wrong = sequence_cross_entropy([[20.0, 0.0, 0.0]], [1], pad_id=9)
    assert wrong > right + 10


def test_cross_entropy_ignores_padding_steps():
    """Добавили шаги с паддингом — средний loss не сдвинулся."""
    real = [[0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
    plain = sequence_cross_entropy(real, [1, 3])
    padded = sequence_cross_entropy(real + [[9.0, 0.0, 0.0, 0.0]] * 5, [1, 3, 0, 0, 0, 0, 0])
    assert padded == pytest.approx(plain)


def test_cross_entropy_of_an_all_padding_batch_is_zero():
    assert sequence_cross_entropy([[1.0, 2.0]] * 3, [0, 0, 0]) == 0.0


def test_cross_entropy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sequence_cross_entropy([[0.0, 1.0], [0.0, 1.0]], [1])


# ----------------------------------------------------------- greedy_decode
def test_greedy_follows_the_argmax_path():
    assert greedy_decode(make_step(DECISIVE), BOS, EOS, ()) == [A, B_TOK]


def test_greedy_stops_at_eos_and_does_not_emit_it():
    out = greedy_decode(make_step(TRAP), BOS, EOS, ())
    assert out == [A, B_TOK]
    assert EOS not in out


def test_greedy_respects_max_len_when_eos_never_comes():
    """Ловушка: без max_len модель, не выучившая eos, крутится вечно."""
    assert greedy_decode(make_step({}), BOS, EOS, (), max_len=5) == [A] * 5


def test_greedy_breaks_a_tie_towards_the_smaller_index():
    tie = {(BOS,): [-5.0, 0.0, 0.0, -5.0], (BOS, A): [3.0, -5.0, -5.0, -5.0]}
    assert greedy_decode(make_step(tie), BOS, EOS, ()) == [A]


# ------------------------------------------------------------- beam_search
def test_beam_of_width_one_equals_greedy():
    """Ширина 1 — это и есть жадное декодирование, шаг в шаг."""
    for table in (TRAP, DECISIVE, {}):
        step = make_step(table)
        assert beam_search(step, BOS, EOS, (), beam_width=1, max_len=6) == greedy_decode(
            step, BOS, EOS, (), max_len=6
        )


def test_beam_escapes_the_trap_that_greedy_falls_into():
    step = make_step(TRAP)
    assert greedy_decode(step, BOS, EOS, ()) == [A, B_TOK]
    assert beam_search(step, BOS, EOS, (), beam_width=2) == [B_TOK, C]


def test_beam_agrees_with_greedy_when_greedy_is_already_optimal():
    step = make_step(DECISIVE)
    assert beam_search(step, BOS, EOS, (), beam_width=3) == [A, B_TOK]


def test_beam_returns_an_empty_sequence_when_eos_wins_immediately():
    step = make_step({(BOS,): [3.0, -5.0, -5.0, -5.0]})
    assert beam_search(step, BOS, EOS, (), beam_width=2) == []


def test_beam_respects_max_len():
    out = beam_search(make_step({}), BOS, EOS, (), beam_width=2, max_len=3)
    assert len(out) == 3
