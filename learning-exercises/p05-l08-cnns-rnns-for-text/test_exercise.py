"""Тесты к уроку «CNN и RNN для текста». Правь exercise.py."""

import math

import pytest

from exercise import (
    conv1d,
    global_max_pool,
    lstm_step,
    pool_hidden,
    rnn_forward,
    rnn_step,
    textcnn_features,
    vanishing_factor,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """Разворачивает список списков в плоский список: pytest.approx не умеет вложенные."""
    return [x for row in M for x in row]


# ------------------------------------------------------------------ conv1d
def test_conv1d_slides_the_kernel_along_the_sequence():
    assert conv1d([[1.0], [2.0], [3.0]], [[1.0], [1.0]]) == APPROX([3.0, 5.0])


def test_conv1d_adds_the_bias_at_every_position():
    assert conv1d([[1.0], [2.0], [3.0]], [[1.0], [1.0]], 0.5) == APPROX([3.5, 5.5])


def test_conv1d_output_length_is_t_minus_k_plus_one():
    seq = [[1.0, 0.0] for _ in range(7)]
    for k in (1, 2, 3, 4):
        kernel = [[1.0, 1.0]] * k
        assert len(conv1d(seq, kernel)) == 7 - k + 1


def test_conv1d_returns_nothing_when_the_kernel_is_wider_than_the_input():
    assert conv1d([[1.0]], [[1.0], [1.0]]) == []


def test_conv1d_fires_hardest_on_the_ngram_it_encodes():
    """Фильтр — это обучаемый детектор n-граммы, его пик стоит на нужном месте."""
    seq = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    kernel = [[1.0, 0.0], [0.0, 1.0]]  # ищет «канал 0, потом канал 1»
    scores = conv1d(seq, kernel)
    assert scores.index(max(scores)) == 1


def test_conv1d_rejects_a_kernel_of_the_wrong_channel_width():
    """Ловушка: zip обрезал бы по короткому и вернул правдоподобную чушь."""
    with pytest.raises(ValueError):
        conv1d([[1.0, 2.0], [3.0, 4.0]], [[1.0]])


# --------------------------------------------------------- global_max_pool
def test_global_max_pool_takes_the_strongest_activation():
    assert global_max_pool([0.0, 3.0, 1.0]) == APPROX(3.0)


def test_global_max_pool_ignores_where_the_peak_sits():
    """Позиционная инвариантность: «not good» в начале и в середине — один признак."""
    assert global_max_pool([9.0, 0.0, 0.0]) == global_max_pool([0.0, 0.0, 9.0])


def test_global_max_pool_rejects_an_empty_feature_map():
    with pytest.raises(ValueError):
        global_max_pool([])


# --------------------------------------------------------- textcnn_features
def test_textcnn_features_returns_one_number_per_filter():
    seq = [[1.0], [2.0], [3.0]]
    filters = [([[1.0], [1.0]], 0.0), ([[1.0]], 0.0)]
    assert textcnn_features(seq, filters) == APPROX([5.0, 3.0])


def test_textcnn_features_are_position_invariant():
    """Один и тот же биграм в разных местах даёт один и тот же признак."""
    kernel = [[1.0, 0.0], [0.0, 1.0]]
    early = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    late = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert textcnn_features(early, [(kernel, 0.0)]) == APPROX(
        textcnn_features(late, [(kernel, 0.0)])
    )


def test_textcnn_features_size_does_not_depend_on_input_length():
    """Ради этого и нужен пулинг: вход любой длины, выход фиксированный."""
    filters = [([[1.0], [1.0]], 0.0)]
    short = textcnn_features([[1.0], [2.0]], filters)
    long = textcnn_features([[1.0]] * 50, filters)
    assert len(short) == len(long) == 1


def test_textcnn_features_never_go_negative():
    """ReLU после свёртки: признак либо есть, либо его нет, «минус два» не бывает."""
    seq = [[-5.0], [-4.0], [-9.0]]
    assert textcnn_features(seq, [([[1.0], [1.0]], 0.0)]) == APPROX([0.0])


def test_textcnn_features_fail_loudly_on_a_filter_that_does_not_fit():
    with pytest.raises(ValueError):
        textcnn_features([[1.0]], [([[1.0], [1.0]], 0.0)])


# ---------------------------------------------------------------- rnn_step
def test_rnn_step_applies_tanh_to_the_affine_part():
    assert rnn_step([1.0], [0.0], [[1.0]], [[1.0]], [0.0]) == APPROX([math.tanh(1.0)])


def test_rnn_step_mixes_in_the_previous_state():
    """Без вклада h_prev это была бы обычная полносвязная сеть, а не RNN."""
    with_state = rnn_step([1.0], [1.0], [[1.0]], [[1.0]], [0.0])
    assert with_state == APPROX([math.tanh(2.0)])


def test_rnn_step_state_stays_inside_the_tanh_range():
    h = rnn_step([10.0], [0.0], [[1.0]], [[1.0]], [0.0])
    assert 0.999 < h[0] < 1.0


def test_rnn_step_returns_a_vector_of_the_hidden_size():
    W_x = [[1.0], [2.0], [3.0]]
    W_h = [[0.0] * 3 for _ in range(3)]
    assert len(rnn_step([1.0], [0.0] * 3, W_x, W_h, [0.0] * 3)) == 3


# ------------------------------------------------------------- rnn_forward
def test_rnn_forward_returns_one_state_per_token():
    seq = [[1.0], [1.0], [1.0]]
    assert len(rnn_forward(seq, [[1.0]], [[0.5]], [0.0])) == 3


def test_rnn_forward_starts_from_zeros_by_default():
    seq = [[1.0]]
    assert flat(rnn_forward(seq, [[1.0]], [[1.0]], [0.0])) == APPROX(
        flat(rnn_forward(seq, [[1.0]], [[1.0]], [0.0], h0=[0.0]))
    )


def test_rnn_forward_depends_on_word_order():
    """Именно этим RNN отличается от bag-of-words: «dog bites man» не равно обратному."""
    W_x, W_h, b = [[1.0]], [[0.9]], [0.0]
    a = rnn_forward([[1.0], [-1.0]], W_x, W_h, b)
    z = rnn_forward([[-1.0], [1.0]], W_x, W_h, b)
    assert flat(a) != APPROX(flat(z))


def test_rnn_forward_reverse_is_the_mirror_of_the_forward_pass():
    """Ловушка: при reverse=True состояния возвращаются в порядке ПОЗИЦИЙ входа."""
    seq = [[1.0], [-2.0], [0.5]]
    W_x, W_h, b = [[1.0]], [[0.7]], [0.1]
    backward = rnn_forward(seq, W_x, W_h, b, reverse=True)
    manual = rnn_forward(seq[::-1], W_x, W_h, b)[::-1]
    assert flat(backward) == APPROX(flat(manual))


def test_rnn_forward_last_state_summarizes_the_whole_prefix():
    seq = [[1.0], [1.0], [1.0]]
    states = rnn_forward(seq, [[1.0]], [[1.0]], [0.0])
    assert states[-1][0] == APPROX(math.tanh(1.0 + states[-2][0]))


def test_rnn_forward_of_an_empty_sequence_is_empty():
    assert rnn_forward([], [[1.0]], [[1.0]], [0.0]) == []


# -------------------------------------------------------------- pool_hidden
def test_pool_hidden_max_is_coordinate_wise():
    """Ловушка: это не «выбрать лучший вектор», а максимум по каждой координате."""
    assert pool_hidden([[1.0, 5.0], [4.0, 2.0]], "max") == APPROX([4.0, 5.0])


def test_pool_hidden_mean_averages_every_state():
    assert pool_hidden([[1.0, 5.0], [4.0, 2.0]], "mean") == APPROX([2.5, 3.5])


def test_pool_hidden_last_takes_only_the_final_state():
    assert pool_hidden([[1.0, 5.0], [4.0, 2.0]], "last") == APPROX([4.0, 2.0])


def test_pool_hidden_last_forgets_an_early_spike_but_max_does_not():
    """Почему для классификации берут max: last помнит в основном хвост."""
    states = [[9.0], [0.0], [0.0], [0.0]]
    assert pool_hidden(states, "last") == APPROX([0.0])
    assert pool_hidden(states, "max") == APPROX([9.0])


def test_pool_hidden_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        pool_hidden([[1.0]], "maxx")


def test_pool_hidden_rejects_an_empty_state_list():
    with pytest.raises(ValueError):
        pool_hidden([], "max")


# --------------------------------------------------------- vanishing_factor
def test_vanishing_factor_of_one_step_is_the_weight_itself():
    assert vanishing_factor(1, 0.9) == APPROX(0.9)


def test_vanishing_factor_kills_the_gradient_over_a_long_sequence():
    assert vanishing_factor(100, 0.9) < 1e-4


def test_vanishing_factor_explodes_above_one():
    """Другая половина проблемы: веса больше единицы разносят обучение."""
    assert vanishing_factor(100, 1.1) > 1000


def test_vanishing_factor_is_flat_exactly_at_one():
    assert vanishing_factor(1000, 1.0) == APPROX(1.0)


def test_vanishing_factor_shrinks_monotonically_with_length():
    values = [vanishing_factor(n, 0.9) for n in (1, 10, 50, 100)]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------- lstm_step
def _gates(bf, bi, bg, bo, w_x=0.0, w_h=0.0):
    """Однонейронная LSTM: всеми гейтами управляем через сдвиг."""
    return {
        "f": ([[w_x]], [[w_h]], [bf]),
        "i": ([[w_x]], [[w_h]], [bi]),
        "g": ([[w_x]], [[w_h]], [bg]),
        "o": ([[w_x]], [[w_h]], [bo]),
    }


def test_lstm_step_carries_the_cell_state_untouched_when_forget_is_open():
    """Главное свойство LSTM: f = 1, i = 0 — и cell state остаётся ровно тем же."""
    gates = _gates(bf=50.0, bi=-50.0, bg=0.0, bo=0.0)
    _, c = lstm_step([0.0], [0.0], [0.42], gates)
    assert c == pytest.approx([0.42], abs=1e-7)


def test_lstm_step_keeps_the_memory_over_a_hundred_steps():
    """Тот же «хайвей» через 100 шагов, где vanishing_factor уже съел бы всё."""
    gates = _gates(bf=50.0, bi=-50.0, bg=0.0, bo=0.0)
    h, c = [0.0], [1.0]
    for _ in range(100):
        h, c = lstm_step([0.0], h, c, gates)
    assert c[0] == pytest.approx(1.0, abs=1e-6)
    assert vanishing_factor(100, 0.9) < 1e-4


def test_lstm_step_wipes_the_cell_state_when_forget_is_shut():
    gates = _gates(bf=-50.0, bi=-50.0, bg=0.0, bo=0.0)
    _, c = lstm_step([0.0], [0.0], [7.0], gates)
    assert c == pytest.approx([0.0], abs=1e-7)


def test_lstm_step_writes_the_candidate_when_the_input_gate_is_open():
    """f = 0, i = 1: cell state становится ровно кандидатом g."""
    gates = _gates(bf=-50.0, bi=50.0, bg=0.5, bo=0.0)
    _, c = lstm_step([0.0], [0.0], [7.0], gates)
    assert c == pytest.approx([math.tanh(0.5)], abs=1e-7)


def test_lstm_step_output_gate_can_hide_the_memory():
    """o = 0 — снаружи ноль, но внутри cell state цел. Гейты независимы."""
    gates = _gates(bf=50.0, bi=-50.0, bg=0.0, bo=-50.0)
    h, c = lstm_step([0.0], [0.0], [0.9], gates)
    assert h == pytest.approx([0.0], abs=1e-7)
    assert c == pytest.approx([0.9], abs=1e-7)


def test_lstm_step_hidden_state_stays_inside_minus_one_and_one():
    gates = _gates(bf=50.0, bi=50.0, bg=50.0, bo=50.0)
    h, _ = lstm_step([0.0], [0.0], [1000.0], gates)
    assert -1.0 < h[0] < 1.0


def test_lstm_step_cell_state_is_free_to_grow_past_one():
    """Ловушка: tanh зажимает h, а c при записи через него не проходит."""
    gates = _gates(bf=50.0, bi=50.0, bg=50.0, bo=0.0)
    _, c = lstm_step([0.0], [0.0], [5.0], gates)
    assert c[0] > 1.0


def test_lstm_step_candidate_uses_tanh_not_sigmoid():
    """Ловушка: g знаковый (tanh), гейты — только от 0 до 1 (sigmoid)."""
    gates = _gates(bf=-50.0, bi=50.0, bg=-2.0, bo=0.0)
    _, c = lstm_step([0.0], [0.0], [0.0], gates)
    assert c[0] < 0  # с сигмоидой вместо tanh кандидат не стал бы отрицательным
    assert c == pytest.approx([math.tanh(-2.0)], abs=1e-7)


def test_lstm_step_reads_both_the_input_and_the_previous_state():
    gates = _gates(bf=-50.0, bi=50.0, bg=0.0, bo=50.0, w_x=1.0, w_h=1.0)
    _, from_x = lstm_step([1.0], [0.0], [0.0], gates)
    _, from_h = lstm_step([0.0], [1.0], [0.0], gates)
    assert from_x == pytest.approx(from_h, abs=1e-7)
    assert from_x[0] > 0
