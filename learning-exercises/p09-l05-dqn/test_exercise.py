"""Тесты к уроку «Deep Q-Networks (DQN)». Правь exercise.py."""

import random

import pytest

from exercise import (
    ReplayBuffer,
    clone_net,
    double_dqn_target,
    dqn_target,
    forward,
    init_net,
    one_hot,
    train_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# крошечная сеть с известными наизусть весами: 1 вход -> 2 скрытых -> 1 выход
TINY = {
    "W1": [[1.0], [-1.0]],
    "b1": [0.0, 0.0],
    "W2": [[1.0, 2.0]],
    "b2": [0.5],
}

# Коридор из шести состояний: действие "вправо" (индекс 1) ведёт к выходу.
# Награда -1 за шаг, терминал — состояние 5. Оптимально из 0: пять шагов.
CORRIDOR = 6


def flat_net(net):
    """Развернуть все веса в плоский список: pytest.approx не умеет вложенное."""
    out = []
    for key in ("W1", "W2"):
        for row in net[key]:
            out.extend(row)
    out.extend(net["b1"])
    out.extend(net["b2"])
    return out


def corridor_step(pos, action):
    """Модель коридора: action 0 — влево, action 1 — вправо."""
    nxt = max(0, pos - 1) if action == 0 else min(CORRIDOR - 1, pos + 1)
    return nxt, -1.0, nxt == CORRIDOR - 1


def corridor_batch():
    """Все переходы коридора, уже в признаках. Это offline-выборка для fitted-Q."""
    batch = []
    for pos in range(CORRIDOR - 1):
        for action in (0, 1):
            nxt, reward, done = corridor_step(pos, action)
            batch.append((one_hot(pos, CORRIDOR), action, reward, one_hot(nxt, CORRIDOR), done))
    return batch


# ----------------------------------------------------------------- one_hot
def test_one_hot_marks_exactly_one_position():
    assert one_hot(2, 4) == [0.0, 0.0, 1.0, 0.0]
    assert one_hot(0, 3) == [1.0, 0.0, 0.0]


def test_one_hot_sums_to_one_and_keeps_its_length():
    for i in range(5):
        vector = one_hot(i, 5)
        assert len(vector) == 5
        assert sum(vector) == APPROX(1.0)


def test_one_hot_rejects_an_index_outside_the_range():
    """Молча вернуть вектор из нулей нельзя: два состояния станут неразличимы."""
    with pytest.raises(ValueError):
        one_hot(5, 5)
    with pytest.raises(ValueError):
        one_hot(-1, 5)


# ---------------------------------------------------------------- init_net
def test_init_net_has_the_requested_shapes():
    net = init_net(4, 3, 2, random.Random(0))
    assert len(net["W1"]) == 3 and all(len(row) == 4 for row in net["W1"])
    assert len(net["W2"]) == 2 and all(len(row) == 3 for row in net["W2"])
    assert len(net["b1"]) == 3 and len(net["b2"]) == 2


def test_init_net_starts_biases_at_zero():
    net = init_net(4, 3, 2, random.Random(0))
    assert net["b1"] == [0.0] * 3
    assert net["b2"] == [0.0] * 2


def test_init_net_breaks_symmetry_between_hidden_units():
    """Одинаковые веса — одинаковые нейроны навсегда, скрытый слой бесполезен."""
    net = init_net(4, 3, 2, random.Random(0))
    assert net["W1"][0] != net["W1"][1]
    assert all(w != 0.0 for row in net["W1"] for w in row)


def test_init_net_is_reproducible_for_the_same_seed():
    a = init_net(4, 3, 2, random.Random(7))
    b = init_net(4, 3, 2, random.Random(7))
    assert flat_net(a) == pytest.approx(flat_net(b), abs=1e-12)


# --------------------------------------------------------------- clone_net
def test_clone_net_copies_every_number():
    net = init_net(4, 3, 2, random.Random(0))
    assert flat_net(clone_net(net)) == pytest.approx(flat_net(net), abs=1e-12)


def test_clone_net_does_not_share_matrix_rows():
    """Главная ловушка урока: dict(net) оставляет строки теми же списками."""
    net = init_net(4, 3, 2, random.Random(0))
    copy = clone_net(net)
    copy["W1"][0][0] = 123.0
    copy["b2"][1] = 456.0
    assert net["W1"][0][0] != 123.0
    assert net["b2"][1] != 456.0


def test_clone_net_survives_training_of_the_original():
    """Target-сеть обязана остаться замороженной, пока учится online-сеть."""
    online = init_net(CORRIDOR, 6, 2, random.Random(1))
    frozen = clone_net(online)
    before = flat_net(frozen)
    train_step(online, frozen, corridor_batch(), gamma=0.9, lr=0.5)
    assert flat_net(frozen) == pytest.approx(before, abs=1e-12)
    assert flat_net(online) != pytest.approx(before, abs=1e-12)


# ----------------------------------------------------------------- forward
def test_forward_computes_the_hand_checked_example():
    q, h = forward(TINY, [3.0])
    assert h == pytest.approx([3.0, 0.0], abs=1e-12)
    assert q == pytest.approx([3.5], abs=1e-12)


def test_forward_relu_clips_negative_pre_activations():
    _, h = forward(TINY, [-2.0])
    assert h == pytest.approx([0.0, 2.0], abs=1e-12)


def test_forward_output_layer_has_no_activation():
    """Q-значения бывают отрицательными: прижимать выход ReLU нельзя."""
    net = {"W1": [[1.0]], "b1": [0.0], "W2": [[-4.0]], "b2": [0.0]}
    q, _ = forward(net, [2.0])
    assert q == pytest.approx([-8.0], abs=1e-12)


def test_forward_returns_one_value_per_action():
    net = init_net(CORRIDOR, 5, 3, random.Random(0))
    q, h = forward(net, one_hot(0, CORRIDOR))
    assert len(q) == 3
    assert len(h) == 5


def test_forward_of_a_zero_input_is_just_the_biases():
    net = init_net(4, 3, 2, random.Random(0))
    net["b2"] = [1.0, -1.0]
    q, h = forward(net, [0.0] * 4)
    assert h == pytest.approx([0.0] * 3, abs=1e-12)
    assert q == pytest.approx([1.0, -1.0], abs=1e-12)


# -------------------------------------------------------------- dqn_target
def test_dqn_target_bootstraps_with_the_best_next_value():
    assert dqn_target(-1.0, 0.9, [-5.0, -2.0, -9.0], False) == APPROX(-1.0 + 0.9 * -2.0)


def test_dqn_target_in_the_terminal_is_the_bare_reward():
    """Иначе значения absorbing state уползают в минус бесконечность."""
    assert dqn_target(-1.0, 0.9, [-5.0, -2.0, -9.0], True) == APPROX(-1.0)


def test_dqn_target_ignores_everything_but_the_max():
    assert dqn_target(0.0, 1.0, [3.0, -100.0], False) == APPROX(
        dqn_target(0.0, 1.0, [3.0, -1.0], False)
    )


# ------------------------------------------------------- double_dqn_target
def test_double_dqn_target_evaluates_the_action_the_online_net_picked():
    assert double_dqn_target(0.0, 1.0, [1.0, 5.0], [7.0, 3.0], False) == APPROX(3.0)


def test_double_dqn_target_equals_the_plain_one_when_both_nets_agree():
    q = [-5.0, -2.0, -9.0]
    assert double_dqn_target(-1.0, 0.9, q, q, False) == APPROX(
        dqn_target(-1.0, 0.9, q, False)
    )


def test_double_dqn_target_in_the_terminal_is_the_bare_reward():
    assert double_dqn_target(-1.0, 0.9, [1.0, 2.0], [3.0, 4.0], True) == APPROX(-1.0)


def test_double_dqn_target_removes_the_maximization_bias():
    """Оба списка — независимый шум вокруг нуля. Истинный max равен нулю.

    Обычная цель берёт max по шуму и в среднем уползает вверх примерно на
    единицу. Double берёт значение из НЕЗАВИСИМОЙ выборки и остаётся у нуля.
    Это и есть та переоценка, которую чинит Hasselt (2016).
    """
    rng = random.Random(0)
    plain, double = [], []
    for _ in range(4000):
        noisy_a = [rng.gauss(0.0, 1.0) for _ in range(4)]
        noisy_b = [rng.gauss(0.0, 1.0) for _ in range(4)]
        plain.append(dqn_target(0.0, 1.0, noisy_b, False))
        double.append(double_dqn_target(0.0, 1.0, noisy_a, noisy_b, False))
    assert sum(plain) / len(plain) > 0.8
    assert abs(sum(double) / len(double)) < 0.1


# --------------------------------------------------------------- train_step
def test_train_step_with_zero_lr_reports_loss_and_changes_nothing():
    net = init_net(CORRIDOR, 6, 2, random.Random(2))
    frozen = clone_net(net)
    before = flat_net(net)
    loss = train_step(net, frozen, corridor_batch(), gamma=0.9, lr=0.0)
    assert loss > 0.0
    assert flat_net(net) == pytest.approx(before, abs=1e-12)


def test_train_step_loss_is_the_mean_half_squared_td_error():
    net = init_net(CORRIDOR, 6, 2, random.Random(2))
    frozen = clone_net(net)
    batch = corridor_batch()
    expected = 0.0
    for x, a, reward, x_next, done in batch:
        q, _ = forward(net, x)
        q_next, _ = forward(frozen, x_next)
        td = q[a] - dqn_target(reward, 0.9, q_next, done)
        expected += 0.5 * td * td
    expected /= len(batch)
    assert train_step(net, frozen, batch, gamma=0.9, lr=0.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_train_step_output_layer_gradient_matches_central_difference():
    """Аналитический градиент по W2 обязан совпасть с численным."""
    frozen = clone_net(init_net(CORRIDOR, 6, 2, random.Random(3)))
    batch = corridor_batch()
    base = init_net(CORRIDOR, 6, 2, random.Random(4))
    lr, h = 1.0, 1e-6

    stepped = clone_net(base)
    train_step(stepped, frozen, batch, gamma=0.9, lr=lr)

    for a in (0, 1):
        for j in (0, 3, 5):
            analytic = (base["W2"][a][j] - stepped["W2"][a][j]) / lr
            up, down = clone_net(base), clone_net(base)
            up["W2"][a][j] += h
            down["W2"][a][j] -= h
            numeric = (
                train_step(up, frozen, batch, gamma=0.9, lr=0.0)
                - train_step(down, frozen, batch, gamma=0.9, lr=0.0)
            ) / (2 * h)
            assert analytic == pytest.approx(numeric, abs=1e-5)


def test_train_step_hidden_layer_gradient_matches_central_difference():
    """И по W1 тоже — иначе ошибка в ReLU-маске останется незамеченной."""
    frozen = clone_net(init_net(CORRIDOR, 6, 2, random.Random(3)))
    batch = corridor_batch()
    base = init_net(CORRIDOR, 6, 2, random.Random(4))
    lr, h = 1.0, 1e-6

    stepped = clone_net(base)
    train_step(stepped, frozen, batch, gamma=0.9, lr=lr)

    checked = 0
    for j in (0, 2, 4):
        for k in (1, 3):
            analytic = (base["W1"][j][k] - stepped["W1"][j][k]) / lr
            up, down = clone_net(base), clone_net(base)
            up["W1"][j][k] += h
            down["W1"][j][k] -= h
            numeric = (
                train_step(up, frozen, batch, gamma=0.9, lr=0.0)
                - train_step(down, frozen, batch, gamma=0.9, lr=0.0)
            ) / (2 * h)
            assert analytic == pytest.approx(numeric, abs=1e-5)
            checked += 1
    assert checked == 6


def test_train_step_bias_gradients_match_central_difference():
    frozen = clone_net(init_net(CORRIDOR, 6, 2, random.Random(3)))
    batch = corridor_batch()
    base = init_net(CORRIDOR, 6, 2, random.Random(4))
    lr, h = 1.0, 1e-6

    stepped = clone_net(base)
    train_step(stepped, frozen, batch, gamma=0.9, lr=lr)

    for key, idx in (("b2", 0), ("b2", 1), ("b1", 0), ("b1", 3)):
        analytic = (base[key][idx] - stepped[key][idx]) / lr
        up, down = clone_net(base), clone_net(base)
        up[key][idx] += h
        down[key][idx] -= h
        numeric = (
            train_step(up, frozen, batch, gamma=0.9, lr=0.0)
            - train_step(down, frozen, batch, gamma=0.9, lr=0.0)
        ) / (2 * h)
        assert analytic == pytest.approx(numeric, abs=1e-5)


def test_train_step_lowers_the_loss_on_a_fixed_batch():
    net = init_net(CORRIDOR, 6, 2, random.Random(5))
    frozen = clone_net(net)
    batch = corridor_batch()
    first = train_step(net, frozen, batch, gamma=0.9, lr=0.2)
    for _ in range(30):
        train_step(net, frozen, batch, gamma=0.9, lr=0.2)
    assert train_step(net, frozen, batch, gamma=0.9, lr=0.0) < first


def test_fitted_q_iteration_learns_to_walk_out_of_the_corridor():
    """Полный DQN в миниатюре: батч фиксирован, target периодически синхронится.

    Из любого состояния коридора оптимально идти вправо (индекс 1), а Q
    первого состояния обязано подойти к -(1 + g + g^2 + g^3 + g^4).
    """
    gamma = 0.9
    net = init_net(CORRIDOR, 12, 2, random.Random(6))
    frozen = clone_net(net)
    batch = corridor_batch()
    for i in range(600):
        if i % 25 == 0:
            frozen = clone_net(net)
        train_step(net, frozen, batch, gamma=gamma, lr=0.3)

    for pos in range(CORRIDOR - 1):
        q, _ = forward(net, one_hot(pos, CORRIDOR))
        assert q[1] > q[0], f"в состоянии {pos} сеть не выбрала «вправо»"

    exact = -sum(gamma ** t for t in range(CORRIDOR - 1))
    q0, _ = forward(net, one_hot(0, CORRIDOR))
    assert max(q0) == pytest.approx(exact, abs=0.5)


# ------------------------------------------------------------- ReplayBuffer
def test_replay_buffer_counts_what_it_holds():
    buf = ReplayBuffer(5)
    assert len(buf) == 0
    buf.push(("a",))
    buf.push(("b",))
    assert len(buf) == 2


def test_replay_buffer_drops_the_oldest_transition_when_full():
    buf = ReplayBuffer(3)
    for name in "abcd":
        buf.push((name,))
    assert len(buf) == 3
    rng = random.Random(0)
    assert set(buf.sample(3, rng)) == {("b",), ("c",), ("d",)}


def test_replay_buffer_sample_returns_distinct_transitions():
    buf = ReplayBuffer(10)
    for i in range(10):
        buf.push((i,))
    drawn = buf.sample(4, random.Random(1))
    assert len(drawn) == 4
    assert len(set(drawn)) == 4


def test_replay_buffer_sample_does_not_consume_the_buffer():
    buf = ReplayBuffer(10)
    for i in range(10):
        buf.push((i,))
    buf.sample(4, random.Random(1))
    assert len(buf) == 10


def test_replay_buffer_sample_is_reproducible_for_the_same_seed():
    """Без этого ни один прогон DQN нельзя повторить."""
    buf = ReplayBuffer(20)
    for i in range(20):
        buf.push((i,))
    assert buf.sample(5, random.Random(2)) == buf.sample(5, random.Random(2))


def test_replay_buffer_decorrelates_a_sequential_stream():
    """Смысл буфера: подряд записанные переходы выходят из него перемешанными."""
    buf = ReplayBuffer(100)
    for i in range(100):
        buf.push((i,))
    drawn = [t[0] for t in buf.sample(20, random.Random(3))]
    assert drawn != sorted(drawn)
    assert max(drawn) - min(drawn) > 20
