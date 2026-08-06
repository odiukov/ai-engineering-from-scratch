"""Tests for the from-scratch DQN: MLP forward/backward, replay buffer, target net.

Covers phases/09-reinforcement-learning/05-dqn/docs/en.md.
The analytic backprop inside train_step is checked against central finite
differences of the squared TD loss, so a wrong sign or a dropped ReLU mask
fails here rather than showing up as a mysteriously flat learning curve.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from main import (  # noqa: E402
    ACTIONS,
    GRID,
    TERMINAL,
    ReplayBuffer,
    clone,
    epsilon_greedy,
    forward,
    init_net,
    reset,
    state_features,
    step,
    train_step,
)

PARAM_KEYS = ("W1", "b1", "W2", "b2")


def param_slots(net):
    """Yield (key, index-path) for every scalar parameter in the net."""
    for j in range(len(net["b1"])):
        yield "b1", (j,)
        for k in range(len(net["W1"][j])):
            yield "W1", (j, k)
    for a in range(len(net["b2"])):
        yield "b2", (a,)
        for j in range(len(net["W2"][a])):
            yield "W2", (a, j)


def read_slot(net, key, path):
    return net[key][path[0]] if len(path) == 1 else net[key][path[0]][path[1]]


def write_slot(net, key, path, value):
    if len(path) == 1:
        net[key][path[0]] = value
    else:
        net[key][path[0]][path[1]] = value


def batch_loss(net, batch, target, gamma):
    """sum_i 0.5 (Q(s_i, a_i; theta) - y_i)^2 with y_i frozen by the target net."""
    total = 0.0
    for s, a, r, s_next, done in batch:
        q, _h = forward(net, state_features(s))
        if done:
            y = r
        else:
            q_next, _ = forward(target, state_features(s_next))
            y = r + gamma * max(q_next)
        total += 0.5 * (q[a] - y) ** 2
    return total


def numerical_grads(net, batch, target, gamma, eps=1e-5):
    grads = {"W1": [], "b1": [], "W2": [], "b2": []}
    grads["b1"] = [0.0] * len(net["b1"])
    grads["W1"] = [[0.0] * len(net["W1"][0]) for _ in net["W1"]]
    grads["b2"] = [0.0] * len(net["b2"])
    grads["W2"] = [[0.0] * len(net["W2"][0]) for _ in net["W2"]]
    for key, path in param_slots(net):
        original = read_slot(net, key, path)
        write_slot(net, key, path, original + eps)
        plus = batch_loss(net, batch, target, gamma)
        write_slot(net, key, path, original - eps)
        minus = batch_loss(net, batch, target, gamma)
        write_slot(net, key, path, original)
        write_slot(grads, key, path, (plus - minus) / (2 * eps))
    return grads


def analytic_grads(before, after, scale):
    """Recover the accumulated gradient from the SGD step: theta -= scale * grad."""
    grads = {
        "b1": [(b - a) / scale for b, a in zip(before["b1"], after["b1"])],
        "W1": [
            [(b - a) / scale for b, a in zip(rb, ra)]
            for rb, ra in zip(before["W1"], after["W1"])
        ],
        "b2": [(b - a) / scale for b, a in zip(before["b2"], after["b2"])],
        "W2": [
            [(b - a) / scale for b, a in zip(rb, ra)]
            for rb, ra in zip(before["W2"], after["W2"])
        ],
    }
    return grads


class TestFeaturesAndForward(unittest.TestCase):
    def test_state_features_are_one_hot_over_the_grid(self) -> None:
        feat = state_features((2, 1))
        self.assertEqual(len(feat), GRID * GRID)
        self.assertEqual(sum(feat), 1.0)
        self.assertEqual(feat[2 * GRID + 1], 1.0)
        self.assertNotEqual(state_features((0, 0)), state_features((0, 1)))

    def test_forward_matches_hand_computed_relu_mlp(self) -> None:
        net = {
            "W1": [[1.0, 0.0], [-1.0, 0.0]],
            "b1": [0.5, 0.25],
            "W2": [[2.0, 3.0], [-1.0, 1.0]],
            "b2": [0.1, -0.1],
        }
        q, h = forward(net, [1.0, 0.0])
        self.assertEqual(h, [1.5, 0.0])  # second unit is clipped by the ReLU
        self.assertAlmostEqual(q[0], 2.0 * 1.5 + 3.0 * 0.0 + 0.1)
        self.assertAlmostEqual(q[1], -1.0 * 1.5 + 1.0 * 0.0 - 0.1)

    def test_clone_is_a_deep_copy(self) -> None:
        net = init_net(4, 3, 2, random.Random(0))
        copy = clone(net)
        copy["W1"][0][0] += 10.0
        copy["b2"][0] += 10.0
        self.assertNotEqual(copy["W1"][0][0], net["W1"][0][0])
        self.assertNotEqual(copy["b2"][0], net["b2"][0])
        self.assertIsNot(copy["W1"][0], net["W1"][0])


class TestEpsilonGreedy(unittest.TestCase):
    def test_epsilon_zero_returns_the_argmax_action_index(self) -> None:
        net = {
            "W1": [[0.0] * (GRID * GRID)],
            "b1": [1.0],
            "W2": [[-5.0], [7.0], [0.0], [3.0]],
            "b2": [0.0] * len(ACTIONS),
        }
        rng = random.Random(0)
        for _ in range(100):
            self.assertEqual(epsilon_greedy(net, (0, 0), rng, 0.0), 1)

    def test_epsilon_one_explores_every_action_index(self) -> None:
        net = init_net(GRID * GRID, 4, len(ACTIONS), random.Random(0))
        rng = random.Random(1)
        picked = {epsilon_greedy(net, (0, 0), rng, 1.0) for _ in range(300)}
        self.assertEqual(picked, set(range(len(ACTIONS))))


class TestBackpropAgainstFiniteDifferences(unittest.TestCase):
    def assert_grads_close(self, got, expected, tol=1e-6) -> None:
        for key in PARAM_KEYS:
            flat_got = got[key] if key.startswith("b") else [v for row in got[key] for v in row]
            flat_exp = (
                expected[key] if key.startswith("b") else [v for row in expected[key] for v in row]
            )
            for g, e in zip(flat_got, flat_exp):
                self.assertAlmostEqual(g, e, delta=tol + 1e-4 * abs(e), msg=f"{key} mismatch")

    def test_terminal_transition_gradient_matches_central_differences(self) -> None:
        rng = random.Random(3)
        net = init_net(GRID * GRID, 6, len(ACTIONS), rng)
        target = clone(net)
        batch = [((1, 2), 2, -1.0, TERMINAL, True)]
        expected = numerical_grads(net, batch, target, 0.99)
        before = clone(net)
        train_step(net, target, batch, 0.99, 1.0)
        self.assert_grads_close(analytic_grads(before, net, 1.0), expected)

    def test_bootstrapped_batch_gradient_matches_central_differences(self) -> None:
        rng = random.Random(5)
        net = init_net(GRID * GRID, 8, len(ACTIONS), rng)
        target = clone(net)
        batch = [
            ((0, 0), 1, -1.0, (1, 0), False),
            ((1, 0), 3, -1.0, (1, 1), False),
            ((3, 2), 3, -1.0, TERMINAL, True),
        ]
        gamma, lr = 0.99, 1.0
        expected = numerical_grads(net, batch, target, gamma)
        before = clone(net)
        train_step(net, target, batch, gamma, lr)
        # train_step applies scale = lr / len(batch), so recover the summed gradient.
        self.assert_grads_close(analytic_grads(before, net, lr / len(batch)), expected)

    def test_gradient_is_zero_when_the_prediction_already_equals_the_target(self) -> None:
        rng = random.Random(9)
        net = init_net(GRID * GRID, 5, len(ACTIONS), rng)
        target = clone(net)
        q, _h = forward(net, state_features((2, 2)))
        batch = [((2, 2), 0, q[0], TERMINAL, True)]  # done -> y = r = current prediction
        before = clone(net)
        loss = train_step(net, target, batch, 0.99, 1.0)
        self.assertAlmostEqual(loss, 0.0, places=12)
        self.assertEqual(net["W2"], before["W2"])
        self.assertEqual(net["b1"], before["b1"])


class TestTrainStepBehavior(unittest.TestCase):
    def test_reported_loss_is_the_mean_half_squared_td_error(self) -> None:
        rng = random.Random(4)
        net = init_net(GRID * GRID, 6, len(ACTIONS), rng)
        target = clone(net)
        batch = [((0, 1), 1, -1.0, (1, 1), False), ((2, 3), 1, -1.0, TERMINAL, True)]
        expected = batch_loss(net, batch, target, 0.99) / len(batch)
        self.assertAlmostEqual(train_step(net, target, batch, 0.99, 0.0), expected, places=12)

    def test_repeated_steps_on_one_batch_drive_the_loss_down(self) -> None:
        rng = random.Random(6)
        net = init_net(GRID * GRID, 8, len(ACTIONS), rng)
        target = clone(net)
        batch = [((0, 0), 2, -1.0, (0, 1), False)]
        losses = [train_step(net, target, batch, 0.99, 0.05) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])))

    def test_target_network_is_never_mutated_by_a_gradient_step(self) -> None:
        rng = random.Random(8)
        net = init_net(GRID * GRID, 6, len(ACTIONS), rng)
        target = clone(net)
        frozen = clone(target)
        batch = [((0, 0), 1, -1.0, (1, 0), False)] * 4
        for _ in range(10):
            train_step(net, target, batch, 0.99, 0.1)
        self.assertEqual(target, frozen)
        self.assertNotEqual(net["W2"], frozen["W2"])


class TestReplayBuffer(unittest.TestCase):
    def test_capacity_is_respected_and_oldest_transitions_are_dropped(self) -> None:
        buffer = ReplayBuffer(3)
        for i in range(7):
            buffer.push((0, i), 0, -1.0, (0, i + 1), False)
        self.assertEqual(len(buffer), 3)
        self.assertEqual([s for s, *_ in buffer.buf], [(0, 4), (0, 5), (0, 6)])

    def test_sampling_returns_the_requested_batch_and_is_reproducible(self) -> None:
        buffer = ReplayBuffer(100)
        for i in range(20):
            buffer.push((0, i), i % 4, -1.0, (0, i + 1), False)
        first = buffer.sample(8, random.Random(2))
        second = buffer.sample(8, random.Random(2))
        self.assertEqual(len(first), 8)
        self.assertEqual(first, second)
        self.assertTrue(all(t in buffer.buf for t in first))


class TestTrainingLoopLearns(unittest.TestCase):
    def test_short_seeded_run_learns_a_downhill_policy_from_the_start_state(self) -> None:
        rng = random.Random(0)
        online = init_net(GRID * GRID, 32, len(ACTIONS), rng)
        target = clone(online)
        buffer = ReplayBuffer(2000)
        batch, gamma, lr, sync_every = 32, 0.99, 0.05, 200
        step_count = 0
        returns_log = []
        for episode in range(220):
            state = reset()
            total = 0.0
            epsilon = max(0.05, 1.0 - episode / 150)
            for _ in range(50):
                action = epsilon_greedy(online, state, rng, epsilon)
                s_next, reward, done = step(state, ACTIONS[action])
                total += reward
                buffer.push(state, action, reward, s_next, done)
                if len(buffer) >= batch:
                    train_step(online, target, buffer.sample(batch, rng), gamma, lr)
                step_count += 1
                if step_count % sync_every == 0:
                    target = clone(online)
                if done:
                    break
                state = s_next
            returns_log.append(total)

        early = sum(returns_log[:20]) / 20
        late = sum(returns_log[-20:]) / 20
        self.assertGreater(late, early + 5.0)
        self.assertLess(late, -6.0 + 1.5)

        q_start, _h = forward(online, state_features(reset()))
        best = ACTIONS[max(range(len(ACTIONS)), key=lambda i: q_start[i])]
        self.assertIn(best, ("down", "right"))


if __name__ == "__main__":
    unittest.main()
