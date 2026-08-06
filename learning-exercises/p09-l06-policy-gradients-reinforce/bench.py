"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_N_FEATURES = 16
_theta = [[_rng.uniform(-1.0, 1.0) for _ in range(_N_FEATURES)] for _ in range(4)]
_features = [0.0] * _N_FEATURES
_features[0] = 1.0
_probs = [0.1, 0.4, 0.2, 0.3]
_logits = [0.7, -1.2, 0.3, 2.0]
_rewards = [-1.0] * 2000


def _traj():
    out = []
    for t in range(60):
        x = [0.0] * _N_FEATURES
        x[t % _N_FEATURES] = 1.0
        out.append((x, t % 4, -1.0))
    return out


BENCH = {
    "softmax": (_logits,),
    "policy_probs": (_theta, _features),
    "sample_action": (_probs, _rng),
    "grad_log_pi": (_probs, 1),
    "returns_to_go": (_rewards, 0.99),
    "grid_rollout": (_theta, _rng),
    "reinforce_grad": (_theta, _traj(), 0.99, -30.0),
    "train_reinforce": (200, 0.05, 0.99, True, _rng),
}
