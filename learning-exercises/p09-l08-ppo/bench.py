"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N_FEAT = 64
_N_ACT = 6
_theta = [[random.gauss(0.0, 0.1) for _ in range(_N_FEAT)] for _ in range(_N_ACT)]
_x = [random.gauss(0.0, 1.0) for _ in range(_N_FEAT)]

_N = 1024  # 8 envs x 128 steps — стандартный роллаут PPO на Atari
_ratios = [math.exp(random.gauss(0.0, 0.15)) for _ in range(_N)]
_advs = [random.gauss(0.0, 1.0) for _ in range(_N)]
_old_logs = [-math.log(_N_ACT) + random.gauss(0.0, 0.1) for _ in range(_N)]
_new_logs = [lp + random.gauss(0.0, 0.05) for lp in _old_logs]

_batch = [
    {
        "x": [random.gauss(0.0, 1.0) for _ in range(_N_FEAT)],
        "a": random.randrange(_N_ACT),
        "log_pi_old": -math.log(_N_ACT),
        "adv": random.gauss(0.0, 1.0),
    }
    for _ in range(128)
]

BENCH = {
    "action_probs": (_theta, _x),
    "importance_ratio": (-1.5, -1.6),
    "clipped_surrogate": (1.1, 0.7),
    "surrogate_gradient_scale": (1.1, 0.7),
    "clip_fraction": (_ratios, _advs),
    "approx_kl": (_old_logs, _new_logs),
    "ppo_actor_step": (_theta, _x, 2, -math.log(_N_ACT), 1.3),
    "ppo_update": (_theta, _batch),
}
