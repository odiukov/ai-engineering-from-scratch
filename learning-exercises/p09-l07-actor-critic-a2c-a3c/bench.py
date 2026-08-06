"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_T = 2048  # длина роллаута PPO/A2C на MuJoCo — реалистичный размер батча
_rewards = [random.gauss(0.0, 1.0) for _ in range(_T)]
_values = [random.gauss(0.0, 1.0) for _ in range(_T)]
_logits = [random.gauss(0.0, 2.0) for _ in range(64)]
_probs = [1.0 / 64] * 64

_N_FEAT = 128
_N_ACT = 8
_theta = [[random.gauss(0.0, 0.1) for _ in range(_N_FEAT)] for _ in range(_N_ACT)]
_w = [0.0] * _N_FEAT
_x = [random.gauss(0.0, 1.0) for _ in range(_N_FEAT)]

BENCH = {
    "softmax": (_logits,),
    "entropy": (_probs,),
    "grad_log_pi": (_probs, 3),
    "discounted_returns": (_rewards, 0.99),
    "td_residuals": (_rewards, _values, 0.99),
    "gae_advantages": (_rewards, _values, 0.99, 0.95),
    "normalize": (_rewards,),
    "actor_critic_step": (_theta, _w, _x, 2, 1.5, 0.7),
}
