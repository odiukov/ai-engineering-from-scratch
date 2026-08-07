"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_CONSTITUTION_FIRST = {
    "id": "no_hedging",
    "text": "The response must directly answer the question asked, without hedging.",
}

_long_response = "The answer is maybe " + "and it has many details " * 30 + "105."
_critique_result = {"principle": "no_hedging", "problems": ["hedging"]}

_prompts = [f"What is {a} * {b}?" for a in range(2, 22) for b in range(2, 12)]
_rewards = [random.choice([0.0, 1.0, 1.1]) for _ in range(20000)]

_policy = [random.gauss(0.0, 0.3) for _ in range(20000)]
_ref = [random.gauss(0.0, 0.3) for _ in range(20000)]
_adv = [random.gauss(0.0, 1.0) for _ in range(20000)]


def _sampler(prompt, rng):
    return str(rng.randrange(200))


BENCH = {
    "critique": (_long_response, _CONSTITUTION_FIRST),
    "revise": (_long_response, _critique_result),
    "reward_math": ("What is 15 * 7?", "10*7=70, 5*7=35, so 105"),
    "reward_format": (_long_response,),
    "total_reward": ("What is 15 * 7?", "<answer>105</answer>"),
    "group_relative_advantage": (_rewards,),
    "grpo_step": (_policy, _ref, _adv),
    "self_improvement_round": (_prompts, _sampler, random.Random(0), 8),
}
