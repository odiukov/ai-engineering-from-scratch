"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# 7 агентов — 5040 перестановок: точный Шепли уже заметно дороже
# сэмплированного, но один вызов ещё укладывается в десятки миллисекунд.
_agents = ["a%d" % i for i in range(7)]
_weights = {a: random.uniform(0, 1) for a in _agents}
_value = lambda s: sum(_weights[a] for a in s) * (1.0 + 0.1 * len(s))

_bids = [random.uniform(0, 1) for _ in range(200)]
_candidates = [i / 50.0 for i in range(51)]
_others = [random.uniform(0, 1) for _ in range(50)]
_reps = [random.uniform(0, 1) for _ in range(500)]

BENCH = {
    "marginal_contributions": (_value, _agents),
    "shapley": (_value, _agents),
    "shapley_sampled": (_value, _agents, 500, random.Random(0)),
    "second_price_auction": (_bids,),
    "bidder_utility": (0.7, _bids, 0),
    "best_bids": (0.7, _others, _candidates),
    "update_reputation": (0.5, 0.8, 0.9),
    "reputation_weighted_pick": (_reps, random.Random(0)),
}
