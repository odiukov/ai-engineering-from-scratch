"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_N = 60
_ANSWERS = ("A", "B", "C", "D")
_opinions = [random.choice(_ANSWERS) for _ in range(_N)]
_weighted = [(a, random.random()) for a in _opinions]
_full_mesh = {i: [j for j in range(_N) if j != i] for i in range(_N)}
_history = [_opinions, ["A"] * _N]

BENCH = {
    "majority_answer": (_opinions,),
    "weighted_answer": (_weighted,),
    "topology_peers": ("full_mesh", _N),
    "critique_ops": (_full_mesh, 3),
    "debate_round": (_opinions, _full_mesh),
    "run_debate": (_opinions, _full_mesh, 3),
    "collapsed_early": (_history,),
    "compare_topologies": (_opinions, 3, "A"),
}
