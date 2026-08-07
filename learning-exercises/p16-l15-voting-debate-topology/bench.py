"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_names = [f"agent-{i:02d}" for i in range(8)]


def _shuffled():
    ballot = list(_names)
    random.shuffle(ballot)
    return ballot


_ballots = [_shuffled() for _ in range(300)]
_approvals = [random.sample(_names, random.randint(1, 5)) for _ in range(300)]

BENCH = {
    "candidates": (_ballots,),
    "plurality_winner": (_ballots,),
    "borda_scores": (_ballots,),
    "borda_winner": (_ballots,),
    "approval_winner": (_approvals,),
    "pairwise_margin": (_ballots, _names[0], _names[1]),
    "condorcet_winner": (_ballots,),
    "condorcet_cycle": (_ballots,),
}
