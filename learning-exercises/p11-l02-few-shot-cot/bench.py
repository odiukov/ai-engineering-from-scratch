"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_vocab = ["apples", "train", "eggs", "coupon", "discount", "hours", "price", "total"]


def _question(i):
    words = " ".join(random.choice(_vocab) for _ in range(12))
    return f"Problem {i}: {words}, how many?"


_examples = [
    {
        "question": _question(i),
        "reasoning": f"Step one gives {i}, step two gives {i * 2}.",
        "answer": str(i % 7),
    }
    for i in range(400)
]

_samples = [
    f"Reasoning path {i} with several steps. The answer is {i % 5}."
    for i in range(300)
]

_answers = [float(i % 5) for i in range(3000)]


def _expand(path):
    return [path[-1] + 1, path[-1] + 3, path[-1] - 1]


def _evaluate(path):
    return float(path[-1])


BENCH = {
    "format_example": (_examples[0],),
    "build_cot_prompt": (_question(1), _examples, 5),
    "extract_answer": (_samples[0] * 200,),
    "select_examples": (_question(2), _examples, 5),
    "select_diverse_examples": (_examples, 7),
    "majority_vote": (_answers,),
    "self_consistency": (_samples,),
    "tree_of_thought": (0, _expand, _evaluate, 3, 6, 4),
}
