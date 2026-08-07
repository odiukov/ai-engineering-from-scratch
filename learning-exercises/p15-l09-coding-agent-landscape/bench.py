"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_tasks = [
    {"id": f"t{i}", "lines": random.randint(1, 40), "solved": random.random() < 0.6}
    for i in range(4000)
]

_scores = {f"agent-{i:04d}": round(random.random(), 4) for i in range(2000)}
_board_a = sorted(_scores.items(), key=lambda kv: (-kv[1], kv[0]))
_board_b = list(reversed(_board_a))

_actions = [{"files": [f"f{j}.py" for j in range(i % 7)]} for i in range(4000)]

_profile_a = {f"axis-{i}": random.random() for i in range(500)}
_profile_b = {f"axis-{i}": random.random() for i in range(500)}

BENCH = {
    "pass_rate": (_tasks,),
    "score_excluding_easy": (_tasks, 10),
    "rank_agents": (_scores,),
    "rank_changes": (_board_a, _board_b),
    "scaffold_delta": (43.2, 59.8),
    "simulate_scaffold": (10_000, 7),
    "blast_radius": (_actions,),
    "compare_axes": (_profile_a, _profile_b),
}
