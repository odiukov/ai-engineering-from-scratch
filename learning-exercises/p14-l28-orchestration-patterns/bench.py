"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_TEMPLATES = (
    "I want a refund for order {}",
    "the app keeps crashing with an error on save, ticket {}",
    "what is the price of the team plan for {} seats",
    "добрый день, вопрос номер {}",
)
_tasks = [random.choice(_TEMPLATES).format(random.randint(1, 9999)) for _ in range(300)]
_log = [random.choice(("billing", "support", "sales")) for _ in range(2000)]
_order = list(range(len(_tasks)))
random.shuffle(_order)

BENCH = {
    "classify": (_tasks[0],),
    "supervisor_route": (_tasks,),
    "swarm_route": (_tasks,),
    "detect_bouncing": (_log,),
    "hierarchical_route": (_tasks,),
    "run_parallel": (_tasks, classify_stub := str.strip, _order),
    "pick_pattern": (4, True, False, True),
    "compare_patterns": (_tasks,),
}
