"""Входные данные для замера скорости."""

import math
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# Договорённость и запрос по всем осям доступа сразу.
_AGREEMENT = ["model_api", "prerelease_checkpoint", "task_scaffold"]
_REQUESTED = ["model_api", "chain_of_thought", "model_weights", "training_data"]

# Полный прогон по HCAST: 189 задач по десять попыток — типичный размер
# сырого результата, из которого строится кривая.
_HOURS = [math.exp(math.log(1.0 / 60.0) + (math.log(8.0) - math.log(1.0 / 60.0))
                   * i / 188.0) for i in range(189)]
_RESULTS = [
    (h, _rng.random() < 1.0 / (1.0 + (h / 2.0) ** 1.5))
    for h in _HOURS
    for _ in range(10)
]

# Длинная кривая: интерполяция обязана находить пересечение за один проход,
# а не сортировать и перебирать заново на каждый уровень надёжности.
_CURVE = [
    (math.exp(0.05 * i), 1.0 - i / 200.0)
    for i in range(1, 200)
]

BENCH = {
    "resolve_access": (_AGREEMENT, _REQUESTED),
    "sample_tasks": ("HCAST", 150, random.Random(1)),
    "run_manifest": ("HCAST", 150, 1, _AGREEMENT, _REQUESTED),
    "success_curve": (_RESULTS,),
    "horizon_at": (_CURVE, 0.5),
    "inject_gaming": (_RESULTS, 0.3, random.Random(2)),
    "doubling_time_days": (7.0, 14.0, 130.8),
    "deployment_gap": (14.0, 8.0, ["idealized_tooling", "user_variance",
                                   "no_real_consequences", "eval_context_gaming"]),
}
