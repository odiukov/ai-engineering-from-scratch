"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 4000 точек: пересчёт среднего внутри цикла вместо одного прохода
# превращает линейную функцию в квадратичную, и это сразу видно
_values = [random.gauss(0.0, 1.0) for _ in range(4000)]
_other = [random.gauss(0.2, 1.0) for _ in range(4000)]
# для бутстрэпа выборка меньше: 200 итераций по 300 точек — десятки мс
_small = [random.gauss(0.0, 1.0) for _ in range(300)]

BENCH = {
    "mean": (_values,),
    "percentile": (_values, 95),
    "variance": (_values,),
    "pearson": (_values, _other),
    "spearman": (_values, _other),
    "welch_t": (_values, _other),
    "cohens_d": (_values, _other),
    "bootstrap_ci": (_small,),
}
