"""
ML-пайплайны — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json
import statistics


def column_medians(rows):
    """Медиана каждого столбца, пропуская None.

    column_medians([[1, 10], [3, None], [5, 30]])  ->  [3.0, 20.0]
    column_medians([[None, 1], [None, 3]])         ->  [0.0, 2.0]

    Медиана чётного числа значений — среднее двух серединных.
    Ловушка: если в столбце вообще нет чисел, медианы не существует —
    верни 0.0, иначе следующий шаг пайплайна упадёт на None.

    Это fit-часть импьютера: статистика, которую потом применят к новым данным.
    """
    n_cols = len(rows[0])
    medians = []
    for j in range(n_cols):
        # None выкидываем ДО подсчёта: median([1, None]) сравнивать не умеет
        known = [row[j] for row in rows if row[j] is not None]
        medians.append(float(statistics.median(known)) if known else 0.0)
    return medians


def fit_step(kind, rows):
    """Обучить один шаг пайплайна и вернуть его состояние (dict).

    fit_step("impute", [[1, 10], [3, None]])  ->  {"medians": [2.0, 10.0]}
    fit_step("scale", [[0.0], [2.0]])         ->  {"means": [1.0], "stds": [1.0]}

    Поддерживаются два вида: "impute" (медианы столбцов) и "scale"
    (средние и стандартные отклонения столбцов, отклонение по всей выборке —
    делим на n, не на n-1). На незнакомый kind брось ValueError.

    Ловушка: нулевое стандартное отклонение. Константный столбец даст std = 0
    и деление на ноль при transform. Замени такой std на 1.0 — тогда столбец
    просто станет нулевым, а не сломает прогон.
    """
    if kind == "impute":
        return {"medians": column_medians(rows)}
    if kind == "scale":
        n = len(rows)
        means, stds = [], []
        for j in range(len(rows[0])):
            col = [row[j] for row in rows]
            m = sum(col) / n
            var = sum((x - m) ** 2 for x in col) / n
            means.append(m)
            # 1.0 вместо 0.0 — единственный способ не делить на ноль,
            # сохранив форму данных: константный столбец уедет в нули
            stds.append(var ** 0.5 or 1.0)
        return {"means": means, "stds": stds}
    raise ValueError(f"неизвестный шаг: {kind}")


def apply_step(kind, state, rows):
    """Применить уже обученный шаг к данным. Новые строки, вход не портим.

    apply_step("impute", {"medians": [2.0]}, [[None], [5]])       ->  [[2.0], [5]]
    apply_step("scale", {"means": [1.0], "stds": [2.0]}, [[3.0]])  ->  [[1.0]]

    Здесь НИЧЕГО не считается заново по rows — только берётся state. В этом
    вся суть: на проде статистики те же, что были на обучении.

    На незнакомый kind — ValueError.
    """
    if kind == "impute":
        med = state["medians"]
        return [
            [med[j] if v is None else v for j, v in enumerate(row)] for row in rows
        ]
    if kind == "scale":
        means, stds = state["means"], state["stds"]
        return [
            [(v - means[j]) / stds[j] for j, v in enumerate(row)] for row in rows
        ]
    raise ValueError(f"неизвестный шаг: {kind}")


def fit_pipeline(steps, rows):
    """Обучить цепочку шагов. Вернуть список [{"kind": ..., "state": ...}, ...].

    fit_pipeline(["impute", "scale"], [[1], [None], [3]])
        ->  [{"kind": "impute", "state": {...}}, {"kind": "scale", "state": {...}}]

    Каждый следующий шаг обучается на ВЫХОДЕ предыдущего, а не на исходных
    rows. Иначе scale увидит None и посчитает статистики не по тем числам,
    которые реально придут ему на вход при transform.
    """
    fitted = []
    current = rows
    for kind in steps:
        state = fit_step(kind, current)
        fitted.append({"kind": kind, "state": state})
        # прогоняем данные дальше: следующий шаг учится на том, что увидит
        current = apply_step(kind, state, current)
    return fitted


def transform_pipeline(fitted, rows):
    """Прогнать данные через уже обученный пайплайн.

    transform_pipeline(fit_pipeline(["scale"], [[0.0], [2.0]]), [[4.0]])  ->  [[3.0]]

    Ловушка ровно одна, зато смертельная: здесь нельзя ничего доучивать.
    Соблазн пересчитать среднее по пришедшим rows — это и есть утечка данных:
    на трёх одинаковых тестовых строках "заново обученный" scale выдаст нули,
    и модель в проде получит совсем не то, на чём училась.
    """
    current = rows
    for step in fitted:
        current = apply_step(step["kind"], step["state"], current)
    return current


def fit_transform_split(steps, rows, n_train):
    """Обучить пайплайн ТОЛЬКО на первых n_train строках, применить к обеим частям.

    Вернуть кортеж (fitted, train_out, test_out).

    fit_transform_split(["scale"], [[0.0], [2.0], [100.0]], 2)
        ->  (fitted, [[-1.0], [1.0]], [[99.0]])

    Тест-часть участвует в transform и НЕ участвует в fit. Проверить легко:
    поменяй хвост rows после n_train — fitted обязан остаться прежним.

    Это дисциплина, ради которой пайплайны и придумали: одна функция, и
    подсмотреть в тест физически негде.
    """
    fitted = fit_pipeline(steps, rows[:n_train])
    # transform, а не fit_transform: обе части идут через одни и те же статистики
    return fitted, transform_pipeline(fitted, rows[:n_train]), transform_pipeline(
        fitted, rows[n_train:]
    )


def dump_pipeline(fitted):
    """Сериализовать обученный пайплайн в JSON-строку.

    dump_pipeline([{"kind": "impute", "state": {"medians": [2.0]}}])
        ->  '[{"kind": "impute", "state": {"medians": [2.0]}}]'

    Артефакт деплоя — это не код модели, а её состояние. Порядок шагов
    обязан сохраниться: JSON-массив упорядочен, dict — нет.
    """
    # sort_keys=True: строка становится детерминированной, её можно
    # сравнивать и хешировать между запусками
    return json.dumps(fitted, sort_keys=True)


def load_pipeline(text):
    """Восстановить пайплайн из JSON-строки, полученной от dump_pipeline.

    load_pipeline('[{"kind": "impute", "state": {"medians": [2.0]}}]')
        ->  [{"kind": "impute", "state": {"medians": [2.0]}}]

    Круговой прогон обязан совпадать: transform через восстановленный пайплайн
    даёт ровно те же числа, что и через исходный. Если не совпадает — в проде
    будет training/serving skew, самая тихая из всех ML-ошибок.
    """
    return json.loads(text)
