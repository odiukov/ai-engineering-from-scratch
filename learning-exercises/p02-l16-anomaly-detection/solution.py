"""
Поиск аномалий — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def zscore_flags(rows, threshold=3.0):
    """Пометить строки, где хоть один признак дальше threshold сигм от среднего.

    zscore_flags([[0.0], [0.0], [0.0], [10.0]], 1.5)  ->  [False, False, False, True]

    Среднее и отклонение считаются по столбцам, отклонение по всей выборке
    (делим на n). Строка помечается, если ЛЮБОЙ её признак вышел за порог.

    Ловушка: константный столбец даёт std = 0 и деление на ноль. Замени такой
    std на 1.0 — все z станут нулями, и столбец просто перестанет голосовать.

    Помни, чем метод плох: сами выбросы входят в выборку, раздувают std и тем
    самым прячут себя. Несколько одинаковых аномалий z-score уже не увидит.
    """
    n = len(rows)
    n_cols = len(rows[0])
    means, stds = [], []
    for j in range(n_cols):
        col = [row[j] for row in rows]
        m = sum(col) / n
        means.append(m)
        # 0.0 -> 1.0: иначе константный столбец обрушит весь прогон
        stds.append((sum((x - m) ** 2 for x in col) / n) ** 0.5 or 1.0)
    return [
        any(abs(row[j] - means[j]) / stds[j] > threshold for j in range(n_cols))
        for row in rows
    ]


def percentile(values, p):
    """Перцентиль p (0..100) с линейной интерполяцией между соседями.

    percentile([1, 2, 3, 4], 25)   ->  1.75
    percentile([1, 2, 3], 50)      ->  2.0
    percentile([1, 2, 3, 4], 100)  ->  4.0

    Позиция считается как p/100 * (n-1) по ОТСОРТИРОВАННОМУ списку. Дробная
    часть — вес правого соседа.

    Ловушка: сортировать надо копию. Перцентиль не должен переставлять
    значения у того, кто его вызвал.
    """
    ordered = sorted(values)  # копия: вход остаётся нетронутым
    pos = (p / 100) * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def iqr_bounds(values, factor=1.5):
    """Границы «нормального» по межквартильному размаху: (нижняя, верхняя).

    iqr_bounds([1, 2, 3, 4])  ->  (-0.5, 5.5)

    Q1 — 25-й перцентиль, Q3 — 75-й, IQR = Q3 - Q1.
    Границы: Q1 - factor*IQR и Q3 + factor*IQR. factor=1.5 — это усы боксплота.

    Нулевой IQR — честный вырожденный случай Tukey: обе границы совпадают с
    квартилями, и любое отличие от центрального значения считается выбросом.
    Нельзя подставлять искусственную единицу: её масштаб произволен, поэтому
    одинаковая форма данных в метрах и миллиметрах дала бы разные флаги.

    Метод устойчив там, где z-score слепнет: перцентили не двигаются от того,
    что один выброс стал в сто раз больше.
    """
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def iqr_flags(rows, factor=1.5):
    """Пометить строки, где хоть один признак вышел за IQR-границы своего столбца.

    iqr_flags([[1.0], [2.0], [3.0], [4.0], [100.0]])  ->  [F, F, F, F, True]

    Границы считаются по столбцам независимо. Отсюда и слабость метода:
    точка, нормальная по каждому признаку по отдельности, но невозможная
    в их сочетании, останется незамеченной.
    """
    n_cols = len(rows[0])
    bounds = [iqr_bounds([row[j] for row in rows], factor) for j in range(n_cols)]
    return [
        any(row[j] < bounds[j][0] or row[j] > bounds[j][1] for j in range(n_cols))
        for row in rows
    ]


def expected_path_length(n):
    """Средняя глубина листа в случайном бинарном дереве из n точек.

    expected_path_length(1)  ->  0.0
    expected_path_length(2)  ->  1.0
    expected_path_length(256)  ->  примерно 10.24

    Формула: 2 * (ln(n-1) + gamma) - 2*(n-1)/n, где gamma = 0.5772156649.
    Для n <= 1 длины пути нет (0.0), для n = 2 формула-приближение врёт,
    точное значение равно 1.0 — зашей его отдельной веткой.

    Зачем: этой величиной нормируют глубину изоляции, иначе оценки на
    выборках разного размера несравнимы.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # приближение гармонического числа H(n-1) ~ ln(n-1) + gamma
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


def build_isolation_tree(rows, max_depth, rng):
    """Одно случайное изолирующее дерево. rng — экземпляр random.Random.

    Узел: {"feature": j, "threshold": t, "left": ..., "right": ...}
    Лист:  {"size": сколько точек в него попало}

    Лист получается, когда max_depth исчерпан, точек не больше одной или все
    значения выбранного признака совпали (делить нечего).

    Признак берётся случайный (rng.randrange), порог — случайное число между
    минимумом и максимумом этого признака (rng.uniform). Влево уходит строго
    меньшее порога.

    Никакой цели тут нет и не должно быть: дерево не учится отличать классы,
    оно просто режет пространство. Аномалии живут в пустоте и отрезаются
    первыми же разрезами.
    """
    if max_depth <= 0 or len(rows) <= 1:
        return {"size": len(rows)}
    feature = rng.randrange(len(rows[0]))
    column = [row[feature] for row in rows]
    low, high = min(column), max(column)
    if low == high:
        # резать по константному признаку бессмысленно: всё уйдёт в одну ветку
        return {"size": len(rows)}
    threshold = rng.uniform(low, high)
    left = [row for row in rows if row[feature] < threshold]
    right = [row for row in rows if row[feature] >= threshold]
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_isolation_tree(left, max_depth - 1, rng),
        "right": build_isolation_tree(right, max_depth - 1, rng),
    }


def path_length(tree, point):
    """Глубина изоляции точки: число разрезов плюс поправка на размер листа.

    path_length({"size": 1}, [0.0])  ->  0.0
    path_length({"feature": 0, "threshold": 5.0,
                 "left": {"size": 1}, "right": {"size": 8}}, [1.0])  ->  1.0

    Спускаемся по дереву, считая шаги. В листе прибавляем
    expected_path_length(размер листа): дерево обрезано по глубине, и внутри
    листа точку пришлось бы изолировать ещё какое-то число разрезов.

    Короткий путь = точку легко отрезать = аномалия.
    """
    depth = 0
    node = tree
    # спуск циклом, а не рекурсией: глубина небольшая, но стек тут ни к чему
    while "feature" in node:
        if point[node["feature"]] < node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
        depth += 1
    return depth + expected_path_length(node["size"])


def isolation_scores(rows, n_trees=50, max_samples=32, seed=0):
    """Оценка аномальности каждой строки лесом изолирующих деревьев.

    Вернуть список чисел из (0, 1): ближе к 1 — аномальнее, около 0.5 — обычно.

    isolation_scores([[0.0], [0.1], [0.2], [50.0]])[3] — самая большая из четырёх

    Каждое дерево строится на подвыборке из min(max_samples, len(rows)) строк
    (rng.sample), глубина ограничена ceil(log2(размер подвыборки)). Оценка:
    2 ** (-средняя глубина / expected_path_length(размер подвыборки)).

    Один seed — один и тот же результат: без этого отладить детектор нельзя.
    Подвыборка не роскошь, а суть метода: маленькие деревья лучше изолируют
    редкое и считаются на порядок быстрее.
    """
    rng = random.Random(seed)
    size = min(max_samples, len(rows))
    max_depth = max(1, math.ceil(math.log2(size))) if size > 1 else 1
    trees = [
        build_isolation_tree(rng.sample(rows, size), max_depth, rng)
        for _ in range(n_trees)
    ]
    norm = expected_path_length(size)
    if norm == 0:
        return [0.5] * len(rows)
    return [
        2.0 ** (-sum(path_length(t, row) for t in trees) / n_trees / norm)
        for row in rows
    ]
