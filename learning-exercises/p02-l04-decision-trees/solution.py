"""
Решающие деревья — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def gini_impurity(labels):
    """Загрязнённость Джини: 1 - сумма(доля_класса^2).

    gini_impurity([1, 1, 1])        ->  0.0    (чистый узел)
    gini_impurity([0, 1])           ->  0.5    (максимум для двух классов)
    gini_impurity([0]*6 + [1]*4)    ->  0.48
    gini_impurity([])               ->  0.0

    Смысл: вероятность ошибиться, если наугад взять объект из узла и наугад
    приписать ему класс по здешнему распределению.

    Ловушка: пустой список. Делить на len(labels) нельзя — верни 0.0.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def entropy(labels):
    """Энтропия узла в битах: -сумма(доля * log2(доля)).

    entropy([1, 1, 1])          ->  0.0   (чистый узел)
    entropy([0, 1])             ->  1.0   (максимум для двух классов)
    entropy([0, 1, 2, 3])       ->  2.0   (максимум для четырёх)
    entropy([])                 ->  0.0

    Ловушка: log2(0) бросает ValueError. Классов с нулевым счётчиком в
    словаре быть не должно, но пустой список обработать надо — верни 0.0.

    Джини и энтропия почти всегда выбирают одно и то же разбиение. Джини
    дешевле (нет логарифма), поэтому он и стоит по умолчанию.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def information_gain(parent, left, right, criterion="gini"):
    """Насколько разбиение снизило загрязнённость.

    Загрязнённость родителя минус ВЗВЕШЕННОЕ среднее по детям, где вес —
    доля объектов, попавших в ребёнка. criterion выбирает меру: "gini" или
    "entropy".

    information_gain([0, 0, 1, 1], [0, 0], [1, 1])        ->  0.5
    information_gain([0, 0, 1, 1], [0, 1], [0, 1])        ->  0.0
    information_gain([0, 1], [], [0, 1])                  ->  0.0

    Ловушка: среднее именно взвешенное. Простое среднее по детям завышает
    выгоду разбиения, которое отрезало один-единственный объект.

    Если один из детей пустой, разбиения на самом деле не произошло —
    верни 0.0.
    """
    measure = gini_impurity if criterion == "gini" else entropy
    n, n_left, n_right = len(parent), len(left), len(right)
    if n_left == 0 or n_right == 0:
        return 0.0
    children = (n_left / n) * measure(left) + (n_right / n) * measure(right)
    return measure(parent) - children


def split_dataset(X, y, feature, threshold):
    """Разрезать выборку по условию X[i][feature] <= threshold.

    Вернуть четвёрку (left_X, left_y, right_X, right_y). Объекты с
    значением РОВНО threshold уходят налево.

    split_dataset([[1.0], [3.0]], [0, 1], 0, 2.0)
        ->  ([[1.0]], [0], [[3.0]], [1])

    Ловушка: строгое < вместо <= переносит граничные объекты направо, и
    порог, найденный best_split, перестаёт делить так, как задумано.
    """
    left_X, left_y, right_X, right_y = [], [], [], []
    for row, label in zip(X, y):
        if row[feature] <= threshold:
            left_X.append(row)
            left_y.append(label)
        else:
            right_X.append(row)
            right_y.append(label)
    return left_X, left_y, right_X, right_y


def best_split(X, y, criterion="gini"):
    """Перебрать все признаки и все пороги, вернуть лучший.

    Вернуть тройку (feature, threshold, gain). Пороги-кандидаты — середины
    между соседними РАЗНЫМИ значениями признака: (v_i + v_{i+1}) / 2.

    best_split([[1.0], [2.0], [5.0], [6.0]], [0, 0, 1, 1])
        ->  (0, 3.5, 0.5)

    Если разбить нечего (все объекты одинаковы, либо ни одно разбиение не
    даёт выигрыша), вернуть (None, None, 0.0).

    Ловушка: перебирать значения самого признака в качестве порогов —
    ошибка. Порог, равный максимальному значению, отправит всё налево и
    даст пустого правого ребёнка.

    Жадность: выбирается лучший ЛОКАЛЬНЫЙ шаг. Оптимальное дерево искать
    NP-трудно, а жадное на практике работает.
    """
    best = (None, None, 0.0)
    for feature in range(len(X[0])):
        values = sorted({row[feature] for row in X})
        # середины между соседними значениями: любой такой порог реально
        # делит выборку на две непустые части
        for lo, hi in zip(values, values[1:]):
            threshold = (lo + hi) / 2.0
            _, left_y, _, right_y = split_dataset(X, y, feature, threshold)
            gain = information_gain(y, left_y, right_y, criterion)
            if gain > best[2]:
                best = (feature, threshold, gain)
    return best


def build_tree(X, y, max_depth=None, min_samples_split=2, depth=0):
    """Построить дерево рекурсивно. Вернуть узел-словарь.

    Формат узлов ровно такой:
      лист       {"leaf": True,  "value": метка}
      развилка   {"leaf": False, "feature": i, "threshold": t,
                  "left": узел, "right": узел}

    Остановка (любое из условий — сразу лист):
      * все метки в узле одинаковы;
      * достигнут max_depth (если он не None);
      * объектов меньше min_samples_split;
      * best_split не нашёл разбиения с положительным выигрышем.

    Метка листа — самый частый класс в узле; при ничьей меньшая метка,
    иначе ответ зависит от порядка словаря.

    build_tree([[1.0], [2.0]], [0, 1])
        ->  {"leaf": False, "feature": 0, "threshold": 1.5,
             "left": {"leaf": True, "value": 0},
             "right": {"leaf": True, "value": 1}}

    Ловушка: без max_depth дерево дорастает до листа на каждый объект. Это
    не обучение, это запоминание — на новых данных такое дерево бесполезно.
    """
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    # sorted даёт детерминированный обход, max берёт первый из максимальных —
    # значит при ничьей выигрывает меньшая метка
    leaf = {"leaf": True, "value": max(sorted(counts), key=counts.get)}

    if len(counts) == 1:
        return leaf
    if (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
        return leaf

    feature, threshold, gain = best_split(X, y)
    if feature is None or gain <= 0:
        return leaf

    left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(left_X, left_y, max_depth, min_samples_split, depth + 1),
        "right": build_tree(right_X, right_y, max_depth, min_samples_split, depth + 1),
    }


def tree_predict(tree, X):
    """Прогнать каждый объект по дереву от корня до листа.

    Вернуть список меток той же длины, что X.

    В развилке идём налево, если x[feature] <= threshold — то же правило,
    что и в split_dataset. Разойдутся правила — дерево начнёт врать на
    граничных значениях.

    tree_predict({"leaf": True, "value": 7}, [[0.0], [1.0]])  ->  [7, 7]

    Предсказание стоит O(глубина) — по одному сравнению на уровень. Именно
    поэтому деревья быстры на инференсе даже на больших выборках.
    """
    labels = []
    for x in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        labels.append(node["value"])
    return labels


def bootstrap_sample(X, y, seed=0):
    """Выборка с возвращением того же размера — основа бэггинга.

    Вернуть пару (X_boot, y_boot). Каждый из len(X) раз берём случайный
    индекс от 0 до len(X)-1, повторы разрешены и обязательны.

    bootstrap_sample([[1.0], [2.0]], [0, 1], seed=0)  ->  два объекта,
        какие-то из исходных двух, возможно один и тот же дважды

    Ловушка: пары (объект, метка) обязаны ехать вместе — берётся ОДИН
    индекс на оба списка. И генератор свой, random.Random(seed), а не
    глобальный random.seed.

    Зачем: каждое дерево леса учится на своей бутстрап-выборке, деревья
    получаются разными, а среднее по разным деревьям снижает дисперсию.
    В выборку попадает примерно 63% исходных объектов, остальные
    (out-of-bag) можно использовать для честной проверки.
    """
    rng = random.Random(seed)
    n = len(X)
    indices = [rng.randrange(n) for _ in range(n)]
    return [X[i] for i in indices], [y[i] for i in indices]
