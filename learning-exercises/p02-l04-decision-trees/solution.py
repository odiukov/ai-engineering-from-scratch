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


def best_split(X, y, criterion="gini", features=None):
    """Перебрать все признаки и все пороги, вернуть лучший.

    Вернуть тройку (feature, threshold, gain). Пороги-кандидаты — середины
    между соседними РАЗНЫМИ значениями признака: (v_i + v_{i+1}) / 2.

    best_split([[1.0], [2.0], [5.0], [6.0]], [0, 0, 1, 1])
        ->  (0, 3.5, 0.5)

    Если разбить нечего (все объекты одинаковы, либо ни одно разбиение не
    даёт выигрыша), вернуть (None, None, 0.0).

    `features` — какие столбцы вообще рассматривать. None означает все.
    Это понадобится случайному лесу: там каждая развилка выбирает лучший
    признак не из всех, а из случайной горстки — именно это делает деревья
    в лесу непохожими друг на друга.

    Ловушка: перебирать значения самого признака в качестве порогов —
    ошибка. Порог, равный максимальному значению, отправит всё налево и
    даст пустого правого ребёнка.

    Жадность: выбирается лучший ЛОКАЛЬНЫЙ шаг. Оптимальное дерево искать
    NP-трудно, а жадное на практике работает.
    """
    best = (None, None, 0.0)
    if features is None:
        features = range(len(X[0]))
    for feature in features:
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


def build_tree(X, y, max_depth=None, min_samples_split=2, depth=0,
               max_features=None, rng=None):
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

    `max_features` и `rng` нужны только случайному лесу: если max_features
    задан, каждая развилка выбирает лучший признак из случайной горстки
    размера max_features, а не из всех. Без них дерево строится как обычно.

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

    features = None
    if max_features is not None:
        n = len(X[0])
        k = min(max_features, n)
        # sorted для детерминизма: sample отдаёт в произвольном порядке,
        # а от порядка перебора зависит, кто выиграет при равном gain
        features = sorted(rng.sample(range(n), k))

    feature, threshold, gain = best_split(X, y, features=features)
    if feature is None or gain <= 0:
        return leaf

    left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(left_X, left_y, max_depth, min_samples_split,
                           depth + 1, max_features, rng),
        "right": build_tree(right_X, right_y, max_depth, min_samples_split,
                            depth + 1, max_features, rng),
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


def fit_random_forest(X, y, n_trees=10, max_depth=None, max_features=None, seed=0):
    """Случайный лес: вырастить n_trees деревьев, каждое на своём бутстрапе.

    Вернуть список деревьев в формате build_tree.

    Два источника непохожести деревьев, и оба обязательны:
      * каждому дереву достаётся своя бутстрап-выборка (bootstrap_sample);
      * каждая развилка выбирает признак из случайной горстки max_features.

    Без второго деревья на похожих данных вырастают почти одинаковыми, и
    голосование ничего не добавляет — все ошибаются в одних и тех же местах.

    max_features=None означает "все признаки", то есть просто бэггинг.
    Классический выбор для классификации — корень из числа признаков.

    Один rng на весь лес: дерево i получает bootstrap_sample с seed+i,
    а развилки тянут признаки из общего генератора.
    """
    rng = random.Random(seed)
    forest = []
    for i in range(n_trees):
        X_boot, y_boot = bootstrap_sample(X, y, seed=seed + i)
        forest.append(build_tree(X_boot, y_boot, max_depth=max_depth,
                                 max_features=max_features, rng=rng))
    return forest


def forest_predict(forest, X):
    """Предсказание леса: каждое дерево голосует, побеждает большинство.

    Вернуть список меток той же длины, что X.

    forest_predict([tree_a, tree_b, tree_c], X)

    При ничьей — меньшая метка, как и в листьях build_tree, иначе ответ
    зависит от порядка словаря.

    Почему это работает: одно дерево переобучается и ошибается уверенно,
    но ошибается по-своему. Усреднение независимых ошибок гасит разброс,
    оставляя смещение на месте. Это и есть bias-variance в действии.
    """
    per_tree = [tree_predict(tree, X) for tree in forest]
    labels = []
    for i in range(len(X)):
        counts = {}
        for preds in per_tree:
            counts[preds[i]] = counts.get(preds[i], 0) + 1
        labels.append(max(sorted(counts), key=counts.get))
    return labels


def feature_importance(tree, X, y, criterion="gini"):
    """Насколько каждый признак снизил загрязнённость. Список длины n_features.

    Важность признака — сумма его выигрышей по всем развилкам, где он
    использовался, ВЗВЕШЕННАЯ долей объектов, дошедших до той развилки.
    Развилка у корня решает судьбу всей выборки, развилка в глубине —
    судьбу горстки; считать их одинаково нельзя.

    Результат нормируется так, чтобы сумма равнялась 1.0. Если дерево —
    один лист, вернуть нули.

    feature_importance(tree_on_feature_0_only, X, y)  ->  [1.0, 0.0]

    Ровно так устроен .feature_importances_ в scikit-learn. И у метода есть
    известный перекос: признаки с большим числом уникальных значений
    выглядят важнее, потому что у них просто больше кандидатов в пороги.
    """
    n_features = len(X[0])
    scores = [0.0] * n_features
    total = len(y)

    def walk(node, node_X, node_y):
        if node["leaf"]:
            return
        left_X, left_y, right_X, right_y = split_dataset(
            node_X, node_y, node["feature"], node["threshold"])
        gain = information_gain(node_y, left_y, right_y, criterion)
        scores[node["feature"]] += gain * len(node_y) / total
        walk(node["left"], left_X, left_y)
        walk(node["right"], right_X, right_y)

    walk(tree, X, y)
    s = sum(scores)
    return [v / s for v in scores] if s > 0 else scores


def variance_reduction(parent, left, right):
    """Критерий для регрессии: насколько разбиение снизило разброс ответов.

    Аналог information_gain, но вместо Gini берётся дисперсия, а вместо
    меток — числа.

    variance_reduction([1.0, 1.0, 5.0, 5.0], [1.0, 1.0], [5.0, 5.0])  ->  4.0

    Формула та же: дисперсия родителя минус взвешенная сумма дисперсий
    детей. Пустой ребёнок даёт вклад 0.

    Классификация спрашивает "стали ли метки однороднее", регрессия —
    "стали ли числа кучнее". Механика одна.
    """
    def var(values):
        if not values:
            return 0.0
        m = sum(values) / len(values)
        return sum((v - m) ** 2 for v in values) / len(values)

    n = len(parent)
    if n == 0:
        return 0.0
    weighted = (len(left) * var(left) + len(right) * var(right)) / n
    return var(parent) - weighted


def build_regression_tree(X, y, max_depth=None, min_samples_split=2, depth=0):
    """Дерево, предсказывающее число. Формат узлов тот же, что у build_tree.

    Отличий от классификации ровно два:
      * критерий разбиения — variance_reduction, а не information_gain;
      * значение листа — СРЕДНЕЕ ответов в нём, а не самый частый класс.

    build_regression_tree([[1.0], [2.0]], [10.0, 20.0])
        ->  {"leaf": False, "feature": 0, "threshold": 1.5,
             "left": {"leaf": True, "value": 10.0},
             "right": {"leaf": True, "value": 20.0}}

    Предсказывать можно тем же tree_predict — он просто возвращает
    значение листа и не знает, метка это или число.

    Отсюда видно ограничение: дерево выдаёт ступеньки, а не гладкую линию.
    Экстраполировать за пределы обучающих данных оно не умеет вообще —
    любой x правее максимума попадёт в тот же крайний лист.
    """
    mean = sum(y) / len(y)
    leaf = {"leaf": True, "value": mean}

    if len(set(y)) == 1:
        return leaf
    if (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
        return leaf

    best = (None, None, 0.0)
    for feature in range(len(X[0])):
        values = sorted({row[feature] for row in X})
        for lo, hi in zip(values, values[1:]):
            threshold = (lo + hi) / 2.0
            _, left_y, _, right_y = split_dataset(X, y, feature, threshold)
            gain = variance_reduction(y, left_y, right_y)
            if gain > best[2]:
                best = (feature, threshold, gain)

    feature, threshold, gain = best
    if feature is None or gain <= 0:
        return leaf

    left_X, left_y, right_X, right_y = split_dataset(X, y, feature, threshold)
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "left": build_regression_tree(left_X, left_y, max_depth,
                                      min_samples_split, depth + 1),
        "right": build_regression_tree(right_X, right_y, max_depth,
                                       min_samples_split, depth + 1),
    }
