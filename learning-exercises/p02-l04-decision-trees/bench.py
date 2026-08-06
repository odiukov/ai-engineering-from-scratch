"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_labels = [random.randint(0, 3) for _ in range(20000)]

_N, _D = 200, 4
_X = [[round(random.uniform(0, 10), 2) for _ in range(_D)] for _ in range(_N)]
_y = [1 if row[0] + row[1] > 10 else 0 for row in _X]

_values = [round(random.uniform(0, 100), 2) for _ in range(_N)]

_left = _labels[:9000]
_right = _labels[9000:]


def _node(feature, threshold, left, right):
    return {"leaf": False, "feature": feature, "threshold": threshold,
            "left": left, "right": right}


def _leaf(value):
    return {"leaf": True, "value": value}


# деревья собраны литералами, а не через build_tree: bench.py обязан
# импортироваться и тогда, когда exercise.py ещё пустая заготовка
_tree4 = _node(0, 5.0,
               _node(1, 5.0, _leaf(0), _node(2, 5.0, _leaf(0), _leaf(1))),
               _node(1, 5.0, _node(3, 5.0, _leaf(0), _leaf(1)), _leaf(1)))
_forest = [
    _tree4,
    _node(1, 4.0, _leaf(0), _node(0, 6.0, _leaf(0), _leaf(1))),
    _node(0, 6.0, _node(1, 3.0, _leaf(0), _leaf(1)), _leaf(1)),
    _node(1, 6.0, _leaf(0), _leaf(1)),
    _node(0, 4.0, _leaf(0), _node(1, 7.0, _leaf(1), _leaf(0))),
]


BENCH = {
    "gini_impurity": (_labels,),
    "entropy": (_labels,),
    "information_gain": (_labels, _left, _right, "gini"),
    "split_dataset": (_X, _y, 0, 5.0),
    "best_split": (_X, _y),
    "build_tree": (_X, _y, 4),
    "tree_predict": ({"leaf": False, "feature": 0, "threshold": 5.0,
                      "left": {"leaf": True, "value": 0},
                      "right": {"leaf": True, "value": 1}}, _X),
    "bootstrap_sample": (_X, _y, 0),
    # лес держим маленьким: n_trees x build_tree и так самый тяжёлый вызов
    "fit_random_forest": (_X, _y, 5, 4, 2, 0),
    "forest_predict": (_forest, _X),
    "feature_importance": (_tree4, _X, _y),
    "variance_reduction": (_values, _values[:100], _values[100:]),
    "build_regression_tree": (_X, _values, 4),
    # лес держим маленьким: n_trees x build_tree и так самый тяжёлый вызов
    "fit_random_forest": (_X, _y, 5, 4, 2, 0),
    "forest_predict": (_forest, _X),
    "feature_importance": (_tree4, _X, _y),
    "variance_reduction": (_values, _values[:100], _values[100:]),
    "build_regression_tree": (_X, _values, 4),
}
