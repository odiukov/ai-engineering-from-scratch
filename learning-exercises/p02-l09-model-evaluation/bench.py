"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_n = 2000
_y_true = [1 if random.random() < 0.3 else 0 for _ in range(_n)]
_y_pred = [t if random.random() < 0.85 else 1 - t for t in _y_true]
_scores = [random.random() * 0.5 + 0.5 * t for t in _y_true]
_reg_true = [random.gauss(50, 10) for _ in range(_n)]
_reg_pred = [v + random.gauss(0, 3) for v in _reg_true]

_X = [[random.random(), random.random()] for _ in range(400)]
_y = [1 if x[0] + x[1] > 1 else 0 for x in _X]
_fit = lambda x_train, y_train: max(set(y_train), key=y_train.count)
_predict = lambda model, x: model

BENCH = {
    "confusion_matrix": (_y_true, _y_pred),
    "accuracy": (_y_true, _y_pred),
    "precision_recall_f1": (_y_true, _y_pred),
    # AUC перебирает пороги: на 2000 уникальных скоров это заметная работа
    "auc_roc": (_y_true[:600], _scores[:600]),
    "regression_metrics": (_reg_true, _reg_pred),
    "kfold_split": (_n, 5),
    "stratified_kfold_split": (_y_true, 5),
    "cross_val_score": (_X, _y, _fit, _predict),
}
