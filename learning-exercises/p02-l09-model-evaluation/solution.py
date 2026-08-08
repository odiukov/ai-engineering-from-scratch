"""
Оценка моделей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def confusion_matrix(y_true, y_pred):
    """Матрица ошибок бинарной задачи. Возвращает кортеж (tp, tn, fp, fn).

    confusion_matrix([1, 1, 0, 0], [1, 0, 0, 0])  ->  (1, 2, 0, 1)
    confusion_matrix([1, 0], [1, 0])              ->  (1, 1, 0, 0)

    tp — предсказали 1 и угадали, tn — предсказали 0 и угадали,
    fp — ложная тревога, fn — пропущенная цель.

    Порядок именно такой: (tp, tn, fp, fn). Перепутать fp и fn — значит поменять
    местами precision и recall, а это разные решения в проде: ложная тревога в
    спам-фильтре и пропущенная опухоль стоят по-разному.
    """
    tp = tn = fp = fn = 0
    # один проход вместо четырёх генераторов: матрица ошибок считается
    # на каждом фолде кросс-валидации, лишние проходы по данным тут не нужны
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    """Доля правильных ответов: (tp + tn) / всего.

    accuracy([1, 1, 0, 0], [1, 0, 0, 0])  ->  0.75
    accuracy([0] * 100, [0] * 100)        ->  1.0

    Ловушка всего урока: при дисбалансе классов accuracy врёт. Если 95%
    объектов нулевые, модель «всегда 0» получит 0.95 и будет бесполезна.
    """
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total else 0.0


def precision_recall_f1(y_true, y_pred):
    """Три метрики одним кортежем: (precision, recall, f1).

    precision_recall_f1([1, 1, 0, 0], [1, 0, 0, 0])  ->  (1.0, 0.5, 0.666...)
    precision_recall_f1([1, 1], [0, 0])              ->  (0.0, 0.0, 0.0)

    precision = tp / (tp + fp) — какая доля тревог была настоящей.
    recall    = tp / (tp + fn) — какую долю целей поймали.
    f1        = гармоническое среднее, наказывает за перекос в любую сторону.

    Ловушка: у модели, которая никогда не говорит «1», знаменатель precision
    равен нулю. Ноль вместо ZeroDivisionError — общее соглашение (так же ведёт
    себя sklearn с zero_division=0).
    """
    tp, _, fp, fn = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # гармоническое среднее: обычное среднее дало бы 0.5 паре (1.0, 0.0),
    # а такая модель бесполезна — f1 честно вернёт 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def auc_roc(y_true, y_scores):
    """Площадь под ROC-кривой по непрерывным скорам. 1.0 — идеал, 0.5 — монетка.

    auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])  ->  1.0
    auc_roc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])  ->  0.0
    auc_roc([0, 1], [0.5, 0.5])                  ->  0.5

    Перебираем пороги по убыванию скора, на каждом считаем (fpr, tpr) и берём
    площадь методом трапеций.

    Ловушки: кривая обязана начинаться в точке (0, 0) — без неё случай
    «все скоры одинаковы» даст 0 вместо честных 0.5. И если в выборке нет
    объектов одного из классов, AUC не определён — возвращаем 0.5.

    Главное свойство: AUC не зависит от порога и от монотонного преобразования
    скоров. Она меряет ранжирование, а не калибровку.
    """
    positives = sum(1 for t in y_true if t == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    points = [(0.0, 0.0)]
    for threshold in sorted(set(y_scores), reverse=True):
        tp = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s >= threshold)
        fp = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= threshold)
        points.append((fp / negatives, tp / positives))

    # пороги идут по убыванию, поэтому tp и fp только растут и точки уже
    # упорядочены по fpr — сортировка тут страховка, а не необходимость
    area = 0.0
    for (fpr_a, tpr_a), (fpr_b, tpr_b) in zip(points, points[1:]):
        area += (fpr_b - fpr_a) * (tpr_a + tpr_b) / 2
    return area


def regression_metrics(y_true, y_pred):
    """Четыре метрики регрессии словарём: mse, rmse, mae, r2.

    regression_metrics([1, 2, 3], [1, 2, 3])  ->  mse 0.0, r2 1.0
    regression_metrics([0, 10], [5, 5])       ->  mse 25.0, rmse 5.0, mae 5.0

    mse наказывает за большие промахи квадратично, mae — линейно, поэтому один
    выброс раздувает mse и почти не трогает mae. rmse — тот же mse, но в
    единицах целевой переменной.

    r2 = 1 - ss_res / ss_tot. Единица — идеал, ноль — «не лучше, чем всегда
    предсказывать среднее», отрицательное значение — хуже среднего.

    Ловушка: если все истинные значения одинаковы, ss_tot равен нулю и r2 не
    определён — договорились возвращать 0.0.
    """
    n = len(y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    mse = ss_res / n

    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)

    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n,
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
    }


def kfold_split(n, k=5, seed=42):
    """Разбиение индексов 0..n-1 на k фолдов. Список пар (train_idx, val_idx).

    kfold_split(4, k=2, seed=0)  ->  [(train, val), (train, val)], по 2 индекса в val
    kfold_split(10, k=5)         ->  5 пар, в каждой val длиной 2

    Каждый индекс попадает в валидацию ровно один раз — в этом весь смысл
    кросс-валидации против одного случайного сплита.

    Ловушка с остатком: 10 объектов на 3 фолда не делятся. Через divmod
    получаем базовый размер и остаток, затем добавляем по одному объекту в
    первые фолды: размеры будут [4, 3, 3], а не [3, 3, 4].

    seed обязателен: сравнивать две модели можно только на одинаковых фолдах.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    base_size, remainder = divmod(n, k)
    folds = []
    for i in range(k):
        start = i * base_size + min(i, remainder)
        end = start + base_size + (1 if i < remainder else 0)
        val = indices[start:end]
        folds.append((indices[:start] + indices[end:], val))
    return folds


def stratified_kfold_split(y, k=5, seed=42):
    """То же разбиение, но доля классов в каждом фолде как в целом наборе.

    stratified_kfold_split([0, 0, 0, 0, 1, 1], k=2)  ->  в каждом val два нуля и одна единица
    stratified_kfold_split([1, 1, 0, 0], k=2)        ->  2 пары (train, val)

    Режем каждый класс отдельно и раскладываем куски по фолдам.

    Зачем: при 5% положительных обычный kfold легко отдаст фолду ноль
    положительных объектов. Recall на таком фолде не определён, а среднее по
    фолдам превращается в шум.
    """
    rng = random.Random(seed)

    by_class = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)

    val_parts = [[] for _ in range(k)]
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        m = len(indices)
        for i in range(k):
            val_parts[i].extend(indices[i * m // k:(i + 1) * m // k])

    all_indices = set(range(len(y)))
    return [(sorted(all_indices - set(val)), sorted(val)) for val in val_parts]


def cross_val_score(X, y, fit_fn, predict_fn, metric_fn=None, k=5, seed=42, stratified=False):
    """Кросс-валидация: список из k оценок, по одной на фолд.

    fit_fn(X_train, y_train) -> модель (любой объект),
    predict_fn(model, x) -> предсказание для одного объекта,
    metric_fn(y_true, y_pred) -> число; по умолчанию accuracy.

    cross_val_score(X, y, fit, predict, k=2)  ->  [0.8, 0.8]
    len(cross_val_score(X, y, fit, predict, k=5)) == 5

    Ловушка: модель обучается ЗАНОВО на каждом фолде. Обучить один раз на всех
    данных и мерить по фолдам — это утечка, оценка получится завышенной.
    """
    folds = stratified_kfold_split(y, k, seed) if stratified else kfold_split(len(X), k, seed)
    metric = metric_fn or accuracy

    scores = []
    for train_idx, val_idx in folds:
        model = fit_fn([X[i] for i in train_idx], [y[i] for i in train_idx])
        predictions = [predict_fn(model, X[i]) for i in val_idx]
        scores.append(metric([y[i] for i in val_idx], predictions))
    return scores
