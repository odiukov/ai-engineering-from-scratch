"""
Несбалансированные данные — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def confusion_counts(y_true, y_pred):
    """Четыре числа матрицы ошибок: (tp, tn, fp, fn). Метки — 0 и 1.

    confusion_counts([1, 0, 1, 0], [1, 0, 0, 0])  ->  (1, 2, 0, 1)
    confusion_counts([1, 1], [0, 0])              ->  (0, 0, 0, 2)

    tp — угадали единицу, tn — угадали ноль, fp — ложная тревога,
    fn — пропущенная единица. Сумма всех четырёх равна длине выборки.

    Всё остальное в уроке считается из этих четырёх чисел, поэтому порядок
    в кортеже важен: перепутанные fp и fn дают зеркальную картину мира.
    """
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1:
            if t == 1:
                tp += 1
            else:
                fp += 1
        elif t == 1:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def precision_recall_f1(y_true, y_pred):
    """Точность, полнота и F1 одним кортежем: (precision, recall, f1).

    precision_recall_f1([1, 0, 1, 0], [1, 0, 0, 0])  ->  (1.0, 0.5, 0.666...)
    precision_recall_f1([1] + [0] * 99, [0] * 100)   ->  (0.0, 0.0, 0.0)

    precision = tp/(tp+fp), recall = tp/(tp+fn),
    f1 = 2*p*r/(p+r) — гармоническое среднее, а не обычное.

    Ловушка: любой из знаменателей может оказаться нулём. Модель, которая
    никогда не говорит "1", даёт tp+fp = 0 — верни 0.0, не падай.

    Второй пример — та самая модель с точностью 99%, которая не ловит ничего.
    F1 честно ставит ей ноль.
    """
    tp, _, fp, fn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def matthews_corrcoef(y_true, y_pred):
    """Коэффициент корреляции Мэтьюса: от -1 до +1.

    matthews_corrcoef([1, 0, 1, 0], [1, 0, 1, 0])   ->  1.0
    matthews_corrcoef([1, 0, 1, 0], [0, 1, 0, 1])   ->  -1.0
    matthews_corrcoef([1] + [0] * 99, [0] * 100)    ->  0.0

    (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).

    Ловушка: у постоянного предсказания одна из скобок равна нулю, и весь
    знаменатель схлопывается. Договорённость — вернуть 0.0.

    MCC хорош тем, что высокий балл требует успеха на ОБОИХ классах сразу.
    Именно поэтому его любят там, где классы различаются в сотни раз.
    """
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def class_weights(y):
    """Веса классов, обратные их частоте: {класс: n / (число классов * счётчик)}.

    class_weights([0] * 950 + [1] * 50)  ->  {0: 0.5263..., 1: 10.0}
    class_weights([0, 0, 1, 1])          ->  {0: 1.0, 1: 1.0}

    На сбалансированных данных все веса равны 1.0 — формула сама это даёт.

    Смысл: ошибка на редком классе становится настолько же дороже, насколько
    он реже. Это дешёвая замена оверсэмплингу — данных не прибавляется,
    а функция потерь ведёт себя так, будто прибавилось.
    """
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    n_classes = len(counts)
    return {c: len(y) / (n_classes * n) for c, n in counts.items()}


def k_nearest(points, index, k):
    """Индексы k ближайших к points[index] соседей. Сама точка не считается.

    k_nearest([[0.0], [1.0], [5.0]], 0, 1)  ->  [1]
    k_nearest([[0.0], [1.0], [5.0]], 0, 5)  ->  [1, 2]   (соседей всего два)

    Расстояние евклидово. При равенстве расстояний раньше идёт меньший индекс.
    Если k больше числа соседей — вернуть всех, сколько есть.

    Сравнивать можно квадраты расстояний: корень монотонен и порядок не меняет,
    зато на длинных векторах экономит вызов sqrt на каждую пару.
    """
    target = points[index]
    distances = []
    for i, other in enumerate(points):
        if i == index:
            continue
        d2 = sum((a - b) ** 2 for a, b in zip(target, other))
        distances.append((d2, i))
    distances.sort()  # кортеж сортируется по расстоянию, затем по индексу
    return [i for _, i in distances[:k]]


def smote(minority, n_synthetic, k=5, seed=0):
    """Сгенерировать n_synthetic синтетических точек редкого класса.

    Алгоритм на каждую точку: взять случайную точку x редкого класса, взять
    случайного из её k ближайших соседей, вернуть x + t*(сосед - x), t ~ U(0,1).

    smote([[0.0, 0.0], [1.0, 1.0]], 3, k=1, seed=0)
        ->  три точки на отрезке между (0,0) и (1,1)

    Это не копирование: новые точки лежат МЕЖДУ реальными, поэтому модель
    видит область, а не одну и ту же точку сто раз. Отсюда и меньший
    переобучающий эффект, чем у простого дублирования.

    k урезается до числа доступных соседей. Меньше двух точек — ValueError:
    интерполировать не с чем.
    """
    if len(minority) < 2:
        raise ValueError("для интерполяции нужно хотя бы две точки")
    rng = random.Random(seed)
    k = min(k, len(minority) - 1)
    synthetic = []
    for _ in range(n_synthetic):
        i = rng.randrange(len(minority))
        neighbors = k_nearest(minority, i, k)
        j = rng.choice(neighbors)
        t = rng.random()
        base, mate = minority[i], minority[j]
        synthetic.append([a + t * (b - a) for a, b in zip(base, mate)])
    return synthetic


def random_oversample(X, y, seed=0):
    """Добить редкие классы дубликатами до размера самого частого. Вернуть (X, y).

    random_oversample([[0.0], [1.0], [2.0]], [0, 0, 1])
        ->  ([[0.0], [1.0], [2.0], [2.0]], [0, 0, 1, 1])

    Сначала идут все исходные строки в исходном порядке, затем дубликаты.
    Если классы уже сбалансированы — данные возвращаются как есть.

    Дубликаты — это буквально те же точки: модель увидит их несколько раз и
    охотно переобучится ровно на них. Ради этого и придумали SMOTE.
    """
    rng = random.Random(seed)
    by_class = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)
    biggest = max(len(idx) for idx in by_class.values())
    X_out, y_out = list(X), list(y)
    for label, indices in sorted(by_class.items()):
        for _ in range(biggest - len(indices)):
            i = rng.choice(indices)
            X_out.append(X[i])
            y_out.append(label)
    return X_out, y_out


def best_threshold(y_true, probs, step=0.01):
    """Перебрать пороги от 0.05 до 0.95 и вернуть (лучший порог, его F1).

    best_threshold([0, 0, 1], [0.1, 0.2, 0.3])  ->  порог около 0.25 c F1 = 1.0

    Предсказание: 1, если вероятность >= порога. При равных F1 побеждает
    меньший порог. Шаг перебора считай через индекс (0.05 + i*step), а не
    накоплением: накопленная сумма float уползёт и последний порог потеряется.

    Порог 0.5 — не закон природы, а значение по умолчанию. На перекошенных
    данных оптимум почти всегда заметно ниже: модель редко бывает уверена
    в редком классе, но ранжирует его выше — этого достаточно.
    """
    n_steps = int(round((0.95 - 0.05) / step)) + 1
    best_t, best_f1 = 0.05, -1.0
    for i in range(n_steps):
        t = 0.05 + i * step
        y_pred = [1 if p >= t else 0 for p in probs]
        f1 = precision_recall_f1(y_true, y_pred)[2]
        # строгое > : первым выигрывает меньший порог
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1
