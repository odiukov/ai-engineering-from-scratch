"""
Смещение и разброс — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def polyval(coeffs, x):
    """Значение полинома в точке. coeffs[i] — коэффициент при x^i.

    polyval([1, 2, 3], 2)   ->  17.0   (1 + 2*2 + 3*4)
    polyval([5], 100)       ->  5.0    (константа)

    Считай схемой Горнера, с конца: ((c2 * x) + c1) * x + c0. Это n умножений
    вместо n возведений в степень, и никаких x**15, которые на x=3 дают
    четырнадцать миллионов и съедают точность.
    """
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def fit_polynomial(xs, ys, degree, l2=0.0):
    """МНК-полином степени degree. Возвращает degree + 1 коэффициент.

    fit_polynomial([0, 1, 2], [1, 3, 5], 1)        ->  [1.0, 2.0]  (y = 1 + 2x)
    fit_polynomial([0, 1, 2, 3], [0, 1, 4, 9], 2)  ->  [0.0, 0.0, 1.0]  (y = x^2)

    Решается нормальное уравнение (A^T A + l2 * I) w = A^T y, где A — матрица
    Вандермонда. l2 — гребневая регуляризация: она тянет коэффициенты к нулю и
    ровно так меняет модель с «разброс» на «смещение».

    Ловушка: свободный член штрафовать нельзя, иначе модель не сможет
    предсказывать даже константу. Штраф идёт по диагонали начиная с индекса 1.
    """
    m = degree + 1
    n = len(xs)
    # степени считаем один раз и переиспользуем: на каждой из n строк нужны
    # все m степеней, пересчитывать их внутри двух вложенных сумм — расточительно
    design = []
    for x in xs:
        row, power = [], 1.0
        for _ in range(m):
            row.append(power)
            power *= x
        design.append(row)

    ata = [[sum(design[r][i] * design[r][j] for r in range(n)) for j in range(m)] for i in range(m)]
    aty = [sum(design[r][i] * ys[r] for r in range(n)) for i in range(m)]
    for i in range(1, m):
        ata[i][i] += l2

    # метод Гаусса с выбором главного элемента: без перестановки строк
    # нулевой ведущий элемент рвёт решение даже на вполне приличных данных
    augmented = [row + [rhs] for row, rhs in zip(ata, aty)]
    for col in range(m):
        pivot = max(range(col, m), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            continue                       # вырожденный столбец: коэффициент остаётся 0
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        head = augmented[col]
        for r in range(m):
            if r == col:
                continue
            factor = augmented[r][col] / head[col]
            for c in range(col, m + 1):
                augmented[r][c] -= factor * head[c]

    return [
        augmented[i][m] / augmented[i][i] if abs(augmented[i][i]) > 1e-12 else 0.0
        for i in range(m)
    ]


def mean_squared_error(y_true, y_pred):
    """Средний квадрат ошибки.

    mean_squared_error([1, 2, 3], [1, 2, 3])  ->  0.0
    mean_squared_error([0, 0], [1, -1])       ->  1.0

    Именно под квадратичную ошибку разложение на смещение и разброс работает
    точно, без приближений.
    """
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def make_dataset(f, n, noise=0.5, seed=0, low=-3.0, high=3.0):
    """Синтетика от известной функции: (xs, ys), где ys = f(x) + шум.

    make_dataset(lambda x: x, 3, noise=0.0, seed=1)  ->  xs == ys
    make_dataset(f, 30, seed=5) == make_dataset(f, 30, seed=5)  ->  True

    Знать истинную f — единственный способ посчитать смещение честно: на
    реальных данных f недоступна, и разложение остаётся теорией.

    seed обязателен: разложение усредняет по сотне обучающих выборок, и каждая
    должна воспроизводиться независимо от остальных.
    """
    # свой генератор, а не random.seed(): иначе один вызов внутри цикла
    # разложения сбрасывал бы состояние глобального random всему процессу
    rng = random.Random(seed)
    xs = [rng.uniform(low, high) for _ in range(n)]
    ys = [f(x) + rng.gauss(0.0, noise) for x in xs]
    return xs, ys


def bias_variance_decomposition(
    f, degree, n_train=30, n_sets=100, noise=0.5, seed=0, l2=0.0, n_test=21, low=-3.0, high=3.0
):
    """Разложить ошибку полинома на смещение и разброс. Словарь bias2/variance/total.

    bias_variance_decomposition(f, 1)["bias2"]  ->  большое: прямая не гнётся
    bias_variance_decomposition(f, 9)["variance"]  ->  большое: полином пляшет

    Обучаем модель на n_sets независимых выборках, предсказываем на общей сетке
    из n_test точек и смотрим на облако предсказаний в каждой точке:

      bias2    = среднее по сетке (среднее предсказание - f(x))^2
      variance = среднее по сетке разброса предсказаний вокруг их среднего
      total    = средний квадрат отклонения предсказаний от f(x)

    Ключевая проверка: total обязано равняться bias2 + variance с точностью до
    машинного нуля. Шума в total нет, потому что сравниваем с истинной f(x), а
    не с зашумлённым y — неустранимая часть ошибки сюда просто не входит.
    """
    grid = [low + i * (high - low) / (n_test - 1) for i in range(n_test)]
    truth = [f(x) for x in grid]

    predictions = []
    for i in range(n_sets):
        xs, ys = make_dataset(f, n_train, noise, seed + i, low, high)
        coeffs = fit_polynomial(xs, ys, degree, l2)
        predictions.append([polyval(coeffs, x) for x in grid])

    mean_pred = [sum(p[j] for p in predictions) / n_sets for j in range(n_test)]

    bias2 = mean_squared_error(truth, mean_pred)
    variance = sum(
        sum((p[j] - mean_pred[j]) ** 2 for p in predictions) / n_sets
        for j in range(n_test)
    ) / n_test
    total = sum(mean_squared_error(truth, p) for p in predictions) / n_sets

    return {"bias2": bias2, "variance": variance, "total": total}


def best_degree(f, degrees, n_train=30, n_sets=100, noise=0.5, seed=0, l2=0.0):
    """Степень с наименьшей суммарной ошибкой — дно U-образной кривой.

    best_degree(lambda x: 2 * x + 1, [1, 2, 5])  ->  1
    best_degree(f, [3])                          ->  3

    Слева от оптимума правит смещение, справа — разброс. Минимум суммы и есть
    та самая «золотая середина» сложности модели.
    """
    return min(
        degrees,
        key=lambda d: bias_variance_decomposition(
            f, d, n_train, n_sets, noise, seed, l2
        )["total"],
    )


def learning_curve(f, degree, sizes, n_repeats=20, noise=0.5, seed=0, n_test=50):
    """Кривая обучения: (train_errors, test_errors) по одному числу на размер.

    learning_curve(f, 1, [10, 40])  ->  ошибки на train растут, на test падают
    len(learning_curve(f, 3, [10, 20, 40])[0]) == 3

    Для каждого размера обучаем n_repeats раз на свежих данных и усредняем MSE
    на обучении и на отдельной тестовой выборке.

    Как читать: обе кривые сошлись на высокой ошибке — это смещение, данных
    добавлять бесполезно. Между кривыми большой зазор — это разброс, данные
    помогут.
    """
    # тестовая выборка одна на всю кривую: если менять её вместе с обучающей,
    # к колебаниям модели добавится колебание самой оценки и кривая замылится
    test_xs, test_ys = make_dataset(f, n_test, noise, seed + 10_000)

    train_errors, test_errors = [], []
    for size in sizes:
        train_scores, test_scores = [], []
        for i in range(n_repeats):
            xs, ys = make_dataset(f, size, noise, seed + i)
            coeffs = fit_polynomial(xs, ys, degree)
            train_scores.append(mean_squared_error(ys, [polyval(coeffs, x) for x in xs]))
            test_scores.append(
                mean_squared_error(test_ys, [polyval(coeffs, x) for x in test_xs])
            )
        train_errors.append(sum(train_scores) / n_repeats)
        test_errors.append(sum(test_scores) / n_repeats)
    return train_errors, test_errors


def diagnose(train_error, test_error, error_threshold=0.5, gap_threshold=0.5):
    """Диагноз по паре ошибок: "bias", "variance" или "good".

    diagnose(0.05, 0.90)  ->  "variance"   (зазор большой)
    diagnose(0.80, 0.85)  ->  "bias"       (обе высокие, зазор мал)
    diagnose(0.10, 0.15)  ->  "good"

    Порядок проверок важен: сначала зазор, потом уровень ошибки. Модель с
    train 0.8 и test 2.0 переобучена, хотя обе ошибки высокие, — лечить её надо
    как разброс, а не как смещение.
    """
    if test_error - train_error > gap_threshold:
        return "variance"
    if train_error > error_threshold:
        return "bias"
    return "good"
