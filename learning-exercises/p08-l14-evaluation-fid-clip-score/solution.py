"""
Оценка генеративных моделей: FID, CLIP score, Elo — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def matmul(A, B):
    """Произведение матриц-списков: (n x m) на (m x p) даёт (n x p).

    matmul([[1, 2], [3, 4]], [[1, 0], [0, 1]])  ->  [[1, 2], [3, 4]]
    matmul([[1, 2]], [[3], [4]])                ->  [[11]]

    Ловушка размеров: число столбцов A обязано совпасть с числом строк B.
    Иначе zip молча обрежет по короткому и выдаст правдоподобный мусор —
    здесь вместо этого нужен ValueError.
    """
    if len(A[0]) != len(B):
        raise ValueError("matmul: len(A[0]) != len(B)")
    n, m, p = len(A), len(B), len(B[0])
    out = [[0.0] * p for _ in range(n)]
    # порядок i-k-j, а не i-j-k: внутренний цикл идёт по строке B подряд,
    # это заметно дружелюбнее к кешу на больших матрицах
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a == 0.0:
                continue
            row = B[k]
            for j in range(p):
                out[i][j] += a * row[j]
    return out


def matrix_inverse(M):
    """Обратная матрица методом Гаусса-Жордана с выбором главного элемента.

    matrix_inverse([[2, 0], [0, 4]])  ->  [[0.5, 0.0], [0.0, 0.25]]
    matrix_inverse([[1, 0], [0, 1]])  ->  [[1.0, 0.0], [0.0, 1.0]]

    Вырожденная матрица (например [[1, 2], [2, 4]]) обращению не подлежит —
    бросай ValueError, а не возвращай матрицу из бесконечностей.

    Нужна дальше для матричного корня, а тот — для FID.
    """
    n = len(M)
    # расширенная матрица [M | I]: справа единичная, слева приводим к ней
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(M)]
    for col in range(n):
        # частичный выбор: строка с наибольшим по модулю элементом в столбце.
        # без него деление на крошечный опорный элемент раздувает ошибку
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("matrix_inverse: матрица вырождена")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [x / piv for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def matrix_sqrt(M, iters=100, tol=1e-13):
    """Квадратный корень из симметричной положительно полуопределённой матрицы.

    Метод Якоби находит собственные значения без обращения M. Затем
    sqrt(M) = V diag(sqrt(lambda)) V^T. Нулевые lambda разрешены: реальные
    ковариации часто вырождены, особенно когда сэмплов меньше размерности.

    matrix_sqrt([[4, 0], [0, 9]])  ->  [[2.0, 0.0], [0.0, 3.0]]
    matrix_sqrt([[1, 0], [0, 1]])  ->  [[1.0, 0.0], [0.0, 1.0]]

    Проверка результата всегда одна: sqrt(M) * sqrt(M) == M.
    Ловушка: итерации через inverse падают на PSD-матрице с нулевым
    собственным значением. Здесь отрицательное значение меньше численного
    допуска — действительно не PSD и даёт ValueError; крошечный минус
    округляется к нулю.
    """
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("matrix_sqrt: matrix must be square")
    if n == 0:
        return []

    # FID строит математически симметричную матрицу; усреднение убирает
    # микроскопическую асимметрию от порядка операций с float.
    A = [[(M[i][j] + M[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max(1.0, max(abs(x) for row in A for x in row))

    for _ in range(iters):
        if n == 1:
            break
        p, q = max(
            ((i, j) for i in range(n) for j in range(i + 1, n)),
            key=lambda ij: abs(A[ij[0]][ij[1]]),
        )
        if abs(A[p][q]) <= tol * scale:
            break

        angle = 0.5 * math.atan2(2.0 * A[p][q], A[q][q] - A[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        for k in range(n):
            if k == p or k == q:
                continue
            akp, akq = A[k][p], A[k][q]
            A[k][p] = A[p][k] = c * akp - s * akq
            A[k][q] = A[q][k] = s * akp + c * akq
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p][q] = A[q][p] = 0.0
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq

    eigenvalues = [A[i][i] for i in range(n)]
    if min(eigenvalues) < -tol * scale:
        raise ValueError("matrix_sqrt: matrix must be positive semidefinite")
    roots = [math.sqrt(max(value, 0.0)) for value in eigenvalues]
    out = [[sum(V[i][k] * roots[k] * V[j][k] for k in range(n))
            for j in range(n)] for i in range(n)]
    return [[(out[i][j] + out[j][i]) / 2.0 for j in range(n)] for i in range(n)]


def mean_vector(vectors):
    """Покоординатное среднее набора векторов признаков.

    mean_vector([[1, 2], [3, 4]])  ->  [2.0, 3.0]
    mean_vector([[5, -5]])         ->  [5.0, -5.0]

    В FID это mu_r и mu_g — центры облаков реальных и сгенерированных
    признаков Inception.
    """
    n = len(vectors)
    d = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(d)]


def covariance(vectors):
    """Выборочная ковариационная матрица набора векторов (делитель n-1).

    covariance([[1, 0], [3, 0]])  ->  [[2.0, 0.0], [0.0, 0.0]]
    covariance([[1, 2]])          ->  [[0.0, 0.0], [0.0, 0.0]]

    Матрица симметрична, на диагонали — дисперсии координат.
    Ловушка делителя: n-1, а не n (несмещённая оценка); при n=1 делить
    не на что — возвращай нули, а не ZeroDivisionError.
    """
    mu = mean_vector(vectors)
    n, d = len(vectors), len(mu)
    denom = max(n - 1, 1)
    cov = [[0.0] * d for _ in range(d)]
    for v in vectors:
        # центрируем один раз на вектор, а не внутри двойного цикла
        c = [v[i] - mu[i] for i in range(d)]
        for i in range(d):
            ci = c[i]
            for j in range(d):
                cov[i][j] += ci * c[j]
    return [[cov[i][j] / denom for j in range(d)] for i in range(d)]


def fid(real_features, gen_features):
    """Frechet Inception Distance между двумя облаками признаков.

    FID = ||mu_r - mu_g||^2 + Tr(S_r) + Tr(S_g)
          - 2 Tr(sqrt(sqrt(S_r) S_g sqrt(S_r))).

    fid(X, X)                        ->  0.0   совпадающие множества
    fid(X, [v + 1 for v in X])       ->  d     сдвиг на 1 по всем d осям

    Свойства, которые обязаны выполняться: неотрицательность, симметрия,
    независимость от ПОРЯДКА сэмплов (это расстояние между распределениями,
    а не между списками).

    Симметричная sandwich-форма выше равна привычному trace(sqrt(S_r S_g)),
    но остаётся PSD и работает для вырожденных ковариаций без inverse.

    Ловушка малого N: на сотне сэмплов FID заметно больше нуля даже для
    двух выборок из одного распределения. Ниже 10 000 сэмплов число
    сравнивать не с чем — ровно так статьи и накручивают метрику.
    """
    mu_r, mu_g = mean_vector(real_features), mean_vector(gen_features)
    cov_r, cov_g = covariance(real_features), covariance(gen_features)
    mean_sq = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    sqrt_r = matrix_sqrt(cov_r)
    middle = matmul(matmul(sqrt_r, cov_g), sqrt_r)
    sqrt_prod = matrix_sqrt(middle)
    trace = lambda M: sum(M[i][i] for i in range(len(M)))
    score = mean_sq + trace(cov_r) + trace(cov_g) - 2 * trace(sqrt_prod)
    # Теоретически FID >= 0; округление собственных значений иногда оставляет
    # -1e-15 на совпадающих облаках.
    return max(score, 0.0)


def clip_score(image_feat, text_feat):
    """Косинусная близость эмбеддингов картинки и текста.

    clip_score([1, 0], [1, 0])   ->  1.0    идеальное совпадение
    clip_score([1, 0], [0, 1])   ->  0.0    ничего общего
    clip_score([1, 0], [-1, 0])  -> -1.0    противоположность

    Метрика НЕ зависит от длины векторов: [2, 0] и [1, 0] дают то же число.
    Ловушка: нулевой вектор даёт деление на ноль — верни 0.0.

    Чем выше, тем лучше картинка следует промпту. Слабость известна:
    у CLIP плохо с композицией («красный куб на синем шаре»), так что
    высокий score ещё не значит, что промпт выполнен.
    """
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    na = math.sqrt(sum(a * a for a in image_feat))
    nb = math.sqrt(sum(b * b for b in text_feat))
    denom = na * nb
    if denom < 1e-12:
        return 0.0
    return dot / denom


def elo_update(r_a, r_b, winner, k=32):
    """Обновление пары рейтингов Elo после одного сравнения A против B.

    winner — строка "a" или "b". Возвращает новую пару (r_a, r_b).

    elo_update(1000, 1000, "a")  ->  (1016.0, 984.0)   ничья по силе
    elo_update(1000, 1000, "b")  ->  (984.0, 1016.0)
    elo_update(1600, 1000, "a")  ->  почти без изменений: победа ожидаемая

    Ожидание для A: 1 / (1 + 10^((r_b - r_a) / 400)).
    Сумма рейтингов сохраняется: сколько один выиграл, столько другой потерял.
    """
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    # дельта считается один раз и вычитается у соперника — иначе сумма
    # рейтингов поплывёт и лидерборд перестанет быть сравнимым
    delta = k * (actual_a - expected_a)
    return r_a + delta, r_b - delta
