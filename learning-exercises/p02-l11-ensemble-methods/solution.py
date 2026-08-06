"""
Ансамбли — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def majority_vote(votes, weights=None):
    """Голосование: метка с наибольшим суммарным весом. Без весов — простое большинство.

    majority_vote([1, 1, -1])                 ->  1
    majority_vote([1, -1], [0.2, 5.0])        ->  -1   (второй голос весомее)

    Ловушка ничьей: при равных суммах побеждает метка, проголосовавшая первой.
    Это соглашение, но оно обязано быть детерминированным — иначе один и тот же
    ансамбль на одном и том же объекте будет отвечать по-разному.

    Зачем в AI: это и есть «hard voting» — простейший способ склеить несколько
    моделей. Ошибки взаимно гасятся, если модели ошибаются в разных местах.
    """
    if weights is None:
        weights = [1.0] * len(votes)

    totals = {}
    for vote, weight in zip(votes, weights):
        totals[vote] = totals.get(vote, 0.0) + weight
    # dict помнит порядок вставки, а max возвращает ПЕРВЫЙ максимум —
    # отсюда и берётся детерминированное правило разрешения ничьей
    return max(totals, key=totals.get)


def vote_accuracy(p, n_models):
    """Точность большинства n независимых моделей с одинаковой точностью p.

    vote_accuracy(0.6, 1)    ->  0.6
    vote_accuracy(0.6, 21)   ->  ~0.826
    vote_accuracy(0.4, 21)   ->  ~0.174  (слабые модели голосуют себе во вред)

    Сумма биномиальных вероятностей по k > n/2: C(n, k) * p^k * (1-p)^(n-k).

    Это теорема Кондорсе и весь смысл ансамблей одной формулой: при p > 0.5
    точность голосования растёт к единице с ростом n, при p < 0.5 — падает к
    нулю. Ключевое условие — НЕЗАВИСИМОСТЬ. Одинаково ошибающиеся модели ничего
    не дают.
    """
    # строго больше половины: при чётном n ничья большинством не считается
    return sum(
        math.comb(n_models, k) * p ** k * (1 - p) ** (n_models - k)
        for k in range(n_models // 2 + 1, n_models + 1)
    )


def bootstrap_indices(n, seed):
    """Индексы бутстрап-выборки: n штук из диапазона 0..n-1, с возвращением.

    len(bootstrap_indices(100, 0)) == 100
    bootstrap_indices(50, 3) == bootstrap_indices(50, 3)   ->  True

    С возвращением — значит индексы повторяются, и примерно 36.8% исходных
    объектов в выборку не попадут вовсе. Эти out-of-bag объекты — бесплатная
    валидация: модель их не видела.

    Ловушка: без возвращения получится перестановка тех же данных, все модели
    ансамбля обучатся на одном и том же и разнообразия не будет.
    """
    rng = random.Random(seed)
    return [rng.randrange(n) for _ in range(n)]


def fit_stump(X, y, weights=None):
    """Пень решений: один порог по одному признаку. Метки — только +1 и -1.

    fit_stump([[0], [1]], [-1, 1])  ->  {"feature": 0, "threshold": 1, "polarity": 1}
    fit_stump([[5], [9]], [1, 1])   ->  пень, отвечающий +1 всем

    Перебираем все признаки, все встречающиеся значения как пороги и обе
    полярности; берём вариант с наименьшей ВЗВЕШЕННОЙ ошибкой. weights — веса
    объектов из AdaBoost, по умолчанию равные.

    Ловушка: полярность обязательна. Без неё пень умеет только «выше порога —
    плюс», и половина разделимых задач становится нерешаемой.
    """
    n = len(X)
    if weights is None:
        weights = [1.0 / n] * n

    best = None
    best_error = float("inf")
    for feature in range(len(X[0])):
        # пороги — сами встречающиеся значения: при сравнении >= минимальное
        # из них даёт «константный» пень, а остальные — все возможные разрезы
        for threshold in sorted({row[feature] for row in X}):
            for polarity in (1, -1):
                error = sum(
                    weight
                    for weight, row, label in zip(weights, X, y)
                    if (polarity if row[feature] >= threshold else -polarity) != label
                )
                # строгое неравенство: при равенстве побеждает найденный раньше,
                # значит результат не зависит от порядка обхода словарей
                if error < best_error:
                    best_error = error
                    best = {"feature": feature, "threshold": threshold, "polarity": polarity}
    return best


def predict_stump(stump, x):
    """Ответ пня на одном объекте: +1 или -1.

    predict_stump({"feature": 0, "threshold": 1, "polarity": 1}, [5])   ->  1
    predict_stump({"feature": 0, "threshold": 1, "polarity": -1}, [5])  ->  -1

    Правило: значение признака не меньше порога — отвечаем polarity, иначе
    минус polarity. Граница включена в правую сторону, так же как при подборе.
    """
    return stump["polarity"] if x[stump["feature"]] >= stump["threshold"] else -stump["polarity"]


def predict_ensemble(ensemble, x):
    """Ответ ансамбля: взвешенное голосование пней. ensemble — список (пень, вес).

    predict_ensemble([(stump_a, 1.0), (stump_b, 1.0)], x)  ->  простое большинство
    predict_ensemble([(stump_a, 0.1), (stump_b, 9.0)], x)  ->  почти всегда мнение b

    Одна функция и для бэггинга (все веса равны), и для AdaBoost (веса — alpha).
    Разница между методами не в способе голосования, а в том, как обучались
    участники.
    """
    votes = [predict_stump(stump, x) for stump, _ in ensemble]
    return majority_vote(votes, [weight for _, weight in ensemble])


def fit_bagging(X, y, n_models=10, seed=0):
    """Бэггинг: n_models пней, каждый на своей бутстрап-выборке. Все веса равны 1.0.

    fit_bagging(X, y, n_models=5, seed=0)  ->  список из 5 пар (пень, 1.0)
    fit_bagging(X, y, seed=1) == fit_bagging(X, y, seed=1)  ->  True

    Разнообразие берётся только из бутстрапа: алгоритм и данные одни и те же,
    отличаются лишь выборки. Бэггинг снижает разброс и почти не трогает
    смещение — усреднение гасит случайные отклонения, но не системные.

    Ловушка: seed один на весь ансамбль, но у каждой модели он обязан быть
    свой. Одинаковый seed для всех даст n копий одного пня.
    """
    ensemble = []
    for i in range(n_models):
        indices = bootstrap_indices(len(X), seed + i)
        stump = fit_stump([X[j] for j in indices], [y[j] for j in indices])
        ensemble.append((stump, 1.0))
    return ensemble


def fit_adaboost(X, y, n_rounds=10):
    """AdaBoost: пни обучаются по очереди, каждый — на ошибках предыдущих.

    fit_adaboost(X, y, n_rounds=5)  ->  список из 5 пар (пень, alpha)
    все alpha > 0, если ошибка каждого пня меньше половины

    Раунд: обучить пень на текущих весах, посчитать взвешенную ошибку err,
    задать вес модели alpha = 0.5 * ln((1 - err) / err), умножить веса объектов
    на exp(-alpha * y * pred) и перенормировать.

    Ловушки: err ровно 0 или 1 взрывает логарифм — зажми его в [1e-10, 1-1e-10].
    И знак в обновлении весов: правильно угаданный объект (y * pred = +1)
    обязан ПОТЕРЯТЬ вес, ошибочный — набрать.

    Бустинг снижает смещение: ансамбль пней собирает границу, которую ни один
    пень поодиночке провести не может.
    """
    n = len(X)
    weights = [1.0 / n] * n
    ensemble = []

    for _ in range(n_rounds):
        stump = fit_stump(X, y, weights)
        predictions = [predict_stump(stump, x) for x in X]

        error = sum(w for w, p, t in zip(weights, predictions, y) if p != t)
        error = min(max(error, 1e-10), 1 - 1e-10)
        alpha = 0.5 * math.log((1 - error) / error)
        ensemble.append((stump, alpha))

        weights = [w * math.exp(-alpha * t * p) for w, t, p in zip(weights, y, predictions)]
        # нормировка обязательна: иначе веса уплывают в ноль или в бесконечность
        # и уже через десяток раундов сравнение с err теряет смысл
        total = sum(weights)
        weights = [w / total for w in weights]

    return ensemble
