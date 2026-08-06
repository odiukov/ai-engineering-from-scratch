"""
Вероятность и распределения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def expected_value(values, probs):
    """Математическое ожидание: среднее, взвешенное вероятностями.

    expected_value([1, 2, 3, 4, 5, 6], [1/6] * 6)  ->  3.5   (честная кость)
    expected_value([0, 1], [0.7, 0.3])             ->  0.3   (Bernoulli p=0.3)

    Ожидание не обязано быть возможным исходом: 3.5 на кости не выпадет
    никогда.

    В ML это и есть loss: средняя ошибка по распределению данных.
    """
    return sum(v * p for v, p in zip(values, probs))


def variance(values, probs):
    """Дисперсия: ожидание квадрата отклонения от среднего.

    variance([1, 2, 3, 4, 5, 6], [1/6] * 6)  ->  2.9166...
    variance([5, 5], [0.5, 0.5])             ->  0.0   (разброса нет)

    Считай через expected_value, а не переписывай формулу заново.
    Ловушка: формула E[X^2] - E[X]^2 математически та же, но на близких
    больших числах теряет точность из-за вычитания почти равных величин.
    """
    mu = expected_value(values, probs)
    # честное E[(X - mu)^2]: отклонения малы, катастрофического
    # вычитания нет — в отличие от E[X^2] - E[X]^2
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probs))


def normal_pdf(x, mu=0.0, sigma=1.0):
    """Плотность нормального распределения в точке x.

    normal_pdf(0)          ->  0.3989...   (1 / sqrt(2*pi))
    normal_pdf(1, 1, 1)    ->  0.3989...   (пик всегда в точке mu)
    normal_pdf(0, 0, 0.1)  ->  3.989...    (плотность БОЛЬШЕ единицы)

    Плотность — не вероятность. Она спокойно превышает 1, вероятность
    получается только интегрированием по отрезку.

    f(x) = 1 / (sigma * sqrt(2*pi)) * exp(-((x - mu) / sigma)^2 / 2)
    """
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    # (x - mu) / sigma считаем ДО возведения в квадрат: так меньше шансов
    # переполнить float на больших отклонениях
    z = (x - mu) / sigma
    return coeff * math.exp(-0.5 * z * z)


def softmax(logits):
    """Превратить произвольные оценки модели в распределение вероятностей.

    softmax([0.0, 0.0])        ->  [0.5, 0.5]
    softmax([1.0, 0.0])        ->  [0.731..., 0.268...]
    softmax([100, 101, 102])   ->  работает, не падает

    Ловушка: наивное exp(z) на logits вида 1000 даёт OverflowError.
    Вычти максимум из всех logits перед exp — результат тот же
    (числитель и знаменатель делятся на одну константу), переполнения нет.
    """
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    """Логарифм softmax, посчитанный без промежуточного softmax.

    log_softmax([0.0, 0.0])       ->  [-0.693..., -0.693...]
    log_softmax([0.0, 1000.0])    ->  [-1000.0, 0.0]

    Ловушка: math.log(softmax(z)[i]) на разнесённых logits даёт -inf,
    потому что softmax уже округлил маленькую вероятность до нуля.
    Считай напрямую: log_softmax(z)_i = z_i - logsumexp(z),
    где logsumexp(z) = max(z) + log(sum(exp(z_j - max(z)))).
    """
    m = max(logits)
    lse = m + math.log(sum(math.exp(z - m) for z in logits))
    return [z - lse for z in logits]


def cross_entropy_loss(logits, target_index):
    """Кросс-энтропия для one-hot цели: минус log вероятности верного класса.

    cross_entropy_loss([0.0, 0.0], 0)      ->  0.693...   (log 2, полное незнание)
    cross_entropy_loss([10.0, 0.0], 0)     ->  0.0000453  (уверен и прав)
    cross_entropy_loss([0.0, 10.0], 0)     ->  10.0000453 (уверен и не прав)

    Всегда >= 0. Ноль недостижим: вероятность 1 требует бесконечного logit.

    Опирайся на log_softmax — именно так это устроено внутри PyTorch,
    и именно поэтому там нет отдельного softmax перед лоссом.
    """
    return -log_softmax(logits)[target_index]


def marginals(joint):
    """Маргинальные распределения из таблицы совместного: (по строкам, по столбцам).

    joint[i][j] — вероятность того, что X = i и одновременно Y = j.

    marginals([[0.4, 0.1], [0.05, 0.45]])  ->  ([0.5, 0.5], [0.45, 0.55])

    Маргинализация — это «просуммировать вторую переменную и забыть о ней».
    Оба списка обязаны в сумме давать 1, если совместное давало 1.
    """
    px = [sum(row) for row in joint]
    # zip(*joint) разворачивает таблицу по столбцам без ручных индексов
    py = [sum(col) for col in zip(*joint)]
    return px, py


def is_independent(joint, tol=1e-9):
    """Независимы ли X и Y: раскладывается ли совместное в произведение маргиналов.

    is_independent([[0.25, 0.25], [0.25, 0.25]])   ->  True
    is_independent([[0.4, 0.1], [0.05, 0.45]])     ->  False  (дождь и зонт)

    Критерий: P(X=i, Y=j) = P(X=i) * P(Y=j) для ВСЕХ пар. Хватает одной
    несовпавшей клетки, чтобы независимости не было.

    Маргиналы бери из marginals, не считай заново.
    """
    px, py = marginals(joint)
    return all(
        abs(joint[i][j] - px[i] * py[j]) < tol
        for i in range(len(px))
        for j in range(len(py))
    )
