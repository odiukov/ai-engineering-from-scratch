"""
Подбор гиперпараметров — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import itertools
import math
import random


def log_uniform(low, high, u):
    """Точка на логарифмической шкале между low и high. u идёт от 0 до 1.

    log_uniform(0.001, 1.0, 0.0)  ->  0.001
    log_uniform(0.001, 1.0, 1.0)  ->  1.0
    log_uniform(0.001, 1.0, 0.5)  ->  0.0316...  (а не 0.5005, как по линейной)

    Ловушка, ради которой это отдельная функция: линейная сетка от 0.001 до 1.0
    отдаёт девять десятых бюджета отрезку [0.1, 1.0], где learning rate почти
    всегда уже бесполезен. Логарифмическая раздаёт бюджет поровну каждому
    порядку величины.

    Правило: learning rate и силу регуляризации ищут в лог-шкале, размер батча
    и глубину дерева — в линейной.
    """
    # умножение на степень отношения вместо 10 ** (...): один вызов pow
    # и никакой потери точности на переводе туда-обратно через log10
    return low * (high / low) ** u


def sample_config(space, seed):
    """Один случайный набор гиперпараметров из описания пространства.

    space = {"lr": ("log_float", 1e-4, 1.0), "depth": ("int", 2, 8)}
    sample_config(space, 0)   ->  {"lr": 0.0027..., "depth": 5}
    sample_config(space, 0) == sample_config(space, 0)   ->  True

    Поддерживаются четыре вида: "float" (равномерно), "log_float" (через
    log_uniform), "int" (целое включительно с обоих концов) и "choice" (элемент
    списка).

    Ловушка: ключи обходятся ОТСОРТИРОВАННЫМИ. Иначе один и тот же seed на
    словарях с разным порядком ключей даст разные конфигурации, и сравнение
    двух запусков перестанет что-либо значить.
    """
    rng = random.Random(seed)

    config = {}
    for name in sorted(space):
        kind = space[name][0]
        if kind == "choice":
            config[name] = rng.choice(space[name][1])
        elif kind == "int":
            config[name] = rng.randint(space[name][1], space[name][2])
        elif kind == "log_float":
            config[name] = log_uniform(space[name][1], space[name][2], rng.random())
        else:
            config[name] = rng.uniform(space[name][1], space[name][2])
    return config


def grid_search(objective, param_grid):
    """Полный перебор сетки. Возвращает (лучший конфиг, лучший score, история).

    grid_search(f, {"a": [1, 2], "b": [3, 4]})  ->  история из 4 пар (конфиг, score)

    objective(config) -> число, чем больше, тем лучше. История — список пар в
    порядке перебора, она нужна, чтобы потом посчитать, сколько уникальных
    значений каждого параметра успели попробовать.

    Ловушка масштаба: шесть параметров по пять значений — это 15 625 обучений.
    Сетка растёт экспоненциально по числу параметров, а полезной информации
    даёт линейно.
    """
    names = sorted(param_grid)
    history = []
    for combo in itertools.product(*(param_grid[name] for name in names)):
        config = dict(zip(names, combo))
        history.append((config, objective(config)))

    best = max(history, key=lambda pair: pair[1])
    return best[0], best[1], history


def random_search(objective, space, n_iter=20, seed=0):
    """Случайный поиск: n_iter независимых конфигураций. Тот же формат ответа.

    random_search(f, space, n_iter=9)  ->  история из 9 пар
    random_search(f, space, n_iter=9, seed=1) == random_search(f, space, n_iter=9, seed=1)

    Каждой конфигурации свой seed — так любой отдельный шаг воспроизводится сам
    по себе, независимо от остальных.

    Зачем: при том же бюджете 9 вычислений сетка 3x3 даёт 3 разных значения
    каждого параметра, а случайный поиск — 9. Если из шести параметров важны
    два, случайный поиск исследует их втрое плотнее.
    """
    history = []
    for i in range(n_iter):
        config = sample_config(space, seed + i)
        history.append((config, objective(config)))

    best = max(history, key=lambda pair: pair[1])
    return best[0], best[1], history


def count_unique(history, name):
    """Сколько разных значений параметра встретилось в истории поиска.

    count_unique(история_сетки_3x3, "lr")       ->  3
    count_unique(история_случайного_из_9, "lr")  ->  9

    Ровно это число и объясняет, почему случайный поиск обычно выигрывает у
    сетки при равном бюджете (Bergstra & Bengio, 2012).
    """
    return len({config[name] for config, _ in history})


def expected_improvement(mu, sigma, best_y):
    """Ожидаемое улучшение над best_y при прогнозе mu с неопределённостью sigma.

    expected_improvement(5.0, 0.0, 4.0)  ->  1.0   (точно знаем: лучше на 1)
    expected_improvement(3.0, 0.0, 4.0)  ->  0.0   (точно знаем: хуже)
    expected_improvement(3.0, 2.0, 4.0)  ->  >0    (может и повезти)

    Формула: EI = (mu - best) * Phi(z) + sigma * phi(z), где z = (mu - best) / sigma,
    Phi — функция распределения нормали (через math.erf), phi — её плотность.

    Ловушки: sigma = 0 обнуляет знаменатель — этот случай считается отдельно и
    даёт max(mu - best, 0). И EI не бывает отрицательной: ухудшение нас не
    интересует, мы просто туда не пойдём.

    Смысл: два слагаемых — это эксплуатация (первое, «модель обещает много») и
    разведка (второе, «мы просто ничего не знаем про эту точку»).
    """
    gain = mu - best_y
    if sigma <= 0:
        return max(gain, 0.0)

    z = gain / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return gain * cdf + sigma * pdf


def surrogate(observations, x, length_scale=1.0):
    """Дешёвая модель дорогой функции. Возвращает (прогноз, неопределённость).

    surrogate([([0.0], 5.0)], [0.0])        ->  (5.0, 0.0)   точка уже измерена
    surrogate([([0.0], 5.0)], [3.0])        ->  (5.0, ~0.99) прогноз тот же, веры нет
    surrogate([([0.0], 0.0), ([1.0], 10.0)], [0.5])  ->  (5.0, ~0.12) ровно посередине

    observations — список пар (вектор координат, score). Прогноз — среднее
    наблюдений с весами exp(-d^2 / (2 * length_scale^2)): близкие точки весят
    больше. Неопределённость — 1 минус вес ближайшего наблюдения: ноль в уже
    проверенной точке, около единицы вдали от всех.

    Ловушка: без наблюдений или когда все веса схлопнулись в ноль, делить не на
    что — возвращаем (0.0, 1.0), то есть «понятия не имею, иди посмотри».

    Настоящий байесовский оптимизатор ставит сюда гауссовский процесс. Идея та
    же: прогноз плюс честное признание незнания.
    """
    if not observations:
        return 0.0, 1.0

    weights = [
        math.exp(-sum((a - b) ** 2 for a, b in zip(x, point)) / (2 * length_scale ** 2))
        for point, _ in observations
    ]
    total = sum(weights)
    if total < 1e-300:
        return 0.0, 1.0

    mu = sum(w * score for w, (_, score) in zip(weights, observations)) / total
    return mu, 1.0 - max(weights)


def bayes_search(objective, space, n_iter=20, n_initial=5, n_candidates=64, length_scale=0.2, seed=0):
    """Байесовский поиск: сначала разведка наугад, дальше — по подсказке суррогата.

    bayes_search(f, space, n_iter=25)  ->  (лучший конфиг, лучший score, история)
    len(история) == n_iter

    Цикл: первые n_initial конфигураций берутся случайно, дальше на каждом шаге
    генерируется n_candidates кандидатов, для каждого суррогат даёт (mu, sigma),
    expected_improvement превращает их в одно число, и берётся максимум.

    Координаты конфигурации нормируются в [0, 1] (лог-параметры — по логарифму),
    иначе learning rate со шкалой 0.001 и глубина со шкалой 8 будут для ядра
    несопоставимы, и length_scale придётся подбирать под каждый параметр.

    Ловушка: seed кандидатов обязан отличаться от seed стартовых точек, иначе
    оптимизатор на каждом шаге предлагает те же самые конфигурации.
    """
    names = sorted(space)

    def to_vector(config):
        vector = []
        for name in names:
            spec = space[name]
            value = config[name]
            if spec[0] == "choice":
                options = spec[1]
                vector.append(options.index(value) / max(len(options) - 1, 1))
            elif spec[0] == "log_float":
                low, high = spec[1], spec[2]
                vector.append(math.log(value / low) / math.log(high / low))
            else:
                low, high = spec[1], spec[2]
                vector.append((value - low) / (high - low) if high > low else 0.0)
        return vector

    observations, history = [], []
    for i in range(n_iter):
        if i < n_initial:
            config = sample_config(space, seed + i)
        else:
            best_so_far = max(score for _, score in history)
            # кандидаты нумеруются со сдвигом, чтобы ни один seed не повторил
            # стартовые точки: иначе оптимизатор будет ходить по своим следам
            candidates = [
                sample_config(space, seed + 1_000_000 + i * n_candidates + j)
                for j in range(n_candidates)
            ]
            scores = []
            for candidate in candidates:
                mu, sigma = surrogate(observations, to_vector(candidate), length_scale)
                scores.append(expected_improvement(mu, sigma, best_so_far))
            config = candidates[scores.index(max(scores))]

        score = objective(config)
        observations.append((to_vector(config), score))
        history.append((config, score))

    best = max(history, key=lambda pair: pair[1])
    return best[0], best[1], history
