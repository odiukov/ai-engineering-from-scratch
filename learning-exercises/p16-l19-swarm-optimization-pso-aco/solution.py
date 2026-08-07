"""
Роевая оптимизация: PSO и ACO — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Классические коэффициенты Кленка–Кеннеди (constriction): инерция 0.729,
# когнитивный и социальный вес по 1.494. С ними рой сходится, а не разлетается.
W_DEFAULT = 0.729
C1_DEFAULT = 1.494
C2_DEFAULT = 1.494


def rastrigin(point, amplitude=10.0):
    """Функция Растригина: гладкая парабола, изрытая косинусными ямами.

    rastrigin([0.0, 0.0])  ->  0.0    (глобальный минимум)
    rastrigin([1.0])       ->  1.0    (локальный минимум, их тут решётка)
    rastrigin([0.5])       ->  20.25  (гребень между ямами)

    Формула: A*n + sum(x_i^2 - A*cos(2*pi*x_i)).

    Это стандартный полигон для роевых алгоритмов: локальных минимумов
    экспоненциально много (по одному на каждую целую точку), и градиентный
    спуск застревает в ближайшей яме. PSO выбирается за счёт того, что
    частицы делятся находками.

    Минимизируем: меньше — лучше. Дальше по файлу это соглашение везде.
    """
    total = amplitude * len(point)
    for x in point:
        total += x * x - amplitude * math.cos(2 * math.pi * x)
    return total


def pso_velocity(v, x, p_best, g_best, w, c1, c2, r1, r2):
    """Новая скорость частицы: инерция + тяга к личному и глобальному лучшему.

    v_new[d] = w*v[d] + c1*r1*(p_best[d]-x[d]) + c2*r2*(g_best[d]-x[d])

    pso_velocity([0.0], [0.0], [1.0], [2.0], 0.5, 1.0, 1.0, 1.0, 1.0)  ->  [3.0]
    pso_velocity([1.0], [5.0], [5.0], [5.0], 0.7, 1.5, 1.5, 0.5, 0.5)  ->  [0.7]

    r1 и r2 — случайные числа из [0, 1), но приходят СНАРУЖИ, готовыми.
    Функция обязана быть чистой: без них PSO невоспроизводим, и отладить
    расходящийся рой невозможно.

    Ловушка: r1 и r2 — по одному скаляру на частицу, а не по одному на
    координату. Так в оригинальной статье Кеннеди–Эберхарта 1995.
    """
    return [
        w * vd + c1 * r1 * (pd - xd) + c2 * r2 * (gd - xd)
        for vd, xd, pd, gd in zip(v, x, p_best, g_best)
    ]


def pso_step(swarm, g_best, fitness, bounds, w, c1, c2, rng):
    """Один шаг роя. Вернуть НОВЫЙ список частиц, вход не трогать.

    Частица — словарь {"x": позиция, "v": скорость,
                       "p_best": личный лучший, "p_best_fit": его значение}.
    bounds — список пар (lo, hi) по каждой координате; позиция обрезается в них.

    Порядок внутри одной частицы: взять r1, r2 из rng, посчитать скорость,
    сдвинуть позицию, обрезать по границам, посчитать fitness, обновить
    p_best ТОЛЬКО если стало лучше (меньше).

    Ловушка: обрезать надо позицию, а не скорость, и уже ПОСЛЕ сложения.
    Иначе частица на границе будет вечно улетать наружу и возвращаться.

    Вторая ловушка: swarm нельзя править на месте. Тест на «c2=0 даёт
    независимый локальный поиск» гоняет один и тот же рой дважды.
    """
    new_swarm = []
    for p in swarm:
        # ровно два вызова rng на частицу — иначе прогон с тем же seed
        # перестанет совпадать сам с собой
        r1, r2 = rng.random(), rng.random()
        v = pso_velocity(p["v"], p["x"], p["p_best"], g_best, w, c1, c2, r1, r2)
        x = [
            min(hi, max(lo, xd + vd))
            for xd, vd, (lo, hi) in zip(p["x"], v, bounds)
        ]
        fit = fitness(x)
        if fit < p["p_best_fit"]:
            p_best, p_best_fit = list(x), fit
        else:
            # копия списка: иначе новая частица будет делить p_best со старой
            p_best, p_best_fit = list(p["p_best"]), p["p_best_fit"]
        new_swarm.append({"x": x, "v": v, "p_best": p_best, "p_best_fit": p_best_fit})
    return new_swarm


def run_pso(fitness, bounds, n_particles, iterations, rng,
            w=W_DEFAULT, c1=C1_DEFAULT, c2=C2_DEFAULT):
    """Полный PSO. Вернуть (лучшая точка, её значение fitness).

    Инициализация: позиция равномерно внутри bounds, скорость равномерно
    в +-10% от ширины диапазона по этой координате.

    run_pso(lambda p: p[0]**2, [(-5.0, 5.0)], 20, 60, random.Random(0))
        ->  ([примерно 0.0], примерно 0.0)

    Реальный аналог: Model Swarms (arXiv:2410.11163) двигает так не точки
    на плоскости, а веса LLM-экспертов в общем подпространстве адаптеров.
    Формула та же, «координата» — дельта весов.

    Ловушка: g_best обновляется по p_best частиц, а не по их текущим x.
    Частица может пролететь минимум насквозь; p_best его помнит, x — нет.
    """
    swarm = []
    for _ in range(n_particles):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        v = [rng.uniform(-0.1 * (hi - lo), 0.1 * (hi - lo)) for lo, hi in bounds]
        swarm.append({"x": x, "v": v, "p_best": list(x), "p_best_fit": fitness(x)})

    best = min(swarm, key=lambda p: p["p_best_fit"])
    g_best, g_best_fit = list(best["p_best"]), best["p_best_fit"]

    for _ in range(iterations):
        swarm = pso_step(swarm, g_best, fitness, bounds, w, c1, c2, rng)
        best = min(swarm, key=lambda p: p["p_best_fit"])
        if best["p_best_fit"] < g_best_fit:
            g_best, g_best_fit = list(best["p_best"]), best["p_best_fit"]
    return g_best, g_best_fit


def tour_length(tour, dist):
    """Длина замкнутого маршрута по матрице расстояний.

    dist = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    tour_length([0, 1, 2], dist)  ->  6.0   (0->1 = 1, 1->2 = 3, 2->0 = 2)
    tour_length([0], dist)        ->  0.0

    Маршрут ЗАМКНУТ: последний город соединён с первым. Забыть это ребро —
    классическая ошибка, из-за которой ACO начинает «оптимизировать»
    незамкнутый путь и находит не тот маршрут.
    """
    n = len(tour)
    return float(sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n)))


def transition_probabilities(pheromone_row, dist_row, allowed, alpha=1.0, beta=2.0):
    """Вероятности перехода муравья из текущего города. Вернуть {город: p}.

    Вес ребра: tau^alpha * (1/d)^beta, где tau — феромон, d — расстояние.
    Нормировка — только по allowed, не по всем городам.

    transition_probabilities([0, 1, 3], [0, 1, 1], [1, 2], 1.0, 0.0)
        ->  {1: 0.25, 2: 0.75}
    transition_probabilities([0, 1, 3], [0, 1, 3], [1, 2], 0.0, 1.0)
        ->  {1: 0.75, 2: 0.25}   (alpha=0: чистая жадность по расстоянию)

    alpha — вес памяти колонии, beta — вес сиюминутной жадности. beta=0
    делает муравья чистым последователем следа, alpha=0 — чистым жадиной.

    Ловушка: если весь феромон нулевой, а alpha>0, сумма весов равна нулю
    и деление падает. Инициализируй матрицу строго положительным tau0.
    """
    weights = {}
    for j in allowed:
        # d>0 гарантировано: город сам себе не allowed, диагональ не трогаем
        weights[j] = (pheromone_row[j] ** alpha) * ((1.0 / dist_row[j]) ** beta)
    total = sum(weights.values())
    if total == 0.0:
        raise ValueError("все веса нулевые: подними tau0 или уменьши alpha")
    return {j: wj / total for j, wj in weights.items()}


def update_pheromone(pheromone, deposits, rho, gate=0.0):
    """Испарение по всей матрице, потом откладывание по прошедшим гейт.

    Вернуть НОВУЮ матрицу. deposits — список пар (маршрут, качество).
    Испарение: tau *= (1 - rho) — на КАЖДОМ ребре, включая непосещённые.
    Откладывание: если quality > gate, прибавить quality на каждое ребро
    маршрута симметрично (i->j и j->i).

    update_pheromone([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [], 0.5)
        ->  [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    update_pheromone([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [([0, 1, 2], 0.5)], 0.5)
        ->  [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

    gate — это quality-gated update из AMRO-S (arXiv:2603.12933): быстрый,
    но неправильный агент не должен накапливать феромон. Без гейта система
    залипает на плохом маршруте, который просто нашли первым.

    Испарение — не косметика. Без него феромон на первом найденном маршруте
    растёт неограниченно, и никакой более короткий его уже не догонит.
    """
    new = [[t * (1.0 - rho) for t in row] for row in pheromone]
    for tour, quality in deposits:
        if quality <= gate:
            continue
        n = len(tour)
        for k in range(n):
            i, j = tour[k], tour[(k + 1) % n]
            new[i][j] += quality
            new[j][i] += quality
    return new


def run_aco(dist, n_ants, iterations, rng,
            alpha=1.0, beta=2.0, rho=0.5, tau0=1.0, gate=0.0):
    """Муравьиный алгоритм на маленьком TSP. Вернуть (маршрут, его длину).

    Каждая итерация: n_ants муравьёв строят маршрут рулеткой по
    transition_probabilities, стартуя из города 0; качество маршрута —
    1/длина; феромон обновляется через update_pheromone.

    run_aco(dist, 8, 30, random.Random(0))  ->  ([0, 1, 2, 3, 4], 12.0)

    Реальный аналог: AMRO-S. Города — типы задач, рёбра — «какой агент
    берёт какой тип», феромон — интерпретируемое свидетельство маршрутизации.

    Ловушка: rng вызывается ровно один раз на выбор города. Лишний вызов
    (например, «на всякий случай» перед проверкой) ломает воспроизводимость.
    """
    n = len(dist)
    pheromone = [[0.0 if i == j else tau0 for j in range(n)] for i in range(n)]
    best_tour, best_len = None, math.inf

    for _ in range(iterations):
        deposits = []
        for _ in range(n_ants):
            tour = [0]
            unvisited = set(range(1, n))
            while unvisited:
                probs = transition_probabilities(
                    pheromone[tour[-1]], dist[tour[-1]], sorted(unvisited), alpha, beta
                )
                # рулетка: один вызов rng, накопленная сумма до попадания
                roll, acc, choice = rng.random(), 0.0, None
                for city, p in probs.items():
                    acc += p
                    if roll < acc:
                        choice = city
                        break
                if choice is None:          # хвост от накопленной ошибки float
                    choice = max(probs, key=probs.get)
                tour.append(choice)
                unvisited.remove(choice)
            length = tour_length(tour, dist)
            deposits.append((tour, 1.0 / length))
            if length < best_len:
                best_tour, best_len = list(tour), length
        pheromone = update_pheromone(pheromone, deposits, rho, gate)
    return best_tour, best_len
