"""
MARL: MADDPG, QMIX, MAPPO — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import itertools
import math


def assignment_cost(starts, pellets, assignment):
    """Суммарная манхэттенская стоимость назначения агентов на цели.

    starts[i] — клетка i-го агента, pellets[j] — клетка j-й цели,
    assignment[i] — номер цели, к которой идёт i-й агент.

    assignment_cost([(0, 0), (2, 0)], [(3, 0), (10, 0)], [0, 1])  ->  11.0
    assignment_cost([(0, 0), (2, 0)], [(3, 0), (10, 0)], [1, 0])  ->  11.0
    assignment_cost([(0, 0), (2, 0)], [(3, 0), (10, 0)], [0, 0])  ->  inf

    Если два агента назначены на одну цель, вторая цель остаётся несобранной
    и задача не выполнена — стоимость math.inf, а не «сколько-то».
    Это то самое связывание агентов, которое ломает value decomposition:
    суммарная стоимость перестаёт быть суммой независимых слагаемых.
    """
    if sorted(assignment) != list(range(len(pellets))):
        return math.inf
    total = 0.0
    for i, j in enumerate(assignment):
        ax, ay = starts[i]
        px, py = pellets[j]
        total += abs(ax - px) + abs(ay - py)
    return total


def independent_assignment(starts, pellets):
    """Каждый агент сам выбирает ближайшую цель. Вернуть (назначение, стоимость).

    Ровно decentralized execution: агент видит только своё наблюдение и
    ничего не знает о выборе соседа.

    independent_assignment([(0, 0), (9, 0)], [(1, 0), (8, 0)])  ->  ([0, 1], 2.0)
    independent_assignment([(0, 0), (2, 0)], [(3, 0), (10, 0)])  ->  ([0, 0], inf)

    При равенстве расстояний берём цель с меньшим индексом — иначе результат
    зависит от порядка перебора и тест перестаёт быть детерминированным.

    Второй пример — это и есть non-stationarity в миниатюре: локально
    оптимальный выбор каждого агента даёт совместно недопустимый исход.
    """
    assignment = []
    for ax, ay in starts:
        best_j, best_d = 0, math.inf
        for j, (px, py) in enumerate(pellets):
            d = abs(ax - px) + abs(ay - py)
            if d < best_d:          # строгое <: при равенстве остаётся меньший j
                best_j, best_d = j, d
        assignment.append(best_j)
    return assignment, assignment_cost(starts, pellets, assignment)


def centralized_assignment(starts, pellets):
    """Централизованное назначение перебором перестановок. (назначение, стоимость).

    Это centralized training из CTDE: на этапе обучения видно всё, поэтому
    можно решать связанную задачу целиком.

    centralized_assignment([(0, 0), (2, 0)], [(3, 0), (10, 0)])  ->  ([0, 1], 11.0)
    centralized_assignment([(0, 0), (9, 0)], [(1, 0), (8, 0)])   ->  ([0, 1], 2.0)

    Перебор факториальный: 10 агентов — 3.6 млн вариантов. Ровно поэтому
    MADDPG не масштабируется дальше ~10 агентов: критик видит все действия.

    При равенстве стоимостей берём лексикографически меньшую перестановку.
    """
    best_assignment, best_cost = None, math.inf
    for perm in itertools.permutations(range(len(pellets))):
        cost = assignment_cost(starts, pellets, list(perm))
        if cost < best_cost:        # строгое <: первая из равных побеждает
            best_assignment, best_cost = list(perm), cost
    return best_assignment, best_cost


def mix(q_values, weights, bias, monotone=True):
    """Смешивающая сеть QMIX: Q_tot из индивидуальных Q_i.

    Q_tot = sum(w_i * q_i) + bias. При monotone=True веса берутся по модулю —
    так статья Rashid 2018 обеспечивает dQ_tot/dQ_i >= 0.

    mix([1.0, 2.0], [0.5, 2.0], 0.5)                   ->  5.0
    mix([1.0, 2.0], [0.5, -2.0], 0.0)                  ->  4.5
    mix([1.0, 2.0], [0.5, -2.0], 0.0, monotone=False)  ->  -3.5

    monotone=False оставлен не для красоты: на нём видно, ЧТО именно ломается,
    если монотонность не обеспечить. Смотри тесты про argmax.

    Реальный QMIX — двухслойная гиперсеть, которая генерит веса из глобального
    состояния и жмёт их через abs. Здесь линейный слой: механика та же.
    """
    total = float(bias)
    for q, w in zip(q_values, weights):
        total += (abs(w) if monotone else w) * q
    return total


def mix_gradient(q_values, weights, monotone=True):
    """Аналитический градиент dQ_tot/dQ_i смешивающей сети.

    mix_gradient([1.0, 2.0], [0.5, -2.0])                  ->  [0.5, 2.0]
    mix_gradient([1.0, 2.0], [0.5, -2.0], monotone=False)  ->  [0.5, -2.0]

    Для линейного слоя градиент от q_values не зависит — аргумент оставлен
    ради проверки численной производной центральной разностью в тестах.

    Неотрицательность каждой компоненты при monotone=True — это ВСЁ, чего
    требует QMIX. Из неё одной следует, что argmax по совместному действию
    раскладывается на независимые argmax по агентам.
    """
    return [abs(w) if monotone else float(w) for w in weights]


def joint_argmax(q_tables, weights, bias, monotone=True):
    """argmax_a Q_tot перебором ВСЕХ совместных действий. Вернуть кортеж индексов.

    q_tables[i] — список Q-значений i-го агента по его действиям.

    joint_argmax([[0.0, 1.0], [0.0, 1.0]], [1.0, 1.0], 0.0)   ->  (1, 1)
    joint_argmax([[0.0, 1.0], [0.0, 1.0]], [1.0, -1.0], 0.0, monotone=False)
        ->  (1, 0)

    Размер перебора — произведение числа действий: 5 агентов по 10 действий
    это уже 100 000 вариантов. Именно этот взрыв QMIX и обходит.

    При равенстве Q_tot берём первый по порядку itertools.product.
    """
    best_actions, best_value = None, -math.inf
    for actions in itertools.product(*[range(len(t)) for t in q_tables]):
        value = mix([t[a] for t, a in zip(q_tables, actions)], weights, bias, monotone)
        if value > best_value:      # строгое >: первый из равных побеждает
            best_actions, best_value = actions, value
    return best_actions


def decentralized_argmax(q_tables):
    """Каждый агент берёт argmax по своей строке. Вернуть кортеж индексов.

    decentralized_argmax([[0.0, 1.0], [3.0, 2.0]])  ->  (1, 0)
    decentralized_argmax([[5.0]])                   ->  (0,)

    Стоимость линейная по числу агентов, никакого перебора совместных
    действий. Это и есть decentralized execution: агент смотрит только на
    свои Q. При равенстве — меньший индекс действия.
    """
    result = []
    for table in q_tables:
        best_a, best_q = 0, -math.inf
        for a, q in enumerate(table):
            if q > best_q:
                best_a, best_q = a, q
        result.append(best_a)
    return tuple(result)


def centralized_advantage(returns, baseline=None):
    """Преимущества относительно централизованной оценки ценности.

    baseline=None означает «взять среднее по returns» — это и есть
    централизованный критик MAPPO, который видит все траектории сразу.

    centralized_advantage([1.0, 2.0, 3.0])       ->  [-1.0, 0.0, 1.0]
    centralized_advantage([1.0, 2.0, 3.0], 0.0)  ->  [1.0, 2.0, 3.0]

    Смысл вычитания базы: сумма преимуществ становится нулевой, а сумма их
    квадратов — минимально возможной по всем базам. Это буквально то
    гашение дисперсии, ради которого MAPPO ставит централизованную V.

    Ловушка: база вычитается, а не делится. Нормировка на стандартное
    отклонение — отдельный приём, к самой идее базы отношения не имеет.
    """
    if not returns:
        return []
    if baseline is None:
        baseline = sum(returns) / len(returns)
    return [r - baseline for r in returns]
