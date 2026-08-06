"""
Динамическое программирование: policy iteration и value iteration — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def transitions(state, action, slip=0.0, grid=4, terminal=(3, 3)):
    """Модель среды: список исходов (next_state, reward, prob) для (s, a).

    Это то, чего у model-free методов НЕТ. Здесь мы «читаем исходники среды».

    transitions((0, 0), "down")             ->  [((1, 0), -1.0, 1.0)]
    transitions((0, 0), "down", slip=0.1)   ->  [((1, 0), -1.0, 0.9),
                                                 ((0, 0), -1.0, 0.05),
                                                 ((0, 1), -1.0, 0.05)]
    transitions((3, 3), "up")               ->  [((3, 3), 0.0, 1.0)]

    slip — вероятность соскользнуть в ПЕРПЕНДИКУЛЯРНОЕ направление (по slip/2
    на каждое из двух), назад агент не едет никогда. Стена не пускает: сдвиг
    обрезается границами сетки, но шаг всё равно стоит -1.

    Исходы с нулевой вероятностью в список не попадают: при slip=0 список
    ровно из одного элемента.

    Сумма вероятностей обязана быть равна 1.0 при любом slip — это первое,
    что стоит проверить, если value iteration «сходится не туда».
    """
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    if state == terminal:
        return [(state, 0.0, 1.0)]

    def move(direction):
        dr, dc = deltas[direction]
        r, c = state
        return (min(max(r + dr, 0), grid - 1), min(max(c + dc, 0), grid - 1))

    perp = ("left", "right") if action in ("up", "down") else ("up", "down")
    candidates = [(action, 1.0 - slip)] + [(d, slip / 2.0) for d in perp]
    # нулевые вероятности выкидываем: они ничего не добавляют в backup,
    # зато засоряют список и мешают сравнивать его в тестах
    return [(move(d), -1.0, p) for d, p in candidates if p > 0.0]


def sup_norm(v_a, v_b):
    """Sup-норма расстояния между двумя value-функциями: max_s |v_a(s) - v_b(s)|.

    sup_norm({1: 0.0, 2: 0.0}, {1: 3.0, 2: -1.0})  ->  3.0
    sup_norm(V, V)                                 ->  0.0

    Именно максимум, а не среднее. Теорема о сжатии сформулирована в
    sup-норме, и критерий остановки DP тоже. Среднее спрячет одно
    несошедшееся состояние среди пятнадцати сошедшихся.
    """
    return max(abs(v_a[s] - v_b[s]) for s in v_a)


def q_value(state, action, V, gamma=0.99, slip=0.0, grid=4, terminal=(3, 3)):
    """Один backup Беллмана для пары (s, a): sum_{s'} p * (r + gamma * V(s')).

    q_value((0, 0), "down", {все нули})         ->  -1.0
    q_value((3, 2), "right", {все нули})        ->  -1.0
    q_value((0, 0), "up", V)                    ->  -1 + gamma * V[(0, 0)]

    Ловушка порядка: gamma умножает ТОЛЬКО V(s'), а не всю скобку. Если
    написать p * gamma * (r + V(s')), награда тоже начнёт дисконтироваться,
    и значения поедут на несколько процентов — ошибка, которую легко
    не заметить глазами.
    """
    return sum(
        p * (r + gamma * V[s_next])
        for s_next, r, p in transitions(state, action, slip, grid, terminal)
    )


def bellman_sweep(V, gamma=0.99, slip=0.0, grid=4, terminal=(3, 3),
                  actions=("up", "down", "left", "right")):
    """Один синхронный проход оператора оптимальности: V'(s) = max_a Q(s, a).

    Вернуть НОВЫЙ dict, старый не менять — это Jacobi-вариант. В терминале
    значение всегда 0.0.

    bellman_sweep({все нули})[(0, 0)]   ->  -1.0
    bellman_sweep({все нули})[(3, 3)]   ->  0.0

    Этот оператор — gamma-сжатие в sup-норме:
        sup_norm(T V1, T V2) <= gamma * sup_norm(V1, V2)
    отсюда и единственность неподвижной точки, и геометрическая скорость
    сходимости value iteration. Проверь это свойство тестом — оно и есть
    причина, по которой DP работает.
    """
    new_V = {}
    for r in range(grid):
        for c in range(grid):
            state = (r, c)
            if state == terminal:
                new_V[state] = 0.0
            else:
                new_V[state] = max(
                    q_value(state, a, V, gamma, slip, grid, terminal) for a in actions
                )
    return new_V


def policy_evaluation(policy, gamma=0.99, slip=0.0, grid=4, terminal=(3, 3),
                      tol=1e-12, max_iter=20000):
    """Найти V^pi: крутить V(s) <- sum_a pi(a|s) Q(s,a), пока не перестанет двигаться.

    policy — функция state -> {action: prob}.

    policy_evaluation(lambda s: {"down": 1.0} if s[0] < 3 else {"right": 1.0})
        ->  V[(0,0)] примерно -5.852 при gamma=0.99 и slip=0

    Здесь max_a НЕТ: политика фиксирована, мы просто усредняем по ней.
    Разница между этой функцией и bellman_sweep — ровно разница между
    «оценить политику» и «улучшить её».

    Обновление на месте (Gauss-Seidel) сходится быстрее Jacobi: соседи,
    посчитанные в этом же проходе, уже несут свежую информацию.
    """
    states = [(r, c) for r in range(grid) for c in range(grid)]
    V = {s: 0.0 for s in states}
    for _ in range(max_iter):
        delta = 0.0
        for state in states:
            if state == terminal:
                continue
            v = 0.0
            for action, pi_a in policy(state).items():
                if pi_a == 0.0:
                    continue
                v += pi_a * q_value(state, action, V, gamma, slip, grid, terminal)
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        if delta < tol:
            break
    return V


def greedy_policy(V, gamma=0.99, slip=0.0, grid=4, terminal=(3, 3),
                  actions=("up", "down", "left", "right")):
    """Шаг policy improvement: {state: argmax_a Q(s, a)} по данному V.

    greedy_policy(V_optimal)[(0, 0)]  ->  "down" или "right"
    greedy_policy({все нули})[(1, 1)] ->  "up"   все Q равны, берём первое

    Тай-брейк обязан быть детерминированным (первое действие в порядке
    `actions`). Иначе argmax будет каждый раз выбирать другое из равных, и
    проверка «политика перестала меняться» в policy_iteration никогда не
    сработает — цикл прокрутится все 100 итераций впустую.
    """
    policy = {}
    for r in range(grid):
        for c in range(grid):
            state = (r, c)
            if state == terminal:
                policy[state] = actions[0]
                continue
            best = actions[0]
            best_q = q_value(state, best, V, gamma, slip, grid, terminal)
            for action in actions[1:]:
                q = q_value(state, action, V, gamma, slip, grid, terminal)
                if q > best_q:  # строгое >, иначе тай-брейк съедет
                    best, best_q = action, q
            policy[state] = best
    return policy


def value_iteration(gamma=0.99, slip=0.0, grid=4, terminal=(3, 3), tol=1e-12,
                    max_iter=20000):
    """Value iteration. Вернуть (V_star, policy, sweeps).

    value_iteration(gamma=0.99)[0][(0, 0)]  ->  примерно -5.852
    value_iteration()[1][(0, 0)]            ->  "down" или "right"

    Схема: гонять bellman_sweep, пока sup_norm(V_new, V) не станет меньше
    tol, потом один раз извлечь жадную политику. Оценка и улучшение слиты
    в один проход — отличие от policy_iteration.

    Чем ближе gamma к единице, тем медленнее сходится: ошибка падает как
    gamma^n, так что 0.99 требует примерно вдвое больше проходов, чем 0.9.
    """
    V = {(r, c): 0.0 for r in range(grid) for c in range(grid)}
    sweeps = 0
    for _ in range(max_iter):
        new_V = bellman_sweep(V, gamma, slip, grid, terminal)
        sweeps += 1
        delta = sup_norm(new_V, V)
        V = new_V
        if delta < tol:
            break
    return V, greedy_policy(V, gamma, slip, grid, terminal), sweeps


def policy_iteration(gamma=0.99, slip=0.0, grid=4, terminal=(3, 3), tol=1e-12,
                     max_outer=100):
    """Policy iteration. Вернуть (V_star, policy, outer_iterations).

    Цикл: policy_evaluation до конца -> greedy_policy -> если политика не
    изменилась, останавливаемся.

    policy_iteration()[2]  ->  небольшое число, обычно 3-6 внешних итераций

    Стартуем с произвольной политики («всегда вверх» — заведомо плохой, что
    и хорошо: видно, что алгоритм её вытаскивает).

    Итог обязан совпасть с value_iteration до tol: у оператора Беллмана одна
    неподвижная точка, и оба алгоритма приходят именно в неё. Разошлись —
    ищи ошибку, а не «особенности алгоритма».
    """
    states = [(r, c) for r in range(grid) for c in range(grid)]
    policy = {s: "up" for s in states}
    V = {s: 0.0 for s in states}
    for outer in range(max_outer):
        # p=policy привязывает текущую политику значением, а не по ссылке:
        # замыкание на переменную цикла — классический источник тихих багов
        V = policy_evaluation(
            lambda s, p=policy: {p[s]: 1.0}, gamma, slip, grid, terminal, tol
        )
        new_policy = greedy_policy(V, gamma, slip, grid, terminal)
        if new_policy == policy:
            return V, policy, outer + 1
        policy = new_policy
    return V, policy, max_outer
