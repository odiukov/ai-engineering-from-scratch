"""
Temporal Difference: Q-learning и SARSA — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import random


def grid_step(state, action, grid=4, terminal=(3, 3)):
    """Один шаг среды 4x4 GridWorld. Вернуть (next_state, reward, done).

    grid_step((0, 0), "down")   ->  ((1, 0), -1.0, False)
    grid_step((3, 2), "right")  ->  ((3, 3), -1.0, True)
    grid_step((3, 3), "up")     ->  ((3, 3),  0.0, True)

    TD-методам, как и Monte Carlo, доступен только этот `step`. Разница с MC
    в другом: TD обновляется ПОСЛЕ КАЖДОГО вызова, не дожидаясь конца
    эпизода. Поэтому TD умеет учиться и в задачах без терминала.
    """
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    if state == terminal:
        return state, 0.0, True
    dr, dc = deltas[action]
    r, c = state
    nr = min(max(r + dr, 0), grid - 1)
    nc = min(max(c + dc, 0), grid - 1)
    return (nr, nc), -1.0, (nr, nc) == terminal


def td_error(reward, gamma, value_next, value_now):
    """TD-ошибка: delta = reward + gamma * value_next - value_now.

    td_error(-1.0, 0.9, -5.0, -5.5)  ->  -0.0 ... точнее -1 + 0.9*(-5) + 5.5 = 0.0
    td_error(0.0, 1.0, 10.0, 0.0)    ->  10.0   приятная неожиданность
    td_error(-1.0, 0.0, -99.0, -1.0) ->  0.0    при gamma=0 будущее не важно

    Это онлайновый аналог (G_t - V(s_t)) из Monte Carlo: вместо полного
    возврата берём «награда за шаг плюс наша же оценка того, куда попали».
    Отсюда bootstrapping: оценка входит в собственную цель.

    Ноль означает не «ошибка исчезла», а «оценка согласована с одним шагом
    среды». Ровно то же условие, что и уравнение Беллмана в уроке 02.
    """
    return reward + gamma * value_next - value_now


def epsilon_greedy_action(q_row, rng, epsilon=0.1):
    """С вероятностью epsilon случайное действие, иначе argmax по q_row.

    q_row — dict {action: q_value}.

    epsilon_greedy_action({"a": 1.0, "b": 2.0}, rng, 0.0)  ->  всегда "b"
    epsilon_greedy_action({"a": 1.0, "b": 2.0}, rng, 1.0)  ->  50/50

    epsilon=0 обязан давать полностью детерминированное поведение. epsilon=1
    превращает агента в равномерно случайного — на таких данных Q-learning
    всё равно выучивает Q*, а SARSA нет, и это главный эксперимент урока.
    """
    if rng.random() < epsilon:
        return rng.choice(list(q_row))
    return max(q_row, key=q_row.get)


def bootstrap_q_learning(q_next_row, done):
    """Оценка V(s') по-Q-learning: max_a' Q(s', a'), либо 0.0 в терминале.

    bootstrap_q_learning({"up": -9.0, "down": -1.0}, False)  ->  -1.0
    bootstrap_q_learning({"up": -9.0, "down": -1.0}, True)   ->   0.0

    max означает «дальше я пойду жадно», независимо от того, что агент
    реально сделает. Именно это делает Q-learning off-policy: цель не
    зависит от поведения.

    Ловушка терминала: если done, то будущего нет и bootstrap строго 0.0.
    Подставить сюда max по строке терминала — обычный источник значений,
    уползающих в минус бесконечность на absorbing state.
    """
    if done:
        return 0.0
    return max(q_next_row.values())


def bootstrap_sarsa(q_next_row, action_next, done):
    """Оценка V(s') по-SARSA: Q(s', a') для ФАКТИЧЕСКИ выбранного a'.

    bootstrap_sarsa({"up": -9.0, "down": -1.0}, "up", False)  ->  -9.0
    bootstrap_sarsa({"up": -9.0, "down": -1.0}, "up", True)   ->   0.0

    Имя алгоритма — это кортеж (s, a, r, s', a'). Пятый элемент здесь:
    берём то действие, которое eps-жадная политика реально выбрала, а не
    лучшее. Поэтому SARSA учит Q^pi текущей политики — вместе со стоимостью
    её случайных вылазок.

    На cliff-walking из-за этого SARSA держится подальше от обрыва, а
    Q-learning идёт по самому краю.
    """
    if done:
        return 0.0
    return q_next_row[action_next]


def bootstrap_expected_sarsa(q_next_row, probs, done):
    """Оценка V(s') по-Expected-SARSA: sum_a' pi(a'|s') * Q(s', a').

    probs — dict {action: prob} той же формы, что q_next_row.

    bootstrap_expected_sarsa({"a": 0.0, "b": 4.0}, {"a": 0.5, "b": 0.5}, False)  ->  2.0
    bootstrap_expected_sarsa({"a": 0.0, "b": 4.0}, {"a": 0.0, "b": 1.0}, False)  ->  4.0

    Та же цель, что у SARSA, но без выборки a': математическое ожидание
    считается точно. Дисперсия падает, смещение не появляется.

    Два предельных случая стоит проверить тестом:
      * probs сосредоточены на argmax  -> получается bootstrap_q_learning;
      * probs равномерны              -> получается среднее по строке.
    """
    if done:
        return 0.0
    return sum(probs[a] * q for a, q in q_next_row.items())


def q_learning(episodes, alpha=0.1, gamma=0.99, epsilon=0.1, rng=None, grid=4,
               terminal=(3, 3), actions=("up", "down", "left", "right"),
               start=(0, 0), max_steps=200):
    """Табличный Q-learning. Вернуть (Q, returns) — таблица и список сумм награды.

    q_learning(3000)[0][(0, 0)]  ->  max по строке около -5.85 (это V*(0,0))

    Один шаг обучения:
      target_bootstrap = bootstrap_q_learning(Q[s'], done)
      Q[s][a] += alpha * td_error(r, gamma, target_bootstrap, Q[s][a])

    returns — недисконтированная сумма награды за эпизод, по одному числу на
    эпизод. По ней рисуют learning curve; она обязана расти.

    Q инициализируем нулями. На сетке с наградой -1 это оптимистичная
    инициализация: непробованное действие выглядит лучше любого пробованного,
    и агент исследует даже при маленьком epsilon.
    """
    rng = rng or random.Random(0)
    Q = {}
    returns = []

    def row(state):
        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}
        return Q[state]

    for _ in range(episodes):
        state = start
        total = 0.0
        for _ in range(max_steps):
            action = epsilon_greedy_action(row(state), rng, epsilon)
            state_next, reward, done = grid_step(state, action, grid, terminal)
            total += reward
            boot = bootstrap_q_learning(row(state_next), done)
            Q[state][action] += alpha * td_error(reward, gamma, boot, Q[state][action])
            state = state_next
            if done:
                break
        returns.append(total)
    return Q, returns


def sarsa(episodes, alpha=0.1, gamma=0.99, epsilon=0.1, rng=None, grid=4,
          terminal=(3, 3), actions=("up", "down", "left", "right"),
          start=(0, 0), max_steps=200):
    """Табличный SARSA. Вернуть (Q, returns) — та же форма, что у q_learning.

    Отличие от q_learning ровно в одной строке: bootstrap берётся не по max,
    а по действию a', которое политика выбрала для следующего шага. Значит
    a' надо выбрать ДО обновления, а потом им же и шагнуть — иначе это уже
    не SARSA.

    sarsa(3000, epsilon=1.0)[0][(0, 0)]
        ->  сильно хуже -5.85: on-policy честно платит за случайные вылазки

    Порядок внутри эпизода: выбрали a, шагнули, выбрали a', обновили Q(s,a),
    перешли к (s', a'). Действие переносится на следующую итерацию, а не
    выбирается заново.
    """
    rng = rng or random.Random(0)
    Q = {}
    returns = []

    def row(state):
        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}
        return Q[state]

    for _ in range(episodes):
        state = start
        action = epsilon_greedy_action(row(state), rng, epsilon)
        total = 0.0
        for _ in range(max_steps):
            state_next, reward, done = grid_step(state, action, grid, terminal)
            total += reward
            # a' выбираем до обновления: он часть цели, а не следствие её
            action_next = None if done else epsilon_greedy_action(
                row(state_next), rng, epsilon
            )
            boot = bootstrap_sarsa(row(state_next), action_next, done)
            Q[state][action] += alpha * td_error(reward, gamma, boot, Q[state][action])
            if done:
                break
            state, action = state_next, action_next
        returns.append(total)
    return Q, returns
