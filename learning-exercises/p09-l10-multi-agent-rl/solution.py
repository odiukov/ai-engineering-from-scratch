"""
Multi-agent RL — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import itertools
import random


def move(pos, action, size=4):
    """Сдвинуть агента на одну клетку, упираясь в стены.

    Действия: "up" уменьшает row, "down" увеличивает row, "left" уменьшает
    col, "right" увеличивает col. pos — кортеж (row, col).

    move((1, 1), "up")     ->  (0, 1)
    move((0, 0), "up")     ->  (0, 0)   (стена, остались на месте)
    move((3, 3), "right")  ->  (3, 3)   (стена справа)

    Ловушка: без обрезки координат агент уедет в (-1, 0), и Q-таблица тихо
    заполнится состояниями, которых в мире нет.
    """
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    dr, dc = deltas[action]
    r, c = pos
    # min/max вместо if-каскада: клетка за стеной просто прижимается к границе
    return (min(max(r + dr, 0), size - 1), min(max(c + dc, 0), size - 1))


def joint_step(state, actions, goal=(3, 3), size=4):
    """Один шаг Markov game на два агента. Вернуть (next_state, reward, done).

    state   — кортеж позиций двух агентов ((r1, c1), (r2, c2)).
    actions — кортеж их действий.

    Награда ОБЩАЯ: -1.0 за каждый шаг, +10.0 в момент, когда ОБА агента
    оказались в goal одновременно.

    joint_step(((2, 3), (3, 2)), ("down", "right"))  ->  (((3, 3), (3, 3)), 10.0, True)
    joint_step(((0, 0), (3, 0)), ("down", "right"))  ->  (((1, 0), (3, 1)), -1.0, False)
    joint_step(((2, 3), (0, 0)), ("down", "down"))   ->  (((3, 3), (1, 0)), -1.0, False)

    Обрати внимание на третий пример: один агент в goal — этого НЕ хватает.
    Именно из-за этого условия задача становится по-настоящему
    кооперативной: пришедший первым обязан ждать второго.
    """
    new1 = move(state[0], actions[0], size)
    new2 = move(state[1], actions[1], size)
    done = new1 == goal and new2 == goal
    return (new1, new2), (10.0 if done else -1.0), done


def joint_actions(actions=("up", "down", "left", "right"), n_agents=2):
    """Все совместные действия: декартова степень множества действий.

    joint_actions(("a", "b"), 2)       ->  [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
    len(joint_actions(("a", "b"), 3))  ->  8
    len(joint_actions())               ->  16

    Порядок обязан быть детерминированным (itertools.product его и даёт),
    иначе argmax по таблице будет прыгать между запусками.

    Здесь наглядно виден главный барьер MARL: размер |A|^n. Четыре действия
    и два агента — 16 вариантов, а десять агентов — миллион. Поэтому CTDE
    держит акторов раздельными и централизует только критика.
    """
    return list(itertools.product(actions, repeat=n_agents))


def epsilon_greedy(q_row, rng, epsilon):
    """Выбрать действие: с вероятностью epsilon случайное, иначе лучшее.

    q_row — словарь действие -> Q-значение.

    epsilon_greedy({"a": 1.0, "b": 5.0}, random.Random(0), 0.0)  ->  "b"
    epsilon_greedy({"a": 1.0, "b": 5.0}, random.Random(0), 1.0)  ->  "a" или "b"

    Случайность обязана идти ЧЕРЕЗ rng, а не через глобальный random:
    иначе один прогон не повторить, и отладить расходящийся MARL невозможно.

    Сначала бросай кубик, потом смотри в таблицу — так порядок обращений к
    rng не зависит от содержимого Q, и одинаковый seed даёт одинаковый прогон.
    """
    if rng.random() < epsilon:
        return rng.choice(list(q_row))
    return max(q_row, key=lambda a: q_row[a])


def q_learning_update(q_row, action, reward, next_row, alpha=0.1, gamma=0.95,
                      done=False):
    """Один шаг Q-learning. Вернуть НОВЫЙ словарь строки Q.

    target = reward, если done, иначе reward + gamma * max(next_row.values())
    q_row[action] += alpha * (target - q_row[action])

    q_learning_update({"a": 0.0}, "a", 1.0, {"a": 0.0}, alpha=0.5, done=True)
        ->  {"a": 0.5}
    q_learning_update({"a": 0.0}, "a", 0.0, {"a": 10.0}, alpha=1.0, gamma=0.5)
        ->  {"a": 5.0}

    Ловушка: bootstrap за терминальным состоянием. Если при done всё равно
    добавить gamma * max(next_row), агент выучит, что после конца эпизода
    его ждёт ещё награда, и значения уплывут.

    Остальные действия строки не меняются: обновляется только выбранное.
    """
    target = reward if done else reward + gamma * max(next_row.values())
    new_row = dict(q_row)
    new_row[action] = q_row[action] + alpha * (target - q_row[action])
    return new_row


def train_independent_q(episodes=800, alpha=0.1, gamma=0.95, epsilon=0.15,
                        max_steps=60, size=4, rng=None):
    """Independent Q-learning: у каждого агента своя таблица.

    Вернуть (Q1, Q2, returns_log): две таблицы вида
    {joint_state: {action: value}} и список суммарных наград по эпизодам.

    Старт: агенты в (0, 0) и (size-1, 0), цель — (size-1, size-1).

    Каждый агент видит полное joint state, но обновляет только СВОЁ действие,
    и общей наградой. Друг для друга агенты — часть среды, и эта среда
    нестационарна: пока сосед учится, оптимальный ответ на него меняется.
    Никаких гарантий сходимости здесь нет, но на плотной согласованной
    награде это работает.

    Всё случайное идёт через rng, чтобы прогон был воспроизводим.
    """
    rng = rng or random.Random(0)
    actions = ("up", "down", "left", "right")
    goal = (size - 1, size - 1)
    start = ((0, 0), (size - 1, 0))
    Q1, Q2 = {}, {}
    returns_log = []

    for _ in range(episodes):
        state = start
        total = 0.0
        for _ in range(max_steps):
            # строки таблиц создаём лениво: посещённых состояний много меньше,
            # чем всех |S|^2 возможных
            row1 = Q1.setdefault(state, dict.fromkeys(actions, 0.0))
            row2 = Q2.setdefault(state, dict.fromkeys(actions, 0.0))
            a1 = epsilon_greedy(row1, rng, epsilon)
            a2 = epsilon_greedy(row2, rng, epsilon)
            next_state, reward, done = joint_step(state, (a1, a2), goal, size)
            total += reward
            next1 = Q1.setdefault(next_state, dict.fromkeys(actions, 0.0))
            next2 = Q2.setdefault(next_state, dict.fromkeys(actions, 0.0))
            Q1[state] = q_learning_update(row1, a1, reward, next1, alpha, gamma, done)
            Q2[state] = q_learning_update(row2, a2, reward, next2, alpha, gamma, done)
            state = next_state
            if done:
                break
        returns_log.append(total)
    return Q1, Q2, returns_log


def train_joint_q(episodes=800, alpha=0.1, gamma=0.95, epsilon=0.15,
                  max_steps=60, size=4, rng=None):
    """Централизованный Q над совместными действиями. Вернуть (Q, returns_log).

    Q имеет вид {joint_state: {joint_action: value}}, где joint_action —
    кортеж из joint_actions(). Один обучающийся объект вместо двух, значит
    среда снова стационарна и обычные гарантии Q-learning возвращаются.

    Цена — размер строки: |A|^n вместо |A|. На двух агентах это 16 вместо 4,
    и уже здесь заметно, что подход не масштабируется.

    Старт и цель те же, что в train_independent_q, чтобы кривые обучения
    были сравнимы.
    """
    rng = rng or random.Random(0)
    all_joint = joint_actions(("up", "down", "left", "right"), 2)
    goal = (size - 1, size - 1)
    start = ((0, 0), (size - 1, 0))
    Q = {}
    returns_log = []

    for _ in range(episodes):
        state = start
        total = 0.0
        for _ in range(max_steps):
            row = Q.setdefault(state, dict.fromkeys(all_joint, 0.0))
            ja = epsilon_greedy(row, rng, epsilon)
            next_state, reward, done = joint_step(state, ja, goal, size)
            total += reward
            next_row = Q.setdefault(next_state, dict.fromkeys(all_joint, 0.0))
            Q[state] = q_learning_update(row, ja, reward, next_row, alpha, gamma, done)
            state = next_state
            if done:
                break
        returns_log.append(total)
    return Q, returns_log


def counterfactual_advantage(q_row, joint_action, agent_index, agent_probs):
    """COMA-advantage: вклад ОДНОГО агента в общий Q.

    A_i = Q(s, a) - sum_{a_i'} pi_i(a_i') * Q(s, (a_i', a_{-i}))

    То есть: «насколько мой ход лучше того, что вышло бы в среднем, если бы я
    сыграл иначе, а сосед — то же самое».

    q_row        — словарь joint_action -> Q-значение.
    joint_action — фактически сыгранная пара.
    agent_index  — чья это координата (0 или 1).
    agent_probs  — словарь действие -> вероятность для ЭТОГО агента.

    counterfactual_advantage({("a","x"): 4.0, ("b","x"): 0.0}, ("a","x"), 0,
                             {"a": 0.5, "b": 0.5})   ->  2.0

    Ключевое свойство: если Q не зависит от моего действия, advantage ровно
    ноль. Это и решает credit assignment — общая награда перестаёт
    начисляться агенту, который на неё не влиял.
    """
    baseline = 0.0
    for alt_action, p in agent_probs.items():
        # подменяем только СВОЮ координату, действие соседа заморожено
        alt = list(joint_action)
        alt[agent_index] = alt_action
        baseline += p * q_row[tuple(alt)]
    return q_row[joint_action] - baseline
