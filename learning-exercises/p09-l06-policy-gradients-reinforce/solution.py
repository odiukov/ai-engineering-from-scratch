"""
Policy gradient: REINFORCE с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def softmax(logits):
    """Превратить произвольные числа в распределение вероятностей.

    Это `torch.nn.functional.softmax` в одну строку, только руками.

    softmax([0.0, 0.0])        ->  [0.5, 0.5]
    softmax([1.0, 0.0])        ->  [0.731..., 0.268...]
    softmax([1001.0, 1000.0])  ->  ровно то же, что softmax([1.0, 0.0])

    Ловушка: наивный exp(l) на logits=1000 даёт OverflowError, а на -1000 —
    ноль во всех компонентах и деление на ноль. Лечится вычитанием максимума:
    softmax(l) = softmax(l - max l) математически тождественно, а численно
    безопасно. Этот сдвиг — обязательная часть ответа.
    """
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def policy_probs(theta, features):
    """pi(.|s) для линейной softmax-политики: по строке theta на каждое действие.

    theta — список из n_actions строк, каждая длиной len(features).

    policy_probs([[0.0, 0.0], [0.0, 0.0]], [1.0, 0.0])  ->  [0.5, 0.5]
    policy_probs([[5.0, 0.0], [0.0, 0.0]], [1.0, 0.0])  ->  [0.993..., 0.006...]

    Нулевая theta даёт равномерную политику — удобная стартовая точка: агент
    ничего не знает и пробует всё одинаково.

    В настоящем PPO вместо скалярного произведения стоит трансформер, но
    softmax-голова и дальнейшая математика ровно те же.
    """
    logits = [sum(w * x for w, x in zip(row, features)) for row in theta]
    return softmax(logits)


def sample_action(probs, rng):
    """Выбрать индекс действия по probs методом обратной CDF.

    sample_action([0.0, 1.0, 0.0], rng)  ->  всегда 1
    sample_action([0.5, 0.5], rng)       ->  примерно 50/50

    Политика в policy gradient СТОХАСТИЧНА по построению: именно поэтому её
    можно дифференцировать по параметрам, а argmax из Q-learning — нельзя.

    Ловушка: из-за ошибок округления сумма probs может оказаться 0.9999999,
    и при x близком к 1 цикл дойдёт до конца. Верни последний индекс, а не
    None — иначе редкий эпизод падает на None вместо действия.
    """
    x = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if x <= cumulative:
            return i
    return len(probs) - 1


def grad_log_pi(probs, action):
    """Градиент log pi(a|s) по ЛОГИТАМ: onehot(action) - probs.

    grad_log_pi([0.25, 0.25, 0.25, 0.25], 1)  ->  [-0.25, 0.75, -0.25, -0.25]
    grad_log_pi([0.9, 0.1], 0)                ->  [0.1, -0.1]

    Формула вся: d/d z_k log softmax(z)[a] = [k == a] - p_k. Выучи наизусть,
    она стоит в каждом policy-gradient-коде мира.

    Два свойства, которые обязаны выполняться и которые стоит проверить
    численно центральной разностью:
      * сумма компонент строго 0 — вероятность перекладывается, а не
        создаётся из ничего;
      * компонента выбранного действия равна 1 - p_a > 0, то есть шаг вверх
        по градиенту делает это действие вероятнее.
    """
    grad = [-p for p in probs]
    grad[action] += 1.0
    return grad


def returns_to_go(rewards, gamma=0.99):
    """Список G_t = r_t + gamma*r_{t+1} + ... для каждого шага.

    returns_to_go([-1.0] * 3, 1.0)  ->  [-3.0, -2.0, -1.0]
    returns_to_go([1.0, 1.0], 0.5)  ->  [1.5, 1.0]

    Именно reward-to-go, а не полная награда эпизода G_0 для всех шагов:
    действие на шаге t не могло повлиять на прошлые награды, и подмешивать
    их — значит добавлять в градиент шум с нулевым средним. Смещения не
    появится, но дисперсия вырастет зря.

    Обратный проход, O(T). Собирать хвостовые суммы заново для каждого t —
    O(T^2) и на длинных эпизодах это доминирует над всем остальным.
    """
    out = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def grid_rollout(theta, rng, grid=4, terminal=(3, 3), start=(0, 0), max_steps=100):
    """Прогнать эпизод на 4x4 GridWorld. Вернуть [(features, action, reward), ...].

    Действия закодированы индексами: 0 up, 1 down, 2 left, 3 right.
    features — one-hot состояния длиной grid*grid, то есть theta имеет
    4 строки по 16 чисел.

    grid_rollout(theta_staircase, rng)  ->  6 шагов, награда -1 на каждом

    В траекторию идут признаки, а не сами координаты: reinforce_grad работает
    с векторами, ему всё равно, была это сетка или кадр Atari.

    max_steps обязателен. Свежая (нулевая) theta даёт равномерную политику, и
    та бродит десятки шагов; политика, застрявшая в углу, не выйдет никогда.
    """
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    state = start
    trajectory = []
    for _ in range(max_steps):
        features = [0.0] * (grid * grid)
        features[state[0] * grid + state[1]] = 1.0
        action = sample_action(policy_probs(theta, features), rng)
        dr, dc = deltas[action]
        nr = min(max(state[0] + dr, 0), grid - 1)
        nc = min(max(state[1] + dc, 0), grid - 1)
        state = (nr, nc)
        trajectory.append((features, action, -1.0))
        if state == terminal:
            break
    return trajectory


def reinforce_grad(theta, trajectory, gamma=0.99, baseline=0.0):
    """Оценка градиента REINFORCE по одному эпизоду. Вернуть матрицу как theta.

    grad[i][j] = sum_t (G_t - baseline) * grad_log_pi(pi(.|s_t), a_t)[i] * x_t[j]

    reinforce_grad(theta, traj, baseline=огромный)  ->  все знаки перевернулись

    Это градиент ПО ВОСХОЖДЕНИЮ: шагать надо theta += lr * grad, с плюсом.
    Минус здесь — самая частая ошибка, после которой агент уверенно учится
    работать хуже.

    Baseline вычитается из G_t и НЕ вносит смещения: E[b * grad log pi] = 0,
    потому что сумма grad_log_pi по действиям равна нулю. Дисперсию же он
    режет в разы — на этом стоят A2C, PPO и GRPO.

    Вероятности пересчитываются из theta прямо здесь, а не берутся
    сохранёнными из rollout: иначе функция перестанет быть настоящим
    градиентом по theta, и численная проверка это поймает. Сурогатная
    функция, чей это градиент: L(theta) = sum_t A_t * log pi_theta(a_t|s_t).
    """
    grad = [[0.0] * len(row) for row in theta]
    returns = returns_to_go([r for _, _, r in trajectory], gamma)
    for (features, action, _), G in zip(trajectory, returns):
        advantage = G - baseline
        probs = policy_probs(theta, features)
        dlog = grad_log_pi(probs, action)
        for i in range(len(theta)):
            scale = advantage * dlog[i]
            if scale == 0.0:
                continue
            row = grad[i]
            for j, x in enumerate(features):
                if x != 0.0:  # one-hot: ненулевая ровно одна координата
                    row[j] += scale * x
    return grad


def train_reinforce(episodes, lr=0.05, gamma=0.99, use_baseline=True, rng=None,
                    grid=4, terminal=(3, 3), max_steps=100):
    """Обучить линейную softmax-политику. Вернуть (theta, returns_log).

    returns_log — недисконтированная сумма награды за эпизод, по одному числу
    на эпизод. Кривая обязана расти: с -60 у случайной политики до почти -6.

    train_reinforce(1500)[1][-100:]  ->  в среднем около -7

    Baseline — бегущее среднее всех наблюдённых G_t. Без него та же lr либо
    учится в разы медленнее, либо разносит политику: градиент масштабируется
    возвратами порядка -60, а не отклонениями от них.

    Стартуем с нулевой theta: равномерная политика, максимальная энтропия,
    ничего не сломано заранее.
    """
    rng = rng or random.Random(0)
    theta = [[0.0] * (grid * grid) for _ in range(4)]
    baseline = 0.0
    seen = 0
    returns_log = []
    for _ in range(episodes):
        trajectory = grid_rollout(theta, rng, grid, terminal, max_steps=max_steps)
        returns = returns_to_go([r for _, _, r in trajectory], gamma)
        b = baseline if use_baseline else 0.0
        grad = reinforce_grad(theta, trajectory, gamma, b)
        for i in range(len(theta)):
            for j in range(len(theta[i])):
                theta[i][j] += lr * grad[i][j]  # ВОСХОЖДЕНИЕ, знак плюс
        # бегущее среднее по всем G_t, а не только по G_0: так baseline
        # ближе к типичному значению возврата на шаге
        for G in returns:
            seen += 1
            baseline += (G - baseline) / seen
        returns_log.append(sum(r for _, _, r in trajectory))
    return theta, returns_log
