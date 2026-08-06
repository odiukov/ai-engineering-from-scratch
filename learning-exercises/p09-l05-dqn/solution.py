"""
Deep Q-Networks (DQN) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def one_hot(index, size):
    """Вектор признаков состояния: нули и одна единица на позиции index.

    one_hot(0, 3)  ->  [1.0, 0.0, 0.0]
    one_hot(2, 3)  ->  [0.0, 0.0, 1.0]

    Так табличное состояние подаётся в сеть. Индекс вне [0, size) — это не
    «просто ноль», а испорченная кодировка, из-за которой два разных
    состояния станут неразличимы. Брось ValueError.

    В настоящем DQN на месте one_hot стоит CNN по кадрам Atari, но роль та
    же: превратить состояние в вектор чисел.
    """
    if not 0 <= index < size:
        raise ValueError(f"index {index} out of range for size {size}")
    features = [0.0] * size
    features[index] = 1.0
    return features


def init_net(n_in, n_hidden, n_out, rng):
    """Собрать сеть n_in -> n_hidden (ReLU) -> n_out. Вернуть dict с W1,b1,W2,b2.

    init_net(4, 3, 2, rng)  ->  {"W1": 3 строки по 4, "b1": [0,0,0],
                                 "W2": 2 строки по 3, "b2": [0,0]}

    Веса — rng.gauss(0, 0.2), смещения — нули. Нулевые веса нельзя: все
    скрытые нейроны считали бы одно и то же и остались бы одинаковыми
    навсегда (см. Phase 3 · 08 про инициализацию).

    rng передаётся снаружи: DQN без фиксированного seed невозможно
    отлаживать, у него слишком много источников случайности (init, epsilon,
    выборка из буфера).
    """
    return {
        "W1": [[rng.gauss(0.0, 0.2) for _ in range(n_in)] for _ in range(n_hidden)],
        "b1": [0.0] * n_hidden,
        "W2": [[rng.gauss(0.0, 0.2) for _ in range(n_hidden)] for _ in range(n_out)],
        "b2": [0.0] * n_out,
    }


def clone_net(net):
    """Глубокая копия сети — из этого делается target network.

    clone_net(net)["W1"][0] is not net["W1"][0]   ->  True

    Ловушка на всё занятие: dict(net) или net.copy() копируют только внешний
    словарь, а строки матриц остаются ТЕМИ ЖЕ списками. Target-сеть тогда
    меняется вместе с online-сетью, target network фактически отключается, и
    обучение начинает расходиться — при этом код выглядит правильным.
    """
    return {
        "W1": [row[:] for row in net["W1"]],
        "b1": net["b1"][:],
        "W2": [row[:] for row in net["W2"]],
        "b2": net["b2"][:],
    }


def forward(net, x):
    """Прямой проход: linear -> ReLU -> linear. Вернуть (q, h).

    q — список Q-значений по действиям, h — скрытая активация (нужна для
    backprop, поэтому возвращаем её, а не выбрасываем).

    forward({"W1": [[1.0], [-1.0]], "b1": [0.0, 0.0],
             "W2": [[1.0, 2.0]], "b2": [0.5]}, [3.0])
        ->  ([3.5], [3.0, 0.0])

    Выходной слой БЕЗ активации: Q-значения бывают любого знака и любой
    величины. Прижать выход сигмоидой или ReLU — классическая ошибка, после
    которой сеть не может выучить отрицательные Q.
    """
    h = []
    for row, b in zip(net["W1"], net["b1"]):
        z = b + sum(w * xi for w, xi in zip(row, x))
        h.append(max(0.0, z))  # ReLU
    q = []
    for row, b in zip(net["W2"], net["b2"]):
        q.append(b + sum(w * hi for w, hi in zip(row, h)))
    return q, h


def dqn_target(reward, gamma, q_next, done):
    """Цель регрессии DQN: reward + gamma * max(q_next), либо reward в терминале.

    q_next — список выходов TARGET-сети на s'.

    dqn_target(-1.0, 0.9, [-5.0, -2.0, -9.0], False)  ->  -2.8
    dqn_target(-1.0, 0.9, [-5.0, -2.0, -9.0], True)   ->  -1.0

    Тот же max, что в табличном Q-learning из урока 04. Новое только одно:
    q_next приходит из ЗАМОРОЖЕННОЙ копии сети. Если подставить сюда online-
    сеть, цель поедет вместе с предсказанием — «погоня за своим хвостом»,
    и loss начнёт колебаться вместо падения.
    """
    if done:
        return reward
    return reward + gamma * max(q_next)


def double_dqn_target(reward, gamma, q_next_online, q_next_target, done):
    """Цель Double DQN: действие выбирает online-сеть, оценивает target-сеть.

    double_dqn_target(0.0, 1.0, [1.0, 5.0], [7.0, 3.0], False)  ->  3.0
        (argmax по online — индекс 1, берём q_next_target[1] = 3.0,
         а обычный dqn_target взял бы max(target) = 7.0)

    Зачем: max по шумным оценкам систематически завышен. Если оба списка —
    независимый шум вокруг нуля, среднее обычной цели уползает вверх, а
    средняя цель Double DQN остаётся около нуля. Выбор и оценка сделаны по
    разным выборкам, и смещение уходит.

    В терминале ведёт себя как dqn_target: bootstrap не нужен вовсе.
    """
    if done:
        return reward
    best = max(range(len(q_next_online)), key=lambda i: q_next_online[i])
    return reward + gamma * q_next_target[best]


def train_step(online, target, batch, gamma=0.99, lr=0.05):
    """Один шаг SGD по TD-ошибке на минибатче. Вернуть средний loss ДО шага.

    batch — список (x, action_index, reward, x_next, done), где x и x_next
    уже векторы признаков.

    Loss на одном примере: 0.5 * (q[a] - y)^2, где y = dqn_target(...).
    Градиенты по всему батчу СНАЧАЛА накапливаются, и только потом
    применяются с шагом lr / len(batch).

    train_step(net, target, batch, lr=0.0)  ->  loss, сеть не изменилась

    Порядок принципиален: если обновлять веса внутри цикла по батчу, поздние
    примеры увидят уже сдвинутые W2, и это будет не градиент минибатча, а
    что-то другое. Проверить всё это проще всего численно: центральная
    разность по одному весу обязана совпасть с (W_до - W_после) / lr.

    Градиенты вручную, ровно то, что делает `loss.backward()` в torch:
      dL/dq[a] = q[a] - y                        (это td_error)
      dL/dW2[a][j] = td * h[j],   dL/db2[a] = td
      dL/dh[j] = td * W2[a][j] * [h[j] > 0]      (ReLU пропускает или нет)
      dL/dW1[j][k] = dL/dh[j] * x[k],  dL/db1[j] = dL/dh[j]
    """
    n_hidden = len(online["b1"])
    n_out = len(online["b2"])
    n_in = len(online["W1"][0])
    dW1 = [[0.0] * n_in for _ in range(n_hidden)]
    db1 = [0.0] * n_hidden
    dW2 = [[0.0] * n_hidden for _ in range(n_out)]
    db2 = [0.0] * n_out
    total_loss = 0.0

    for x, a, reward, x_next, done in batch:
        q, h = forward(online, x)
        q_next, _ = forward(target, x_next)
        y = dqn_target(reward, gamma, q_next, done)
        td = q[a] - y
        total_loss += 0.5 * td * td

        db2[a] += td
        for j in range(n_hidden):
            dW2[a][j] += td * h[j]

        for j in range(n_hidden):
            if h[j] <= 0.0:
                continue  # ReLU закрыт, градиент дальше не идёт
            grad_h = td * online["W2"][a][j]
            db1[j] += grad_h
            for k in range(n_in):
                dW1[j][k] += grad_h * x[k]

    scale = lr / len(batch)
    for j in range(n_hidden):
        online["b1"][j] -= scale * db1[j]
        for k in range(n_in):
            online["W1"][j][k] -= scale * dW1[j][k]
    for a in range(n_out):
        online["b2"][a] -= scale * db2[a]
        for j in range(n_hidden):
            online["W2"][a][j] -= scale * dW2[a][j]
    return total_loss / len(batch)


class ReplayBuffer:
    """Кольцевой буфер переходов — первый из трёх трюков DQN.

    buf = ReplayBuffer(3)
    buf.push(("a",)); buf.push(("b",)); buf.push(("c",)); buf.push(("d",))
    len(buf)      ->  3
    buf.sample(2, rng)  ->  два РАЗНЫХ перехода из ("b",), ("c",), ("d",)

    Зачем он нужен: подряд идущие переходы почти одинаковы, и градиенты по
    ним сильно коррелированы. Случайная выборка из буфера разрывает эту
    корреляцию и заодно позволяет переиспользовать редкие удачные переходы
    много раз. Без буфера нейросетевой Q-learning на Atari расходится.
    """

    def __init__(self, capacity):
        """Создать пустой буфер на capacity переходов."""
        self.capacity = capacity
        self.items = []

    def push(self, transition):
        """Добавить переход, вытеснив самый старый, если места нет."""
        if len(self.items) >= self.capacity:
            # pop(0) на списке это O(n); для учебного буфера нормально,
            # в production здесь collections.deque(maxlen=...) или индекс по кругу
            self.items.pop(0)
        self.items.append(transition)

    def sample(self, batch, rng):
        """Вернуть batch РАЗНЫХ переходов (выборка без повторов)."""
        return rng.sample(self.items, batch)

    def __len__(self):
        """Сколько переходов сейчас лежит в буфере."""
        return len(self.items)
