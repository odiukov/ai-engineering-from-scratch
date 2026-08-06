"""
Sim-to-real transfer — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Мир для всех функций один и тот же — «обрыв» 4x6:

      col 0  1  2  3  4  5
  row 0  .  .  .  .  .  .
  row 1  .  .  .  .  .  .
  row 2  .  .  .  .  .  .
  row 3  S  X  X  X  X  G

S = старт (3, 0), G = цель (3, 5), X = обрыв. Шаг стоит -1. Шаг в обрыв
стоит -20 и возвращает агента на старт, эпизод при этом НЕ кончается.
slip — вероятность, что моторы «проскользнут» и агент уедет перпендикулярно
задуманному направлению. Это и есть параметр, который в sim один, а в real
другой.
"""

import random


def perpendicular(action):
    """Два направления, перпендикулярных заданному.

    perpendicular("up")     ->  ("left", "right")
    perpendicular("down")   ->  ("left", "right")
    perpendicular("left")   ->  ("up", "down")

    Порядок фиксированный, чтобы rng.choice по этому кортежу давал
    воспроизводимый результат.
    """
    return ("left", "right") if action in ("up", "down") else ("up", "down")


def slip_step(state, action, slip, rng, rows=4, cols=6):
    """Один шаг «симулятора». Вернуть (next_state, reward, done).

    С вероятностью slip действие подменяется на случайное перпендикулярное.
    Дальше обычный сдвиг с упором в стены. Обрыв (3, 1)..(3, 4) даёт -20.0 и
    отправляет агента на старт (3, 0), НЕ завершая эпизод. Цель (3, 5) даёт
    -1.0 и done=True. Всё остальное — -1.0 и done=False.

    slip_step((2, 0), "right", 0.0, random.Random(0))  ->  ((2, 1), -1.0, False)
    slip_step((2, 1), "down", 0.0, random.Random(0))   ->  ((3, 0), -20.0, False)
    slip_step((2, 5), "down", 0.0, random.Random(0))   ->  ((3, 5), -1.0, True)

    Ловушка: кубик на slip надо бросать ВСЕГДА, даже при slip = 0.0. Иначе
    последовательность обращений к rng зависит от slip, и один и тот же seed
    даст разные прогоны на разных slip — сравнивать политики станет нечем.
    """
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    start, goal = (rows - 1, 0), (rows - 1, cols - 1)
    cliff = {(rows - 1, c) for c in range(1, cols - 1)}

    # кубик бросается безусловно: порядок обращений к rng не должен
    # зависеть от значения slip
    if rng.random() < slip:
        action = rng.choice(perpendicular(action))
    dr, dc = deltas[action]
    r, c = state
    nr = min(max(r + dr, 0), rows - 1)
    nc = min(max(c + dc, 0), cols - 1)
    if (nr, nc) in cliff:
        return start, -20.0, False
    return (nr, nc), -1.0, (nr, nc) == goal


def randomize(rng, ranges):
    """Domain randomization: сэмплировать каждый параметр из своего диапазона.

    ranges — словарь имя -> (low, high). Вернуть словарь имя -> значение,
    равномерно взятое из [low, high].

    randomize(random.Random(0), {"slip": (0.0, 0.0)})   ->  {"slip": 0.0}
    randomize(random.Random(0), {"slip": (0.2, 0.2)})   ->  {"slip": 0.2}

    Порядок обхода параметров обязан быть детерминированным (обычный dict в
    Python его и сохраняет), иначе одинаковый seed даст разные наборы.

    Так выглядит DR в Isaac Lab: массы, коэффициенты трения, задержки
    моторов, положение камеры — каждое со своим диапазоном, новый сэмпл на
    каждый эпизод.
    """
    return {name: rng.uniform(low, high) for name, (low, high) in ranges.items()}


def epsilon_greedy(q_row, rng, epsilon):
    """Выбрать действие: с вероятностью epsilon случайное, иначе лучшее.

    q_row — словарь действие -> Q-значение.

    epsilon_greedy({"a": 1.0, "b": 5.0}, random.Random(0), 0.0)  ->  "b"

    Кубик бросается ПЕРВЫМ, до обращения к таблице: так порядок вызовов rng
    не зависит от содержимого Q.
    """
    if rng.random() < epsilon:
        return rng.choice(list(q_row))
    return max(q_row, key=lambda a: q_row[a])


def train_q(slip_range=(0.0, 0.0), episodes=3000, alpha=0.2, gamma=0.98,
            epsilon=0.15, max_steps=100, rng=None):
    """Q-learning в «симуляторе». Вернуть таблицу {state: {action: value}}.

    На КАЖДЫЙ эпизод slip сэмплируется заново через randomize из
    slip_range. slip_range=(0.0, 0.0) — обучение без рандомизации,
    slip_range=(0.0, 0.3) — domain randomization.

    Ровно в этом и весь урок: одна и та же функция, один и тот же бюджет
    шагов, разница только в ширине диапазона. Узкое обучение даст политику,
    идеальную на slip=0 и катастрофическую на slip=0.4; широкое — чуть
    хуже дома и в разы лучше «на железе».

    Таблица растёт лениво: строка создаётся при первом визите в состояние.
    """
    rng = rng or random.Random(0)
    actions = ("up", "down", "left", "right")
    start = (3, 0)
    Q = {}

    for _ in range(episodes):
        # новый «экземпляр симулятора» на каждый эпизод
        slip = randomize(rng, {"slip": slip_range})["slip"]
        state = start
        for _ in range(max_steps):
            row = Q.setdefault(state, dict.fromkeys(actions, 0.0))
            action = epsilon_greedy(row, rng, epsilon)
            next_state, reward, done = slip_step(state, action, slip, rng)
            next_row = Q.setdefault(next_state, dict.fromkeys(actions, 0.0))
            target = reward if done else reward + gamma * max(next_row.values())
            new_row = dict(row)
            new_row[action] = row[action] + alpha * (target - row[action])
            Q[state] = new_row
            state = next_state
            if done:
                break
    return Q


def evaluate(Q, slip, rng, episodes=200, max_steps=100):
    """Средняя сумма награды жадной политики при заданном slip.

    Действия берутся строго жадно (epsilon = 0). Состояние, которого нет в
    таблице, считается строкой из нулей — политика обязана хоть что-то
    ответить и на клетке, где никогда не была. Это буквально то, что
    происходит с роботом на железе.

    evaluate(train_q(), 0.0, random.Random(1))  ->  -7.0  (кратчайший путь)

    Эпизод обрывается по max_steps: при большом slip политика может не
    добраться до цели вообще, и без предела прогон не закончится.
    """
    actions = ("up", "down", "left", "right")
    start = (3, 0)
    total = 0.0
    for _ in range(episodes):
        state = start
        episode_total = 0.0
        for _ in range(max_steps):
            row = Q.get(state, dict.fromkeys(actions, 0.0))
            action = epsilon_greedy(row, rng, 0.0)
            state, reward, done = slip_step(state, action, slip, rng)
            episode_total += reward
            if done:
                break
        total += episode_total
    return total / episodes


def sweep(Q, slips, rng, episodes=200):
    """Прогнать одну политику по списку slip. Вернуть {slip: средняя награда}.

    sweep(Q, [0.0, 0.5], random.Random(1))  ->  {0.0: ..., 0.5: ...}

    Это обязательный отчёт перед деплоем: «в поддержке обучения политика
    почти оптимальна, за её пределами деградирует плавно». Одно число на
    одном slip ничего не доказывает.
    """
    return {slip: evaluate(Q, slip, rng, episodes) for slip in slips}


def widen_range(slip_range, score, target, step=0.05, cap=0.9):
    """Один шаг ADR-куррикулума: расширить диапазон, если политика справилась.

    Справилась (score >= target) — верхняя граница растёт на step, но не выше
    cap. Не справилась — диапазон возвращается без изменений. Нижняя граница
    не двигается никогда.

    widen_range((0.0, 0.1), -9.0, -12.0)              ->  (0.0, 0.15)
    widen_range((0.0, 0.1), -30.0, -12.0)             ->  (0.0, 0.1)
    widen_range((0.0, 0.88), -9.0, -12.0, cap=0.9)    ->  (0.0, 0.9)

    Награды здесь отрицательные, так что «справилась» — это score БОЛЬШЕ
    порога. Перепутать знак легко, и тогда куррикулум будет расширяться
    ровно тогда, когда политика сломалась.

    Так работает Automatic Domain Randomization из OpenAI Dactyl: диапазон
    ползёт вверх сам, по мере того как политика его осваивает.
    """
    low, high = slip_range
    if score >= target:
        return (low, min(high + step, cap))
    return (low, high)
