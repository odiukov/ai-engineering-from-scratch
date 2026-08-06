"""
Actor-critic: A2C и A3C

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p09-l07-actor-critic-a2c-a3c
Разбор:  /check-code p09-l07-actor-critic-a2c-a3c
"""

import math


def softmax(logits):
    """Превращает логиты в распределение вероятностей действий.

    softmax([0.0, 0.0])        ->  [0.5, 0.5]
    softmax([1.0, 1.0, 1.0])   ->  [1/3, 1/3, 1/3]
    softmax([0.0, 1000.0])     ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум логитов
    перед возведением в экспоненту — результат математически тот же, а
    переполнения нет.

    Это `torch.nn.functional.softmax` в одну строку. Actor выдаёт логиты,
    softmax делает из них политику pi(a|s), из которой сэмплируют действие.
    """
    raise NotImplementedError


def entropy(probs):
    """Энтропия политики H(pi) = -sum p * log p. В натах.

    entropy([0.5, 0.5])       ->  0.6931...  (== log 2, максимум для 2 действий)
    entropy([1.0, 0.0])       ->  0.0        (детерминированная политика)

    Ловушка: log(0) это -inf. Нулевые вероятности просто пропускай — предел
    p*log(p) при p -> 0 равен нулю.

    Зачем в AI: entropy bonus c_e * H в лоссе A2C не даёт политике
    схлопнуться в детерминированную и перестать исследовать среду.
    """
    raise NotImplementedError


def grad_log_pi(probs, action):
    """Градиент log pi(action | s) по логитам. Список той же длины, что probs.

    grad_log_pi([0.5, 0.5], 0)      ->  [0.5, -0.5]
    grad_log_pi([0.2, 0.8], 1)      ->  [-0.2, 0.2]

    Формула короткая: onehot(action) - probs. Сумма компонент всегда 0 —
    softmax нормирован, поднять одно действие можно только опустив остальные.

    Это то, что PyTorch считает за тебя в `log_softmax(z)[a].backward()`.
    Именно этот вектор умножается на advantage в policy gradient.
    """
    raise NotImplementedError


def discounted_returns(rewards, gamma=0.99, last_value=0.0):
    """Monte Carlo возвраты G_t для каждого шага, с bootstrap на конце.

    discounted_returns([1.0, 1.0], gamma=0.5)              ->  [1.5, 1.0]
    discounted_returns([0.0], gamma=0.9, last_value=10.0)  ->  [9.0]
    discounted_returns([1.0, 2.0], gamma=0.0)              ->  [1.0, 2.0]

    G_t = r_t + gamma * G_{t+1}, где G_T = last_value (bootstrap через
    критика, если эпизод обрезали, и 0.0 если он честно завершился).

    Считать надо ОДНИМ проходом справа налево. Наивный двойной цикл даёт
    O(T^2) и на роллауте из 2048 шагов это уже заметно.
    """
    raise NotImplementedError


def td_residuals(rewards, values, gamma=0.99, last_value=0.0):
    """TD-остатки delta_t = r_t + gamma * V(s_{t+1}) - V(s_t). Список длины T.

    td_residuals([1.0], [0.0], gamma=0.9)              ->  [1.0]
    td_residuals([1.0], [1.0], gamma=0.0)              ->  [0.0]  (критик точен)
    td_residuals([0.0, 0.0], [1.0, 1.0], gamma=1.0)    ->  [0.0, -1.0]

    values[t] — оценка критика в состоянии s_t. Для последнего шага
    V(s_{t+1}) взять из last_value, потому что values такой же длины, как
    rewards, и values[T] не существует.

    delta_t — это одношаговая оценка advantage. Смещённая (использует V), зато
    с куда меньшей дисперсией, чем G_t - V(s_t).
    """
    raise NotImplementedError


def gae_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    """GAE(lambda): вернуть кортеж (advantages, returns).

    advantages[t] = sum_l (gamma*lam)^l * delta_{t+l}
    returns[t]    = advantages[t] + values[t]   (цель для критика)

    gae_advantages([1.0, 1.0], [0.0, 0.0], gamma=1.0, lam=0.0)  ->  ([1.0, 1.0], [1.0, 1.0])
    gae_advantages([1.0, 1.0], [0.0, 0.0], gamma=1.0, lam=1.0)  ->  ([2.0, 1.0], [2.0, 1.0])

    Две крайности стоит проверить руками:
      lam = 0  —  чистый TD, advantages == td_residuals (максимум смещения,
                  минимум дисперсии);
      lam = 1  —  чистый Monte Carlo, advantages == discounted_returns минус
                  values (нет смещения, максимум дисперсии).
    lam = 0.95 — дефолт 2026 года, ручка между этими двумя.

    Считается одним проходом справа налево, как и возвраты.
    """
    raise NotImplementedError


def normalize(xs):
    """Привести список к нулевому среднему и единичному стандартному отклонению.

    normalize([1.0, 2.0, 3.0])  ->  [-1.2247..., 0.0, 1.2247...]
    normalize([5.0])            ->  [5.0]        (одному элементу нормировать нечего)
    normalize([2.0, 2.0])       ->  [0.0, 0.0]   (без деления на ноль!)

    Ловушка: у константного списка sd == 0. Прибавь к знаменателю 1e-8, иначе
    ZeroDivisionError на первом же роллауте, где все advantage совпали.

    Нормировка advantage по батчу — одна из самых дешёвых стабилизаций A2C и
    PPO: масштаб градиента перестаёт зависеть от масштаба награды.
    """
    raise NotImplementedError


def actor_critic_step(theta, w, x, action, advantage, target_value,
                      lr_a=0.05, lr_v=0.1):
    """Один совместный шаг актора и критика. Вернуть (new_theta, new_w).

    theta — матрица [n_actions][n_features], логиты считаются как theta[a] . x.
    w     — вектор критика длины n_features, V(s) = w . x.

    Актор поднимается ПО градиенту advantage * log pi(action):
        theta[i][j] += lr_a * advantage * grad_log_pi[i] * x[j]
    Критик спускается ПО градиенту (target - V)^2:
        w[j] += lr_v * (target_value - V(s)) * x[j]

    actor_critic_step([[0.0], [0.0]], [0.0], [1.0], 0, 1.0, 2.0)
        ->  ([[0.025], [-0.025]], [0.2])

    Знаки принципиальны: актор МАКСИМИЗИРУЕТ, критик МИНИМИЗИРУЕТ. Перепутать
    их — самая частая ошибка, и она даёт молча деградирующую политику.

    Функция ничего не мутирует: theta и w на входе остаются прежними.
    """
    raise NotImplementedError
