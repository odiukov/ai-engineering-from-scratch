"""
Monte Carlo: обучение по полным эпизодам

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p09-l03-monte-carlo-methods
Разбор:  /check-code p09-l03-monte-carlo-methods
"""

import random


def grid_step(state, action, grid=4, terminal=(3, 3)):
    """Один шаг среды 4x4 GridWorld. Вернуть (next_state, reward, done).

    grid_step((0, 0), "down")   ->  ((1, 0), -1.0, False)
    grid_step((0, 0), "up")     ->  ((0, 0), -1.0, False)   стена
    grid_step((3, 3), "up")     ->  ((3, 3),  0.0, True)    absorbing

    Monte Carlo не знает модель среды: ему доступен только этот `step`, и
    вероятности переходов он никогда не увидит. Это и есть разница между
    model-based DP из урока 02 и model-free методами.
    """
    raise NotImplementedError


def returns_from(trajectory, gamma=0.99):
    """Список G_t для каждого шага траектории [(state, action, reward), ...].

    returns_from([(s, a, 1.0), (s, a, 1.0)], gamma=0.5)  ->  [1.5, 1.0]
    returns_from([(s, a, -1.0)] * 6, gamma=1.0)          ->  [-6, -5, -4, -3, -2, -1]

    Считать надо ОДНИМ проходом с конца по рекуррентности
        G_t = r_t + gamma * G_{t+1}
    Наивный двойной цикл (для каждого t суммировать хвост) даёт O(T^2) и на
    эпизодах в тысячу шагов становится узким местом всего обучения.
    """
    raise NotImplementedError


def incremental_mean(mean, value, count):
    """Обновить среднее одним новым наблюдением: mean + (value - mean) / count.

    count — номер этого наблюдения, начиная с 1.

    incremental_mean(0.0, 10.0, 1)   ->  10.0    первое наблюдение
    incremental_mean(10.0, 20.0, 2)  ->  15.0
    incremental_mean(0.0, 100.0, 100) -> 1.0     сотое почти не двигает

    Именно эта форма — «старое значение плюс шаг в сторону цели» — потом
    станет TD-обновлением: замени 1/count на константу alpha и получишь
    урок 04. Хранить все returns в списке не нужно.
    """
    raise NotImplementedError


def rollout(policy, rng, grid=4, terminal=(3, 3), start=(0, 0), max_steps=200):
    """Прогнать эпизод. Вернуть траекторию [(state, action, reward), ...].

    policy — функция (state, rng) -> action.

    rollout(lambda s, r: "down" if s[0] < 3 else "right", rng)
        ->  6 шагов, награда -1 на каждом

    В траекторию попадает состояние ДО шага. Терминал как состояние в
    список не входит: из него действий уже не делают.

    max_steps обязателен: случайная политика на этой сетке иногда бродит
    сотни шагов, а политика вроде «всегда вверх» не выйдет никогда, и MC
    просто зависнет.
    """
    raise NotImplementedError


def mc_evaluate(policy, episodes, gamma=0.99, rng=None, first_visit=True,
                grid=4, terminal=(3, 3), max_steps=200):
    """Оценить V^pi усреднением наблюдённых returns. Вернуть (V, counts).

    mc_evaluate(optimal_policy, 100)[0][(0, 0)]  ->  ровно -5.852 (дисперсии нет)
    mc_evaluate(uniform_policy, 5000)[0][(0, 0)] ->  около DP-ответа

    first_visit=True — считать состояние один раз за эпизод (первое
    посещение), False — каждое посещение. Первое проще анализировать
    (наблюдения независимы), второе выжимает больше данных из эпизода.

    В V попадают только посещённые состояния — это принципиальное отличие
    от DP: непосещённое состояние остаётся без оценки навсегда. Отсюда и
    вся возня с exploration.
    """
    raise NotImplementedError


def constant_alpha_mc(policy, episodes, alpha=0.1, gamma=0.99, rng=None,
                      grid=4, terminal=(3, 3), max_steps=200):
    """То же, но с постоянным шагом: V(s) <- V(s) + alpha * (G - V(s)).

    constant_alpha_mc(pol, 50, alpha=1.0)  ->  V(s) равен G последнего визита
    constant_alpha_mc(pol, 500, alpha=0.1) ->  экспоненциально сглаженное среднее

    Отличие от mc_evaluate ровно в одном: вместо 1/count стоит фиксированная
    alpha. Из-за этого старые эпизоды экспоненциально забываются, и оценка
    начинает СЛЕДИТЬ за меняющейся политикой вместо усреднения по всей её
    истории. В MC control политика меняется каждый эпизод, так что это не
    приятная мелочь, а необходимость.

    Оценка всегда остаётся выпуклой комбинацией наблюдённых G (при
    0 < alpha <= 1 и старте с 0), то есть вылететь за диапазон returns она
    не может. Если вылетела — alpha больше единицы.
    """
    raise NotImplementedError


def epsilon_greedy_action(q_row, rng, epsilon=0.1):
    """Выбрать действие: с вероятностью epsilon случайное, иначе argmax по Q.

    q_row — dict {action: q_value}.

    epsilon_greedy_action({"a": 1.0, "b": 2.0}, rng, 0.0)  ->  всегда "b"
    epsilon_greedy_action({"a": 1.0, "b": 2.0}, rng, 1.0)  ->  50/50

    При epsilon=0 поведение обязано стать полностью детерминированным —
    это самый быстрый способ проверить, что случайность не подмешивается
    где-то ещё.

    Ловушка: при epsilon=1 действие выбирается равномерно среди ВСЕХ
    действий, включая жадное. Поэтому жадное берётся с вероятностью
    (1 - epsilon) + epsilon/len(q_row), а не (1 - epsilon).
    """
    raise NotImplementedError


def mc_control(episodes, gamma=0.99, epsilon=0.1, rng=None, grid=4,
               terminal=(3, 3), actions=("up", "down", "left", "right"),
               max_steps=200):
    """MC control: оценить Q по эпизодам, вести себя epsilon-жадно. Вернуть (Q, greedy).

    Это policy iteration из урока 02, у которого шаг «оценить» заменён на
    «прогнать эпизоды и усреднить». Модель не нужна.

    mc_control(10000)[1][(0, 0)]  ->  "down" или "right"

    Q — dict {state: {action: value}}, greedy — dict {state: action}.
    Стартовое Q = 0 при отрицательных наградах работает как оптимистичная
    инициализация: непопробованное действие выглядит лучше любого
    попробованного, и агент сам себя заставляет исследовать.

    Ловушка: без epsilon > 0 жадная политика с нулевого Q заперта в первом
    же действии по тай-брейку и целые куски сетки не увидит никогда.
    """
    raise NotImplementedError
