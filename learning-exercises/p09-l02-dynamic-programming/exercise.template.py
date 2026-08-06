"""
Динамическое программирование: policy iteration и value iteration

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p09-l02-dynamic-programming
Разбор:  /check-code p09-l02-dynamic-programming
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
    raise NotImplementedError


def sup_norm(v_a, v_b):
    """Sup-норма расстояния между двумя value-функциями: max_s |v_a(s) - v_b(s)|.

    sup_norm({1: 0.0, 2: 0.0}, {1: 3.0, 2: -1.0})  ->  3.0
    sup_norm(V, V)                                 ->  0.0

    Именно максимум, а не среднее. Теорема о сжатии сформулирована в
    sup-норме, и критерий остановки DP тоже. Среднее спрячет одно
    несошедшееся состояние среди пятнадцати сошедшихся.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
