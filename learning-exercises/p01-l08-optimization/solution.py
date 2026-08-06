"""
Оптимизация — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def rosenbrock(params):
    """Функция Розенброка — классический полигон для оптимизаторов.

    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

    rosenbrock([1.0, 1.0])   ->  0.0    (единственный минимум)
    rosenbrock([0.0, 0.0])   ->  1.0
    rosenbrock([-1.0, 1.0])  ->  4.0

    Значение всегда >= 0: сумма двух квадратов. Минимум лежит на дне узкого
    изогнутого оврага — найти его легко, дойти по нему трудно.
    """
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(params):
    """Аналитический градиент функции Розенброка: [df/dx, df/dy].

    rosenbrock_gradient([1.0, 1.0])  ->  [0.0, 0.0]   (в минимуме)
    rosenbrock_gradient([0.0, 0.0])  ->  [-2.0, 0.0]

    df/dx = -2*(1 - x) - 400*x*(y - x^2)
    df/dy = 200*(y - x^2)

    Ловушка: в df/dx множитель 400, а не 200 — внутренняя производная
    (y - x^2) по x равна -2x, и двойка уходит в коэффициент.
    """
    x, y = params
    gap = y - x * x
    return [-2 * (1 - x) - 400 * x * gap, 200 * gap]


def sgd_momentum_step(params, grads, velocity, lr, momentum=0.9):
    """Один шаг SGD с моментом. Возвращает (новые params, новая velocity).

    sgd_momentum_step([1.0], [2.0], [0.0], 0.1)        ->  ([0.8], [2.0])
    sgd_momentum_step([0.8], [2.0], [2.0], 0.1)        ->  ([0.42], [3.8])
    sgd_momentum_step([1.0], [2.0], [0.0], 0.1, 0.0)   ->  ([0.8], [2.0])

    v = momentum * v + grad;  w = w - lr * v.

    При momentum = 0 это обычный градиентный спуск. Момент копит скорость
    в стабильном направлении и гасит зигзаг поперёк оврага.

    Функция чистая: входные списки не менять, возвращать новые.
    """
    new_velocity = [momentum * v + g for v, g in zip(velocity, grads)]
    # новые списки, а не in-place: вызывающий может хранить историю шагов
    new_params = [p - lr * v for p, v in zip(params, new_velocity)]
    return new_params, new_velocity


def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Один шаг Adam. Возвращает (новые params, новый m, новый v).

    t — номер ЭТОГО шага, начиная с 1 (нужен для bias correction).

    adam_step([1.0], [2.0], [0.0], [0.0], 1, lr=0.1)      ->  ([0.9], ...)
    adam_step([1.0], [1e6], [0.0], [0.0], 1, lr=0.1)      ->  ([0.9], ...)

    m = beta1*m + (1-beta1)*g          первый момент, «куда»
    v = beta2*v + (1-beta2)*g^2        второй момент, «насколько сильно»
    m_hat = m / (1 - beta1^t)          bias correction
    v_hat = v / (1 - beta2^t)
    w = w - lr * m_hat / (sqrt(v_hat) + eps)

    Смысл деления на sqrt(v_hat): у каждого веса свой эффективный learning
    rate. Первый шаг из нулевого состояния имеет длину примерно lr — и это
    не зависит от масштаба градиента. Забудешь bias correction — первый шаг
    съёжится в (1 - beta1) = 10 раз.
    """
    new_m = [beta1 * mi + (1 - beta1) * g for mi, g in zip(m, grads)]
    new_v = [beta2 * vi + (1 - beta2) * g * g for vi, g in zip(v, grads)]
    # деление на (1 - beta^t) компенсирует старт из нулей: без него
    # ранние шаги систематически занижены
    m_hat = [mi / (1 - beta1 ** t) for mi in new_m]
    v_hat = [vi / (1 - beta2 ** t) for vi in new_v]
    new_params = [
        p - lr * mh / (math.sqrt(vh) + eps)
        for p, mh, vh in zip(params, m_hat, v_hat)
    ]
    return new_params, new_m, new_v


def minimize_momentum(grad_fn, start, lr, momentum=0.9, steps=1000):
    """Прогнать SGD с моментом заданное число шагов. Вернуть финальную точку.

    minimize_momentum(lambda p: [2*p[0]], [10.0], 0.05)  ->  примерно [0.0]

    Скорость стартует нулевой. Шаг делай через sgd_momentum_step, не
    переписывай формулу.
    """
    params = list(start)
    velocity = [0.0] * len(params)
    for _ in range(steps):
        params, velocity = sgd_momentum_step(
            params, grad_fn(params), velocity, lr, momentum
        )
    return params


def minimize_adam(grad_fn, start, lr=0.01, steps=1000):
    """Прогнать Adam заданное число шагов. Вернуть финальную точку.

    minimize_adam(lambda p: [2*p[0]], [10.0], 0.1, 500)  ->  примерно [0.0]

    Оба момента стартуют нулевыми, счётчик t — с единицы. Шаг делай через
    adam_step.
    """
    params = list(start)
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    for step in range(1, steps + 1):
        params, m, v = adam_step(params, grad_fn(params), m, v, step, lr)
    return params


def exponential_decay(lr0, step, decay=0.999):
    """Экспоненциальное затухание learning rate: lr0 * decay^step.

    exponential_decay(0.1, 0)          ->  0.1
    exponential_decay(0.1, 1000)       ->  0.0367...
    exponential_decay(0.1, 10, 0.5)    ->  0.00009765625

    На нулевом шаге расписание обязано вернуть исходный lr, а не уже
    уменьшенный — иначе первый шаг тихо теряет множитель.
    """
    return lr0 * decay ** step


def cosine_annealing(lr_max, lr_min, step, total_steps):
    """Косинусное затухание — расписание из современного обучения трансформеров.

    cosine_annealing(0.1, 0.0, 0, 1000)     ->  0.1     (старт)
    cosine_annealing(0.1, 0.0, 500, 1000)   ->  0.05    (середина)
    cosine_annealing(0.1, 0.0, 1000, 1000)  ->  0.0     (финиш)

    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * step / total_steps))

    В отличие от экспоненциального, косинус в начале падает медленно,
    в середине быстро, в конце снова медленно — модель успевает
    и продвинуться, и осесть.
    """
    cos_term = math.cos(math.pi * step / total_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos_term)
