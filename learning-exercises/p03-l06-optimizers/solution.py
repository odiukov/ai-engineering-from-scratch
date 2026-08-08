"""
Оптимизаторы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def sgd_step(params, grads, lr):
    """Шаг обычного градиентного спуска. Вернуть НОВЫЙ список параметров.

    sgd_step([1.0, 2.0], [0.5, -1.0], 0.1)  ->  [0.95, 2.1]

    Правило целиком: w -= lr * g.

    Один learning rate на все параметры — это и есть главная слабость
    SGD. Веса живут в разных масштабах, а шаг им положен один.
    """
    return [w - lr * g for w, g in zip(params, grads)]


def momentum_step(params, grads, velocity, lr, beta=0.9):
    """Шаг SGD с моментом. Вернуть (новые параметры, новая скорость).

    momentum_step([1.0], [1.0], [0.0], 0.1)  ->  ([0.9], [1.0])
    momentum_step([0.9], [1.0], [1.0], 0.1)  ->  ([0.71], [1.9])

    Правило:
        v = beta * v + g
        w -= lr * v

    beta = 0.9 хранит примерно десять последних градиентов (1/(1 - 0.9)).
    Совпадающие по направлению градиенты складываются и разгоняют шаг,
    противоположные гасят друг друга. Именно так лечится болтанка поперёк
    узкого оврага: «поперёк» меняет знак каждый шаг, «вдоль» — нет.

    Второй пример показывает разгон: тот же градиент 1.0, а шаг уже 0.19
    вместо 0.1.
    """
    new_velocity = [beta * v + g for v, g in zip(velocity, grads)]
    return [w - lr * v for w, v in zip(params, new_velocity)], new_velocity


def bias_correct(moment, beta, t):
    """Поправка на холодный старт скользящего среднего: moment / (1 - beta^t).

    bias_correct(0.1, 0.9, 1)   ->  1.0
    bias_correct(0.1, 0.9, 100) ->  0.10000...

    Зачем: на шаге 1 среднее m = (1 - beta)*g, то есть при beta = 0.9 оно
    в десять раз меньше настоящего градиента — просто потому, что
    стартовало с нуля. Деление на (1 - beta^t) возвращает масштаб.

    Второй пример показывает, что поправка сама себя выключает: 0.9^100
    практически ноль, знаменатель равен единице. Она важна первые десять
    шагов и бесполезна после пятидесяти.
    """
    return moment / (1.0 - beta**t)


def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Шаг Adam. Вернуть (параметры, m, v) — все новые списки.

    t — номер ЭТОГО шага, считая с единицы.

    adam_step([1.0], [1.0], [0.0], [0.0], 1, lr=0.1)  ->  (~[0.9], [0.1], [0.001])

    Правило:
        m = beta1*m + (1 - beta1)*g          первый момент, среднее градиента
        v = beta2*v + (1 - beta2)*g^2        второй момент, средний квадрат
        w -= lr * bias_correct(m, beta1, t) / (sqrt(bias_correct(v, beta2, t)) + eps)

    На первом шаге при ненулевом g поправки дают шаг
    lr * |g| / (|g| + eps): он лишь приближается к lr, когда |g| >> eps.
    При g = 0 шаг нулевой, а при крошечном градиенте eps заметно уменьшает
    его. Поэтому Adam почти нормирует величину, но не игнорирует её буквально.

    Ловушка: t начинается с 1, а не с 0. При t = 0 получится деление на
    ноль (1 - beta^0 = 0).
    """
    new_m = [beta1 * mi + (1.0 - beta1) * g for mi, g in zip(m, grads)]
    new_v = [beta2 * vi + (1.0 - beta2) * g * g for vi, g in zip(v, grads)]
    new_params = []
    for w, mi, vi in zip(params, new_m, new_v):
        m_hat = bias_correct(mi, beta1, t)
        v_hat = bias_correct(vi, beta2, t)
        new_params.append(w - lr * m_hat / (math.sqrt(v_hat) + eps))
    return new_params, new_m, new_v


def adamw_step(
    params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
):
    """Шаг AdamW: Adam плюс отвязанное затухание весов.

    adamw_step([1.0], [0.0], [0.0], [0.0], 1, lr=0.1, weight_decay=0.5)  ->  ([0.95], [0.0], [0.0])

    Правило: сначала обычный шаг Adam, потом отдельно
        w -= lr * weight_decay * w

    Ключевое слово — «отдельно». Если вместо этого добавить lambda*w к
    градиенту (классическая L2), то адаптивный делитель sqrt(v_hat)
    отмасштабирует и регуляризацию тоже: параметры с шумным градиентом
    получат её меньше, с тихим — больше. Это не то, чего хотели.

    Пример проверяет именно развязку: градиент нулевой, Adam ничего не
    делает, а вес всё равно ужимается на lr*weight_decay.

    AdamW — оптимизатор по умолчанию для BERT, GPT, LLaMA и Stable Diffusion.
    """
    stepped, new_m, new_v = adam_step(params, grads, m, v, t, lr, beta1, beta2, eps)
    decayed = [w - lr * weight_decay * old for w, old in zip(stepped, params)]
    return decayed, new_m, new_v


def run_sgd(grad_fn, start, lr, steps):
    """Прогнать обычный SGD. Вернуть параметры после steps шагов.

    grad_fn(params) -> список градиентов той же длины.

    run_sgd(lambda p: [2 * p[0]], [1.0], 0.1, 1)  ->  [0.8]

    Скучная, но нужная обёртка: без неё не сравнить оптимизаторы честно,
    на одной и той же задаче и с одного и того же старта.
    """
    params = list(start)
    for _ in range(steps):
        params = sgd_step(params, grad_fn(params), lr)
    return params


def run_momentum(grad_fn, start, lr, steps, beta=0.9):
    """Прогнать SGD с моментом. Вернуть параметры после steps шагов.

    Скорость стартует с нулей — по одному нулю на параметр.

    run_momentum(lambda p: [2 * p[0]], [1.0], 0.1, 1)  ->  [0.8]

    На первом шаге момент ещё пуст и совпадает с обычным SGD; разница
    накапливается дальше.
    """
    params = list(start)
    velocity = [0.0] * len(params)
    for _ in range(steps):
        params, velocity = momentum_step(params, grad_fn(params), velocity, lr, beta)
    return params


def run_adam(grad_fn, start, lr, steps):
    """Прогнать Adam. Вернуть параметры после steps шагов.

    Оба момента стартуют с нулей, счётчик t идёт с единицы.

    run_adam(lambda p: [2 * p[0]], [1.0], 0.1, 1)  ->  примерно [0.9]

    Сравни с run_sgd на том же входе: SGD дал 0.8, потому что послушался
    величины градиента, Adam — примерно 0.9, потому что |g| намного больше
    eps и первый шаг почти равен lr.

    На плохо обусловленной задаче (одна координата в миллион раз круче
    другой) SGD обязан выбрать lr по самой крутой, иначе разойдётся, — и
    по пологой после этого не двигается вовсе. Adam делит каждую
    координату на её собственный второй момент и идёт по обеим с
    одинаковой скоростью. Тесты это меряют.
    """
    params = list(start)
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    for t in range(1, steps + 1):
        params, m, v = adam_step(params, grad_fn(params), m, v, t, lr)
    return params


def noisy_grad(grad_fn, params, sigma, seed):
    """Градиент с шумом: к каждой координате прибавить random.gauss(0, sigma).

    noisy_grad(lambda p: [1.0], [0.0], 0.0, seed=0)  ->  [1.0]
    noisy_grad(f, p, 0.1, seed=1) == noisy_grad(f, p, 0.1, seed=1)  ->  True

    Буква S в SGD — это «стохастический»: настоящий градиент считается по
    случайному мини-батчу и всегда шумит. Шум мешает сходиться, но
    помогает не залипать в узких минимумах.

    seed — параметр, а не глобальный вызов random.seed(). Заводи
    random.Random(seed) локально, иначе испортишь генератор всем вокруг и
    ни один замер нельзя будет повторить.
    """
    rng = random.Random(seed)
    return [g + rng.gauss(0.0, sigma) for g in grad_fn(params)]
