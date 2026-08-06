"""
Отладка нейросетей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def has_nan_or_inf(values):
    """Есть ли хоть одно NaN или бесконечность в списке любой вложенности.

    has_nan_or_inf([1.0, 2.0])              ->  False
    has_nan_or_inf([1.0, float("nan")])     ->  True
    has_nan_or_inf([[1.0], [float("inf")]]) ->  True
    has_nan_or_inf([])                      ->  False

    Градиенты приходят матрицами, поэтому спускайся рекурсивно, а не
    только по верхнему уровню.

    Ловушка: NaN не равен сам себе, поэтому `v == float("nan")` всегда
    False. Проверять надо math.isnan, а бесконечность — math.isinf.
    """
    for v in values:
        if isinstance(v, (list, tuple)):
            if has_nan_or_inf(v):
                return True
        elif math.isnan(v) or math.isinf(v):
            return True
    return False


def find_bad_gradients(named_grads):
    """Имена слоёв, у которых в градиенте завелись NaN или inf. Список, отсортированный.

    find_bad_gradients({"fc1": [1.0], "fc2": [float("nan")]})  ->  ["fc2"]
    find_bad_gradients({"fc1": [1.0]})                         ->  []

    named_grads — словарь {имя слоя: градиент любой вложенности}.
    Сортировка обязательна: порядок обхода словаря не должен просачиваться
    в вывод, иначе отчёт будет прыгать от запуска к запуску.

    Первый слой, где всплыл NaN, — это и есть место аварии. Дальше по
    сети NaN распространяется сам, и виноватым выглядит кто угодно.
    """
    return sorted(name for name, grads in named_grads.items() if has_nan_or_inf(grads))


def dead_relu_fractions(activations):
    """Доля нулевых выходов у каждого нейрона по батчу.

    activations — список образцов, каждый образец список выходов нейронов.

    dead_relu_fractions([[0.0, 1.0], [0.0, 2.0]])  ->  [1.0, 0.0]
    dead_relu_fractions([[0.0, 1.0], [3.0, 2.0]])  ->  [0.5, 0.0]

    Считаем по столбцам, а не по строкам: нейрон — это столбец, один и
    тот же индекс во всех образцах. Перепутаешь — получишь долю нулей
    внутри образца, что не значит ровным счётом ничего.

    Пустой батч — пустой список, без деления на ноль.
    """
    if not activations:
        return []
    n_samples = len(activations)
    n_neurons = len(activations[0])
    return [
        sum(1 for sample in activations if sample[i] == 0.0) / n_samples
        for i in range(n_neurons)
    ]


def dead_neurons(activations, threshold=1.0):
    """Индексы нейронов, у которых доля нулей не меньше threshold.

    dead_neurons([[0.0, 1.0], [0.0, 2.0]])              ->  [0]
    dead_neurons([[0.0, 1.0], [3.0, 2.0]], 0.5)         ->  [0]
    dead_neurons([[0.0, 1.0], [3.0, 2.0]])              ->  []

    threshold=1.0 значит «молчит на всех образцах без исключения» — это
    строго мёртвый нейрон. На практике тревожатся уже при 0.5.

    Мёртвый ReLU не воскресает сам: выход ноль, производная ноль, градиент
    до его весов не доходит. Лечится снижением lr, LeakyReLU или
    переинициализацией входящих весов.
    """
    fractions = dead_relu_fractions(activations)
    return [i for i, fraction in enumerate(fractions) if fraction >= threshold]


def numeric_gradient(f, params, h=1e-4):
    """Численный градиент f(params) -> число по каждому параметру.

    numeric_gradient(lambda p: p[0] ** 2, [3.0])  ->  [6.0] с точностью до 1e-6

    Центральная разность: (f(x+h) - f(x-h)) / (2h). Шаг h=1e-4, а не 1e-8:
    слишком мелкий h убивает точность на вычитании близких чисел, слишком
    крупный — на самой аппроксимации. 1e-4 это разумная середина для float64.

    Входной список не менять: копируй его на каждой координате.
    """
    out = []
    for i in range(len(params)):
        up, down = list(params), list(params)
        up[i] += h
        down[i] -= h
        out.append((f(up) - f(down)) / (2 * h))
    return out


def relative_difference(a, b):
    """Относительное расхождение двух чисел: |a - b| / max(|a|, |b|, 1e-8).

    relative_difference(1.0, 1.0)  ->  0.0
    relative_difference(1.0, 2.0)  ->  0.5
    relative_difference(0.0, 0.0)  ->  0.0

    Знаменатель с 1e-8 — защита от деления на ноль, когда оба градиента
    нулевые. Без него gradient checking падает ровно на тех слоях, где
    всё как раз в порядке.

    Порог для вердикта: меньше 1e-5 — совпало, больше 1e-3 — почти
    наверняка баг в обратном проходе.
    """
    return abs(a - b) / max(abs(a), abs(b), 1e-8)


def gradient_check(f, analytic, params, h=1e-4):
    """Максимальное относительное расхождение аналитического градиента с численным.

    gradient_check(lambda p: p[0] ** 2, [6.0], [3.0])  ->  примерно 0 (меньше 1e-9)
    gradient_check(lambda p: p[0] ** 2, [3.0], [3.0])  ->  0.5   (градиент вдвое мал)

    analytic — то, что вернул твой backward; функция сравнивает его с
    numeric_gradient по каждому параметру и отдаёт худший случай.

    Это единственный способ поймать ошибку в обратном проходе: forward
    может быть верным, лосс убывать, а градиент — транспонированным.
    Сеть при этом учится, только плохо, и ничего не падает.
    """
    numeric = numeric_gradient(f, params, h=h)
    return max(
        (relative_difference(a, n) for a, n in zip(analytic, numeric)),
        default=0.0,
    )


def can_overfit_one_batch(loss_fn, params, steps=500, lr=0.1, tol=1e-3):
    """Способна ли модель загнать лосс одного батча почти в ноль. True/False.

    can_overfit_one_batch(lambda p: (p[0] - 3.0) ** 2, [0.0])        ->  True
    can_overfit_one_batch(lambda p: (p[0] - 3.0) ** 2 + 1.0, [0.0])  ->  False
    can_overfit_one_batch(lambda p: 5.0, [0.0])                      ->  False

    Внутри — обычный градиентный спуск на numeric_gradient, steps шагов,
    без всяких батчей и перемешивания. Возвращает loss_fn(params) < tol.

    Самая важная проверка во всём глубоком обучении: она занимает 30
    секунд и ловит сломанный лосс, сломанный backward, слишком маленькую
    модель, оптимизатор, не подключённый к параметрам, и разъехавшиеся
    данные с метками. Не прошла — дальше учить бессмысленно.
    """
    current = list(params)
    for _ in range(steps):
        grads = numeric_gradient(loss_fn, current)
        current = [p - lr * g for p, g in zip(current, grads)]
        if has_nan_or_inf(current):
            return False
    return loss_fn(current) < tol


def diagnose_loss_curve(losses):
    """Вердикт по кривой лосса одной строкой.

    diagnose_loss_curve([1.0])                       ->  "NOT_ENOUGH_DATA"
    diagnose_loss_curve([1.0, float("nan")])         ->  "NAN_OR_INF"
    diagnose_loss_curve([1.0, 0.9, 0.8, 0.7])        ->  "HEALTHY"
    diagnose_loss_curve([1.0, 1.0, 1.0, 1.0])        ->  "NOT_DECREASING"
    diagnose_loss_curve([1.0, 5.0, 1.0, 5.0])        ->  "OSCILLATING"

    Порядок проверок принципиален и идёт от самого фатального к самому
    безобидному: сначала NaN (обучение мертво), потом колебания (lr велик),
    потом отсутствие прогресса (lr мал или сеть не подключена).

    Колебания ловим так: лосс хотя бы раз вырос больше чем вдвое по
    сравнению с предыдущим шагом. Застой — последнее значение не меньше
    чем на 1% ниже первого.
    """
    if len(losses) < 2:
        return "NOT_ENOUGH_DATA"
    if has_nan_or_inf(losses):
        return "NAN_OR_INF"
    if any(b > 2 * a for a, b in zip(losses, losses[1:])):
        return "OSCILLATING"
    if losses[-1] >= losses[0] * 0.99:
        return "NOT_DECREASING"
    return "HEALTHY"
