"""
Расписания learning rate и разогрев — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def constant_schedule(step, lr=0.01):
    """Постоянный learning rate: сколько дали, столько и вернули.

    constant_schedule(0, lr=0.05)     ->  0.05
    constant_schedule(9999, lr=0.05)  ->  0.05

    Годится для отладки и совсем маленьких моделей. Для всего, что учится
    дольше часа, это плохой выбор: в начале шаг слишком мал, в конце —
    слишком велик, и лосс болтается вокруг минимума, не попадая в него.
    """
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1):
    """Ступенчатое затухание: lr * gamma^(step // step_size).

    step_decay_schedule(0, lr=0.1, step_size=100)    ->  0.1
    step_decay_schedule(99, lr=0.1, step_size=100)   ->  0.1
    step_decay_schedule(100, lr=0.1, step_size=100)  ->  0.010000000000000002

    Деление обязано быть целочисленным (//): именно оно делает график
    лестницей, а не плавной кривой.

    Так учили ResNet-50: lr=0.1 и деление на 10 на эпохах 30, 60, 90.
    Минус — точки спада приходится подбирать под каждую задачу заново.
    """
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=0.0):
    """Косинусный отжиг: плавный спуск от lr до lr_min за total_steps шагов.

    cosine_schedule(0, lr=0.1, total_steps=100)     ->  0.1
    cosine_schedule(50, lr=0.1, total_steps=100)    ->  0.05   (ровно половина)
    cosine_schedule(100, lr=0.1, total_steps=100)   ->  0.0
    cosine_schedule(500, lr=0.1, total_steps=100)   ->  0.0   (после конца — пол)

    Формула: lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * step / total_steps)).

    Ловушка — шаги за пределами total_steps: cos продолжит колебаться и
    lr поедет обратно вверх. Отсекай такие шаги на lr_min явно.

    Дефолт современных запусков: настраивать нечего, кроме lr и lr_min.
    """
    if step >= total_steps:
        return lr_min
    progress = step / total_steps
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def linear_warmup(step, lr=0.01, warmup_steps=100):
    """Линейный разогрев: от нуля до lr за warmup_steps шагов.

    linear_warmup(0, lr=0.1, warmup_steps=10)   ->  0.0
    linear_warmup(5, lr=0.1, warmup_steps=10)   ->  0.05
    linear_warmup(10, lr=0.1, warmup_steps=10)  ->  0.1
    linear_warmup(50, lr=0.1, warmup_steps=10)  ->  0.1   (дальше — потолок)

    warmup_steps=0 означает «разогрева нет», и делить на ноль нельзя —
    возвращай сразу lr.

    Зачем: Adam на первых шагах оценивает среднее и дисперсию градиента
    по мусорной статистике. Большой lr в этот момент уводит модель в
    случайную сторону. Разогрев даёт статистике устояться.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return lr
    return lr * step / warmup_steps


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=0.0):
    """Разогрев плюс косинус — то, чем учат Llama, GPT и почти всё остальное.

    warmup_cosine_schedule(0, lr=0.1, total_steps=100, warmup_steps=10)    ->  0.0
    warmup_cosine_schedule(10, lr=0.1, total_steps=100, warmup_steps=10)   ->  0.1
    warmup_cosine_schedule(100, lr=0.1, total_steps=100, warmup_steps=10)  ->  0.0

    До warmup_steps работает linear_warmup, после — cosine_schedule, но
    отсчёт косинуса начинается заново: шаг step - warmup_steps из
    total_steps - warmup_steps. Если этого не сделать, пик придётся не на
    конец разогрева и график получится с изломом вниз.

    Типичный разогрев — 1-5% всех шагов. У Llama 3 это 2000 шагов.
    """
    if step < warmup_steps:
        return linear_warmup(step, lr=lr, warmup_steps=warmup_steps)
    return cosine_schedule(
        step - warmup_steps,
        lr=lr,
        total_steps=max(total_steps - warmup_steps, 1),
        lr_min=lr_min,
    )


def one_cycle_schedule(step, lr=0.01, total_steps=1000):
    """Политика 1cycle: линейно вверх до середины, линейно вниз до конца.

    one_cycle_schedule(0, lr=0.1, total_steps=100)   ->  0.004  (это lr / 25)
    one_cycle_schedule(50, lr=0.1, total_steps=100)  ->  0.1    (пик ровно в середине)
    one_cycle_schedule(99, lr=0.1, total_steps=100)  ->  0.0020098000000000017

    Первая фаза идёт от lr/25 до lr, вторая — от lr до lr/10000.

    Высокий lr в середине работает как регуляризация: шум сбивает
    оптимизатор с плохих узких минимумов. Цена — нужно заранее знать
    total_steps, иначе фазы не сойдутся.
    """
    mid = max(total_steps // 2, 1)
    if step < mid:
        low = lr / 25.0
        return low + (lr - low) * step / mid
    progress = min((step - mid) / max(total_steps - mid, 1), 1.0)
    return lr * (1 - progress) + (lr / 10000.0) * progress


def lr_curve(schedule_fn, total_steps):
    """Значения lr на шагах 0..total_steps-1 одним списком.

    lr_curve(lambda s: 0.1, 3)  ->  [0.1, 0.1, 0.1]

    schedule_fn — функция одного аргумента. Остальные параметры расписания
    привязывай снаружи лямбдой:
        lr_curve(lambda s: cosine_schedule(s, lr=0.1, total_steps=50), 50)

    Из такого списка видно всё: где пик, где излом, где пол.
    """
    return [schedule_fn(step) for step in range(total_steps)]


def peak_step(lrs):
    """Номер шага с максимальным lr. При равных максимумах — самый ранний.

    peak_step([0.1, 0.1, 0.1])   ->  0
    peak_step([0.0, 0.5, 0.2])   ->  1

    Проверка на здравый смысл для разогрева: пик обязан приходиться
    ровно на конец разогрева, ни раньше, ни позже.
    """
    best = 0
    for i, value in enumerate(lrs):
        if value > lrs[best]:
            best = i
    return best


def descend(schedule_fn, start=10.0, steps=50):
    """Градиентный спуск по f(x) = x^2 с заданным расписанием.

    Возвращает список из steps + 1 позиций, начиная со start.
    Производная f равна 2x, поэтому шаг: x -= schedule_fn(t) * 2 * x.

    descend(lambda s: 0.1, 1.0, 2)  ->  [1.0, 0.8, 0.64]
    descend(lambda s: 0.0, 5.0, 3)  ->  [5.0, 5.0, 5.0, 5.0]

    Множитель за шаг равен (1 - 2*lr). При lr > 1 его модуль больше
    единицы — и вот тебе расхождение из ничего, на самой выпуклой задаче
    на свете. Разогрев спасает ровно от этого.
    """
    x = start
    history = [x]
    for t in range(steps):
        x -= schedule_fn(t) * 2 * x
        history.append(x)
    return history
