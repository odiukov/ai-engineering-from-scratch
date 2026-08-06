"""
Расписания learning rate и разогрев

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p03-l09-learning-rate-schedules
Разбор:  /check-code p03-l09-learning-rate-schedules
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def lr_curve(schedule_fn, total_steps):
    """Значения lr на шагах 0..total_steps-1 одним списком.

    lr_curve(lambda s: 0.1, 3)  ->  [0.1, 0.1, 0.1]

    schedule_fn — функция одного аргумента. Остальные параметры расписания
    привязывай снаружи лямбдой:
        lr_curve(lambda s: cosine_schedule(s, lr=0.1, total_steps=50), 50)

    Из такого списка видно всё: где пик, где излом, где пол.
    """
    raise NotImplementedError


def peak_step(lrs):
    """Номер шага с максимальным lr. При равных максимумах — самый ранний.

    peak_step([0.1, 0.1, 0.1])   ->  0
    peak_step([0.0, 0.5, 0.2])   ->  1

    Проверка на здравый смысл для разогрева: пик обязан приходиться
    ровно на конец разогрева, ни раньше, ни позже.
    """
    raise NotImplementedError


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
    raise NotImplementedError
