"""
Диффузионные модели: DDPM с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Линейное расписание шума beta_t: список из T чисел от beta_start до beta_end.

    linear_beta_schedule(3, 0.0, 1.0)  ->  [0.0, 0.5, 1.0]
    linear_beta_schedule(1, 0.2, 0.9)  ->  [0.2]

    Это то самое расписание из DDPM (Ho et al., 2020): T=1000, от 1e-4 до 0.02.
    Формула: beta_start + (beta_end - beta_start) * t / (T - 1).

    Ловушка: делитель именно T - 1, а не T. Иначе последний beta не дотянет до
    beta_end. И при T = 1 деление на ноль — этот случай разбери отдельно.
    """
    if T == 1:
        # единственный шаг: расписание вырождается в стартовое значение
        return [beta_start]
    step = (beta_end - beta_start) / (T - 1)
    return [beta_start + step * t for t in range(T)]


def alpha_bars_from_betas(betas):
    """Кумулятивные произведения alpha_bar_t = prod_{s<=t} (1 - beta_s).

    alpha_bars_from_betas([0.0, 0.5])   ->  [1.0, 0.5]
    alpha_bars_from_betas([0.1, 0.1])   ->  [0.9, 0.81]

    alpha_bar — единственная величина, которая нужна для замкнутой формулы
    q(x_t | x_0). Именно она превращает T шагов зашумления в одно умножение.

    Считай накопитель в цикле, а не math.prod на каждом префиксе: иначе
    получится O(T^2) вместо O(T).
    """
    out = []
    cum = 1.0
    for b in betas:
        cum *= (1.0 - b)
        out.append(cum)
    return out


def forward_sample(x0, t, alpha_bars, rng):
    """Замкнутая формула прямого процесса: вернуть (x_t, eps) за один шаг.

    q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I )

    forward_sample(3.0, 0, [1.0], rng)  ->  (3.0, eps)  — при alpha_bar = 1 шума нет

    eps — тот самый гауссов шум, который потом придётся угадывать сети. Верни
    его вторым элементом: без него не посчитать loss.

    Весь смысл DDPM здесь: чтобы получить x_t, не нужно прогонять t шагов
    цепочки. Одна выборка eps ~ N(0, 1) и одна строка арифметики.
    """
    a_bar = alpha_bars[t]
    eps = rng.gauss(0.0, 1.0)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1.0 - a_bar) * eps
    return x_t, eps


def sinusoidal_embedding(t, dim=8):
    """Синусоидальное вложение номера шага t: список длины dim.

    sinusoidal_embedding(0, 4)  ->  [0.0, 1.0, 0.0, 1.0]

    Пары (sin, cos) с частотами freq_i = 1 / 10000^(i / (half - 1)),
    где half = dim // 2. Как позиционное кодирование в трансформере.

    Зачем: сеть eps_theta(x_t, t) обязана знать, какой уровень шума она сейчас
    разгребает. Сырое число t работает только на игрушечных задачах —
    в реальных U-Net всегда вложение.

    Ловушка: при dim = 2 получается half = 1 и показатель i / (half - 1) делит
    на ноль. Защити знаменатель через max(half - 1, 1).
    """
    out = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000 ** (i / max(half - 1, 1)))
        out.append(math.sin(t * freq))
        out.append(math.cos(t * freq))
    # срез нужен для нечётного dim: пар получилось half, значений 2 * half
    return out[:dim]


def ddpm_loss(eps, eps_hat):
    """Простой loss из DDPM: средний квадрат ошибки предсказания шума.

    ddpm_loss([1.0, -1.0], [1.0, -1.0])  ->  0.0
    ddpm_loss([0.0], [2.0])              ->  4.0

    Весь вариационный вывод с KL по каждому шагу сворачивается вот в это.
    Ho явно выбросил коэффициенты перед слагаемыми — и качество выросло.

    Среднее, а не сумма: иначе loss зависел бы от размерности данных.
    """
    n = len(eps)
    return sum((a - b) ** 2 for a, b in zip(eps, eps_hat)) / n


def predict_x0(x_t, t, eps_hat, alpha_bars):
    """Оценка чистого x_0 по зашумлённому x_t и предсказанному шуму.

    predict_x0(3.0, 0, 0.0, [1.0])  ->  3.0

    Это обращение замкнутой формулы:
        x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps
        =>  x_0 = (x_t - sqrt(1 - a_bar) * eps) / sqrt(a_bar)

    Ровно эта подстановка превращает апостериорное q(x_{t-1} | x_t, x_0)
    в реализуемый обратный шаг: настоящего x_0 у нас нет, берём оценку.

    Ловушка: при a_bar близком к нулю делитель крошечный, и оценка x_0
    разлетается. Так и должно быть — на больших t сигнала в x_t почти нет.
    """
    a_bar = alpha_bars[t]
    return (x_t - math.sqrt(1.0 - a_bar) * eps_hat) / math.sqrt(a_bar)


def reverse_step(x_t, t, eps_hat, betas, alpha_bars, rng):
    """Один шаг обратного процесса: из x_t получить x_{t-1}.

    mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(1 - beta_t)
    x_{t-1} = mean + sqrt(beta_t) * z,   z ~ N(0, 1)

    reverse_step(0.0, 0, 0.0, [0.1], [0.9], rng)  ->  0.0

    На последнем шаге (t == 0) шум НЕ добавляется — иначе итоговая картинка
    получит лишнюю порцию зерна поверх готового результата. Проверь, что при
    t == 0 функция вообще не трогает rng.

    Формула страшная, но это чистая алгебра: подставили predict_x0 в среднее
    апостериорного распределения и упростили.
    """
    beta_t = betas[t]
    alpha_t = 1.0 - beta_t
    mean = (x_t - beta_t / math.sqrt(1.0 - alpha_bars[t]) * eps_hat) / math.sqrt(alpha_t)
    if t == 0:
        return mean
    # sigma_t = sqrt(beta_t) — вариант из статьи; учёная дисперсия даёт чуть лучше
    return mean + math.sqrt(beta_t) * rng.gauss(0.0, 1.0)


def sample_chain(eps_model, betas, alpha_bars, rng):
    """Сэмплирование: из чистого шума прогнать обратную цепочку до x_0.

    eps_model(x, t) — функция, возвращающая предсказанный шум.

    sample_chain(lambda x, t: 0.0, [0.1], [0.9], rng)  ->  число

    Старт: x_T ~ N(0, 1). Дальше t от T-1 вниз до 0, на каждом шаге
    reverse_step. Возвращается финальный x_0.

    Это весь inference диффузионной модели. Медленно ровно потому, что цикл
    в T итераций — отсюда DDIM, DPM-Solver и дистилляция в 1-4 шага.
    """
    T = len(betas)
    x = rng.gauss(0.0, 1.0)
    for t in range(T - 1, -1, -1):
        x = reverse_step(x, t, eps_model(x, t), betas, alpha_bars, rng)
    return x
