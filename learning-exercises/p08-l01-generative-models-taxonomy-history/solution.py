"""
Генеративные модели: таксономия и история — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def model_family(name):
    """Номер семейства генеративных моделей (1..5) по имени модели.

    model_family("PixelCNN")  ->  1
    model_family("VAE")       ->  2
    model_family("StyleGAN")  ->  3

    Пять семейств из урока:
      1 — explicit density, tractable   (autoregressive, normalizing flows)
      2 — explicit density, approximate (VAE, diffusion)
      3 — implicit density              (GAN)
      4 — score-based / continuous-time (score SDE, flow matching)
      5 — token-based autoregressive over discrete codes

    Функция обязана знать ровно эти имена (регистр и пробелы по краям не
    важны): pixelcnn, wavenet, gpt, realnvp, glow, vae, beta-vae, ddpm,
    gan, dcgan, stylegan, score sde, flow matching, rectified flow,
    parti, audiolm, vall-e, musenet.

    Незнакомое имя — ValueError, а не None: молчаливый None утечёт в
    следующую функцию и всплывёт уже там, в совсем другом месте.
    """
    # таблица живёт внутри функции, а не в модуле: так её видно рядом с
    # docstring и не приходится держать глобальное состояние
    table = {
        "pixelcnn": 1, "wavenet": 1, "gpt": 1, "realnvp": 1, "glow": 1,
        "vae": 2, "beta-vae": 2, "ddpm": 2,
        "gan": 3, "dcgan": 3, "stylegan": 3,
        "score sde": 4, "flow matching": 4, "rectified flow": 4,
        "parti": 5, "audiolm": 5, "vall-e": 5, "musenet": 5,
    }
    key = name.strip().lower()
    if key not in table:
        raise ValueError(f"unknown model: {name!r}")
    return table[key]


def has_explicit_density(family):
    """Можно ли у семейства спросить «насколько вероятна вот эта точка?».

    has_explicit_density(1)  ->  True
    has_explicit_density(3)  ->  False

    Семейства 1, 2 и 5 пишут log p(x) — точно или через нижнюю оценку
    (ELBO), — поэтому плотность в точке достать можно. Семейство 3 (GAN)
    даёт только сэмплы: сгенерировать умеет, оценить чужую точку — нет.
    Семейство 4 учит score, то есть градиент логарифма плотности; из
    градиента само значение p(x) без интегрирования не получить.

    Номер вне 1..5 — ValueError.
    """
    if family not in (1, 2, 3, 4, 5):
        raise ValueError(f"family must be 1..5, got {family!r}")
    return family in (1, 2, 5)


def histogram_density(samples, x, bin_width=0.25):
    """Явная плотность по гистограмме: доля точек в корзине вокруг x.

    histogram_density([0.0, 0.1, 5.0], 0.05, bin_width=1.0)  ->  0.6666...
    histogram_density([0.0, 0.1, 5.0], 99.0, bin_width=1.0)  ->  0.0

    Корзина — полуинтервал [x - w/2, x + w/2). Делить надо на n * w, а не
    на n: иначе получится доля точек, а не плотность, и интеграл по всей
    оси совпадёт с единицей только случайно, при w == 1.

    Пустой список samples — ValueError, делить не на что.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    lo, hi = x - bin_width / 2, x + bin_width / 2
    count = sum(1 for s in samples if lo <= s < hi)
    return count / (len(samples) * bin_width)


def kde_density(samples, x, bandwidth=0.3):
    """Приближённая плотность: гауссово ядро, поставленное в каждую точку.

    kde_density([0.0], 0.0, bandwidth=1.0)   ->  0.3989...  (это 1/sqrt(2*pi))
    kde_density([0.0], 10.0, bandwidth=1.0)  ->  ~7.7e-23

    Формула: (1 / (n * h)) * sum_i phi((x - s_i) / h), где phi — плотность
    стандартного нормального, exp(-u^2 / 2) / sqrt(2 * pi).

    В отличие от гистограммы KDE гладкая и строго положительная везде —
    её можно логарифмировать и делить на неё, не проверяя на ноль.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    norm = 1.0 / math.sqrt(2 * math.pi)
    total = 0.0
    for s in samples:
        u = (x - s) / bandwidth
        total += norm * math.exp(-0.5 * u * u)
    return total / (len(samples) * bandwidth)


def integrate_density(density_fn, samples, lo, hi, steps=200):
    """Интеграл плотности по отрезку [lo, hi] методом трапеций.

    integrate_density(kde_density, [0.0], -20, 20, steps=4000)  ->  ~1.0

    density_fn вызывается как density_fn(samples, x).
    Трапеции: на каждом шаге берётся (f(a) + f(b)) / 2 * (b - a).

    Это и есть проверка «а плотность ли это»: по всей оси интеграл обязан
    давать единицу. У неявного генератора такой проверки просто нет —
    интегрировать нечего.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    width = (hi - lo) / steps
    # значения на границах считаем по одному разу, а не дважды на шаг:
    # density_fn может быть дорогой (KDE — это проход по всем samples)
    total = 0.5 * (density_fn(samples, lo) + density_fn(samples, hi))
    for i in range(1, steps):
        total += density_fn(samples, lo + width * i)
    return total * width


def implicit_generator(samples, k, rng, sigma=0.1):
    """Неявный генератор: k новых точек, каждая — обучающая плюс шум.

    rng = random.Random(0)
    implicit_generator([1.0, 2.0], 3, rng)  ->  три числа рядом с 1.0 или 2.0

    Ровно то, что делает GAN: сэмплы выдавать умеет, ответить «насколько
    вероятна вот эта точка» — нет. Плотности здесь нет вообще, ни точной,
    ни приближённой.

    rng — обязательный параметр (random.Random). Глобальный random
    использовать нельзя: и тесты, и замер обязаны быть воспроизводимы.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    return [rng.choice(samples) + rng.gauss(0.0, sigma) for _ in range(k)]


def sampling_cost(num_steps, step_cost):
    """Стоимость сэмплирования: число шагов умножить на стоимость шага.

    sampling_cost(50, 0.06)  ->  3.0    (50 шагов SDXL по 60 мс — это 3 с)
    sampling_cost(1, 0.03)   ->  0.03   (GAN: один forward, и всё)

    Вся арифметика инференса генеративных моделей сводится к этому
    произведению. У GAN num_steps == 1 по построению, у диффузии — 20-50,
    у дистиллированной — 1-4.

    Отрицательные аргументы — ValueError.
    """
    if num_steps < 0 or step_cost < 0:
        raise ValueError("num_steps and step_cost must be non-negative")
    return num_steps * step_cost


def speedup_source(base_steps, base_step_cost, new_steps, new_step_cost):
    """Откуда взялось ускорение: "steps", "step_cost", "both" или "none".

    speedup_source(50, 0.06, 4, 0.06)   ->  "steps"
    speedup_source(50, 0.06, 50, 0.01)  ->  "step_cost"
    speedup_source(50, 0.06, 4, 0.01)   ->  "both"
    speedup_source(50, 0.06, 50, 0.06)  ->  "none"

    Из production-заметки урока: любое «в 100 раз быстрее диффузии» — это
    либо меньше шагов при той же цене шага, либо та же схема при более
    дешёвом шаге. Всё остальное — маркетинг.

    Считать строго: равные значения ускорением не считаются. Если по
    сумме (шаги * цена шага) быстрее не стало — ответ "none", даже когда
    шагов формально меньше.
    """
    if sampling_cost(new_steps, new_step_cost) >= sampling_cost(base_steps, base_step_cost):
        return "none"
    fewer = new_steps < base_steps
    cheaper = new_step_cost < base_step_cost
    if fewer and cheaper:
        return "both"
    if fewer:
        return "steps"
    if cheaper:
        return "step_cost"
    return "none"
