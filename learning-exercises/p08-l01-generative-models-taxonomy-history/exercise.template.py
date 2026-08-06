"""
Генеративные модели: таксономия и история

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p08-l01-generative-models-taxonomy-history
Разбор:  /check-code p08-l01-generative-models-taxonomy-history
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
    raise NotImplementedError


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
    raise NotImplementedError


def histogram_density(samples, x, bin_width=0.25):
    """Явная плотность по гистограмме: доля точек в корзине вокруг x.

    histogram_density([0.0, 0.1, 5.0], 0.05, bin_width=1.0)  ->  0.6666...
    histogram_density([0.0, 0.1, 5.0], 99.0, bin_width=1.0)  ->  0.0

    Корзина — полуинтервал [x - w/2, x + w/2). Делить надо на n * w, а не
    на n: иначе получится доля точек, а не плотность, и интеграл по всей
    оси совпадёт с единицей только случайно, при w == 1.

    Пустой список samples — ValueError, делить не на что.
    """
    raise NotImplementedError


def kde_density(samples, x, bandwidth=0.3):
    """Приближённая плотность: гауссово ядро, поставленное в каждую точку.

    kde_density([0.0], 0.0, bandwidth=1.0)   ->  0.3989...  (это 1/sqrt(2*pi))
    kde_density([0.0], 10.0, bandwidth=1.0)  ->  ~7.7e-23

    Формула: (1 / (n * h)) * sum_i phi((x - s_i) / h), где phi — плотность
    стандартного нормального, exp(-u^2 / 2) / sqrt(2 * pi).

    В отличие от гистограммы KDE гладкая и строго положительная везде —
    её можно логарифмировать и делить на неё, не проверяя на ноль.
    """
    raise NotImplementedError


def integrate_density(density_fn, samples, lo, hi, steps=200):
    """Интеграл плотности по отрезку [lo, hi] методом трапеций.

    integrate_density(kde_density, [0.0], -20, 20, steps=4000)  ->  ~1.0

    density_fn вызывается как density_fn(samples, x).
    Трапеции: на каждом шаге берётся (f(a) + f(b)) / 2 * (b - a).

    Это и есть проверка «а плотность ли это»: по всей оси интеграл обязан
    давать единицу. У неявного генератора такой проверки просто нет —
    интегрировать нечего.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def sampling_cost(num_steps, step_cost):
    """Стоимость сэмплирования: число шагов умножить на стоимость шага.

    sampling_cost(50, 0.06)  ->  3.0    (50 шагов SDXL по 60 мс — это 3 с)
    sampling_cost(1, 0.03)   ->  0.03   (GAN: один forward, и всё)

    Вся арифметика инференса генеративных моделей сводится к этому
    произведению. У GAN num_steps == 1 по построению, у диффузии — 20-50,
    у дистиллированной — 1-4.

    Отрицательные аргументы — ValueError.
    """
    raise NotImplementedError


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
    raise NotImplementedError
