"""
Генерация изображений — GAN

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p04-l09-image-generation-gans
Разбор:  /check-code p04-l09-image-generation-gans
"""

import math


def sigmoid(x):
    """Сигмоида: логит дискриминатора -> вероятность «это настоящее».

    sigmoid(0)  ->  0.5
    sigmoid(10) ->  0.99995...

    Ловушка: math.exp(-x) при x = -1000 падает с OverflowError. Для x < 0
    считай через e^x / (1 + e^x) — то же самое, но без переполнения.

    0.5 на всех входах — это ровно то равновесие, которое доказал Goodfellow:
    D больше не отличает настоящее от сгенерированного.
    """
    raise NotImplementedError


def bce_with_logits(logit, target):
    """Бинарная кросс-энтропия, посчитанная сразу из логита. target — 0 или 1.

    bce_with_logits(0.0, 1)    ->  0.6931...  (= log 2)
    bce_with_logits(0.0, 0)    ->  0.6931...
    bce_with_logits(100.0, 1)  ->  ~0.0       (угадал уверенно)
    bce_with_logits(100.0, 0)  ->  ~100.0     (ошибся уверенно)

    Наивный путь sigmoid -> log даёт log(0) = -inf на уверенном промахе.
    Устойчивая форма, которую и использует F.binary_cross_entropy_with_logits:

        max(x, 0) - x * t + log(1 + exp(-|x|))

    Проверь её вручную на x = 0: 0 - 0 + log 2. Сходится.
    """
    raise NotImplementedError


def discriminator_loss(real_logits, fake_logits):
    """Лосс дискриминатора: он хочет сказать «1» настоящему и «0» фейку.

    L_D = mean BCE(real_logits, 1) + mean BCE(fake_logits, 0)

    discriminator_loss([0.0], [0.0])       ->  1.3862...  (= 2 * log 2)
    discriminator_loss([100.0], [-100.0])  ->  ~0.0       (D всё видит)

    Две средние складываются, а не усредняются между собой — так написано
    в оригинале и так делает эталонный код DCGAN.

    2*log2 = 1.386 — это цифра равновесия. Если L_D уехал к нулю и там
    залип, D победил, градиент генератора умер, обучение встало.
    """
    raise NotImplementedError


def generator_loss(fake_logits, non_saturating=True):
    """Лосс генератора. Две формы одной и той же цели — обмануть D.

    non_saturating=True (так делают все):
        L_G = mean BCE(fake_logits, 1) = mean -log D(G(z))
    non_saturating=False (исходный минимакс из статьи 2014):
        L_G = mean log(1 - D(G(z)))

    generator_loss([0.0])                       ->  0.6931...
    generator_loss([0.0], non_saturating=False) ->  -0.6931...
    generator_loss([100.0])                     ->  ~0.0  (D поверил в фейк)

    Обрати внимание на знаки: несатурирующая версия положительна и падает к
    нулю по мере успеха, исходная отрицательна и падает к минус
    бесконечности. Сравнивать их численно бессмысленно — сравнивают
    градиенты, см. generator_loss_grad.
    """
    raise NotImplementedError


def generator_loss_grad(fake_logit, non_saturating=True):
    """Производная лосса генератора по логиту D. Одно число, не список.

    non_saturating=True:   d/dx [-log sigmoid(x)]      = sigmoid(x) - 1
    non_saturating=False:  d/dx [log(1 - sigmoid(x))]  = -sigmoid(x)

    generator_loss_grad(-10.0)                       ->  -0.99995...
    generator_loss_grad(-10.0, non_saturating=False) ->  -0.0000453...

    Вот и весь ответ на вопрос «почему все переписали лосс». В начале
    обучения D уверенно отвергает фейки, логиты сильно отрицательные.
    Исходная форма даёт градиент 4.5e-5 — генератор не двигается. Новая
    даёт почти -1 — генератор получает полный сигнал.
    """
    raise NotImplementedError


def conv_transpose_output_size(in_size, kernel, stride, padding, output_padding=0):
    """Размер стороны после nn.ConvTranspose2d. Целое число.

        out = (in - 1) * stride - 2 * padding + kernel + output_padding

    conv_transpose_output_size(1, 4, 1, 0)   ->  4    (первый слой G: z -> 4x4)
    conv_transpose_output_size(4, 4, 2, 1)   ->  8    (комбинация k4-s2-p1 удваивает)
    conv_transpose_output_size(16, 4, 2, 1)  ->  32

    Правило DCGAN «kernel=4, stride=2, padding=1» держится именно на этой
    формуле: 2*in - 2 - 2 + 4 = 2*in, ровно удвоение при любом входе.
    Комбинации, где kernel не делится на stride, дают шахматные артефакты —
    их видно на сгенерированных картинках невооружённым глазом.
    """
    raise NotImplementedError


def power_iteration_sigma(matrix, rng, iters=50):
    """Наибольшее сингулярное число матрицы методом степенных итераций.

    matrix — список строк. rng — экземпляр random.Random (случайность идёт
    только отсюда, глобальный random трогать нельзя: результат обязан быть
    воспроизводимым).

    Схема:
        v = случайный вектор длины n, нормировать
        повторить iters раз:
            u = normalize(W  @ v)
            v = normalize(W^T @ u)
        sigma = u @ (W @ v)

    power_iteration_sigma([[3.0, 0.0], [0.0, 1.0]], random.Random(0))  ->  3.0
    power_iteration_sigma([[2.0, 0.0], [0.0, 2.0]], random.Random(0))  ->  2.0

    Это и есть spectral norm: разделив W на sigma, получаешь слой с
    константой Липшица 1. Дискриминатор перестаёт становиться сколь угодно
    крутым, и «D побеждает всухую» лечится одной строкой.
    """
    raise NotImplementedError


def mode_collapse_score(samples):
    """Средняя попарная евклидова дистанция между сэмплами. Индикатор коллапса.

    samples — список векторов одинаковой длины (сплющенные картинки).

    mode_collapse_score([[0.0], [1.0]])        ->  1.0
    mode_collapse_score([[0.0, 0.0], [3.0, 4.0]])  ->  5.0
    mode_collapse_score([[1.0], [1.0], [1.0]]) ->  0.0   (полный коллапс)

    Меньше одного сэмпла — пар нет, возвращай 0.0.

    Ноль здесь означает, что генератор нашёл одну картинку, которая
    обманывает D, и выдаёт только её. Числа сравнивают не с абсолютным
    порогом, а с той же величиной на настоящих данных: упало вдвое от
    реального разброса — это коллапс.
    """
    raise NotImplementedError
