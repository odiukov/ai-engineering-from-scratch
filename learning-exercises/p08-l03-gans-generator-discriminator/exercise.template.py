"""
GAN: генератор против дискриминатора

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p08-l03-gans-generator-discriminator
Разбор:  /check-code p08-l03-gans-generator-discriminator
"""

import math


def sigmoid(x):
    """Сигмоида: логит превращается в вероятность из (0, 1).

    sigmoid(0.0)    ->  0.5
    sigmoid(-800.0) ->  ~0.0

    Дискриминатор выдаёт логит, а вероятность «это настоящее» получается
    вот так. Наивная формула 1 / (1 + exp(-x)) на x = -800 падает с
    OverflowError. Разбери два случая: при x >= 0 бери 1 / (1 + exp(-x)),
    при x < 0 бери exp(x) / (1 + exp(x)) — математически то же самое,
    численно безопасно.
    """
    raise NotImplementedError


def binary_cross_entropy(p, target):
    """BCE на одной паре: -[t * log(p) + (1 - t) * log(1 - p)].

    binary_cross_entropy(0.5, 1.0)  ->  0.6931...   (это log 2)
    binary_cross_entropy(0.9, 1.0)  ->  0.1053...
    binary_cross_entropy(0.1, 1.0)  ->  2.3025...

    Это лосс дискриминатора на одном примере: t = 1 для настоящих, t = 0
    для подделок.

    p приходит от сигмоиды и после округления может оказаться ровно 0.0
    или 1.0 — log(0) уронит весь прогон. Зажми p в [eps, 1 - eps] с
    eps = 1e-12 и верни большое конечное число вместо inf.
    """
    raise NotImplementedError


def binary_cross_entropy_grad(p, target):
    """Производная BCE по p: (p - t) / (p * (1 - p)).

    binary_cross_entropy_grad(0.5, 1.0)  ->  -2.0
    binary_cross_entropy_grad(0.5, 0.0)  ->  +2.0
    binary_cross_entropy_grad(0.9, 1.0)  ->  -1.1111...

    Знак читается сам: если p меньше цели, производная отрицательна —
    значит лосс падает, когда p растёт. Ровно это и нужно дискриминатору.

    Тот же зажим p в [eps, 1 - eps], иначе знаменатель обнулится. Сверь
    результат центральной разностью от binary_cross_entropy.
    """
    raise NotImplementedError


def discriminator_loss(d_real, d_fake):
    """Лосс дискриминатора: цель 1 на настоящих, цель 0 на подделках.

    Две пачки усредняются отдельно, потом складываются:
      loss = mean(BCE(p, 1) по d_real) + mean(BCE(p, 0) по d_fake)

    discriminator_loss([0.5], [0.5])  ->  1.3862...   (это 2 * log 2)
    discriminator_loss([1.0], [0.0])  ->  ~0.0        (идеальный D)

    Число 2 * log 2 ≈ 1.386 — то самое равновесие из урока: G попал в
    распределение данных, D не может лучше монетки. Увидел эту цифру в
    логах — это успех, а не поломка.

    Пустая пачка — ValueError: усреднять не по чему.
    """
    raise NotImplementedError


def generator_loss(d_fake, non_saturating=True):
    """Лосс генератора по вероятностям, которые D выдал на его подделках.

    non_saturating=True :  -mean(log(p))        <- форма, которой все пользуются
    non_saturating=False:   mean(log(1 - p))    <- ванильный минимакс из статьи

    generator_loss([0.5])                        ->   0.6931...
    generator_loss([0.5], non_saturating=False)  ->  -0.6931...

    Обе формы хотят одного и того же: чтобы D(G(z)) рос. Оптимум у них
    совпадает, разница живёт в градиенте — см. generator_loss_grad.

    Аргумент логарифма зажми, как в BCE: p == 0 или p == 1 бывает.
    """
    raise NotImplementedError


def generator_loss_grad(d_fake, non_saturating=True):
    """Производные лосса генератора по ЛОГИТУ дискриминатора. Список той же длины.

    p = sigmoid(logit), поэтому dp/dlogit = p * (1 - p), и после
    усреднения по батчу:
      non_saturating=True :  d(-log p) / dlogit     = -(1 - p) / n
      non_saturating=False:  d(log(1 - p)) / dlogit = -p / n

    generator_loss_grad([0.5])                        ->  [-0.5]
    generator_loss_grad([0.5], non_saturating=False)  ->  [-0.5]

    Вот здесь и живёт весь смысл урока. Когда D уверенно ловит подделку
    (p -> 0), у ванильной формы градиент стремится к нулю — генератор
    глохнет именно тогда, когда ему нужнее всего учиться. У non-saturating
    формы градиент в этот момент стремится к -1, то есть остаётся сильным.

    Считай по логиту, а не по p: сатурация — это заслуга сигмоиды, и по p
    её не видно. Сверь центральной разностью от generator_loss(sigmoid(...)).
    """
    raise NotImplementedError


def optimal_discriminator(p_data, p_gen):
    """Оптимальный дискриминатор в точке: p_data / (p_data + p_gen).

    optimal_discriminator(0.3, 0.3)  ->  0.5
    optimal_discriminator(0.4, 0.0)  ->  1.0
    optimal_discriminator(0.0, 0.4)  ->  0.0

    Это тот D*, к которому сходится обучение D при замороженном G. Главное
    следствие: если G точно попал в p_data, то D* равен 0.5 ВЕЗДЕ,
    градиента для G больше нет, игра в равновесии.

    Обе плотности нулевые — точка не встречается ни там, ни там, отвечать
    нечего, верни 0.5. Отрицательная плотность — ValueError.
    """
    raise NotImplementedError


def is_mode_collapse(samples, threshold=0.0, min_share=0.1):
    """Схлопнулся ли генератор на одну моду двухмодового распределения.

    Мода A — это samples < threshold, мода B — все остальные. Схлопыванием
    считается ситуация, когда доля МЕНЬШЕЙ моды меньше min_share.

    is_mode_collapse([-2.0, -2.0, -2.0, 2.0])  ->  False   (доля 0.25)
    is_mode_collapse([-2.0] * 99 + [2.0])      ->  True    (доля 0.01)

    Классический симптом из урока: одна из двух настоящих мод перестала
    порождаться. Дискриминатор перестаёт её поправлять, потому что больше
    не видит её среди подделок, и G остаётся в своём углу навсегда.

    Пустой список — ValueError.
    """
    raise NotImplementedError
