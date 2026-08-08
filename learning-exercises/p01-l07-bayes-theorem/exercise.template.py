"""
Теорема Байеса

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p01-l07-bayes-theorem
Разбор:  /check-code p01-l07-bayes-theorem
"""

import math


def bayes_posterior(prior, likelihood, false_positive_rate):
    """Апостериорная вероятность гипотезы после ОДНОГО положительного результата.

    prior              — P(H), вера до наблюдения;
    likelihood         — P(+|H), тест ловит настоящий случай;
    false_positive_rate — P(+|не H), тест срабатывает впустую.

    bayes_posterior(0.0001, 0.99, 0.01)  ->  0.00980...  (болезнь редкая)
    bayes_posterior(0.3, 0.05, 0.001)    ->  0.9554...   (письмо про «lottery»)
    bayes_posterior(0.5, 0.9, 0.1)       ->  0.9

    Знаменатель — полная вероятность улики:
    P(+) = P(+|H) * P(H) + P(+|не H) * P(не H).

    Главный урок: при редкой гипотезе даже точный тест даёт в основном
    ложные срабатывания. Prior решает.
    """
    raise NotImplementedError


def sequential_posterior(prior, likelihood, false_positive_rate, n_positive):
    """Последовательное обновление: n независимых положительных результатов подряд.

    sequential_posterior(0.0001, 0.99, 0.01, 0)  ->  0.0001   (данных нет)
    sequential_posterior(0.0001, 0.99, 0.01, 1)  ->  0.00980...
    sequential_posterior(0.0001, 0.99, 0.01, 2)  ->  0.495...

    Вчерашний posterior становится сегодняшним prior. Второй тест поднимает
    веру с 1% почти до 50% — именно поэтому назначают подтверждающий тест.

    Это же и есть online learning: модель дообучается на новых данных,
    не переобучаясь с нуля.
    """
    raise NotImplementedError


def mle_probability(successes, total):
    """Оценка максимального правдоподобия: просто доля успехов.

    mle_probability(7, 10)  ->  0.7
    mle_probability(0, 10)  ->  0.0

    Ловушка MLE: ноль наблюдений даёт вероятность РОВНО ноль. В наивном
    Байесе одно такое слово обнуляет произведение целиком, и класс
    становится невозможным независимо от остальных улик.
    """
    raise NotImplementedError


def laplace_probability(count, total, vocab_size, alpha=1.0):
    """Сглаженная оценка: к каждому счётчику добавили alpha «воображаемых» наблюдений.

    laplace_probability(0, 10, 5)            ->  0.0666...  (не ноль!)
    laplace_probability(7, 10, 5, alpha=0)   ->  0.7        (это чистый MLE)
    laplace_probability(3, 6, 4, alpha=1)    ->  0.4

    Формула: (count + alpha) / (total + alpha * vocab_size). Знаменатель
    растёт на alpha * vocab_size, а не на alpha — иначе сумма по всем
    словам перестанет быть единицей.

    Это MLE, посчитанный на подправленных счётчиках: alpha — тот самый prior
    из Байеса, зашитый в данные.
    """
    raise NotImplementedError


def beta_update(params, successes, failures):
    """Сопряжённое обновление Beta-распределения. Возвращает (a, b).

    params — пара (a, b) априорного Beta.

    beta_update((1, 1), 7, 3)  ->  (8, 4)
    beta_update((8, 4), 5, 5)  ->  (13, 9)

    Никаких интегралов: успехи прибавляются к a, неудачи к b. Beta(1, 1) —
    равномерный prior, «мнения нет».

    Порядок данных не важен: (1,1) + 12 успехов и 8 неудач разом даёт
    ту же (13, 9), что и два обновления подряд.
    """
    raise NotImplementedError


def beta_mean(params):
    """Среднее Beta(a, b) — оценка вероятности с учётом prior.

    beta_mean((1, 1))   ->  0.5
    beta_mean((8, 4))   ->  0.666...
    beta_mean((13, 9))  ->  0.5909...

    Формула a / (a + b). Чем больше a + b, тем увереннее (уже) распределение.
    """
    raise NotImplementedError


def beta_map(params):
    """Мода Beta(a, b) — MAP-оценка вероятности.

    beta_map((9, 5))   ->  0.666...   (7 орлов из 10 при prior Beta(2,2))
    beta_map((8, 4))   ->  0.7
    beta_map((1, 1))   ->  0.5        (особый случай, см. ниже)
    beta_map((1, 3))   ->  0.0        (максимум на левой границе)
    beta_map((3, 1))   ->  1.0        (максимум на правой границе)

    Формула (a - 1) / (a + b - 2) годится только при a > 1 и b > 1.
    Если один параметр не больше 1, единственная мода лежит на границе.
    При a = b = 1 плотность равномерна — договоримся возвращать 0.5.
    При a < 1 и b < 1 распределение U-образно и имеет сразу две моды,
    0 и 1, поэтому единственной числовой MAP-оценки нет: это ValueError.

    MAP — это MLE с prior. Ровно так же L2-регуляризация — это гауссов
    prior на веса модели.
    """
    raise NotImplementedError


def naive_bayes_predict(documents, labels, text, alpha=1.0):
    """Наивный Байес: обучиться на documents/labels и предсказать класс для text.

    Слова — то, что даёт text.lower().split(). Считаем в ЛОГАХ.

    docs = ["win free money", "meeting at noon"]
    labels = ["spam", "ham"]
    naive_bayes_predict(docs, labels, "free money")   ->  "spam"
    naive_bayes_predict(docs, labels, "noon meeting") ->  "ham"

    score(c) = log P(c) + сумма log P(слово | c) по всем словам text,
    P(слово | c) — laplace_probability со словарём из обучающих текстов.

    Две ловушки. Первая: произведение сотен вероятностей улетает в 0.0 и
    все классы становятся равны — поэтому логи, а не умножение. Вторая:
    без сглаживания незнакомое слово обнулит класс насмерть.

    При равенстве очков выбираем класс, меньший лексикографически, —
    чтобы ответ не зависел от порядка обучающих примеров.
    """
    raise NotImplementedError
