"""
Теорема Байеса — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
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
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    return likelihood * prior / evidence


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
    posterior = prior
    for _ in range(n_positive):
        # цикл, а не формула с возведением в степень: так видно, что
        # каждый шаг — обычный Байес, просто prior уже обновлён
        posterior = bayes_posterior(posterior, likelihood, false_positive_rate)
    return posterior


def mle_probability(successes, total):
    """Оценка максимального правдоподобия: просто доля успехов.

    mle_probability(7, 10)  ->  0.7
    mle_probability(0, 10)  ->  0.0

    Ловушка MLE: ноль наблюдений даёт вероятность РОВНО ноль. В наивном
    Байесе одно такое слово обнуляет произведение целиком, и класс
    становится невозможным независимо от остальных улик.
    """
    return successes / total


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
    return mle_probability(count + alpha, total + alpha * vocab_size)


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
    a, b = params
    return (a + successes, b + failures)


def beta_mean(params):
    """Среднее Beta(a, b) — оценка вероятности с учётом prior.

    beta_mean((1, 1))   ->  0.5
    beta_mean((8, 4))   ->  0.666...
    beta_mean((13, 9))  ->  0.5909...

    Формула a / (a + b). Чем больше a + b, тем увереннее (уже) распределение.
    """
    a, b = params
    return a / (a + b)


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
    a, b = params
    if a <= 0 or b <= 0:
        raise ValueError("параметры Beta должны быть положительными")
    if a < 1 and b < 1:
        raise ValueError("U-образная Beta имеет две моды: 0 и 1")
    if a == 1 and b == 1:
        # плоская плотность: максимум не единственный, берём середину по соглашению
        return 0.5
    if a <= 1:
        return 0.0
    if b <= 1:
        return 1.0
    return (a - 1) / (a + b - 2)


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
    class_docs = {}
    word_counts = {}
    class_word_totals = {}
    vocab = set()

    for doc, label in zip(documents, labels):
        class_docs[label] = class_docs.get(label, 0) + 1
        word_counts.setdefault(label, {})
        class_word_totals.setdefault(label, 0)
        for word in doc.lower().split():
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            class_word_totals[label] += 1
            vocab.add(word)

    total_docs = sum(class_docs.values())
    vocab_size = len(vocab)
    words = text.lower().split()

    best_class, best_score = None, float("-inf")
    # sorted — чтобы результат не зависел от порядка вставки в словарь
    for cls in sorted(class_docs):
        score = math.log(class_docs[cls] / total_docs)
        for word in words:
            count = word_counts[cls].get(word, 0)
            p = laplace_probability(count, class_word_totals[cls], vocab_size, alpha)
            score += math.log(p)
        if score > best_score:
            best_score, best_class = score, cls
    return best_class
