"""
Наивный Байес — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def build_vocabulary(docs):
    """Словарь корпуса: отсортированный список уникальных слов.

    build_vocabulary([["free", "money"], ["money", "meeting"]])
        ->  ["free", "meeting", "money"]
    build_vocabulary([[], ["a"]])  ->  ["a"]

    Сортировка нужна не для красоты: индекс слова в этом списке — номер
    столбца в матрице признаков, и он обязан быть одинаковым при обучении
    и при инференсе. Множество в Python порядок не гарантирует.
    """
    words = set()
    for doc in docs:
        words.update(doc)
    return sorted(words)


def bag_of_words(tokens, vocab):
    """Вектор счётчиков: сколько раз каждое слово словаря встретилось в токенах.

    bag_of_words(["free", "free", "money"], ["free", "meeting", "money"])
        ->  [2, 0, 1]

    Слова, которых нет в словаре, просто игнорируются — на проде такое
    случается постоянно, и падать из-за этого нельзя.
    """
    # dict вместо vocab.index(w) в цикле: index — это O(len(vocab)) на слово,
    # на словаре в 50 тысяч слов разница видна невооружённым глазом
    position = {w: i for i, w in enumerate(vocab)}
    counts = [0] * len(vocab)
    for token in tokens:
        i = position.get(token)
        if i is not None:
            counts[i] += 1
    return counts


def class_log_priors(labels):
    """Логарифмы априорных вероятностей классов: {класс: log P(класс)}.

    class_log_priors(["spam", "ham", "ham"])  ->  {"ham": log(2/3), "spam": log(1/3)}

    P(класс) — это просто доля класса в обучающей выборке. Логарифм берём
    сразу, потому что дальше всё считается сложением, а не умножением.
    """
    total = len(labels)
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {c: math.log(n / total) for c, n in counts.items()}


def feature_log_probs(X, y, alpha=1.0):
    """Логарифмы P(слово | класс) со сглаживанием Лапласа: {класс: [log p, ...]}.

    Для класса c: p_j = (сумма счётчиков слова j по документам класса c + alpha)
                        / (сумма ВСЕХ счётчиков класса c + alpha * размер словаря)

    feature_log_probs([[80, 60, 10], [5, 10, 100]], ["spam", "ham"], 1.0)
        ->  {"spam": [log(81/153), log(61/153), log(11/153)],
             "ham":  [log(6/118),  log(11/118), log(101/118)]}

    Ловушка, ради которой сглаживание и придумали: слово, ни разу не
    встретившееся в классе, даст p = 0 и log(0) = -inf. Одно такое слово
    обнуляет всю сумму, сколько бы доводов ни было в другую сторону.

    alpha * размер словаря в знаменателе — не украшение: без него сумма
    вероятностей по словарю перестанет быть единицей.
    """
    n_features = len(X[0])
    totals = {}
    for row, label in zip(X, y):
        acc = totals.setdefault(label, [0.0] * n_features)
        for j, v in enumerate(row):
            acc[j] += v
    log_probs = {}
    for label, acc in totals.items():
        denom = sum(acc) + alpha * n_features
        log_probs[label] = [math.log((v + alpha) / denom) for v in acc]
    return log_probs


def fit_multinomial_nb(X, y, alpha=1.0):
    """Обучить мультиномиальный наивный Байес. Один проход по данным.

    Вернуть словарь модели с ключами:
        "classes"    — отсортированный список классов,
        "log_priors" — {класс: log P(класс)},
        "log_probs"  — {класс: [log P(слово_j | класс), ...]}

    fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"])["classes"]  ->  ["a", "b"]

    Всё обучение — это подсчёт частот. Ни оптимизации, ни итераций.
    Поэтому NB тренируется на миллионе документов за секунды.
    """
    return {
        "classes": sorted(set(y)),
        "log_priors": class_log_priors(y),
        "log_probs": feature_log_probs(X, y, alpha),
    }


def log_scores(model, x):
    """Ненормированные логарифмы P(класс | x): {класс: log P(класс) + сумма}.

    Формула: log P(c) + сумма_j x_j * log P(слово_j | c).

    log_scores(fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"]), [1, 0])
        ->  {"a": ..., "b": ...}, где значение "a" больше

    Множитель x_j важен: слово, встретившееся дважды, даёт вдвое больше
    доводов. И наоборот — слово с нулевым счётчиком не даёт ничего, поэтому
    мультиномиальный NB отсутствие слова не наказывает (этим занят Бернулли).

    Знаменатель P(x) одинаков для всех классов, поэтому его не считают:
    на сравнение классов он не влияет.
    """
    scores = {}
    for c in model["classes"]:
        lp = model["log_probs"][c]
        # sum по ненулевым: на разреженном тексте это на порядок меньше работы
        total = model["log_priors"][c]
        for j, v in enumerate(x):
            if v:
                total += v * lp[j]
        scores[c] = total
    return scores


def predict(model, X):
    """Класс с наибольшим log_scores для каждой строки X.

    predict(fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"]), [[1, 0], [0, 1]])
        ->  ["a", "b"]

    При точном равенстве выигрывает класс, идущий раньше в model["classes"] —
    предсказание обязано быть детерминированным.
    """
    out = []
    for x in X:
        scores = log_scores(model, x)
        # строгое > : первый из classes выигрывает ничью
        best = model["classes"][0]
        for c in model["classes"][1:]:
            if scores[c] > scores[best]:
                best = c
        out.append(best)
    return out


def predict_proba(model, x):
    """Нормированные вероятности классов: {класс: P(класс | x)}, сумма равна 1.

    predict_proba(fit_multinomial_nb([[2, 0], [0, 2]], ["a", "b"]), [1, 0])
        ->  {"a": 0.75, "b": 0.25}

    Нормировать надо в логарифмах: exp от -800 обнулится, и все вероятности
    станут 0/0. Вычти максимум, потом возводи в экспоненту — это и есть
    приём log-sum-exp.

    Помни: у наивного Байеса числа получаются уверенные, но плохо
    откалиброванные. 0.99 здесь не значит "в 99 случаях из 100".
    """
    scores = log_scores(model, x)
    top = max(scores.values())
    # сдвиг на максимум: exp никогда не переполнится и не занулится целиком
    exps = {c: math.exp(v - top) for c, v in scores.items()}
    total = sum(exps.values())
    return {c: v / total for c, v in exps.items()}
