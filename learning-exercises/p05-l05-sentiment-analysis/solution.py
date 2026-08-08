"""
Анализ тональности — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def apply_negation(tokens):
    """Разметка области отрицания: токены после «not» получают префикс NOT_.

    Отрицание включается на слове из
        {"not", "no", "never", "nor", "none", "nothing", "neither"}
    и выключается на первом знаке из
        {".", "!", "?", ",", ";"}.

    apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
        ->  ['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']

    Ловушка: само слово-отрицание и сам знак препинания в вывод попадают
    БЕЗ префикса. Префикс получают только слова между ними.

    Зачем: для bag-of-words "not good" и "not bad" — это {not, good} и
    {not, bad}, и модель учит good/bad по отдельности. После разметки
    good и NOT_good становятся разными признаками с разными весами.
    Три строчки препроцессинга дают измеримый прирост F1.
    """
    NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
    NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}

    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
        elif token in NEGATION_WORDS:
            negate = True
            out.append(token)
        else:
            out.append(f"NOT_{token}" if negate else token)
    return out


def train_nb(docs_by_class, vocab, alpha=1.0):
    """Обучение мультиномиального Naive Bayes. Вернуть (priors, word_probs).

    docs_by_class — {метка: список токенизированных документов},
    vocab — множество или список слов, которые модель вообще знает.

    priors[cls]         = доля документов класса cls,
    word_probs[cls][w]  = (count[w] + alpha) / (всего слов класса + alpha * |vocab|)

    train_nb({"pos": [["good"]], "neg": [["bad"]]}, ["good", "bad"])
        ->  ({'pos': 0.5, 'neg': 0.5},
             {'pos': {'good': 2/3, 'bad': 1/3}, 'neg': {'good': 1/3, 'bad': 2/3}})

    Разбор знаменателя: в классе pos одно слово, словарь из двух, alpha=1,
    значит делим на 1 + 1*2 = 3, а в числителе 1 + 1 = 2.

    Ловушка: alpha (сглаживание Лапласа) нужен не для точности, а чтобы
    слово, ни разу не встреченное в классе, не получило вероятность 0.
    Ноль под логарифмом уносит весь счёт в минус бесконечность, и один
    незнакомый токен перечёркивает весь остальной документ.

    Ловушка: токены вне vocab не считаются вообще — ни в числителе, ни в
    знаменателе. Иначе вероятности по словарю перестанут давать в сумме 1.

    alpha обязан быть положительным. При alpha=0 невиданные слова снова
    получают ноль, а отрицательное значение вообще создаёт «вероятности»
    вне допустимого диапазона, поэтому оба случая дают ValueError.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    vocab = list(dict.fromkeys(vocab))  # порядок сохраняем, дубликаты убираем
    total_docs = sum(len(docs) for docs in docs_by_class.values())

    priors, word_probs = {}, {}
    for cls, docs in docs_by_class.items():
        priors[cls] = len(docs) / total_docs
        counts = {w: 0 for w in vocab}
        for doc in docs:
            for token in doc:
                if token in counts:  # OOV молча пропускаем
                    counts[token] += 1
        denom = sum(counts.values()) + alpha * len(vocab)
        word_probs[cls] = {w: (counts[w] + alpha) / denom for w in vocab}
    return priors, word_probs


def predict_nb(doc, priors, word_probs):
    """Предсказание Naive Bayes: метка с наибольшим log-счётом.

    Счёт класса: log(prior) + сумма log(P(слово | класс)) по токенам
    документа. Токены, которых нет в word_probs[cls], пропускаются.

    priors = {"pos": 0.5, "neg": 0.5}
    probs  = {"pos": {"good": 0.9, "bad": 0.1}, "neg": {"good": 0.1, "bad": 0.9}}
    predict_nb(["good"], priors, probs)          ->  'pos'
    predict_nb(["zzz"], priors, probs)           ->  'neg'  (решает только prior и порядок)

    Ловушка: складываем ЛОГАРИФМЫ, а не перемножаем вероятности. Сто слов
    по 0.01 дают 1e-200 — число, которое float округлит до нуля, и все
    классы станут неотличимы.

    Ловушка: при равных счётах нужен детерминированный порядок, иначе
    один и тот же документ будет классифицироваться по-разному.
    """
    scores = {}
    for cls in priors:
        s = math.log(priors[cls])
        for token in doc:
            p = word_probs[cls].get(token)
            if p is not None:
                s += math.log(p)
        scores[cls] = s
    # сортировка по (-счёт, метка) снимает неоднозначность при равенстве
    return min(scores, key=lambda c: (-scores[c], str(c)))


def sigmoid(x):
    """Сигмоида с обрезкой аргумента до [-20, 20].

    sigmoid(0)      ->  0.5
    sigmoid(-1000)  ->  sigmoid(-20), то есть примерно 2.06e-9

    Обрезка — не косметика: math.exp(1000) роняет программу с
    OverflowError. Ровно это делает np.clip(x, -20, 20) в коде урока.
    """
    if x > 20:
        x = 20.0
    elif x < -20:
        x = -20.0
    return 1.0 / (1.0 + math.exp(-x))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01, w0=None, b0=0.0):
    """Логистическая регрессия полным градиентным спуском. Вернуть (w, b).

    X — список строк-признаков, y — список меток 0/1. Стартовые веса w0 по
    умолчанию нулевые.

    Функция потерь (её и минимизируем):
        L = -mean(y*log(p) + (1-y)*log(1-p)) + (l2/2) * sum(w^2)

    Градиенты, где err = p - y:
        dL/dw = X^T @ err / n + l2 * w
        dL/db = mean(err)

    train_lr([[1.0], [-1.0]], [1, 0], epochs=200, lr=0.5, l2=0.0)
        ->  (w с положительной единственной компонентой, b около нуля)

    Ловушка: L2 штрафует ТОЛЬКО веса, не сдвиг b. Сдвиг задаёт базовую
    долю положительного класса, и загонять его в ноль вредно.

    Ловушка: деление на n обязательно. Без него шаг зависит от размера
    выборки, и lr, подобранный на 100 примерах, взорвётся на 100 000.

    L2 здесь принципиален: текстовые признаки разреженные, без штрафа
    модель просто запомнит обучающие примеры.
    """
    n = len(y)
    dim = len(X[0]) if X else 0
    w = [0.0] * dim if w0 is None else list(w0)
    b = float(b0)

    for _ in range(epochs):
        # ошибки считаем разом на всей выборке: это full-batch спуск,
        # градиент — усреднённый по выборке, а не по одному примеру
        errs = [sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b) - yi
                for row, yi in zip(X, y)]
        grad_w = [
            sum(row[k] * e for row, e in zip(X, errs)) / n + l2 * w[k]
            for k in range(dim)
        ]
        grad_b = sum(errs) / n
        w = [w[k] - lr * grad_w[k] for k in range(dim)]
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    """Метки 0/1 логистической регрессии по порогу 0.5.

    predict_lr([[1.0], [-1.0]], [2.0], 0.0)  ->  [1, 0]

    Порог 0.5 по вероятности — это порог 0 по логиту, потому что
    sigmoid(0) = 0.5. Считать сигмоиду ради сравнения не обязательно, но
    так виднее связь с вероятностью.

    Ловушка: ровно 0.5 относим к классу 1 (>=, не >). Иначе результат
    зависит от того, как округлился последний бит.
    """
    return [1 if sigmoid(sum(xi * wi for xi, wi in zip(row, w)) + b) >= 0.5 else 0
            for row in X]


def evaluate(y_true, y_pred):
    """Метрики бинарной классификации для положительного класса 1.

    Вернуть словарь с ключами tp, fp, tn, fn, precision, recall, f1, accuracy.

    evaluate([1, 1, 0, 0], [1, 0, 0, 0])
        ->  {'tp': 1, 'fp': 0, 'tn': 2, 'fn': 1,
             'precision': 1.0, 'recall': 0.5, 'f1': 0.666..., 'accuracy': 0.75}

    Ловушка: знаменатели бывают нулевые. Модель, которая никогда не
    говорит «1», даёт tp + fp = 0 — это не деление на ноль, а precision 0.0.

    precision — «из того, что я назвал положительным, сколько правда»,
    recall — «из того, что правда положительное, сколько я нашёл»,
    f1 — их гармоническое среднее (штрафует перекос сильнее обычного).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def macro_f1(y_true, y_pred):
    """Среднее F1 по обоим классам, с равным весом.

    Считается как среднее f1 из evaluate для класса 1 и того же evaluate
    на инвертированных метках (тогда положительным становится класс 0).

    macro_f1([1, 1, 0, 0], [1, 1, 0, 0])  ->  1.0
    macro_f1([1] * 9 + [0], [1] * 10)     ->  примерно 0.4736

    Второй пример — весь смысл метрики. Классификатор, который всегда
    отвечает «положительный», получает accuracy 0.9 и выглядит рабочим.
    Macro-F1 показывает 0.47, потому что по классу 0 модель не нашла
    ничего. На несбалансированных данных отчитываться accuracy нельзя.
    """
    flip = lambda ys: [1 - y for y in ys]
    pos = evaluate(y_true, y_pred)["f1"]
    neg = evaluate(flip(y_true), flip(y_pred))["f1"]
    return (pos + neg) / 2
