"""
Функции потерь — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def mse(predictions, targets):
    """Средний квадрат ошибки — базовый loss для регрессии.

    mse([1.0, 2.0], [1.0, 2.0])  ->  0.0
    mse([0.0, 0.0], [1.0, 3.0])  ->  5.0

    Формула: среднее по (p - t)^2.

    Квадрат наказывает большие промахи непропорционально: ошибка 10 стоит
    в сто раз дороже ошибки 1. Для регрессии это плюс, для классификации —
    минус, и следующие функции показывают почему.
    """
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def mse_gradient(predictions, targets):
    """Градиент MSE по каждому предсказанию: список той же длины.

    mse_gradient([0.0, 0.0], [1.0, 3.0])  ->  [-1.0, -3.0]

    Формула: 2*(p - t)/n.

    Ловушка — деление на n. Забудешь его, и градиент вырастет ровно во
    столько раз, сколько примеров в батче; вместе с ним поедет и
    подобранный learning rate.
    """
    n = len(predictions)
    return [2.0 * (p - t) / n for p, t in zip(predictions, targets)]


def binary_cross_entropy(predictions, targets, eps=1e-15):
    """Бинарная кросс-энтропия, усреднённая по примерам.

    binary_cross_entropy([0.9], [1.0])  ->  0.10536...
    binary_cross_entropy([0.5], [1.0])  ->  0.69314...
    binary_cross_entropy([0.0], [1.0])  ->  ~34.5, а НЕ бесконечность

    Формула: -(t*log(p) + (1 - t)*log(1 - p)), среднее по примерам.

    Ловушка здесь главная во всём уроке: log(0) это -inf, а math.log(0.0)
    вообще бросает ValueError. Модель законно может выдать 0.0 или 1.0.
    Зажми p в [eps, 1 - eps] перед логарифмом — тогда самый уверенный
    промах стоит конечных ~34.5, а обучение не разваливается на NaN.

    Сравни числа: p = 0.9 стоит 0.105, p = 0.5 стоит 0.693. Cross-entropy
    жёстко штрафует неуверенность там, где MSE почти не отличает 0.5 от 0.9.
    """
    total = 0.0
    for p, t in zip(predictions, targets):
        p = max(eps, min(1.0 - eps, p))
        total += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))
    return total / len(predictions)


def bce_gradient(predictions, targets, eps=1e-15):
    """Градиент BCE по каждому предсказанию: -(t/p) + (1 - t)/(1 - p).

    bce_gradient([0.5], [1.0])  ->  [-2.0]
    bce_gradient([0.9], [1.0])  ->  [-1.11111...]

    Градиент обязан соответствовать именно зажатому loss. Внутри интервала
    [eps, 1 - eps] действует обычная формула. За его границами clip постоянен,
    поэтому производная по исходному p равна нулю; в самих изломах здесь тоже
    выбираем нулевую производную. И, как в mse_gradient, делим на размер батча.

    При t = 1 и крошечном p чуть выше eps градиент равен -1/p и огромен.
    Но ровно 0 уже лежит на плоском участке защитного clip и даёт ноль.
    На практике BCE чаще объединяют с логитами в устойчивую функцию, чтобы
    не создавать такой плоский участок по вероятности.
    """
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        if p <= eps or p >= 1.0 - eps:
            grads.append(0.0)
        else:
            grads.append((-(t / p) + (1.0 - t) / (1.0 - p)) / n)
    return grads


def softmax(logits):
    """Логиты -> распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([1000.0, 1000.0]) ->  [0.5, 0.5]

    Вычти максимум перед exp: результат тот же, переполнения нет.
    """
    shift = max(logits)
    exps = [math.exp(v - shift) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def categorical_cross_entropy(logits, target_index):
    """Кросс-энтропия по сырым логитам и индексу верного класса.

    categorical_cross_entropy([0.0, 0.0], 0)          ->  0.69314...
    categorical_cross_entropy([1000.0, 0.0], 0)       ->  0.0
    categorical_cross_entropy([0.0, 1000.0], 0)       ->  1000.0

    Считать через -log(softmax(logits)[target]) можно, но плохо: softmax
    уже потерял точность, и на уверенно неверном ответе получится -log(0.0).
    Устойчивый путь — log-sum-exp:

        loss = logsumexp(logits) - logits[target],
        где logsumexp(v) = max(v) + log(sum(exp(v_i - max(v))))

    Это ровно та же величина, но без промежуточного деления и без
    логарифма от нуля. Третий пример — проверка: наивная реализация там
    даёт inf или падает, устойчивая возвращает честные 1000.0.
    """
    shift = max(logits)
    log_sum_exp = shift + math.log(sum(math.exp(v - shift) for v in logits))
    return log_sum_exp - logits[target_index]


def cce_gradient(logits, target_index):
    """Градиент кросс-энтропии по логитам: softmax(logits), минус 1 в верном классе.

    cce_gradient([0.0, 0.0], 0)  ->  [-0.5, 0.5]

    Вся производная softmax + cross-entropy схлопывается в одну строку:
    p_i - y_i, где y — one-hot. Это не совпадение, а причина, по которой
    softmax и кросс-энтропию всегда ставят в паре.
    """
    grads = softmax(logits)
    grads[target_index] -= 1.0
    return grads


def label_smoothed_cce(logits, target_index, alpha=0.1):
    """Кросс-энтропия со сглаженными метками.

    Вместо one-hot цель: (1 - alpha + alpha/K) у верного класса и
    alpha/K у остальных, где K — число классов.

    label_smoothed_cce([2.0, 0.0], 0, alpha=0.0)  ->  0.12692...   (обычная CCE)
    label_smoothed_cce([2.0, 0.0], 0, alpha=0.2)  ->  0.32692...   (та же уверенность дороже)

    Зачем: чтобы softmax выдал ровно 1.0, логит должен уйти в
    бесконечность. Модель разгоняет логиты, становится самоуверенной и
    хрупкой. Сглаживание ставит достижимую цель 0.9 вместо 1.0 и держит
    логиты в разумном диапазоне. Так учат GPT и почти все большие модели.

    Считай через log-вероятности из categorical_cross_entropy, а не через
    log(softmax(...)) — по той же причине устойчивости.
    """
    k = len(logits)
    loss = 0.0
    for i in range(k):
        weight = alpha / k + (1.0 - alpha if i == target_index else 0.0)
        # -log(p_i) переиспользуем: это и есть CCE с целью в классе i
        loss += weight * categorical_cross_entropy(logits, i)
    return loss
