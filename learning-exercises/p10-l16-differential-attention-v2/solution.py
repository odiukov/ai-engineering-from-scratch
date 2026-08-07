"""
Differential Attention (V2) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Идея урока: softmax никогда не выдаёт ноль, поэтому каждый нерелевантный
токен получает свою крошку вероятности, и на 128k токенов эти крошки
складываются в шумовой пол. Differential Transformer считает ДВЕ карты
внимания и вычитает вторую из первой: общий шум сокращается, сигнал —
нет, потому что он есть только в одной ветке.

Матрица — список строк. K[i] это ключ i-го токена.
"""

import math


def softmax(row):
    """Строка логитов -> распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([0.0, 1000.0])    ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум строки
    перед экспонентой — сумма нормируется, результат тот же.

    Заодно тут видно источник шумового пола: exp(x) > 0 всегда, значит
    ноль в весах не получить в принципе.
    """
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def attention_weights(q, K):
    """Одна карта внимания: softmax(q . k_i / sqrt(d)) по всем ключам.

    attention_weights([1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])  ->  [0.5, 0.5]

    Делитель sqrt(d), где d = len(q). Без него дисперсия скалярного
    произведения растёт с размерностью, softmax насыщается и градиент
    умирает.

    Веса всегда неотрицательны и в сумме дают 1 — именно это свойство
    differential attention и ломает, причём намеренно.
    """
    scale = math.sqrt(len(q))
    scores = [sum(a * b for a, b in zip(q, k)) / scale for k in K]
    return softmax(scores)


def attend(weights, V):
    """Взвешенная сумма строк значений: out[j] = sum_i w_i * V[i][j].

    attend([0.5, 0.5], [[1.0, 0.0], [3.0, 4.0]])  ->  [2.0, 2.0]
    attend([1.0, -1.0], [[1.0, 0.0], [3.0, 4.0]]) ->  [-2.0, -4.0]

    Веса МОГУТ быть отрицательными: после вычитания двух softmax это
    норма. Никакой проверки «сумма равна единице» здесь быть не должно.
    """
    d_v = len(V[0])
    return [sum(w * v[j] for w, v in zip(weights, V)) for j in range(d_v)]


def diff_weights(w1, w2, lam):
    """Разностная карта внимания: w1 - lambda * w2 поэлементно.

    diff_weights([0.6, 0.4], [0.5, 0.5], 1.0)  ->  [0.1, -0.1]
    diff_weights([0.6, 0.4], [0.5, 0.5], 0.0)  ->  [0.6, 0.4]

    lambda — обучаемый скаляр на голову, в реальной модели он равен
    exp(lq1.lk1) - exp(lq2.lk2) + lambda_init и может быть отрицательным.

    Проверяемое свойство: сумма разностных весов равна 1 - lambda, потому
    что каждая из двух карт сама по себе суммируется в единицу.
    """
    return [a - lam * b for a, b in zip(w1, w2)]


def diff_attention(Q1, K1, Q2, K2, V, lam):
    """Differential attention целиком: две ветки, вычитание, умножение на V.

    Модуль модели: MultiheadDiffAttn.forward.

    Q1/Q2 — списки запросов двух веток (по строке на позицию), K1/K2 —
    их ключи. Возвращает список выходных строк, по одной на запрос.

    При lam = 0 ответ совпадает с обычным вниманием по ветке 1 — это
    первый тест, который стоит написать.

    В V1 для этого делили размерность головы пополам, в V2 удваивают
    число Q-голов и оставляют KV-головы как есть. Математика одна и та
    же, отличается только бухгалтерия параметров.
    """
    out = []
    for q1, q2 in zip(Q1, Q2):
        w = diff_weights(attention_weights(q1, K1), attention_weights(q2, K2), lam)
        out.append(attend(w, V))
    return out


def signal_to_noise(weights, signal_idx):
    """Отношение сигнал/шум карты внимания.

    Модуль вычисляется как |w[signal_idx]| делить на среднее |w| по всем
    остальным позициям.

    signal_to_noise([0.7, 0.1, 0.1, 0.1], 0)  ->  7.0
    signal_to_noise([0.5, 0.0, 0.0, 0.0], 0)  ->  inf

    Ловушка: если шум сократился ровно в ноль, делить нельзя — верни
    math.inf. Это не аварийный случай, а идеальный: ровно к нему и
    стремится differential attention.
    """
    noise = [abs(w) for i, w in enumerate(weights) if i != signal_idx]
    if not noise:
        return math.inf
    mean_noise = sum(noise) / len(noise)
    if mean_noise == 0:
        return math.inf
    return abs(weights[signal_idx]) / mean_noise


def best_lambda(q1, K1, q2, K2, signal_idx, lambdas):
    """Перебор lambda: какое значение даёт максимальный signal/noise.

    lambdas — список кандидатов, например [i / 100 for i in range(101)].
    При равенстве побеждает первый в списке.

    Смысл: у пары веток, делящих один и тот же шум, есть lambda, при
    которой шумовые веса сокращаются ТОЧНО в ноль. Она равна отношению
    нормировочных сумм двух softmax. Обучаемый lambda в настоящей модели
    ищет ровно её.
    """
    w1 = attention_weights(q1, K1)
    w2 = attention_weights(q2, K2)
    best, best_snr = lambdas[0], -math.inf
    for lam in lambdas:
        snr = signal_to_noise(diff_weights(w1, w2, lam), signal_idx)
        if snr > best_snr:
            best, best_snr = lam, snr
    return best


def attention_param_count(hidden, heads, head_dim, variant="baseline"):
    """Параметры одного блока внимания: baseline против DIFF V1 и V2.

    Возвращает словарь с ключами q, k, v, o, lam, total.

    baseline: Wq = Wk = Wv = Wo = hidden * heads * head_dim.
    v1: те же матрицы (размерность головы урезана вдвое внутри) плюс
        lambda-параметры — 4 вектора длины head_dim на голову.
    v2: Wq удвоена, Wk/Wv/Wo без изменений, плюс те же lambda.

    Смысловая точка: k и v одинаковы во всех трёх вариантах, значит
    KV-кэш DIFF V2 равен KV-кэшу обычной модели. Именно это и делает V2
    пригодной для продакшена, а V1 была нет.

    Неизвестный variant -> ValueError.
    """
    proj = hidden * heads * head_dim
    lam = 4 * heads * head_dim  # lq1, lk1, lq2, lk2 на каждую голову
    if variant == "baseline":
        counts = {"q": proj, "k": proj, "v": proj, "o": proj, "lam": 0}
    elif variant == "v1":
        counts = {"q": proj, "k": proj, "v": proj, "o": proj, "lam": lam}
    elif variant == "v2":
        counts = {"q": 2 * proj, "k": proj, "v": proj, "o": proj, "lam": lam}
    else:
        raise ValueError(f"неизвестный вариант: {variant!r}")
    counts["total"] = sum(counts.values())
    return counts
