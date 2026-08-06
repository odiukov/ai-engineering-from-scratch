"""
GPT и causal language modeling — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(logits, temperature=1.0):
    """Softmax с температурой: логиты -> распределение вероятностей.

    softmax([0.0, 0.0])          ->  [0.5, 0.5]
    softmax([1.0, 0.0], 0.5)     ->  [0.881..., 0.119...]  (температура заостряет)
    softmax([0.0, float("-inf")])->  [1.0, 0.0]

    Температура делит логиты ПЕРЕД экспонентой: T < 1 заостряет, T > 1
    размазывает. T -> 0 это greedy, T -> inf это равномерное распределение.
    T <= 0 бессмысленна — брось ValueError.

    Ловушка: exp большого логита переполняется. Вычти максимум строки перед
    экспонентой — сумма не изменится, а переполнения не будет. Значение
    -inf при этом даёт ровно 0.0, на этом и держится causal-маска.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in logits]
    # вычитание максимума — стандартный приём численной устойчивости:
    # softmax(x) == softmax(x - c) для любой константы c
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def causal_mask(n):
    """Треугольная маска n x n: 0.0 туда, куда смотреть можно, -inf в будущее.

    causal_mask(3)  ->  [[0.0, -inf, -inf],
                         [0.0,  0.0, -inf],
                         [0.0,  0.0,  0.0]]

    M[i][j] = 0.0 при j <= i и -inf при j > i. Маску складывают с сырыми
    attention-скорами ДО softmax: exp(-inf) = 0, значит будущие позиции
    получают ровно нулевой вес.

    В torch это одна строка `torch.tril`. Самая дешёвая строчка в истории AI.
    """
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]


def prefix_average_matrix(n):
    """Матрица префиксного среднего: строка i усредняет позиции 0..i.

    prefix_average_matrix(3)  ->  [[1.0, 0.0, 0.0],
                                   [0.5, 0.5, 0.0],
                                   [1/3, 1/3, 1/3]]

    Нижнетреугольная матрица, каждая строка делится на число своих единиц.
    Умножь её на последовательность — получишь причинное среднее одним
    матричным умножением, без всякого цикла.

    Зачем: с этого начинается вывод attention. Треугольник не «прикрутили»
    к attention сверху, он остался здесь от обычного среднего.
    """
    return [[1.0 / (i + 1) if j <= i else 0.0 for j in range(n)] for i in range(n)]


def causal_attention_weights(scores, temperature=1.0):
    """Сырые attention-скоры -> причинные веса: маска, потом softmax по строке.

    causal_attention_weights([[0.0, 5.0],
                              [0.0, 0.0]])  ->  [[1.0, 0.0], [0.5, 0.5]]

    Порядок операций принципиален: сначала прибавить causal_mask, потом
    softmax. Наоборот нельзя — softmax никогда не выдаёт точный ноль, и
    обнулять его результат руками значит ломать нормировку строки.

    Свойства результата: строка i имеет нули правее диагонали и суммируется
    в единицу. Именно поэтому позиция i не видит будущего.
    """
    n = len(scores)
    mask = causal_mask(n)
    return [
        softmax([s + m for s, m in zip(row, mrow)], temperature)
        for row, mrow in zip(scores, mask)
    ]


def cross_entropy_shifted(logits_per_pos, token_ids):
    """Средний next-token loss: предсказание позиции i сверяется с токеном i+1.

    Вход: logits_per_pos[i] — вектор логитов длиной vocab_size,
    token_ids[i] — настоящий токен на позиции i.

    cross_entropy_shifted([[0.0, 0.0], [0.0, 0.0]], [0, 1])  ->  0.693... (= ln 2)

    Сдвиг на единицу — это и есть вся разметка языковой модели: входы
    token_ids[:-1], цели token_ids[1:]. Последняя позиция цели не имеет,
    в сумму она не входит.

    Ловушка: -log(0) это бесконечность. Подпирай вероятность снизу
    маленькой константой вроде 1e-12, иначе один уверенно неправильный
    токен превратит весь loss в inf.

    Ориентир: необученная модель на словаре V даёт примерно ln V.
    """
    if len(token_ids) < 2:
        raise ValueError("need at least two tokens to shift by one")
    total = 0.0
    for i in range(len(token_ids) - 1):
        probs = softmax(logits_per_pos[i])
        total += -math.log(max(probs[token_ids[i + 1]], 1e-12))
    # среднее, а не сумма: иначе loss зависит от длины и его нельзя
    # сравнивать между батчами разной длины
    return total / (len(token_ids) - 1)


def top_k_filter(probs, k):
    """Оставить k самых вероятных токенов, остальным ноль, и перенормировать.

    top_k_filter([0.5, 0.3, 0.2], 2)  ->  [0.625, 0.375, 0.0]
    top_k_filter([0.5, 0.3, 0.2], 9)  ->  [0.5, 0.3, 0.2]   (k больше словаря)

    Длина результата не меняется — отброшенные токены получают 0.0, а не
    исчезают. Так индексы остаются индексами словаря.

    k < 1 бессмысленно — брось ValueError.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    # сортируем индексы по убыванию вероятности; при равенстве побеждает
    # меньший индекс — иначе результат зависел бы от порядка сортировки
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    keep = set(order[:k])
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]


def top_p_filter(probs, p):
    """Nucleus sampling: минимальный набор токенов с суммой >= p, перенормировать.

    top_p_filter([0.6, 0.3, 0.1], 0.9)   ->  [0.666..., 0.333..., 0.0]
    top_p_filter([0.6, 0.3, 0.1], 0.01)  ->  [1.0, 0.0, 0.0]

    Отличие от top-k: размер набора не фиксирован, он подстраивается под
    форму распределения. На остром распределении останется один токен, на
    плоском — почти весь словарь.

    Хотя бы один токен остаётся всегда, даже при крошечном p.
    p вне (0, 1] бессмысленно — брось ValueError.

    Ловушка чистой арифметики: 0.6 + 0.3 в double это 0.8999999999999999,
    и строгое сравнение с 0.9 не срабатывает. Сравнивай с допуском.
    """
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    keep = []
    cum = 0.0
    for i in order:
        keep.append(i)
        cum += probs[i]
        # допуск, иначе накопленная ошибка сложения оставит лишний токен
        if cum >= p - 1e-9:
            break
    keep = set(keep)
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]


def min_p_filter(probs, min_p):
    """Оставить токены с p >= min_p * max(p), перенормировать.

    min_p_filter([0.9, 0.05, 0.05], 0.1)        ->  [1.0, 0.0, 0.0]
    min_p_filter([0.34, 0.33, 0.33], 0.1)       ->  [0.34, 0.33, 0.33]

    Порог задаётся относительно пика, а не абсолютно. Поэтому на остром
    распределении фильтр режет жёстко, а на плоском не режет почти ничего —
    это и делает min-p аккуратнее top-p на длинных хвостах.

    min_p = 0 не отбрасывает ничего. min_p вне [0, 1] — ValueError.
    """
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")
    threshold = min_p * max(probs)
    keep = [i for i, pi in enumerate(probs) if pi >= threshold]
    total = sum(probs[i] for i in keep)
    keep = set(keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]
