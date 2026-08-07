"""
Предобучение mini-GPT — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Конфигурация GPT-2 Small. Числа голов здесь нет намеренно: на количество
# параметров она не влияет — головы делят одну и ту же матрицу 768x768.
GPT2_SMALL = {
    "vocab_size": 50257,
    "embed_dim": 768,
    "num_layers": 12,
    "max_seq_len": 1024,
    "ff_dim": 3072,
}


def softmax(scores):
    """Превращает произвольные числа в распределение вероятностей.

    softmax([0.0, 0.0])   ->  [0.5, 0.5]
    softmax([1000.0, 0.0]) -> примерно [1.0, 0.0]
    softmax([])            -> []

    Обязательно вычитай максимум перед exp: math.exp(1000) — это
    OverflowError, а не «очень большое число». Ответ от вычитания не
    меняется, потому что softmax(x - c) == softmax(x) для любой константы.
    """
    if not scores:
        return []
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def layer_norm(vec, gamma, beta, eps=1e-5):
    """Нормализация по признакам: среднее 0, дисперсия 1, потом gamma и beta.

    layer_norm([1.0, 3.0], [1.0, 1.0], [0.0, 0.0])  ->  [-1.0, 1.0]
    layer_norm([5.0, 5.0], [1.0, 1.0], [0.0, 0.0])  ->  [0.0, 0.0]

    Дисперсия считается по самому вектору (делим на n, а не на n-1).
    eps под корнем спасает от деления на ноль, когда все компоненты равны —
    именно так выглядит вход у мёртвого нейрона.

    Свойство, ради которого это работает: результат не зависит ни от
    сдвига, ни от масштаба входа. Активации могут разъезжаться по
    величине от слоя к слою, LayerNorm возвращает их в один диапазон.
    """
    n = len(vec)
    mean = sum(vec) / n
    var = sum((x - mean) ** 2 for x in vec) / n
    denom = math.sqrt(var + eps)
    return [g * (x - mean) / denom + b for x, g, b in zip(vec, gamma, beta)]


def causal_attention(Q, K, V):
    """Self-attention с причинной маской. Q, K, V — списки строк (seq_len x d).

    Позиция i смотрит только на позиции 0..i. Возвращает список строк той
    же длины, что V.

    causal_attention([[1.0]], [[1.0]], [[7.0, 8.0]])  ->  [[7.0, 8.0]]

    Формула: scores = Q @ K^T / sqrt(d_k), маска, softmax, веса @ V.

    Маску можно не строить: вместо того чтобы прибавлять -inf к будущим
    позициям и потом их занулять, мы просто не считаем их. Результат тот
    же, работы вдвое меньше.

    Деление на sqrt(d_k) — не украшение. Без него скалярные произведения
    растут вместе с размерностью, softmax насыщается, и градиент исчезает.
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    out = []
    for i, q in enumerate(Q):
        # только 0..i — это и есть причинная маска
        scores = [sum(a * b for a, b in zip(q, K[j])) / scale for j in range(i + 1)]
        weights = softmax(scores)
        out.append([sum(w * V[j][c] for j, w in enumerate(weights)) for c in range(len(V[0]))])
    return out


def cross_entropy(logits_rows, targets):
    """Средний -log вероятности правильного токена по всем позициям.

    logits_rows — список строк логитов, targets — список индексов.

    cross_entropy([[0.0, 0.0]], [0])       ->  0.6931...  (это ln 2)
    cross_entropy([[100.0, 0.0]], [0])     ->  примерно 0.0

    На старте обучения модель ничего не знает, распределение равномерное,
    и loss равен ln(vocab_size): для байтового словаря из 256 это 5.545.
    Если первый шаг показал что-то сильно другое — ищи ошибку, а не удачу.

    Считай через log-sum-exp с вычитанием максимума: возводить exp в
    степень сырого логита нельзя, переполнение прилетит на первом же
    крупном значении.
    """
    total = 0.0
    for row, target in zip(logits_rows, targets):
        top = max(row)
        # log(sum(exp)) устойчиво: сдвиг на максимум выносится из-под лога
        log_sum = top + math.log(sum(math.exp(s - top) for s in row))
        total += log_sum - row[target]
    return total / len(targets)


def d_cross_entropy(logits_rows, targets):
    """Градиент cross_entropy по логитам: (softmax - one_hot) / число позиций.

    d_cross_entropy([[0.0, 0.0]], [0])  ->  [[-0.5, 0.5]]

    Красота этой формулы в том, что весь backward через softmax и
    логарифм сворачивается в одно вычитание. Именно поэтому все
    фреймворки считают softmax и cross-entropy одной операцией.

    Сумма каждой строки градиента равна нулю: вероятности обязаны
    остаться вероятностями, поднять один логит можно только за счёт
    остальных. Это первое, что стоит проверить, если сеть не учится.
    """
    n = len(targets)
    grad = []
    for row, target in zip(logits_rows, targets):
        probs = softmax(row)
        probs[target] -= 1.0
        grad.append([p / n for p in probs])
    return grad


def count_parameters(vocab_size, embed_dim, num_layers, max_seq_len, ff_dim, tie_weights=True):
    """Сколько обучаемых чисел в GPT-подобной модели.

    Считаем всё: эмбеддинги токенов и позиций, по 12 блоков из внимания и
    FFN со смещениями, два LayerNorm в блоке и финальный LayerNorm.

    count_parameters(**GPT2_SMALL)                    ->  124439808
    count_parameters(**GPT2_SMALL, tie_weights=False) ->  163037184

    Число голов внимания в подсчёте не участвует: 12 голов по 64 — это та
    же матрица 768x768, просто нарезанная.

    Осторожно с эталонным числом. В таблице урока строка «per-block» дана
    без смещений, а итог 124 438 272 забывает финальный LayerNorm. Честная
    сумма — 124 439 808, ровно её и печатает
    sum(p.numel() for p in GPT2LMHeadModel(...).parameters()).

    Weight tying: голова предсказания переиспользует матрицу эмбеддингов,
    экономя vocab_size * embed_dim параметров — для GPT-2 это 38 миллионов.
    """
    tok = vocab_size * embed_dim
    pos = max_seq_len * embed_dim
    # 4 матрицы внимания + их смещения
    attn = 4 * embed_dim * embed_dim + 4 * embed_dim
    # расширение до ff_dim и обратно, у каждой матрицы своё смещение
    ffn = embed_dim * ff_dim + ff_dim + ff_dim * embed_dim + embed_dim
    norms = 2 * (2 * embed_dim)
    per_block = attn + ffn + norms
    total = tok + pos + num_layers * per_block + 2 * embed_dim
    if not tie_weights:
        total += vocab_size * embed_dim
    return total


def top_k_top_p(probs, top_k=None, top_p=None):
    """Обрезает хвост распределения и перенормирует остаток.

    top_k оставляет k самых вероятных токенов, top_p (nucleus) — самый
    короткий набор, чья суммарная вероятность дошла до p.

    top_k_top_p([0.6, 0.3, 0.1], top_k=1)    ->  [1.0, 0.0, 0.0]
    top_k_top_p([0.6, 0.3, 0.1], top_p=0.85) ->  [0.6666..., 0.3333..., 0.0]

    Хотя бы один токен остаётся всегда, даже если p меньше вероятности
    самого вероятного токена — иначе генерация просто встанет.

    Разница между ними: top_k всегда режет до k штук, top_p подстраивается
    под форму распределения. Когда модель уверена, nucleus оставит один
    токен; когда сомневается — двадцать.
    """
    keep = set(range(len(probs)))
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    if top_k is not None:
        keep &= set(order[:max(1, top_k)])
    if top_p is not None:
        nucleus = []
        acc = 0.0
        for i in order:
            nucleus.append(i)
            acc += probs[i]
            if acc >= top_p:
                break
        keep &= set(nucleus)
    filtered = [p if i in keep else 0.0 for i, p in enumerate(probs)]
    total = sum(filtered)
    return [p / total for p in filtered]


def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Выбирает следующий токен: температура, обрезка хвоста, розыгрыш.

    rng — экземпляр random.Random, чтобы генерация была воспроизводимой.

    sample_next_token([0.0, 100.0], None, temperature=0.0)  ->  1

    temperature == 0 — это жадный выбор argmax. Делить логиты на ноль
    нельзя, случай надо разобрать отдельно; rng при этом не нужен вообще.
    Ниже единицы распределение заостряется, выше — размывается.

    Порядок операций важен: сначала температура, потом softmax, потом
    обрезка. Обрезать сырые логиты по top_p бессмысленно — они не
    вероятности и не суммируются в единицу.
    """
    if temperature == 0:
        return max(range(len(logits)), key=lambda i: (logits[i], -i))
    probs = softmax([x / temperature for x in logits])
    probs = top_k_top_p(probs, top_k, top_p)
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1  # страховка от накопленной ошибки округления
