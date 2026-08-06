"""
Собираем трансформер с нуля — капстоун — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(scores):
    """Softmax, устойчивый к большим логитам: вычти максимум перед exp.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([1000.0, 999.0])  ->  [0.7310..., 0.2689...]

    Ловушка: math.exp(1000) бросает OverflowError. Сдвиг на максимум
    математически ничего не меняет, но делает самый большой аргумент exp
    равным нулю.

    Это кирпич, на котором стоит вся остальная модель.
    """
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def linear(x, W):
    """Слой без bias: y[i] = сумма W[i][j] * x[j]. W — список строк.

    linear([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])  ->  [1.0, 2.0]
    linear([1.0, 2.0], [[1.0, 1.0]])              ->  [3.0]

    Строка W — один выходной нейрон, поэтому len(результата) == len(W), а
    len(W[0]) == len(x). Такое расположение весов (out x in) совпадает с
    torch.nn.Linear.weight, и именно поэтому у torch там transpose.

    Ловушка: перепутаешь строки со столбцами — заработает только на
    квадратных матрицах, и потом молча сломается на первом же неквадратном
    слое.
    """
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: делим вектор на его среднеквадратичное, потом масштабируем weight.

    rms_norm([3.0, 4.0], [1.0, 1.0])       ->  [0.8485..., 1.1313...]
    rms_norm([1.0, 1.0, 1.0], [1.0]*3)     ->  [1.0, 1.0, 1.0]
    rms_norm([2.0, 2.0], [3.0, 3.0])       ->  [3.0, 3.0]

    Формула: rms = sqrt(mean(x^2) + eps), выход = weight[i] * x[i] / rms.

    Ловушка: в отличие от LayerNorm среднее НЕ вычитается. Поэтому
    [1, 1, 1] остаётся [1, 1, 1], а не превращается в нули. Это не
    упрощение ради лени — так делают Llama, Mistral, Gemma: на одно
    прохождение по вектору меньше, а качество то же.

    eps нужен ради нулевого вектора: без него первое же деление на нуль.
    """
    ms = sum(xi * xi for xi in x) / len(x)
    rms = math.sqrt(ms + eps)
    return [w * xi / rms for w, xi in zip(weight, x)]


def swiglu_ffn(x, W1, W3, W2):
    """SwiGLU-блок: W2 @ (silu(W1 @ x) * (W3 @ x)).

    silu(t) = t / (1 + exp(-t)) — гладкая ReLU, она же swish.

    swiglu_ffn([1.0], [[0.0]], [[1.0]], [[1.0]])  ->  [0.0]
        (W1 даёт 0, silu(0) = 0, всё произведение — ноль)

    Три матрицы вместо двух: W1 считает «значение», W3 — «ворота», их
    поэлементное произведение и есть gating. Отсюда и лишняя треть
    параметров, из-за которой скрытый слой берут 2/3 от обычного 4*d.

    Ловушка: exp(-t) при большом отрицательном t уходит в OverflowError.
    Разбери знак t отдельно, как в устойчивой сигмоиде.

    Зачем: с 2020 года SwiGLU вытеснил ReLU-FFN во всех серьёзных LLM.
    """
    gate_in = linear(x, W1)
    value = linear(x, W3)
    hidden = []
    for t, v in zip(gate_in, value):
        # устойчивая сигмоида: при t < 0 считаем exp(t), а не exp(-t)
        if t >= 0:
            sig = 1.0 / (1.0 + math.exp(-t))
        else:
            e = math.exp(t)
            sig = e / (1.0 + e)
        hidden.append(t * sig * v)
    return linear(hidden, W2)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    """Causal multi-head attention над последовательностью X (список векторов).

    Вернуть список из len(X) векторов той же размерности.

    Шаги: проекции Q, K, V каждой позиции; разрезание каждого вектора на
    n_heads подряд идущих кусков по d_head; внутри головы — causal softmax
    с масштабом 1/sqrt(d_head); склейка голов обратно; проекция Wo.

    Свойства, которые обязаны держаться:
      выход позиции 0 == linear(linear(X[0], Wv), Wo)
        — единственный доступный токен получает вес 1.0;
      выход позиции i не зависит от X[j] при j > i
        — это и есть causal-маска, и это главный тест урока.

    Ловушка: масштаб 1/sqrt(d_head), а не 1/sqrt(d_model). На одной голове
    разницы нет, на четырёх — уже вдвое, и модель учится заметно хуже.

    Ловушка вторая: голова h берёт координаты [h*d_head, (h+1)*d_head), а
    не каждую n_heads-ю. Перепутаешь — сеть всё равно обучится, но головы
    перестанут быть независимыми подпространствами.
    """
    n = len(X)
    d_model = len(Wq)
    d_head = d_model // n_heads
    Q = [linear(x, Wq) for x in X]
    K = [linear(x, Wk) for x in X]
    V = [linear(x, Wv) for x in X]
    scale = 1.0 / math.sqrt(d_head)
    merged = [[0.0] * d_model for _ in range(n)]
    for h in range(n_heads):
        lo, hi = h * d_head, (h + 1) * d_head
        for i in range(n):
            # причинность: ключи только до текущей позиции включительно
            scores = [
                sum(a * b for a, b in zip(Q[i][lo:hi], K[j][lo:hi])) * scale
                for j in range(i + 1)
            ]
            weights = softmax(scores)
            for j, w in enumerate(weights):
                for c in range(lo, hi):
                    merged[i][c] += w * V[j][c]
    return [linear(m, Wo) for m in merged]


def transformer_block(X, block):
    """Pre-norm блок: X + attn(RMSNorm(X)), затем результат + ffn(RMSNorm(...)).

    block — словарь с ключами "n_heads", "norm1", "wq", "wk", "wv", "wo",
    "norm2", "w1", "w3", "w2" (см. init_params).

    Свойство, которое стоит проверить первым: если "wo" и "w2" — нулевые
    матрицы, блок обязан вернуть X без изменений. Оба residual-а
    прозрачны, значит подключены правильно.

    Порядок важен: норма ВНУТРИ residual-а (pre-norm), а не после него.
    Post-norm из оригинальной статьи 2017 года требует warmup и разваливается
    на глубоких стопках; с 2019 года все нормальные LLM — pre-norm.
    """
    normed = [rms_norm(x, block["norm1"]) for x in X]
    attn = multi_head_attention(
        normed, block["wq"], block["wk"], block["wv"], block["wo"], block["n_heads"]
    )
    mid = [[a + b for a, b in zip(x, delta)] for x, delta in zip(X, attn)]
    normed2 = [rms_norm(x, block["norm2"]) for x in mid]
    ffn = [swiglu_ffn(h, block["w1"], block["w3"], block["w2"]) for h in normed2]
    return [[a + b for a, b in zip(x, delta)] for x, delta in zip(mid, ffn)]


def init_params(vocab_size, d_model, n_heads, n_layers, block_size, rng, scale=0.02):
    """Случайная инициализация всех весов модели. rng — объект random.Random.

    Вернуть словарь:
      "tok_emb": vocab_size строк по d_model   — таблица эмбеддингов токенов
      "pos_emb": block_size строк по d_model   — обучаемые позиционные
      "blocks":  список из n_layers словарей, в каждом:
            "n_heads": n_heads
            "norm1":   d_model единиц
            "wq", "wk", "wv", "wo": d_model строк по d_model
            "norm2":   d_model единиц
            "w1", "w3": hidden строк по d_model, где hidden = 2 * d_model
            "w2":       d_model строк по hidden
      "norm_f":  d_model единиц

    Отдельного lm_head нет: он привязан (tied) к "tok_emb", логиты
    считаются как linear(h, params["tok_emb"]). Это экономит
    vocab_size * d_model параметров и заодно улучшает качество.

    Все матрицы — rng.gauss(0, scale), все веса норм — ровно 1.0.

    Ловушка: rng приходит параметром. Глобальный random сделал бы тесты
    невоспроизводимыми, а по инициализации проверяется самое важное
    свойство урока — что loss на старте равен ln(vocab_size).
    """
    def mat(rows, cols):
        return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

    hidden = 2 * d_model
    blocks = []
    for _ in range(n_layers):
        blocks.append({
            "n_heads": n_heads,
            "norm1": [1.0] * d_model,
            "wq": mat(d_model, d_model),
            "wk": mat(d_model, d_model),
            "wv": mat(d_model, d_model),
            "wo": mat(d_model, d_model),
            "norm2": [1.0] * d_model,
            "w1": mat(hidden, d_model),
            "w3": mat(hidden, d_model),
            "w2": mat(d_model, hidden),
        })
    return {
        "tok_emb": mat(vocab_size, d_model),
        "pos_emb": mat(block_size, d_model),
        "blocks": blocks,
        "norm_f": [1.0] * d_model,
    }


def gpt_forward(tokens, params):
    """Полный forward: эмбеддинги -> блоки -> финальная норма -> связанный lm_head.

    Вернуть список из len(tokens) векторов логитов длиной vocab_size.

    Порядок в точности как в схеме урока:
      x = tok_emb[token] + pos_emb[позиция]
      x = каждый блок по очереди
      x = rms_norm(x, params["norm_f"])
      логиты = linear(x, params["tok_emb"])     <- тот самый tied lm_head

    Ловушка: pos_emb индексируется позицией в последовательности, а не
    значением токена. Перепутаешь — модель потеряет порядок слов и
    останется на уровне bag-of-words.

    Причинность обязана дожить до логитов: подмени последний токен, и
    логиты всех предыдущих позиций не должны шевельнуться.
    """
    X = [
        [t + p for t, p in zip(params["tok_emb"][tok], params["pos_emb"][i])]
        for i, tok in enumerate(tokens)
    ]
    for block in params["blocks"]:
        X = transformer_block(X, block)
    X = [rms_norm(x, params["norm_f"]) for x in X]
    return [linear(x, params["tok_emb"]) for x in X]


def cross_entropy_next_token(logits, tokens):
    """Средняя кросс-энтропия предсказания следующего токена (сдвиг на один).

    Логиты позиции i предсказывают tokens[i+1]; последняя позиция цели не
    имеет и в среднее не входит. Делим на len(tokens) - 1.

    cross_entropy_next_token([[0.0, 0.0], [0.0, 0.0]], [0, 1])  ->  0.6931...
        (ln 2: два равновероятных варианта)
    cross_entropy_next_token([[0.0, 50.0], [0.0, 0.0]], [0, 1])  ->  ~0.0
        (уверенно и верно предсказал единицу)

    Ловушка: не считай softmax, а потом log — на больших логитах это
    ноль под логарифмом. Используй log-sum-exp: loss = lse(row) - row[цель],
    где lse(row) = max + ln(сумма exp(row - max)).

    Свежая модель со случайными весами обязана дать примерно
    ln(vocab_size): она ещё ничего не знает и распределяет вес ровно.
    Это первая проверка любого обучения — если на старте не ln(V), где-то
    ошибка в forward, и учить бессмысленно.
    """
    total = 0.0
    for i in range(len(tokens) - 1):
        row = logits[i]
        top = max(row)
        lse = top + math.log(sum(math.exp(v - top) for v in row))
        total += lse - row[tokens[i + 1]]
    return total / (len(tokens) - 1)
