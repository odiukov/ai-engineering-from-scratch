"""
Разбор архитектур открытых моделей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Шесть ручек, которые отличают Llama 3, Mixtral и DeepSeek-V3 от GPT-2:
RMSNorm вместо LayerNorm, RoPE вместо обучаемых позиций, SwiGLU вместо
GELU, GQA/MLA вместо MHA, MoE вместо плотного MLP, pre-norm вместо
post-norm. Здесь ты собираешь руками первые пять — каждая в реальном коде
модели занимает один класс на 20 строк.

Матрица — список строк: M[i][j] это элемент строки i, столбца j.
Конфиг — тот самый config.json из HuggingFace, ключи один в один.
"""

import math


def rms_norm(x, gamma, eps=1e-5):
    """RMSNorm: делим на корень из среднего квадрата, умножаем на gamma.

    Модуль модели: LlamaRMSNorm.

    rms_norm([3.0, 4.0], [1.0, 1.0], 0.0)  ->  [0.848..., 1.131...]
    rms_norm([1.0, 1.0], [2.0, 2.0], 0.0)  ->  [2.0, 2.0]

    Формула: x_i / sqrt(mean(x^2) + eps) * gamma_i.

    Главное отличие от LayerNorm: среднее НЕ вычитается и сдвига (beta)
    нет. Поэтому rms_norm не центрирует вход — постоянная добавка ко всем
    координатам меняет результат, а у LayerNorm нет.
    """
    # среднее квадрата, а не дисперсия: вычитать среднее здесь нечего
    ms = sum(v * v for v in x) / len(x)
    scale = math.sqrt(ms + eps)
    return [v / scale * g for v, g in zip(x, gamma)]


def rope_rotate(vec, pos, theta=10000.0):
    """RoPE: поворот пар координат на угол, зависящий от позиции.

    Модуль модели: apply_rotary_pos_emb.

    rope_rotate([1.0, 0.0], 0)  ->  [1.0, 0.0]   (нулевая позиция — тождество)
    rope_rotate([1.0, 0.0], 1, theta=1.0)  ->  [cos(1), sin(1)]

    Координаты берутся парами (0,1), (2,3), ...: пара номер i крутится на
    угол pos * theta^(-2i/d). Формула поворота:
        x' = x*cos - y*sin
        y' = x*sin + y*cos

    Нечётная длина вектора -> ValueError: крутить нечего.

    Ради чего это: скалярное произведение повёрнутых q и k зависит только
    от РАЗНОСТИ позиций. Отсюда и экстраполяция за длину обучения — в
    отличие от таблицы обучаемых позиций GPT-2, которая просто кончается.
    """
    d = len(vec)
    if d % 2 != 0:
        raise ValueError(f"длина вектора должна быть чётной, получено {d}")
    out = []
    for i in range(d // 2):
        # частота падает с номером пары: младшие пары крутятся быстро,
        # старшие почти стоят — так один вектор кодирует и близкие, и
        # далёкие расстояния
        angle = pos * theta ** (-2.0 * i / d)
        c, s = math.cos(angle), math.sin(angle)
        x, y = vec[2 * i], vec[2 * i + 1]
        out.append(x * c - y * s)
        out.append(x * s + y * c)
    return out


def softmax(logits):
    """Логиты -> распределение вероятностей.

    softmax([0.0, 0.0])     ->  [0.5, 0.5]
    softmax([0.0, 1000.0])  ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум перед
    экспонентой — результат математически тот же, переполнения нет.
    """
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def swiglu_mlp(x, W_gate, W_up, W_down):
    """MLP-блок современной модели: SwiGLU вместо GELU.

    Модуль модели: LlamaMLP.

    Формула: down( silu(gate(x)) * up(x) ), где silu(z) = z * sigmoid(z).
    Три матрицы вместо двух: gate и up идут параллельно, их произведение
    поэлементное, потом down возвращает в hidden.

    swiglu_mlp([1.0], [[0.0]], [[1.0]], [[1.0]])  ->  [0.0]
        (silu(0) = 0, значит вся ветка гасится)

    Ловушка: наивная sigmoid(z) = 1/(1+exp(-z)) падает с OverflowError на
    больших отрицательных z. Разбери знак: при z < 0 считай exp(z)/(1+exp(z)).

    Именно поэтому intermediate_size у Llama 3 равен 14336, а не 4*4096:
    трёх матриц вместо двух, размер урезан до 8/3 * hidden.
    """
    def matvec(M, v):
        return [sum(a * b for a, b in zip(row, v)) for row in M]

    def sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    gate = matvec(W_gate, x)
    up = matvec(W_up, x)
    hidden = [g * sigmoid(g) * u for g, u in zip(gate, up)]
    return matvec(W_down, hidden)


def top_k_route(logits, k):
    """Роутер MoE: выбрать k экспертов и раздать им веса.

    Модуль модели: MixtralSparseMoeBlock (часть с router).

    Возвращает пару (indices, weights): indices отсортированы по
    возрастанию, weights — softmax ТОЛЬКО по выбранным логитам.

    top_k_route([0.0, 0.0], 2)      ->  ([0, 1], [0.5, 0.5])
    top_k_route([3.0, 0.0, 1.0], 1) ->  ([0], [1.0])

    Ловушка: нормировать надо по выбранным логитам, а не по всем — иначе
    веса не дадут в сумме единицу и выход блока просядет по норме.

    При равных логитах побеждает меньший индекс: маршрутизация обязана
    быть воспроизводимой, иначе два прогона дадут разные выходы.
    """
    if not 1 <= k <= len(logits):
        raise ValueError(f"k должен быть в [1, {len(logits)}], получено {k}")
    # сортируем по убыванию логита, при равенстве — по возрастанию индекса
    ranked = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    chosen = sorted(ranked[:k])
    weights = softmax([logits[i] for i in chosen])
    return chosen, weights


def moe_block(x, experts, router_logits, k):
    """Блок Mixture-of-Experts: взвешенная сумма выходов k экспертов.

    Модуль модели: MixtralSparseMoeBlock целиком.

    experts — список троек (W_gate, W_up, W_down) для swiglu_mlp.
    Невыбранные эксперты НЕ вызываются: в этом весь смысл разреженности,
    671B параметров всего и 37B активных на токен.

    Свойство, которое стоит проверить руками: при k = len(experts) и
    одинаковых логитах ответ равен среднему выходов всех экспертов.
    """
    indices, weights = top_k_route(router_logits, k)
    out = None
    for idx, w in zip(indices, weights):
        W_gate, W_up, W_down = experts[idx]
        y = swiglu_mlp(x, W_gate, W_up, W_down)
        if out is None:
            out = [w * v for v in y]
        else:
            out = [o + w * v for o, v in zip(out, y)]
    return out


def param_count(config):
    """Разбор числа параметров по компонентам прямо из config.json.

    Возвращает словарь с ключами: embedding, attention, mlp, norm, head,
    total. attention/mlp/norm — суммы по ВСЕМ слоям.

    Считаем так:
      head_dim  = hidden_size / num_attention_heads
      attention = Wq(hidden*heads*head_dim) + Wk и Wv(hidden*kv_heads*head_dim)
                  + Wo(heads*head_dim*hidden)   — вот где GQA экономит
      mlp       = 3 * hidden * intermediate     — три матрицы SwiGLU
      norm      = 2 * hidden на слой + один финальный hidden
      head      = vocab * hidden, ноль при tie_word_embeddings

    На конфиге Llama 3 8B total выходит 8_030_261_248 — те самые «8B».

    Смещений (bias) в современных моделях нет, поэтому их и не считаем.
    """
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    vocab = config["vocab_size"]
    head_dim = hidden // heads

    per_attn = (
        hidden * heads * head_dim  # Wq
        + 2 * hidden * kv_heads * head_dim  # Wk и Wv, ужатые GQA
        + heads * head_dim * hidden  # Wo
    )
    per_mlp = 3 * hidden * inter
    embedding = vocab * hidden
    head = 0 if config.get("tie_word_embeddings", False) else vocab * hidden

    counts = {
        "embedding": embedding,
        "attention": per_attn * layers,
        "mlp": per_mlp * layers,
        "norm": 2 * hidden * layers + hidden,
        "head": head,
    }
    counts["total"] = sum(counts.values())
    return counts


def kv_cache_bytes(config, seq_len, bytes_per_elem=2):
    """Размер KV-кэша одной последовательности в байтах.

    Формула урока:
      2 * num_hidden_layers * num_key_value_heads * head_dim * seq_len * bytes

    Двойка — это K и V. bytes_per_elem = 2 для BF16, 1 для FP8.

    На Llama 3 8B при 131072 токенах и BF16 выходит 17_179_869_184 байт,
    то есть 17.2 ГБ — БОЛЬШЕ, чем сами веса в BF16 (16 ГБ). Ровно это и
    гонит всех с MHA на GQA и MLA.
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = hidden // heads
    layers = config["num_hidden_layers"]
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per_elem
