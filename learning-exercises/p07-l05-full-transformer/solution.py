"""
Полный трансформер — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def layer_norm(x, eps=1e-5):
    """LayerNorm одного вектора: вычесть среднее, поделить на стандартное отклонение.

    Дисперсия считается по всем координатам вектора (без поправки Бесселя),
    eps добавляется под корень.

    layer_norm([1.0, 2.0, 3.0])  ->  примерно [-1.2247, 0.0, 1.2247]
    layer_norm([5.0, 5.0])       ->  примерно [0.0, 0.0]

    Пустой вектор — ValueError.

    Ключевое свойство: LayerNorm стирает и сдвиг, и масштаб. Прибавь ко
    всем координатам константу — результат не изменится. Именно это
    свойство отличает его от RMSNorm ниже.
    """
    if not x:
        raise ValueError("layer_norm of an empty vector")
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    denom = math.sqrt(var + eps)
    return [(v - mean) / denom for v in x]


def rms_norm(x, eps=1e-6):
    """RMSNorm: поделить на корень из среднего квадрата. БЕЗ вычитания среднего.

    rms_norm([3.0, 4.0])  ->  примерно [0.8485, 1.1314]   (RMS равен 3.5355)
    rms_norm([2.0, 2.0])  ->  примерно [1.0, 1.0]

    Пустой вектор — ValueError.

    На векторе с нулевым средним RMSNorm и LayerNorm совпадают: вычитать
    было нечего. Как только среднее не ноль, они расходятся — и это не
    баг, а вся суть замены. Одно вычитание меньше на каждый слой, а
    эмпирическая устойчивость та же (Zhang & Sennrich 2019).
    """
    if not x:
        raise ValueError("rms_norm of an empty vector")
    ms = sum(v * v for v in x) / len(x)
    denom = math.sqrt(ms + eps)
    return [v / denom for v in x]


def silu(x):
    """SiLU (она же Swish): x * sigmoid(x).

    silu(0.0)   ->  0.0
    silu(10.0)  ->  примерно 10.0   (почти как ReLU)
    silu(-2.0)  ->  примерно -0.238 (в отличие от ReLU, не ровно ноль)

    Ловушка: math.exp(-x) при x = -1000 улетает в OverflowError. Разбери
    знак x отдельно, как в уроке про сигмоиду.

    Небольшой отрицательный «провал» и есть причина, по которой SiLU и её
    гейтовый вариант SwiGLU выигрывают у ReLU: градиент не умирает
    полностью на отрицательной стороне.
    """
    if x >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-x))
    else:
        # для отрицательных считаем через e^x / (1 + e^x): то же самое,
        # но exp получает отрицательный аргумент и не переполняется
        e = math.exp(x)
        sigmoid = e / (1.0 + e)
    return x * sigmoid


def ffn_swiglu(x, W1, W2, W3):
    """SwiGLU-FFN одного вектора: W2 @ (silu(W1 @ x) * (W3 @ x)).

    Умножение здесь поэлементное (это и есть «гейт»).
    Формы: W1 и W3 это (hidden, d), W2 это (d, hidden), x длины d.
    Результат — вектор длины d.

    ffn_swiglu([1.0], [[0.0]], [[1.0]], [[1.0]])  ->  [0.0]

    Ловушка: W3 — не опечатка и не дубль W1. Ветка silu(W1 x) решает
    «насколько пропускать», ветка W3 x — «что именно пропускать».
    Обнули W3 — выход станет нулевым при любых W1 и W2. Обнули W1 — выход
    тоже нулевой, потому что silu(0) = 0.

    Три матрицы вместо двух — поэтому в 2026 берут расширение 2.6x, а не
    4x: суммарное число параметров совпадает.
    """
    gate = [silu(sum(w * v for w, v in zip(row, x))) for row in W1]
    up = [sum(w * v for w, v in zip(row, x)) for row in W3]
    hidden = [g * u for g, u in zip(gate, up)]
    return [sum(w * h for w, h in zip(row, hidden)) for row in W2]


def pre_norm_sublayer(rows, sublayer, norm=layer_norm):
    """Pre-norm подслой: out = rows + sublayer(norm(каждая строка)).

    rows — матрица (n, d), sublayer — функция матрицы в матрицу той же формы.
    Нормализуется ВХОД подслоя, а к результату прибавляется НЕтронутый rows.

    pre_norm_sublayer([[1.0, 2.0]], lambda m: [[0.0, 0.0]])  ->  [[1.0, 2.0]]

    Обрати внимание на пример: подслой, который вернул нули, не меняет
    ничего. Резидуальный поток проходит насквозь, и именно поэтому
    pre-norm стек тренируется на 100 слоях без разогрева. Это дефолт 2026
    года: Llama, Qwen, GPT-3+, Mistral.
    """
    normed = [norm(row) for row in rows]
    delta = sublayer(normed)
    return [[a + b for a, b in zip(row, d)] for row, d in zip(rows, delta)]


def post_norm_sublayer(rows, sublayer, norm=layer_norm):
    """Post-norm подслой: out = norm(rows + sublayer(rows)) по строкам.

    Порядок из статьи 2017 года: сначала residual, потом нормализация.

    post_norm_sublayer([[1.0, 2.0]], lambda m: [[0.0, 0.0]])
        ->  примерно [[-1.0, 1.0]]

    Сравни с примером выше: здесь нулевой подслой всё равно меняет вход,
    потому что норма стоит на выходе и переписывает резидуальный поток.
    Отсюда и известная проблема post-norm: сигнал приходится продавливать
    через каждый слой заново, глубокие стеки без warmup не сходятся
    (Xiong et al. 2020).
    """
    delta = sublayer(rows)
    return [norm([a + b for a, b in zip(row, d)]) for row, d in zip(rows, delta)]


def transformer_block(rows, sublayers, norm=layer_norm):
    """Блок трансформера: применить подслои по очереди в pre-norm обвязке.

    sublayers — список функций. У энкодера их два: [self_attention, ffn].
    У декодера три: [masked_self_attention, cross_attention, ffn].
    Пустой список подслоёв возвращает вход как есть.

    transformer_block([[1.0, 2.0]], [])  ->  [[1.0, 2.0]]

    Ровно эта разница — два подслоя против трёх — и отделяет BERT от
    декодера T5. Всё остальное в блоке одинаковое.
    """
    out = rows
    for sublayer in sublayers:
        out = pre_norm_sublayer(out, sublayer, norm)
    return out


def block_params(d_model, ffn_ratio=4.0, swiglu=False, cross_attention=False):
    """Сколько весов в одном блоке (нормы и bias не считаем).

    Внимание:        4 * d^2      (проекции Q, K, V, O)
    Cross-attention: ещё 4 * d^2  (у декодера)
    FFN на ReLU:     2 * ffn_ratio * d^2   (две матрицы)
    FFN на SwiGLU:   3 * ffn_ratio * d^2   (три матрицы)

    block_params(512)                              ->  3145728
    block_params(512, cross_attention=True)        ->  4194304

    Округляй до целого: ffn_ratio бывает нецелым (2.6).

    Проверь на себе табличку из урока: SwiGLU с расширением 2.6 стоит
    почти столько же, сколько ReLU с расширением 4. Поэтому при переходе
    на SwiGLU и уменьшают расширение — чтобы бюджет параметров не поехал.
    """
    attention = 4 * d_model * d_model
    if cross_attention:
        attention *= 2
    matrices = 3 if swiglu else 2
    ffn = matrices * ffn_ratio * d_model * d_model
    return round(attention + ffn)
