"""
Fine-tuning с LoRA и QLoRA — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random

# Матрица — список строк: W[i][j] это вес из входа j в выход i.
# W имеет размер d_out x d_in, A — rank x d_in, B — d_out x rank.
# В коде урока (PyTorch) те же матрицы записаны транспонированно, потому что
# nn.Linear умножает строку-вектор справа: x @ A @ B. Математика та же.

# Число уровней у 4-битного веса: 16 штук, симметрично вокруг нуля.
# Отсюда деление absmax на 7 — это и есть "simulated NF4" из урока.
QUANT_LEVELS = 7


def linear(W, x):
    """Обычный линейный слой: y = W x. Вернуть список длины d_out.

    linear([[1, 2], [3, 4]], [1, 1])  ->  [3.0, 7.0]
    linear([[1, 0], [0, 1]], [5, 9])  ->  [5.0, 9.0]

    Это тот самый замороженный слой, поверх которого LoRA потом добавит
    свою поправку. Внутри ничего умного: строка W скалярно умножается на x.
    """
    return [float(sum(w * xi for w, xi in zip(row, x))) for row in W]


def init_lora(d_in, d_out, rank, seed=0):
    """Инициализация адаптера: A — случайная, B — нулевая. Вернуть (A, B).

    A, B = init_lora(4, 3, 2, seed=0)
    len(A), len(A[0])  ->  (2, 4)      # rank x d_in
    len(B), len(B[0])  ->  (3, 2)      # d_out x rank
    B                  ->  [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    A заполняется из нормального распределения с масштабом 1/sqrt(rank),
    B — строго нулями. Порядок важен: произведение B @ A в начале равно нулю,
    значит адаптер сначала НЕ меняет выход базовой модели, и обучение
    стартует ровно с её поведения.

    Случайность берётся из random.Random(seed), а не из глобального random:
    иначе один и тот же seed давал бы разные адаптеры и повторить обучение
    было бы невозможно.

    Соответствует peft.LoraConfig(r=rank, ...) + get_peft_model().
    """
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(rank)
    A = [[rng.gauss(0.0, 1.0) * scale for _ in range(d_in)] for _ in range(rank)]
    B = [[0.0 for _ in range(rank)] for _ in range(d_out)]
    return A, B


def lora_forward(W, A, B, alpha, x):
    """Прямой проход с адаптером: y = W x + (alpha / rank) * B (A x).

    lora_forward([[1, 0], [0, 1]], [[1, 1]], [[2], [0]], 1, [1, 2])
        ->  [7.0, 2.0]

    Разбор: W x = [1, 2]; A x = [3]; B (A x) = [6, 0]; alpha/rank = 1.

    Скобки принципиальны. Считать (B A) x значит собрать полную матрицу
    d_out x d_in — ровно то, чего LoRA избегает. Считать B (A x) значит
    два умножения на маленькие матрицы: сначала d_in -> rank, потом
    rank -> d_out. Именно поэтому адаптер почти ничего не стоит по памяти.

    rank берётся как len(A) — число строк A.
    """
    scaling = alpha / len(A)
    h = linear(A, x)              # d_in -> rank, узкое место
    delta = linear(B, h)          # rank -> d_out
    base = linear(W, x)
    return [b + scaling * d for b, d in zip(base, delta)]


def merge_lora(W, A, B, alpha):
    """Вплавить адаптер в веса: W' = W + (alpha / rank) * B A. Вернуть новую матрицу.

    merge_lora([[1, 0], [0, 1]], [[1, 1]], [[2], [0]], 1)  ->  [[3.0, 2.0], [0.0, 1.0]]

    После слияния адаптера нет: модель того же размера, что исходная, и на
    инференсе делает одно матричное умножение вместо трёх. Цена — адаптер
    больше не отцепить и не подменить другим.

    Исходную W трогать нельзя: её же держат другие адаптеры. Возвращай новый
    список списков, а не правь переданный на месте.

    Соответствует peft model.merge_and_unload().
    """
    rank = len(A)
    scaling = alpha / rank
    merged = []
    for i, row in enumerate(W):
        # B[i] — строка длины rank, A[r] — строка длины d_in
        delta = [
            scaling * sum(B[i][r] * A[r][j] for r in range(rank))
            for j in range(len(row))
        ]
        merged.append([w + d for w, d in zip(row, delta)])
    return merged


def count_trainable(d_in, d_out, rank):
    """Сколько параметров учит LoRA против полного дообучения.

    count_trainable(4096, 4096, 16)
        ->  {'full': 16777216, 'lora': 131072, 'ratio': 0.0078125}

    full = d_in * d_out — полное дообучение трогает каждый вес.
    lora = rank * (d_in + d_out) — только A и B.
    ratio = lora / full.

    Ловушка: экономия не бесконечна. При rank = d_in * d_out / (d_in + d_out)
    адаптер становится ровно такого же размера, что и полная матрица, и
    смысл теряется. Для квадратной 4096x4096 это rank = 2048.
    """
    full = d_in * d_out
    lora = rank * (d_in + d_out)
    return {"full": full, "lora": lora, "ratio": lora / full if full else 0.0}


def quantize_dequantize(matrix, block_size=8):
    """Симуляция 4-битного веса: absmax-квантование по блокам и обратно.

    quantize_dequantize([[-3.5, -1.5, 0.0, 2.0, 3.5]], block_size=5)
        ->  [[-3.5, -1.5, 0.0, 2.0, 3.5]]     # значения кратны шагу, потерь нет
    quantize_dequantize([[0.0, 0.0]], block_size=2)  ->  [[0.0, 0.0]]

    На каждый блок из block_size весов считается свой масштаб
    scale = max(|w|) / 7, вес округляется до целого q в диапазоне [-8, 7] и
    восстанавливается как q * scale. Форма матрицы сохраняется.

    Ловушка: пустой блок из одних нулей даёт scale = 0 и деление на ноль.
    Такой блок восстанавливается нулями.

    Два следствия, которые и объясняют, почему в QLoRA блоки маленькие:
    самый большой по модулю вес блока восстанавливается ТОЧНО, а один
    выброс в блоке съедает точность всех своих соседей. В bitsandbytes
    block_size = 64, а NF4 вдобавок ставит уровни по квантилям нормального
    распределения вместо равномерной сетки.
    """
    flat = [v for row in matrix for v in row]
    out = []
    for start in range(0, len(flat), block_size):
        block = flat[start : start + block_size]
        scale = max(abs(v) for v in block) / QUANT_LEVELS
        if scale == 0.0:
            out.extend(0.0 for _ in block)
            continue
        for v in block:
            q = max(-8, min(7, round(v / scale)))
            out.append(q * scale)

    # обратно в форму matrix
    result, pos = [], 0
    for row in matrix:
        result.append(out[pos : pos + len(row)])
        pos += len(row)
    return result


def lora_grads(W, A, B, alpha, x, target):
    """Градиенты MSE по A и B для одного примера. Вернуть (loss, grad_A, grad_B).

    lora_grads([[0.0]], [[1.0]], [[1.0]], 1.0, [2.0], [0.0])
        ->  (4.0, [[8.0]], [[8.0]])

    loss = среднее по выходам от (pred - target)^2, где pred = lora_forward(...).

    Вывод по цепному правилу, s = alpha / rank:
        h = A x
        pred = W x + s * B h
        e_i = d loss / d pred_i = 2 * (pred_i - target_i) / d_out
        grad_B[i][r] = e_i * s * h[r]
        grad_A[r][j] = s * x[j] * sum_i e_i * B[i][r]

    W в градиентах не участвует вообще — он заморожен, и это весь смысл LoRA.

    Проверь себя: при B = 0 (то есть сразу после init_lora) grad_A выходит
    нулевым, а grad_B — нет. Первый шаг обучения двигает только B; A
    начинает получать градиент лишь после того, как B отошёл от нуля.

    Ни A, ни B, ни W функция не меняет.
    """
    rank = len(A)
    d_out = len(W)
    scaling = alpha / rank

    h = linear(A, x)
    pred = lora_forward(W, A, B, alpha, x)
    errors = [2.0 * (p - t) / d_out for p, t in zip(pred, target)]
    loss = sum((p - t) ** 2 for p, t in zip(pred, target)) / d_out

    grad_B = [[errors[i] * scaling * h[r] for r in range(rank)] for i in range(d_out)]
    # по A градиент собирается через B: вклад входа j в выход i идёт через r
    back = [sum(errors[i] * B[i][r] for i in range(d_out)) for r in range(rank)]
    grad_A = [[scaling * back[r] * xj for xj in x] for r in range(rank)]
    return loss, grad_A, grad_B


def train_lora(W, data, rank=2, alpha=4.0, lr=0.05, epochs=50, seed=0):
    """Обучить адаптер на замороженной W. Вернуть (A, B, losses).

    data — список пар (x, target). losses — средний loss за каждую эпоху,
    длина списка равна epochs.

    W, A, B = ..., адаптер учится обычным SGD:
        для каждого примера посчитать градиенты и шагнуть против них.

    Ключевое свойство, которое стоит проверить руками: W после обучения
    БАЙТ В БАЙТ такая же, как была. Всё выученное лежит в A и B — те самые
    10-100 МБ адаптера, которые выкладывают на Hugging Face Hub отдельно от
    базовой модели.

    Порядок примеров не перемешивается: обучение обязано быть воспроизводимым,
    а вся случайность урока сидит в seed для init_lora.
    """
    A, B = init_lora(len(W[0]), len(W), rank, seed)
    losses = []
    for _ in range(epochs):
        total = 0.0
        for x, target in data:
            loss, grad_A, grad_B = lora_grads(W, A, B, alpha, x, target)
            total += loss
            # новые списки, а не правка на месте: lora_grads считает от
            # текущих A и B, и портить их посреди шага нельзя
            A = [[a - lr * g for a, g in zip(row, grow)] for row, grow in zip(A, grad_A)]
            B = [[b - lr * g for b, g in zip(row, grow)] for row, grow in zip(B, grad_B)]
        losses.append(total / len(data))
    return A, B, losses
