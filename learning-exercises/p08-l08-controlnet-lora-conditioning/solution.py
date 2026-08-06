"""
ControlNet, LoRA и обусловливание — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def matvec(M, x):
    """Умножение матрицы на вектор: M @ x.

    matvec([[1, 2], [3, 4]], [1, 1])   ->  [3, 7]
    matvec([[1, 0], [0, 1]], [5, -2])  ->  [5, -2]

    Базовый кирпич: и замороженный слой W, и множители LoRA — это он.
    Длина результата равна числу СТРОК M, а не длине x.
    """
    return [sum(m * v for m, v in zip(row, x)) for row in M]


def lora_delta(A, B, alpha=1.0):
    """Полная матрица поправки LoRA: alpha * B @ A.

    A — матрица r x d_in, B — матрица d_out x r.

    lora_delta([[1.0, 2.0]], [[3.0], [4.0]])       ->  [[3.0, 6.0], [4.0, 8.0]]
    lora_delta([[1.0, 2.0]], [[0.0], [0.0]])       ->  [[0.0, 0.0], [0.0, 0.0]]

    На инференсе эту матрицу материализовать НЕ надо (см. lora_forward) — она
    нужна только чтобы посмотреть на дельту глазами: её ранг не может быть
    больше r, сколько бы ни было d.

    Заодно тут видно, почему B инициализируют нулями: при B = 0 дельта ровно
    нулевая, и адаптер на шаге 0 не портит предобученную модель.
    """
    r = len(A)
    d_in = len(A[0])
    return [
        [alpha * sum(row_b[k] * A[k][j] for k in range(r)) for j in range(d_in)]
        for row_b in B
    ]


def lora_forward(W, A, B, x, alpha=1.0):
    """Проход через слой с адаптером: (W + alpha * B @ A) @ x.

    lora_forward([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], [[0.0], [0.0]], [2.0, 3.0])
        ->  [2.0, 3.0]        (B нулевая — ровно исходный слой)

    Считай в порядке B @ (A @ x), а не (B @ A) @ x. Первый вариант — это
    d_in*r + d_out*r операций, второй — d_out*d_in*r плюс лишняя матрица в
    памяти. Ради этого LoRA и придумали.

    alpha — рантайм-регулятор силы адаптера. 0.5-1.5 нормально, 2-3 ломает
    картинку.
    """
    base = matvec(W, x)
    correction = matvec(B, matvec(A, x))
    return [b + alpha * c for b, c in zip(base, correction)]


def merge_lora(W, A, B, alpha=1.0):
    """Вплавить адаптер в веса: вернуть новую матрицу W + alpha * B @ A.

    merge_lora([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], [[0.0], [0.0]])
        ->  [[1.0, 0.0], [0.0, 1.0]]

    Так делают ради 3-5% скорости на шаг: сложения в рантайме больше нет.
    Цена — alpha застывает навсегда, и обратно адаптер уже не отклеить.
    Поэтому в продакшене держат обе версии.

    Ловушка: W менять на месте нельзя, база должна остаться замороженной.
    """
    delta = lora_delta(A, B, alpha)
    return [[w + d for w, d in zip(row_w, row_d)] for row_w, row_d in zip(W, delta)]


def matrix_rank(M, tol=1e-9):
    """Ранг матрицы методом Гаусса с выбором ведущего элемента.

    matrix_rank([[1.0, 0.0], [0.0, 1.0]])   ->  2
    matrix_rank([[1.0, 2.0], [2.0, 4.0]])   ->  1   (вторая строка — первая x2)
    matrix_rank([[0.0, 0.0]])               ->  0

    Нужен, чтобы своими глазами увидеть главное свойство LoRA: у дельты
    B @ A ранг не выше r, сколько бы ни было измерений.

    tol — АБСОЛЮТНЫЙ порог: элемент меньше него считается нулём. Для матриц с
    элементами порядка единицы 1e-9 в самый раз; для сильно отмасштабированных
    порог придётся поднять.
    """
    rows = [list(row) for row in M]
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows else 0
    rank = 0
    for col in range(n_cols):
        # ведущий элемент — максимальный по модулю: без этого метод неустойчив
        pivot, best = None, tol
        for i in range(rank, n_rows):
            if abs(rows[i][col]) > best:
                pivot, best = i, abs(rows[i][col])
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        head = rows[rank][col]
        for i in range(rank + 1, n_rows):
            factor = rows[i][col] / head
            if factor:
                for j in range(col, n_cols):
                    rows[i][j] -= factor * rows[rank][j]
        rank += 1
    return rank


def lora_param_count(d_out, d_in, r):
    """Сколько параметров у полного дообучения и сколько у LoRA. Вернуть пару.

    lora_param_count(640, 640, 16)  ->  (409600, 20480)
    lora_param_count(4, 4, 4)       ->  (16, 32)

    Полное: d_out * d_in. LoRA: r * (d_out + d_in) — это A размера r x d_in
    плюс B размера d_out x r.

    Второй пример показывает границу: при r, сравнимом с d, никакой экономии
    нет, LoRA становится дороже полного слоя. Смысл есть только при r << d.
    """
    return d_out * d_in, r * (d_out + d_in)


def lora_grads(W, A, B, x, target, alpha=1.0):
    """Градиенты loss = sum_i (pred_i - target_i)^2 по A и B. Вернуть (grad_A, grad_B).

    W заморожена, по ней градиент не считаем вовсе — в этом весь смысл LoRA.

    lora_grads([[1.0]], [[1.0]], [[0.0]], [1.0], [1.0])  ->  ([[0.0]], [[0.0]])
        (предсказание уже точное — градиенты нулевые)

    Loss без множителя 1/2, поэтому в градиентах остаётся честная двойка.
    Проверь себя центральной разностью: сдвинь один элемент A на +-h и посмотри,
    как меняется loss.

    Заметь ещё одно: при B = 0 градиент по A тоже нулевой. Поэтому нулями
    инициализируют ТОЛЬКО B, а A заполняют случайными числами — иначе адаптер
    никогда не сдвинется с места.
    """
    pred = lora_forward(W, A, B, x, alpha)
    err = [2.0 * (p - t) for p, t in zip(pred, target)]
    ax = matvec(A, x)
    r = len(A)
    d_out = len(B)
    grad_B = [[alpha * err[i] * ax[k] for k in range(r)] for i in range(d_out)]
    # по A градиент идёт через B: сначала свернём ошибку с k-м столбцом B
    grad_A = [
        [alpha * sum(err[i] * B[i][k] for i in range(d_out)) * xj for xj in x]
        for k in range(r)
    ]
    return grad_A, grad_B


def apply_controls(base_out, side_outs, gates):
    """Сложение выходов ControlNet-ов с выходом замороженной базы.

    base_out + sum_k gates[k] * side_outs[k]

    apply_controls([1.0, 2.0], [[10.0, 10.0]], [0.0])   ->  [1.0, 2.0]
    apply_controls([1.0, 2.0], [[10.0, 10.0]], [0.5])   ->  [6.0, 7.0]

    gates[k] = 0 — это наша версия zero-convolution: до обучения ControlNet
    ничего не меняет, база работает как была. Нулевой гейт и есть гарантия
    «не сломаем предобученную модель».

    Практика: сумма весов ~1.0 — безопасный дефолт. Pose 1.0 + Depth 1.0 уже
    перебор, картинка начинает спорить сама с собой.

    Ловушка: base_out менять на месте нельзя, он ещё нужен вызывающему.
    """
    out = list(base_out)
    for side, gate in zip(side_outs, gates):
        for i in range(len(out)):
            out[i] += gate * side[i]
    return out
