"""
CLIP и контрастивное предобучение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def l2_normalize(v):
    """Привести вектор к единичной длине. Новый список.

    l2_normalize([3.0, 4.0])   ->  [0.6, 0.8]
    l2_normalize([0.0, -2.0])  ->  [0.0, -1.0]

    Обе башни CLIP нормируют свой выход, и только поэтому скалярное
    произведение сразу равно косинусу — отдельного деления в лоссе нет.

    Нулевой вектор — ValueError: направления у него нет, а деление на ноль
    даст nan, который потом молча отравит весь батч.
    """
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in v]


def cosine_similarity(a, b):
    """Косинус угла между векторами: от -1 (противоположны) до 1 (совпадают).

    cosine_similarity([1.0, 0.0], [0.0, 1.0])   ->  0.0
    cosine_similarity([1.0, 0.0], [10.0, 0.0])  ->  1.0

    Ключевое свойство: длина векторов не влияет. Изображение «ярче» или
    подпись длиннее — сходство то же самое, важно только направление.

    Разная размерность — ValueError.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same dim")
    ua, ub = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(ua, ub))


def similarity_matrix(images, texts, temperature=1.0):
    """Матрица S[i][j] = cosine_similarity(image_i, text_j) / temperature.

    similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [[1.0, 0.0]]

    Это ровно та матрица, которую CLIP считает на каждом шаге: строки —
    картинки батча, столбцы — подписи. По замыслу правильная пара стоит на
    диагонали, все остальные N-1 клеток строки — негативы.

    temperature (тау) делает распределение резче или мягче. CLIP стартует
    с 0.07 и учит её в лог-пространстве. temperature <= 0 — ValueError.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return [[cosine_similarity(img, txt) / temperature for txt in texts]
            for img in images]


def infonce_loss(S):
    """Симметричный InfoNCE по квадратной матрице сходств. Одно число.

    infonce_loss([[0.0, 0.0], [0.0, 0.0]])    ->  0.6931  (это log 2)
    infonce_loss([[10.0, 0.0], [0.0, 10.0]])  ->  почти 0

    Считается по строкам и по столбцам, потом полусумма:
      row_i = -S[i][i] + logsumexp(S[i][:])     картинка ищет свою подпись
      col_j = -S[j][j] + logsumexp(S[:][j])     подпись ищет свою картинку
      L     = 0.5 * (mean(row) + mean(col))

    Ловушка: exp(S) при больших логитах даёт OverflowError, а при 1/тау=100
    логиты порядка сотен — обычное дело. Считай logsumexp через вычитание
    максимума строки: математически то же, численно живое.

    Матрица обязана быть квадратной — пары идут по диагонали. Иначе ValueError.
    """
    def logsumexp(xs):
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))

    n = len(S)
    if n == 0 or any(len(row) != n for row in S):
        raise ValueError("S must be a non-empty square matrix")
    rows = [logsumexp(S[i]) - S[i][i] for i in range(n)]
    cols = [logsumexp([S[i][j] for i in range(n)]) - S[j][j] for j in range(n)]
    return 0.5 * (sum(rows) / n + sum(cols) / n)


def infonce_grad(S):
    """Аналитический градиент infonce_loss по каждому элементу S.

    Матрица того же размера, что S.

    infonce_grad([[0.0, 0.0], [0.0, 0.0]])  ->  [[-0.25, 0.25], [0.25, -0.25]]

    Вывод короткий: производная -log softmax_i(S)[j] по S[i][j] равна
    softmax_i(S)[j] - [i == j]. Такая же поправка приходит по столбцам.
    Делим на N (усреднение) и на 2 (полусумма двух направлений):

      dL/dS[i][j] = 0.5/N * (P_row[i][j] + P_col[i][j] - 2*[i == j])

    где P_row[i] — softmax строки i, а P_col[j] — softmax столбца j.

    Проверь себя центральной разностью — тесты именно это и делают.
    Полезное следствие: сумма ВСЕХ элементов градиента равна нулю, потому
    что прибавление одной константы ко всей матрице лосс не меняет.
    """
    def softmax(xs):
        m = max(xs)
        e = [math.exp(x - m) for x in xs]
        total = sum(e)
        return [x / total for x in e]

    n = len(S)
    if n == 0 or any(len(row) != n for row in S):
        raise ValueError("S must be a non-empty square matrix")
    p_row = [softmax(S[i]) for i in range(n)]
    # softmax по столбцу считаем один раз на столбец: p_col[j][i]
    p_col = [softmax([S[i][j] for i in range(n)]) for j in range(n)]
    grad = []
    for i in range(n):
        line = []
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            line.append(0.5 / n * (p_row[i][j] + p_col[j][i] - 2.0 * delta))
        grad.append(line)
    return grad


def sigmoid_pairwise_loss(S, bias=0.0):
    """Парный сигмоидный лосс SigLIP. Одно число.

    Каждая клетка матрицы — независимая бинарная классификация «это пара?».
    Метка +1 на диагонали, -1 везде ещё:

      L = -1/N * sum_ij log sigmoid( y_ij * (S[i][j] + bias) ),  y_ij = +-1

    sigmoid_pairwise_loss([[0.0, 0.0], [0.0, 0.0]])  ->  1.3863  (это 2*log 2)

    Делим на N (размер батча), а не на N^2 — так в статье SigLIP.

    Зачем это вместо softmax: softmax требует всю строку целиком, то есть
    all-gather эмбеддингов между всеми GPU. Здесь каждая клетка считается
    сама по себе, и батч 32k+ обходится дёшево.

    Ловушка та же: math.exp(1000) падает. Считай log sigmoid как
    min(x, 0) - log1p(exp(-|x|)) — одна формула на оба знака.
    """
    def log_sigmoid(x):
        return min(x, 0.0) - math.log1p(math.exp(-abs(x)))

    n = len(S)
    if n == 0 or any(len(row) != n for row in S):
        raise ValueError("S must be a non-empty square matrix")
    total = 0.0
    for i in range(n):
        for j in range(n):
            y = 1.0 if i == j else -1.0
            total += log_sigmoid(y * (S[i][j] + bias))
    return -total / n


def zero_shot_classify(image, class_vectors, class_names):
    """Zero-shot классификация: ближайший по косинусу текстовый эмбеддинг.

    Вернуть кортеж (имя класса, косинус).

    zero_shot_classify([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], ["cat", "dog"])
        ->  ("cat", 1.0)

    Никакого обучения на целевых классах: эмбеддишь «a photo of a {class}»
    и берёшь argmax. Так CLIP делает ImageNet, ни разу не увидев его меток.

    При равенстве побеждает первый класс — детерминированность важнее
    красоты. Пустой список классов или несовпадение длин — ValueError.
    """
    if not class_vectors:
        raise ValueError("need at least one class")
    if len(class_vectors) != len(class_names):
        raise ValueError("class_vectors and class_names must match in length")
    best_i = 0
    best_score = cosine_similarity(image, class_vectors[0])
    for i in range(1, len(class_vectors)):
        score = cosine_similarity(image, class_vectors[i])
        if score > best_score:  # строго >, поэтому при ничьей остаётся первый
            best_i, best_score = i, score
    return (class_names[best_i], best_score)


def prompt_ensemble(embeddings):
    """Усреднить эмбеддинги нескольких шаблонов и снова отнормировать.

    prompt_ensemble([[1.0, 0.0], [0.0, 1.0]])  ->  [0.7071, 0.7071]

    CLIP в статье усреднял 80 шаблонов на класс («a photo of a {}»,
    «a painting of a {}», «a low resolution photo of a {}»...) и получал
    +3 пункта на ImageNet. Усреднение гасит шум конкретной формулировки.

    Два шага, оба обязательны: сначала каждый шаблон на единичную сферу,
    иначе длинный вектор перевесит остальные; потом нормировать среднее,
    иначе класс с «размазанными» шаблонами системно проиграет классу с
    согласованными — просто потому, что его средний вектор короче.
    """
    if not embeddings:
        raise ValueError("need at least one embedding")
    dim = len(embeddings[0])
    acc = [0.0] * dim
    for e in embeddings:
        if len(e) != dim:
            raise ValueError("all embeddings must have the same dim")
        for i, x in enumerate(l2_normalize(e)):
            acc[i] += x
    return l2_normalize(acc)
