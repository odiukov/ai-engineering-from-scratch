"""
CNN и RNN для текста — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def conv1d(sequence, kernel, bias=0.0):
    """Одномерная свёртка по последовательности эмбеддингов, один выходной канал.

    sequence — список из T векторов длины d (эмбеддинги токенов),
    kernel — список из k векторов длины d (обучаемый детектор k-граммы).
    Ответ — список из T - k + 1 чисел: скалярное произведение окна на ядро
    плюс bias. К отрицательным значениям ReLU здесь ещё не применяется.

    conv1d([[1.0], [2.0], [3.0]], [[1.0], [1.0]])       ->  [3.0, 5.0]
    conv1d([[1.0], [2.0], [3.0]], [[1.0], [1.0]], 0.5)  ->  [3.5, 5.5]
    conv1d([[1.0]], [[1.0], [1.0]])                     ->  []   (ядро шире входа)

    Это `nn.Conv1d(d, 1, kernel_size=k)` без транспонирования: PyTorch хочет
    [batch, channels, seq_len], а здесь честный [seq_len, channels], потому
    что так читаемее.

    Ловушка: если длина вектора в ядре не совпадает с размерностью
    эмбеддинга, zip молча обрежет по короткому и вернёт правдоподобную чушь.
    Такой вызов должен падать с ValueError.
    """
    d = len(sequence[0]) if sequence else (len(kernel[0]) if kernel else 0)
    for row in kernel:
        if len(row) != d:
            raise ValueError("kernel width must match the embedding dim")

    k = len(kernel)
    out = []
    for i in range(len(sequence) - k + 1):
        total = bias
        for a in range(k):
            # внутренний цикл по каналам: обычное скалярное произведение
            total += sum(w * kv for w, kv in zip(sequence[i + a], kernel[a]))
        out.append(total)
    return out


def global_max_pool(values):
    """Глобальный max-pooling: из карты признаков остаётся одно число.

    global_max_pool([0.0, 3.0, 1.0])  ->  3.0

    Пулинг делает представление независимым от позиции: фраза "not good"
    зажигает один и тот же признак и в начале отзыва, и в середине. И он же
    даёт фиксированный размер выхода при любой длине входа.

    Ловушка: на пустом списке максимума нет. Бросай ValueError, а не
    возвращай ноль — ноль неотличим от честного нулевого признака.
    """
    if not values:
        raise ValueError("global_max_pool of an empty feature map")
    return max(values)


def textcnn_features(sequence, filters):
    """Признаки TextCNN: свёртка -> ReLU -> глобальный max-pool по каждому фильтру.

    filters — список пар (kernel, bias). Ответ — список чисел длиной
    len(filters); в сети это то, что уходит в `nn.Linear`.

    seq = [[1.0], [2.0], [3.0]]
    textcnn_features(seq, [([[1.0], [1.0]], 0.0), ([[1.0]], 0.0)])  ->  [5.0, 3.0]

    ReLU здесь обязателен: карта признаков должна отвечать «этот n-грамм
    есть» или «его нет», а не «его минус два».

    Фильтры разной ширины (2, 3, 4) ловят разные масштабы: биграммы,
    триграммы, четвёрки. Именно поэтому в статье Kim (2014) их несколько.

    Ловушка: фильтр шире последовательности даёт пустую карту, и пулинг по
    ней падает. Это правильное поведение, в PyTorch будет тот же краш.
    """
    out = []
    for kernel, bias in filters:
        feature_map = conv1d(sequence, kernel, bias)
        out.append(global_max_pool([v if v > 0 else 0.0 for v in feature_map]))
    return out


def rnn_step(x, h_prev, W_x, W_h, b):
    """Один шаг RNN: h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b).

    W_x — матрица (d_h, d_in), W_h — матрица (d_h, d_h), b — вектор длины d_h.

    rnn_step([1.0], [0.0], [[1.0]], [[1.0]], [0.0])  ->  [0.7615941...]  (tanh(1))
    rnn_step([0.0], [0.0], [[1.0]], [[1.0]], [0.0])  ->  [0.0]

    Веса общие для всех шагов — это и есть «рекуррентность». Из-за общего W_h
    градиент по времени превращается в произведение одинаковых множителей, и
    отсюда растёт затухание.

    tanh зажимает состояние в (-1, 1), так что взорваться может градиент, но
    не само состояние.
    """
    h = []
    for i in range(len(b)):
        total = b[i]
        total += sum(W_x[i][j] * x[j] for j in range(len(x)))
        total += sum(W_h[i][j] * h_prev[j] for j in range(len(h_prev)))
        h.append(math.tanh(total))
    return h


def rnn_forward(sequence, W_x, W_h, b, h0=None, reverse=False):
    """Прогон RNN по последовательности. Ответ — список из T скрытых состояний.

    h0 по умолчанию — нулевой вектор длины len(b).

    rnn_forward([[1.0], [1.0]], [[1.0]], [[0.0]], [0.0])
        ->  [[0.7615...], [0.7615...]]

    При reverse=True последовательность читается справа налево, но список
    состояний возвращается в исходном порядке позиций — чтобы его можно было
    поэлементно склеить с прямым проходом и получить BiRNN. Состояние на
    позиции i при этом видит правый контекст, а не левый.

    Проверка на понимание: rnn_forward(seq, ..., reverse=True) обязана
    совпасть с rnn_forward(seq[::-1], ...)[::-1].
    """
    h = list(h0) if h0 is not None else [0.0] * len(b)
    order = range(len(sequence) - 1, -1, -1) if reverse else range(len(sequence))

    states = []
    for i in order:
        h = rnn_step(sequence[i], h, W_x, W_h, b)
        states.append(h)
    if reverse:
        states.reverse()  # вернуть выравнивание по позициям входа
    return states


def pool_hidden(states, mode="max"):
    """Свёртка последовательности состояний в один вектор: max, mean или last.

    pool_hidden([[1.0, 5.0], [4.0, 2.0]], "max")   ->  [4.0, 5.0]
    pool_hidden([[1.0, 5.0], [4.0, 2.0]], "mean")  ->  [2.5, 3.5]
    pool_hidden([[1.0, 5.0], [4.0, 2.0]], "last")  ->  [4.0, 2.0]

    max и mean — покоординатные, а не «выбрать один из векторов». Для
    классификации max обычно выигрывает у last: последнее состояние длинной
    последовательности помнит в основном её хвост.

    Ловушка: неизвестный mode и пустой список состояний — ValueError. Тихо
    возвращённый last вместо опечатанного "maxx" стоит потом дня отладки.
    """
    if not states:
        raise ValueError("pool_hidden of an empty state list")
    if mode == "last":
        return list(states[-1])
    if mode == "max":
        return [max(s[i] for s in states) for i in range(len(states[0]))]
    if mode == "mean":
        return [sum(s[i] for s in states) / len(states) for i in range(len(states[0]))]
    raise ValueError(f"unknown pooling mode: {mode}")


def vanishing_factor(seq_len, recurrent_weight=0.9):
    """Во сколько раз ослабнет градиент, пройдя seq_len шагов назад по времени.

    vanishing_factor(1, 0.9)    ->  0.9
    vanishing_factor(100, 0.9)  ->  2.65e-05
    vanishing_factor(100, 1.1)  ->  13780.6...

    Это тот же множитель, что и произведение производных из урока про цепное
    правило, только все множители одинаковые — веса RNN общие по времени.

    Меньше единицы — градиент затухает и модель не учит дальние зависимости.
    Больше единицы — взрывается и обучение разваливается. Ровно единица
    невозможна на практике. LSTM обходит выбор: у неё есть аддитивный
    «хайвей» cell state, по которому градиент течёт без этого множителя.
    """
    return recurrent_weight ** seq_len


def lstm_step(x, h_prev, c_prev, gates):
    """Один шаг LSTM. Вернуть кортеж (h, c) — новое скрытое и новое cell state.

    gates — словарь с ключами "f", "i", "g", "o"; в каждом тройка
    (W_x, W_h, b) той же формы, что у rnn_step.

    Уравнения (везде покоординатно):
        f = sigmoid(W_xf x + W_hf h_prev + b_f)     что забыть
        i = sigmoid(W_xi x + W_hi h_prev + b_i)     что записать
        g = tanh(   W_xg x + W_hg h_prev + b_g)     что именно записать
        o = sigmoid(W_xo x + W_ho h_prev + b_o)     что показать наружу
        c = f * c_prev + i * g
        h = o * tanh(c)

    Ловушка: у гейтов f, i, o — сигмоида (это «ворота», их значение от 0 до
    1), а у кандидата g — tanh (это содержимое, оно знаковое). Перепутать
    легко, а сеть после этого просто не учится.

    Ловушка: c не проходит через tanh при записи, поэтому по модулю он
    свободно уходит за единицу. Зажат только h.

    Главное свойство: при f = 1 и i = 0 получается c = c_prev — ровно то же
    число, сколько шагов ни делай. Это и есть «хайвей», по которому градиент
    течёт без множителя из vanishing_factor. Плоское умножение на W_h у RNN
    такого не умеет.

    Это `nn.LSTMCell` руками. PyTorch держит все четыре матрицы одним
    тензором ради скорости, здесь они разложены по ключам ради читаемости.
    """

    def affine(triple):
        W_x, W_h, b = triple
        out = []
        for i in range(len(b)):
            total = b[i]
            total += sum(W_x[i][j] * x[j] for j in range(len(x)))
            total += sum(W_h[i][j] * h_prev[j] for j in range(len(h_prev)))
            out.append(total)
        return out

    def sigmoid(z):
        # обрезка ровно та же, что в np.clip(z, -20, 20): math.exp(1000)
        # роняет программу с OverflowError, а гейт и так уже насыщен
        z = 20.0 if z > 20 else (-20.0 if z < -20 else z)
        return 1.0 / (1.0 + math.exp(-z))

    f = [sigmoid(z) for z in affine(gates["f"])]
    i_gate = [sigmoid(z) for z in affine(gates["i"])]
    g = [math.tanh(z) for z in affine(gates["g"])]
    o = [sigmoid(z) for z in affine(gates["o"])]

    c = [f[k] * c_prev[k] + i_gate[k] * g[k] for k in range(len(c_prev))]
    h = [o[k] * math.tanh(c[k]) for k in range(len(c))]
    return h, c
