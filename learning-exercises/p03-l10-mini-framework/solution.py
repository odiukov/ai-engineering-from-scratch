"""
Собственный мини-фреймворк — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


class Layer:
    """Базовый интерфейс слоя: forward, backward, parameters.

    Сам по себе ничего не считает — это контракт, который обязаны
    выполнять Linear, ReLU, Sigmoid и Sequential. Ровно та же тройка
    методов, что у torch.nn.Module.

    forward(x)      принимает список чисел, возвращает список чисел
    backward(grad)  принимает dL/d(выход), возвращает dL/d(вход)
    parameters()    возвращает список троек (values, index, grads)

    Тройка (values, index, grads) — это ссылка на один скалярный параметр:
    values[index] — само число, grads[index] — накопленный по нему
    градиент. Оптимизатору больше ничего знать не нужно, поэтому он
    работает с любой архитектурой.
    """

    def forward(self, x):
        """Прямой проход. Базовый класс не умеет — переопредели в наследнике."""
        raise NotImplementedError

    def backward(self, grad):
        """Обратный проход. Базовый класс не умеет — переопредели в наследнике."""
        raise NotImplementedError

    def parameters(self):
        """Обучаемые параметры слоя. У слоя без весов их нет — пустой список.

        Layer().parameters()  ->  []

        ReLU и Sigmoid этот метод не переопределяют именно поэтому.
        """
        return []


class Linear(Layer):
    """Полносвязный слой: y = W @ x + b.

    Веса кладём построчно: W[i][j] — вес от входа j к выходу i, то есть
    W имеет out_features строк по in_features чисел. Инициализация —
    Kaiming, N(0, sqrt(2 / in_features)), потому что дальше почти всегда
    стоит ReLU. Смещения — нули.

    lin = Linear(2, 3, seed=0)
    len(lin.forward([1.0, 2.0]))   ->  3
    len(lin.parameters())          ->  9    (6 весов + 3 смещения)

    Градиенты НАКАПЛИВАЮТСЯ (+=), а не перезаписываются. Это не описка:
    так один и тот же слой можно прогнать по нескольким образцам и
    получить сумму градиентов. Обнуляет их отдельный вызов zero_grads —
    ровно поэтому в PyTorch optimizer.zero_grad() отдельная строка.
    """

    def __init__(self, in_features, out_features, seed=0):
        """Создать слой in_features -> out_features с Kaiming-инициализацией."""
        rng = random.Random(seed)
        std = math.sqrt(2.0 / in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weights = [
            [rng.gauss(0.0, std) for _ in range(in_features)] for _ in range(out_features)
        ]
        self.biases = [0.0] * out_features
        self.weight_grads = [[0.0] * in_features for _ in range(out_features)]
        self.bias_grads = [0.0] * out_features
        self.input = None

    def forward(self, x):
        """y[i] = sum_j W[i][j] * x[j] + b[i]. Вход запоминается для backward."""
        # без входа обратный проход невозможен: dL/dW[i][j] = grad[i] * x[j]
        self.input = list(x)
        return [
            sum(w * v for w, v in zip(row, x)) + b
            for row, b in zip(self.weights, self.biases)
        ]

    def backward(self, grad):
        """dL/dW += grad * x, dL/db += grad, вернуть dL/dx = W^T @ grad."""
        input_grad = [0.0] * self.in_features
        for i in range(self.out_features):
            g = grad[i]
            self.bias_grads[i] += g
            row, grad_row = self.weights[i], self.weight_grads[i]
            for j in range(self.in_features):
                grad_row[j] += g * self.input[j]
                input_grad[j] += g * row[j]
        return input_grad

    def parameters(self):
        """Все веса и смещения как тройки (values, index, grads)."""
        params = []
        for i in range(self.out_features):
            row, grad_row = self.weights[i], self.weight_grads[i]
            params.extend((row, j, grad_row) for j in range(self.in_features))
            params.append((self.biases, i, self.bias_grads))
        return params


class ReLU(Layer):
    """Активация ReLU: max(0, x) поэлементно.

    r = ReLU()
    r.forward([-1.0, 2.0])   ->  [0.0, 2.0]
    r.backward([5.0, 5.0])   ->  [0.0, 5.0]

    Обратный проход пропускает градиент только там, где вход был
    положительным. Маску надо запомнить в forward — по одному лишь
    градиенту восстановить её нельзя.
    """

    def __init__(self):
        """Маска положительных входов, заполняется в forward."""
        self.mask = None

    def forward(self, x):
        """Отрицательное обнуляется, положительное проходит как есть."""
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [v if v > 0 else 0.0 for v in x]

    def backward(self, grad):
        """Градиент проходит только сквозь открытые нейроны."""
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Layer):
    """Активация Sigmoid: 1 / (1 + e^-x) поэлементно.

    s = Sigmoid()
    s.forward([0.0])       ->  [0.5]
    s.backward([1.0])      ->  [0.25]   (максимум производной)

    Производная выражается через сам выход: s' = s * (1 - s). Поэтому в
    forward запоминаем выход, а не вход — второй раз exp считать незачем.

    Большие по модулю входы роняют math.exp в OverflowError. Зажми
    аргумент в разумные пределы прежде, чем считать экспоненту.
    """

    def __init__(self):
        """Выход последнего forward, нужен для производной."""
        self.output = None

    def forward(self, x):
        """Сжимает любое число в (0, 1)."""
        # зажим спасает от OverflowError на входах вроде -1000
        self.output = [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, v)))) for v in x]
        return self.output

    def backward(self, grad):
        """s' = s * (1 - s), считаем через сохранённый выход."""
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Sequential(Layer):
    """Контейнер: цепочка слоёв, которая сама является слоем.

    model = Sequential(Linear(2, 4, seed=0), ReLU(), Linear(4, 1, seed=1))
    len(model.forward([1.0, 1.0]))  ->  1

    Прямой проход идёт слева направо, обратный — справа налево. Это
    паттерн «композит»: снаружи Sequential неотличим от одного слоя,
    поэтому вложить Sequential в Sequential можно без единой правки.

    parameters() просто склеивает списки вложенных слоёв — порядок
    сохраняется, что позволяет оптимизатору быть тупым и универсальным.
    """

    def __init__(self, *layers):
        """Слои перечисляются позиционно, как в torch.nn.Sequential."""
        self.layers = list(layers)

    def forward(self, x):
        """Прогнать вход через все слои слева направо."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """Прогнать градиент через все слои справа налево."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """Параметры всех вложенных слоёв одним плоским списком."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def mse_loss(predicted, target):
    """Среднеквадратичная ошибка между двумя списками одной длины.

    mse_loss([1.0], [1.0])        ->  0.0
    mse_loss([2.0, 0.0], [0.0, 0.0])  ->  2.0

    Делим на длину, а не на 2: тогда градиент равен 2*(p - t)/n, и это
    ровно то, что возвращает mse_grad.
    """
    n = len(predicted)
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / n


def mse_grad(predicted, target):
    """Градиент mse_loss по предсказанию: 2 * (p - t) / n.

    mse_grad([2.0, 0.0], [0.0, 0.0])  ->  [2.0, 0.0]

    Это тот самый вектор, который скармливается model.backward(). В
    PyTorch его считает loss.backward(), здесь — ты руками.
    """
    n = len(predicted)
    return [2.0 * (p - t) / n for p, t in zip(predicted, target)]


def zero_grads(params):
    """Обнулить накопленные градиенты всех параметров. Ничего не возвращает.

    Аналог optimizer.zero_grad(). Забыть его — классическая ошибка:
    градиенты от прошлых батчей продолжают складываться, шаг получается
    в разы больше задуманного, и лосс начинает болтаться.
    """
    for _values, index, grads in params:
        grads[index] = 0.0


def sgd_step(params, lr):
    """Шаг SGD: values[index] -= lr * grads[index]. Ничего не возвращает.

    Оптимизатор не знает ни про слои, ни про архитектуру — он видит
    плоский список троек. Именно поэтому один и тот же SGD работает и
    с двухслойной сетью, и с трансформером.
    """
    for values, index, grads in params:
        values[index] -= lr * grads[index]


def xor_dataset():
    """Четыре образца XOR: вход из двух чисел, цель из одного.

    xor_dataset()[0]  ->  ([0.0, 0.0], [0.0])
    xor_dataset()[1]  ->  ([0.0, 1.0], [1.0])

    XOR не решается одним линейным слоем — это и есть проверка, что
    скрытый слой с нелинейностью реально работает.
    """
    return [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]


def train_xor(seed=0, hidden=16, epochs=4000, lr=0.3):
    """Обучить сеть 2 -> hidden -> 1 на XOR. Вернуть (model, final_loss).

    Архитектура: Linear(2, hidden) -> ReLU -> Linear(hidden, 1) -> Sigmoid.
    Слоям даются разные seed (seed и seed + 1), иначе при hidden == 2
    оба слоя получат одинаковые веса.

    model, loss = train_xor()
    loss < 0.01                       ->  True
    model.forward([1.0, 0.0])[0] > 0.5  ->  True

    Порядок внутри эпохи: zero_grads -> forward -> mse_grad -> backward
    -> sgd_step, по одному образцу за раз. Переставь zero_grads после
    backward — и обучение сломается молча, без единого исключения.

    final_loss — средний mse_loss по всем четырём образцам в последней эпохе.
    """
    model = Sequential(
        Linear(2, hidden, seed=seed),
        ReLU(),
        Linear(hidden, 1, seed=seed + 1),
        Sigmoid(),
    )
    params = model.parameters()
    data = xor_dataset()

    total = 0.0
    for _ in range(epochs):
        total = 0.0
        for x, target in data:
            zero_grads(params)
            predicted = model.forward(x)
            total += mse_loss(predicted, target)
            model.backward(mse_grad(predicted, target))
            sgd_step(params, lr)
    return model, total / len(data)
