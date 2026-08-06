"""
Многослойные сети и прямой проход — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def sigmoid(z):
    """Сигмоида: сжимает любое число в диапазон (0, 1).

    sigmoid(0.0)  ->  0.5
    sigmoid(10.0) ->  0.99995...

    Формула: 1 / (1 + e^(-z)).
    Ловушка: math.exp(-z) при z = -1000 бросает OverflowError. Зажми z в
    [-500, 500] перед вычислением — на этих концах сигмоида уже неотличима
    от 0 и 1, зато exp не переполняется.
    """
    # зажим дешевле, чем разбор случаев, и читается однозначно
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def layer_forward(weights, biases, inputs):
    """Прямой проход одного слоя: список активаций, по одной на нейрон.

    weights — матрица (n_neurons, n_inputs), строка на нейрон.
    biases  — вектор длины n_neurons.

    layer_forward([[1.0, 1.0]], [0.0], [0.0, 0.0])  ->  [0.5]
    layer_forward([[0.0], [0.0]], [0.0, 0.0], [7.0])  ->  [0.5, 0.5]

    Каждый нейрон: z = сумма(w_i * x_i) + b, потом sigmoid(z).
    Порядок индексов не случаен: строк столько, сколько нейронов в этом
    слое, столбцов — сколько нейронов было в предыдущем. Перепутаешь —
    получишь либо ошибку длины, либо, что хуже, тихо неверные числа.
    """
    return [
        sigmoid(sum(w * x for w, x in zip(row, inputs)) + b)
        for row, b in zip(weights, biases)
    ]


def network_forward(layers, inputs):
    """Прямой проход всей сети: выход последнего слоя.

    layers — список слоёв, каждый слой это пара (weights, biases).

    net = [([[1.0, 1.0]], [0.0]), ([[1.0]], [0.0])]
    network_forward(net, [0.0, 0.0])  ->  [0.62245...]

    Выход слоя k становится входом слоя k+1 — и всё. Никакого обучения
    здесь не происходит: прямой проход это чистое вычисление.
    """
    current = inputs
    for weights, biases in layers:
        current = layer_forward(weights, biases, current)
    return current


def predict_binary(output, threshold=0.5):
    """Перевод выхода сигмоиды в класс 0 или 1.

    predict_binary(0.73)  ->  1
    predict_binary(0.12)  ->  0
    predict_binary(0.5)   ->  1

    Ровно на пороге отвечаем 1 — то же соглашение, что у step в уроке 01.
    Порог 0.5 не священен: на несбалансированных данных его двигают.
    """
    return 1 if output >= threshold else 0


def xor_forward(x1, x2):
    """XOR на сети 2-2-1 с выставленными руками весами. Вернуть число.

    xor_forward(0, 0)  ->  ~0.0
    xor_forward(0, 1)  ->  ~1.0
    xor_forward(1, 1)  ->  ~0.0

    Скрытый слой: weights [[20, 20], [-20, -20]], biases [-10, 30].
    Выходной слой: weights [[20, 20]], biases [-30].

    Веса по 20 нужны, чтобы сигмоида работала почти как ступенька: на
    z = 10 она даёт 0.99995. Первый скрытый нейрон — это OR, второй —
    NAND, выходной — AND. Ровно схема из урока 01, но гладкая, поэтому
    у неё есть производная и её можно будет обучить.
    """
    hidden = ([[20.0, 20.0], [-20.0, -20.0]], [-10.0, 30.0])
    output = ([[20.0, 20.0]], [-30.0])
    return network_forward([hidden, output], [x1, x2])[0]


def layer_shapes(layers):
    """Размерности матриц весов слоёв: список пар (нейронов, входов).

    net = [([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [0.0] * 3), ([[1.0, 1.0, 1.0]], [0.0])]
    layer_shapes(net)  ->  [(3, 2), (1, 3)]

    Это первый инструмент отладки в deep learning. Соседние пары обязаны
    стыковаться: число входов слоя k+1 равно числу нейронов слоя k. Если
    в списке рядом стоят (3, 2) и (1, 4) — где-то опечатка в архитектуре.
    """
    return [(len(weights), len(weights[0])) for weights, _ in layers]


def count_parameters(sizes):
    """Сколько обучаемых чисел в сети с такими размерами слоёв.

    count_parameters([2, 3, 1])            ->  13
    count_parameters([784, 256, 128, 10])  ->  235146
    count_parameters([5])                  ->  0

    sizes — список размеров: вход, скрытые, выход. На переходе
    a -> b получается a*b весов и b смещений.

    Второй пример — классическая сеть для MNIST. 235 тысяч параметров
    для распознавания цифр 28x28; у GPT-3 их 175 миллиардов.
    """
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def init_network(sizes, seed=0):
    """Сеть со случайными весами: список слоёв (weights, biases).

    Веса — random.uniform(-1, 1), смещения — нули.

    net = init_network([2, 3, 1], seed=0)
    layer_shapes(net)  ->  [(3, 2), (1, 3)]
    init_network([2, 3, 1], seed=0) == init_network([2, 3, 1], seed=0)  ->  True

    seed обязателен: без него два прогона дают разные сети, и любой
    замер «стало лучше или хуже» превращается в гадание. Заводи
    random.Random(seed) локально, а не дёргай random.seed() — глобальный
    seed портит генератор всем остальным.
    """
    rng = random.Random(seed)
    layers = []
    for n_inputs, n_neurons in zip(sizes, sizes[1:]):
        weights = [[rng.uniform(-1.0, 1.0) for _ in range(n_inputs)] for _ in range(n_neurons)]
        layers.append((weights, [0.0] * n_neurons))
    return layers
