"""
Собственный мини-фреймворк

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p03-l10-mini-framework
Разбор:  /check-code p03-l10-mini-framework
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
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x):
        """y[i] = sum_j W[i][j] * x[j] + b[i]. Вход запоминается для backward."""
        raise NotImplementedError

    def backward(self, grad):
        """dL/dW += grad * x, dL/db += grad, вернуть dL/dx = W^T @ grad."""
        raise NotImplementedError

    def parameters(self):
        """Все веса и смещения как тройки (values, index, grads)."""
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x):
        """Отрицательное обнуляется, положительное проходит как есть."""
        raise NotImplementedError

    def backward(self, grad):
        """Градиент проходит только сквозь открытые нейроны."""
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x):
        """Сжимает любое число в (0, 1)."""
        raise NotImplementedError

    def backward(self, grad):
        """s' = s * (1 - s), считаем через сохранённый выход."""
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x):
        """Прогнать вход через все слои слева направо."""
        raise NotImplementedError

    def backward(self, grad):
        """Прогнать градиент через все слои справа налево."""
        raise NotImplementedError

    def parameters(self):
        """Параметры всех вложенных слоёв одним плоским списком."""
        raise NotImplementedError


def mse_loss(predicted, target):
    """Среднеквадратичная ошибка между двумя списками одной длины.

    mse_loss([1.0], [1.0])        ->  0.0
    mse_loss([2.0, 0.0], [0.0, 0.0])  ->  2.0

    Делим на длину, а не на 2: тогда градиент равен 2*(p - t)/n, и это
    ровно то, что возвращает mse_grad.
    """
    raise NotImplementedError


def mse_grad(predicted, target):
    """Градиент mse_loss по предсказанию: 2 * (p - t) / n.

    mse_grad([2.0, 0.0], [0.0, 0.0])  ->  [2.0, 0.0]

    Это тот самый вектор, который скармливается model.backward(). В
    PyTorch его считает loss.backward(), здесь — ты руками.
    """
    raise NotImplementedError


def zero_grads(params):
    """Обнулить накопленные градиенты всех параметров. Ничего не возвращает.

    Аналог optimizer.zero_grad(). Забыть его — классическая ошибка:
    градиенты от прошлых батчей продолжают складываться, шаг получается
    в разы больше задуманного, и лосс начинает болтаться.
    """
    raise NotImplementedError


def sgd_step(params, lr):
    """Шаг SGD: values[index] -= lr * grads[index]. Ничего не возвращает.

    Оптимизатор не знает ни про слои, ни про архитектуру — он видит
    плоский список троек. Именно поэтому один и тот же SGD работает и
    с двухслойной сетью, и с трансформером.
    """
    raise NotImplementedError


def xor_dataset():
    """Четыре образца XOR: вход из двух чисел, цель из одного.

    xor_dataset()[0]  ->  ([0.0, 0.0], [0.0])
    xor_dataset()[1]  ->  ([0.0, 1.0], [1.0])

    XOR не решается одним линейным слоем — это и есть проверка, что
    скрытый слой с нелинейностью реально работает.
    """
    raise NotImplementedError


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
    raise NotImplementedError
