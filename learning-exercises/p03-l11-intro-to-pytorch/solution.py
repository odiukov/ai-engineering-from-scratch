"""
Знакомство с PyTorch: собираем autograd руками — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


class Tensor:
    """Двумерный тензор с лентой автодифференцирования — наш torch.Tensor.

    data — список строк одинаковой длины, grad — такой же формы, забитый
    нулями. requires_grad включает накопление градиента, как одноимённый
    флаг в PyTorch.

    t = Tensor([[1.0, 2.0]], requires_grad=True)
    t.shape                ->  (1, 2)
    t.add(t).data          ->  [[2.0, 4.0]]
    t.mul(3.0).data        ->  [[3.0, 6.0]]

    Что делает настоящий PyTorch и чего не делает эта версия: тут только
    2D, только float, нет device, нет dtype и нет ускорения. Зато лента
    ровно та же — каждая операция запоминает, как протолкнуть градиент
    назад, а backward обходит ленту в обратном топологическом порядке.
    """

    def __init__(self, data, requires_grad=False):
        """Создать тензор из списка строк. Градиент сразу нулевой.

        Tensor([[1.0, 2.0], [3.0, 4.0]]).shape  ->  (2, 2)

        Данные копируются построчно: иначе два тензора будут делить один
        список и правка одного молча испортит другой.
        """
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.data else 0
        self.shape = (self.rows, self.cols)
        self.requires_grad = requires_grad
        self.grad = [[0.0] * self.cols for _ in range(self.rows)]
        # родители по ленте и функция, разливающая градиент по ним
        self.parents = []
        self.backward_fn = None

    def zero_grad(self):
        """Обнулить накопленный градиент. Аналог optimizer.zero_grad()."""
        self.grad = [[0.0] * self.cols for _ in range(self.rows)]

    def add(self, other):
        """Поэлементное сложение. Аналог torch.add / оператора +.

        Tensor([[1.0, 2.0]]).add(Tensor([[10.0, 20.0]])).data  ->  [[11.0, 22.0]]

        Форма other либо совпадает, либо это одна строка — тогда она
        транслируется на все строки, как bias в nn.Linear. Это единственный
        broadcasting, который мы поддерживаем.
        """
        broadcast = other.rows == 1 and self.rows != 1
        out = Tensor(
            [
                [a + other.data[0 if broadcast else i][j] for j, a in enumerate(row)]
                for i, row in enumerate(self.data)
            ]
        )
        out.parents = [self, other]

        def backward_fn():
            # сложение пропускает градиент в оба слагаемых без изменений,
            # но у транслированной строки вклады со всех строк складываются
            for i in range(self.rows):
                for j in range(self.cols):
                    g = out.grad[i][j]
                    self.grad[i][j] += g
                    other.grad[0 if broadcast else i][j] += g

        out.backward_fn = backward_fn
        return out

    def mul(self, other):
        """Умножение: поэлементно на тензор той же формы или на число.

        Tensor([[1.0, 2.0]]).mul(3.0).data                    ->  [[3.0, 6.0]]
        Tensor([[1.0, 2.0]]).mul(Tensor([[4.0, 5.0]])).data   ->  [[4.0, 10.0]]

        Число — не тензор, у него нет градиента; заворачивать его в Tensor
        не нужно, но и забывать про него в backward нельзя.
        """
        if isinstance(other, Tensor):
            out = Tensor(
                [
                    [a * other.data[i][j] for j, a in enumerate(row)]
                    for i, row in enumerate(self.data)
                ]
            )
            out.parents = [self, other]

            def backward_fn():
                # производная произведения: каждому множителю достаётся другой
                for i in range(self.rows):
                    for j in range(self.cols):
                        g = out.grad[i][j]
                        self.grad[i][j] += g * other.data[i][j]
                        other.grad[i][j] += g * self.data[i][j]
        else:
            scalar = float(other)
            out = Tensor([[a * scalar for a in row] for row in self.data])
            out.parents = [self]

            def backward_fn():
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += out.grad[i][j] * scalar

        out.backward_fn = backward_fn
        return out

    def matmul(self, other):
        """Матричное умножение (m, k) @ (k, n) -> (m, n). Аналог torch.matmul.

        Tensor([[1.0, 2.0]]).matmul(Tensor([[1.0], [1.0]])).data  ->  [[3.0]]

        Порядок сомножителей важен: в forward пишут x @ W, где x это
        (batch, in_features), а W это (in_features, out_features).
        Перепутаешь — получишь либо исключение по форме, либо, что хуже,
        случайно подходящие размеры и молча неверный результат.
        """
        n = other.cols
        out = Tensor(
            [
                [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(n)]
                for i in range(self.rows)
            ]
        )
        out.parents = [self, other]

        def backward_fn():
            # dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
            for i in range(self.rows):
                for j in range(n):
                    g = out.grad[i][j]
                    if g == 0.0:
                        continue
                    for k in range(self.cols):
                        self.grad[i][k] += g * other.data[k][j]
                        other.grad[k][j] += g * self.data[i][k]

        out.backward_fn = backward_fn
        return out

    def backward(self):
        """Прогнать градиент назад по ленте. Возвращает None.

        Стартовый градиент — единицы по всей форме тензора, то есть это
        ровно t.sum().backward() в PyTorch. Настоящий torch требует, чтобы
        корень был скаляром, либо явного аргумента gradient=.

        t = Tensor([[3.0]], requires_grad=True)
        y = t.mul(t)
        y.backward()
        t.grad  ->  [[6.0]]   (производная x^2 это 2x)

        Обход обязан идти в обратном топологическом порядке: узел отдаёт
        градиент родителям только после того, как собрал его от всех своих
        потребителей. Наивный обход в глубину даст неверный результат на
        любом ромбе вроде y = x * x.
        """
        order = []
        seen = set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        self.grad = [[1.0] * self.cols for _ in range(self.rows)]
        for node in reversed(order):
            if node.backward_fn is not None:
                node.backward_fn()


class Module:
    """База для слоёв: сама находит свои параметры. Аналог torch.nn.Module.

    В PyTorch присвоение nn.Parameter или подмодуля в __init__ регистрирует
    его автоматически, и model.parameters() собирает всё рекурсивно.
    Повторяем: обходим __dict__, берём Tensor с requires_grad и
    рекурсивно спускаемся во вложенные Module.

    Именно поэтому в PyTorch не нужно вручную перечислять веса — а в
    мини-фреймворке из урока 10 приходилось.
    """

    def forward(self, x):
        """Прямой проход. База не умеет — переопредели в наследнике."""
        raise NotImplementedError

    def parameters(self):
        """Все обучаемые тензоры модуля и вложенных модулей.

        len(Linear(2, 3).parameters())  ->  2   (веса и смещения)

        Порядок — порядок присвоения атрибутов в __init__ (в Python
        словарь его сохраняет), чтобы список был воспроизводим.
        """
        found = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                found.append(value)
            elif isinstance(value, Module):
                found.extend(value.parameters())
        return found

    def zero_grad(self):
        """Обнулить градиенты всех параметров. Аналог optimizer.zero_grad()."""
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Полносвязный слой x @ W + b. Аналог torch.nn.Linear.

    W имеет форму (in_features, out_features), b — (1, out_features) и
    транслируется на все строки батча.

    layer = Linear(2, 3, seed=0)
    layer.weight.shape                    ->  (2, 3)
    layer.forward(Tensor([[1.0, 2.0]])).shape  ->  (1, 3)

    Инициализация — Kaiming, N(0, sqrt(2 / in_features)); PyTorch по
    умолчанию берёт похожую Kaiming uniform. Смещения — нули.
    """

    def __init__(self, in_features, out_features, seed=0):
        """Создать слой с воспроизводимыми весами: свой random.Random(seed)."""
        rng = random.Random(seed)
        std = math.sqrt(2.0 / in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            [[rng.gauss(0.0, std) for _ in range(out_features)] for _ in range(in_features)],
            requires_grad=True,
        )
        self.bias = Tensor([[0.0] * out_features], requires_grad=True)

    def forward(self, x):
        """x @ W + b. Смещение транслируется на все строки батча."""
        return x.matmul(self.weight).add(self.bias)


def randn(rows, cols, seed=0, scale=1.0):
    """Матрица rows x cols из N(0, scale) как обычный список списков.

    len(randn(2, 3))     ->  2
    len(randn(2, 3)[0])  ->  3

    Аналог torch.randn, только без тензора: результат годится и как
    данные, и как аргумент Tensor(...). Свой random.Random(seed) —
    чтобы два вызова с одним seed совпали.
    """
    rng = random.Random(seed)
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def mse_loss(predicted, target):
    """Квадраты ошибок поэлементно, как Tensor. Аналог nn.MSELoss.

    mse_loss(Tensor([[3.0]]), Tensor([[1.0]])).data  ->  [[4.0]]

    Возвращается ТЕНЗОР той же формы, а не число: усреднения тут нет,
    его берёт на себя backward, который стартует с единиц и тем самым
    суммирует. То есть это nn.MSELoss(reduction='sum') с точностью до
    отсутствующего деления на размер батча.

    Строится через add и mul, поэтому лента градиента не рвётся.
    """
    diff = predicted.add(target.mul(-1.0))
    return diff.mul(diff)


def sgd_step(params, lr):
    """Шаг SGD прямо по data тензоров. Возвращает None. Аналог optim.SGD.

    Обновление идёт в обход ленты: меняем data, а не строим новый тензор.
    В PyTorch ровно поэтому шаг оптимизатора завёрнут в torch.no_grad() —
    обновление весов не должно попадать в граф.
    """
    for p in params:
        for i in range(p.rows):
            for j in range(p.cols):
                p.data[i][j] -= lr * p.grad[i][j]


def fit_line(seed=0, steps=200, lr=0.05, slope=3.0, intercept=-1.0):
    """Обучить Linear(1, 1) на y = slope * x + intercept. Вернуть (w, b).

    w, b = fit_line()
    abs(w - 3.0) < 0.05        ->  True
    abs(b - (-1.0)) < 0.05     ->  True

    Данные — 16 точек x из N(0, 1), цель считается по формуле точно, без
    шума, поэтому идеальная модель достижима.

    Порядок в цикле: zero_grad -> forward -> loss -> backward -> sgd_step.
    Убери zero_grad — градиенты сложатся с прошлого шага, эффективный lr
    поедет вверх, и обучение развалится без единого исключения. Это ошибка
    номер один в PyTorch-коде.
    """
    layer = Linear(1, 1, seed=seed)
    xs = randn(16, 1, seed=seed + 100)
    x = Tensor(xs)
    target = Tensor([[slope * row[0] + intercept] for row in xs])
    params = layer.parameters()

    for _ in range(steps):
        layer.zero_grad()
        predicted = layer.forward(x)
        mse_loss(predicted, target).backward()
        sgd_step(params, lr / len(xs))

    return layer.weight.data[0][0], layer.bias.data[0][0]
