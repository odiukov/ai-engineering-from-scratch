"""
Знакомство с PyTorch: собираем autograd руками

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p03-l11-intro-to-pytorch
Разбор:  /check-code p03-l11-intro-to-pytorch
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
        raise NotImplementedError

    def zero_grad(self):
        """Обнулить накопленный градиент. Аналог optimizer.zero_grad()."""
        raise NotImplementedError

    def add(self, other):
        """Поэлементное сложение. Аналог torch.add / оператора +.

        Tensor([[1.0, 2.0]]).add(Tensor([[10.0, 20.0]])).data  ->  [[11.0, 22.0]]

        Форма other либо совпадает, либо это одна строка — тогда она
        транслируется на все строки, как bias в nn.Linear. Это единственный
        broadcasting, который мы поддерживаем.
        """
        raise NotImplementedError

    def mul(self, other):
        """Умножение: поэлементно на тензор той же формы или на число.

        Tensor([[1.0, 2.0]]).mul(3.0).data                    ->  [[3.0, 6.0]]
        Tensor([[1.0, 2.0]]).mul(Tensor([[4.0, 5.0]])).data   ->  [[4.0, 10.0]]

        Число — не тензор, у него нет градиента; заворачивать его в Tensor
        не нужно, но и забывать про него в backward нельзя.
        """
        raise NotImplementedError

    def matmul(self, other):
        """Матричное умножение (m, k) @ (k, n) -> (m, n). Аналог torch.matmul.

        Tensor([[1.0, 2.0]]).matmul(Tensor([[1.0], [1.0]])).data  ->  [[3.0]]

        Порядок сомножителей важен: в forward пишут x @ W, где x это
        (batch, in_features), а W это (in_features, out_features).
        Перепутаешь — получишь либо исключение по форме, либо, что хуже,
        случайно подходящие размеры и молча неверный результат.
        """
        raise NotImplementedError

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
        raise NotImplementedError


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
        raise NotImplementedError

    def zero_grad(self):
        """Обнулить градиенты всех параметров. Аналог optimizer.zero_grad()."""
        raise NotImplementedError


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
        raise NotImplementedError

    def forward(self, x):
        """x @ W + b. Смещение транслируется на все строки батча."""
        raise NotImplementedError


def randn(rows, cols, seed=0, scale=1.0):
    """Матрица rows x cols из N(0, scale) как обычный список списков.

    len(randn(2, 3))     ->  2
    len(randn(2, 3)[0])  ->  3

    Аналог torch.randn, только без тензора: результат годится и как
    данные, и как аргумент Tensor(...). Свой random.Random(seed) —
    чтобы два вызова с одним seed совпали.
    """
    raise NotImplementedError


def mse_loss(predicted, target):
    """Квадраты ошибок поэлементно, как Tensor. Аналог nn.MSELoss.

    mse_loss(Tensor([[3.0]]), Tensor([[1.0]])).data  ->  [[4.0]]

    Возвращается ТЕНЗОР той же формы, а не число: усреднения тут нет,
    его берёт на себя backward, который стартует с единиц и тем самым
    суммирует. То есть это nn.MSELoss(reduction='sum') с точностью до
    отсутствующего деления на размер батча.

    Строится через add и mul, поэтому лента градиента не рвётся.
    """
    raise NotImplementedError


def sgd_step(params, lr):
    """Шаг SGD прямо по data тензоров. Возвращает None. Аналог optim.SGD.

    Обновление идёт в обход ленты: меняем data, а не строим новый тензор.
    В PyTorch ровно поэтому шаг оптимизатора завёрнут в torch.no_grad() —
    обновление весов не должно попадать в граф.
    """
    raise NotImplementedError


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
    raise NotImplementedError
