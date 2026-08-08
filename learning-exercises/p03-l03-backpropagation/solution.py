"""
Backpropagation с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import random


def sigmoid(z):
    """Сигмоида: 1 / (1 + e^(-z)), зажатая по z в [-500, 500].

    sigmoid(0.0)  ->  0.5

    Зажим спасает от OverflowError на больших по модулю z.
    Её производная выражается через саму функцию: s'(z) = s(z)*(1 - s(z)).
    Это и есть причина, почему сигмоиду удобно дифференцировать вручную:
    forward уже посчитал всё, что нужно backward.
    """
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def init_params(n_inputs, n_hidden, seed=0):
    """Параметры сети n_inputs -> n_hidden -> 1. Вернуть словарь.

    Ключи ровно такие, дальше на них опирается всё остальное:
      "w1" — матрица (n_hidden, n_inputs)
      "b1" — вектор длины n_hidden
      "w2" — вектор длины n_hidden (выходной нейрон один)
      "b2" — одно число

    p = init_params(2, 4, seed=0)
    len(p["w1"]), len(p["w1"][0]), len(p["w2"])  ->  (4, 2, 4)
    init_params(2, 4, seed=0) == init_params(2, 4, seed=0)  ->  True

    Веса каждого слоя используют свой fan-in:
      W1: random.uniform(-scale1, scale1), scale1 = sqrt(2 / n_inputs)
      w2: random.uniform(-scale2, scale2), scale2 = sqrt(2 / n_hidden)
    Смещения: нули.

    Масштаб не выдуман: он держит z в районе нуля, где у сигмоиды есть
    производная. Возьмёшь веса покрупнее — нейроны сразу упрутся в 0 или 1,
    производная станет нулевой, и обучение не начнётся.
    """
    rng = random.Random(seed)
    scale1 = (2.0 / n_inputs) ** 0.5
    scale2 = (2.0 / n_hidden) ** 0.5
    return {
        "w1": [[rng.uniform(-scale1, scale1) for _ in range(n_inputs)] for _ in range(n_hidden)],
        "b1": [0.0] * n_hidden,
        "w2": [rng.uniform(-scale2, scale2) for _ in range(n_hidden)],
        "b2": 0.0,
    }


def forward(params, x):
    """Прямой проход. Вернуть кэш — словарь с ключами "z1", "a1", "z2", "a2".

    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    forward(p, [0.0])["a1"]  ->  [0.5]
    forward(p, [0.0])["a2"]  ->  0.62245...

    Считается так:
      z1 = W1 * x + b1,   a1 = sigmoid(z1)
      z2 = w2 . a1 + b2,  a2 = sigmoid(z2)

    Кэш возвращается целиком не из щедрости: backward без сохранённых
    промежуточных значений пришлось бы считать forward заново. Это и есть
    размен памяти на скорость, на котором держится весь backprop.
    """
    z1 = [
        sum(w * xi for w, xi in zip(row, x)) + b
        for row, b in zip(params["w1"], params["b1"])
    ]
    a1 = [sigmoid(z) for z in z1]
    z2 = sum(w * a for w, a in zip(params["w2"], a1)) + params["b2"]
    return {"z1": z1, "a1": a1, "z2": z2, "a2": sigmoid(z2)}


def loss_for_params(params, x, target):
    """MSE на одном примере: (a2 - target)^2.

    p = {"w1": [[1.0]], "b1": [0.0], "w2": [0.0], "b2": 0.0}
    loss_for_params(p, [0.0], 0.5)  ->  0.0

    Отдельная функция от параметров нужна не ради красоты: именно её
    численно дифференцирует numeric_gradient, чтобы проверить backward.
    """
    return (forward(params, x)["a2"] - target) ** 2


def backward(params, x, target):
    """Аналитические градиенты dL/dp. Вернуть словарь той же формы, что params.

    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    backward(p, [1.0], 1.0)["b2"]  ->  -0.1426..., то есть число отрицательное:
    чтобы уменьшить loss, b2 надо увеличивать.

    Цепное правило по шагам:
      dL/da2 = 2*(a2 - target)
      dL/dz2 = dL/da2 * a2*(1 - a2)
      dL/dw2[i] = dL/dz2 * a1[i]
      dL/db2 = dL/dz2
      dL/dz1[i] = dL/dz2 * w2[i] * a1[i]*(1 - a1[i])
      dL/dw1[i][j] = dL/dz1[i] * x[j]
      dL/db1[i] = dL/dz1[i]

    Ловушка: w2 нужен ДО обновления. Если сначала подправить w2, а потом
    считать градиенты скрытого слоя, они будут посчитаны по чужим весам —
    ошибка тихая, тесты на численный градиент её и ловят.
    """
    cache = forward(params, x)
    a1, a2 = cache["a1"], cache["a2"]

    d_z2 = 2.0 * (a2 - target) * a2 * (1.0 - a2)
    g_w2 = [d_z2 * a for a in a1]

    g_w1, g_b1 = [], []
    for i, a in enumerate(a1):
        # локальная производная сигмоиды берётся из forward — exp заново не считаем
        d_z1 = d_z2 * params["w2"][i] * a * (1.0 - a)
        g_w1.append([d_z1 * xi for xi in x])
        g_b1.append(d_z1)

    return {"w1": g_w1, "b1": g_b1, "w2": g_w2, "b2": d_z2}


def numeric_gradient(params, x, target, h=1e-5):
    """Численный градиент центральной разностью. Форма как у backward.

    Для каждого параметра p: (L(p + h) - L(p - h)) / (2h).

    p = init_params(2, 3, seed=0)
    numeric_gradient(p, [1.0, 0.0], 1.0)["b2"]  ->  почти то же, что backward(...)["b2"]

    Это gradient checking — единственный честный способ убедиться, что
    аналитический backward написан верно. Медленно (по два прямых прохода
    на каждый параметр), поэтому только для отладки, но один раз прогнать
    обязательно.

    Ловушка: параметр надо вернуть на место ТОЧНО. Не `p += h`, потом
    `p -= h` — сохрани старое значение и запиши обратно, иначе накопится
    дрейф и следующие производные поедут.
    """
    grads = {"w1": [], "b1": [], "w2": [], "b2": 0.0}

    for row in params["w1"]:
        row_grads = []
        for j, old in enumerate(row):
            row[j] = old + h
            up = loss_for_params(params, x, target)
            row[j] = old - h
            down = loss_for_params(params, x, target)
            row[j] = old  # ровно исходное значение, а не old + h - h
            row_grads.append((up - down) / (2.0 * h))
        grads["w1"].append(row_grads)

    for key in ("b1", "w2"):
        vec = params[key]
        for i, old in enumerate(vec):
            vec[i] = old + h
            up = loss_for_params(params, x, target)
            vec[i] = old - h
            down = loss_for_params(params, x, target)
            vec[i] = old
            grads[key].append((up - down) / (2.0 * h))

    old = params["b2"]
    params["b2"] = old + h
    up = loss_for_params(params, x, target)
    params["b2"] = old - h
    down = loss_for_params(params, x, target)
    params["b2"] = old
    grads["b2"] = (up - down) / (2.0 * h)

    return grads


def sgd_step(params, grads, lr):
    """Шаг градиентного спуска. Вернуть НОВЫЙ словарь параметров.

    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    g = {"w1": [[2.0]], "b1": [1.0], "w2": [0.0], "b2": -4.0}
    sgd_step(p, g, 0.5)  ->  {"w1": [[0.0]], "b1": [-0.5], "w2": [1.0], "b2": 2.0}

    Правило одно на все параметры: p -= lr * dL/dp.
    Исходный словарь трогать нельзя — тест это проверяет. Чужие
    параметры, изменённые «по дороге», это классический источник багов,
    когда одну и ту же сеть считают в двух местах.
    """
    return {
        "w1": [
            [w - lr * g for w, g in zip(row, g_row)]
            for row, g_row in zip(params["w1"], grads["w1"])
        ],
        "b1": [b - lr * g for b, g in zip(params["b1"], grads["b1"])],
        "w2": [w - lr * g for w, g in zip(params["w2"], grads["w2"])],
        "b2": params["b2"] - lr * grads["b2"],
    }


def train_xor(seed=0, n_hidden=4, lr=1.0, epochs=4000):
    """Обучить сеть 2-n_hidden-1 на XOR. Вернуть (params, loss).

    loss — сумма MSE по всем четырём примерам после обучения.

    params, loss = train_xor()
    loss < 0.05  ->  True
    forward(params, [0.0, 1.0])["a2"] > 0.5  ->  True

    Порядок внутри цикла: init один раз, дальше на каждый пример
    forward -> backward -> sgd_step. Обновление после каждого примера
    (online SGD), а не раз в эпоху — так сеть быстрее ломает симметрию
    и не застревает в точке, где все четыре ответа равны 0.5.
    """
    xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
    params = init_params(2, n_hidden, seed=seed)
    for _ in range(epochs):
        for x, target in xor_data:
            params = sgd_step(params, backward(params, x, target), lr)
    loss = sum(loss_for_params(params, x, t) for x, t in xor_data)
    return params, loss
