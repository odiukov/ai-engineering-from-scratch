"""
Генерация 3D: гауссовы сплаты и подгонка градиентом — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def gaussian_value(x, y, gauss):
    """Вклад одного сплата в пиксель (x, y).

    Сплат — словарь {"pos": [px, py], "sigma": s, "color": c}.
    Значение: c * exp(-((x-px)^2 + (y-py)^2) / (2 * s^2)).

    g = {"pos": [1.0, 1.0], "sigma": 1.0, "color": 0.8}
    gaussian_value(1, 1, g)  ->  0.8    в центре ослабления нет
    gaussian_value(2, 1, g)  ->  0.485  = 0.8 * exp(-0.5)

    Ловушка порядка: pos — это [x, y], то есть [столбец, строка]. Перепутать
    с [row, col] — самая частая ошибка при переносе на изображение.
    """
    px, py = gauss["pos"]
    dx, dy = x - px, y - py
    sigma = gauss["sigma"]
    # d2 считаем без math.hypot и без возведения в 0.5: корень тут же
    # возвелся бы обратно в квадрат
    d2 = dx * dx + dy * dy
    return gauss["color"] * math.exp(-d2 / (2 * sigma * sigma))


def render(size, gaussians):
    """Отрисовать квадратное изображение size x size как сумму сплатов.

    render(2, [])  ->  [[0.0, 0.0], [0.0, 0.0]]
    render(2, [{"pos": [0.0, 0.0], "sigma": 1.0, "color": 1.0}])
        ->  [[1.0, 0.606...], [0.606..., 0.368...]]

    Индексация img[y][x]: сначала строка, потом столбец.

    Настоящий 3D-GS сортирует гауссианы по глубине и альфа-композитит их по
    порядку; наша 2D-игрушка просто складывает — этого хватает, чтобы
    увидеть дифференцируемый рендер.
    """
    img = [[0.0] * size for _ in range(size)]
    # внешний цикл по сплатам, а не по пикселям: словарь распаковывается
    # один раз на сплат, а не size*size раз
    for g in gaussians:
        for y in range(size):
            row = img[y]
            for x in range(size):
                row[x] += gaussian_value(x, y, g)
    return img


def image_mse(a, b):
    """Средний квадрат разности двух изображений одинакового размера.

    image_mse([[1.0]], [[1.0]])  ->  0.0
    image_mse([[0.0, 0.0]], [[1.0, 3.0]])  ->  5.0   (1 + 9) / 2

    Это и есть функция потерь подгонки сплатов: MSE между рендером и целью.
    """
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            d = va - vb
            total += d * d
            count += 1
    return total / count


def color_gradients(size, gaussians, target):
    """Аналитические производные MSE по яркости каждого сплата.

    dL/dc_i = (2 / N) * sum по пикселям (pred - target) * K_i(x, y),
    где K_i — гауссово ядро БЕЗ множителя цвета, N = size * size.

    Вернуть список чисел, по одному на сплат, в том же порядке.

    Проверять такое надо всегда одним способом: центральной разностью по
    той же яркости. Если аналитика и численная производная разошлись —
    ошибка в аналитике, а не в разности.
    """
    pred = render(size, gaussians)
    n = size * size
    grads = []
    for g in gaussians:
        # тот же сплат с единичной яркостью и есть ядро K_i
        kernel = dict(g)
        kernel["color"] = 1.0
        acc = 0.0
        for y in range(size):
            for x in range(size):
                acc += (pred[y][x] - target[y][x]) * gaussian_value(x, y, kernel)
        grads.append(2.0 * acc / n)
    return grads


def fit_colors(size, gaussians, target, lr=2.0, steps=300):
    """Градиентный спуск по яркостям сплатов. Позиции и sigma не трогаем.

    Вернуть НОВЫЙ список сплатов — входной изменять нельзя, иначе повторный
    запуск с теми же аргументами даст другой ответ.

    Задача линейная по цвету, поэтому спуск сходится к точному ответу: если
    target отрисован из тех же позиций, восстановятся исходные яркости.

    Ловушка знака: шаг идёт ПРОТИВ градиента, c -= lr * grad.
    """
    # копии словарей: pos остаётся общим списком, но мы его и не меняем
    current = [dict(g) for g in gaussians]
    for _ in range(steps):
        grads = color_gradients(size, current, target)
        for g, grad in zip(current, grads):
            g["color"] -= lr * grad
    return current


def alpha_composite(layers):
    """Альфа-композиция списка слоёв спереди назад.

    layers — список пар (color, alpha), первый элемент ближе всех к камере.
    Итог: sum по i от color_i * alpha_i * произведения (1 - alpha_j) по j < i.

    alpha_composite([(1.0, 1.0), (0.0, 1.0)])  ->  1.0   передний непрозрачный
    alpha_composite([(0.0, 1.0), (1.0, 1.0)])  ->  0.0   порядок решает всё
    alpha_composite([(1.0, 0.5), (1.0, 1.0)])  ->  1.0

    Это ровно то, что делает настоящий 3D-GS после сортировки гауссиан по
    глубине. Порядок принципиален: композиция не коммутативна.
    """
    out = 0.0
    transmittance = 1.0
    for color, alpha in layers:
        out += transmittance * alpha * color
        # прозрачность копится произведением, а не суммой
        transmittance *= 1.0 - alpha
    return out


def prune_gaussians(gaussians, min_color=0.01):
    """Убрать сплаты, чья яркость по модулю меньше min_color.

    prune_gaussians([{"pos": [0, 0], "sigma": 1, "color": 0.001}])  ->  []

    Порядок оставшихся сохраняется. Сравнивать надо модуль: сплат с цветом
    -0.5 вносит вклад не меньше, чем +0.5.

    Без прунинга обучение 3D-GS разрастается до 10 миллионов сплатов и
    начинает переобучаться — прунинг и densification в паре держат размер.
    """
    return [g for g in gaussians if abs(g["color"]) >= min_color]


def split_gaussian(gauss, offset, shrink=1.6):
    """Densification: заменить один сплат двумя поменьше, разведёнными вбок.

    offset — сдвиг [dx, dy]; дети встают в pos + offset и pos - offset,
    sigma делится на shrink, цвет наследуется без изменений.

    g = {"pos": [4.0, 4.0], "sigma": 2.0, "color": 0.6}
    split_gaussian(g, [1.0, 0.0])
        ->  два сплата в [5.0, 4.0] и [3.0, 4.0], sigma 1.25, color 0.6

    Родителя мутировать нельзя — он ещё нужен для сравнения до/после.
    Середина между детьми обязана совпасть с позицией родителя, иначе
    densification начнёт уводить сцену.
    """
    px, py = gauss["pos"]
    dx, dy = offset
    sigma = gauss["sigma"] / shrink
    return [
        # новые списки pos, а не срезы родительского: иначе дети поедут
        # вместе с родителем при любой правке
        {"pos": [px + dx, py + dy], "sigma": sigma, "color": gauss["color"]},
        {"pos": [px - dx, py - dy], "sigma": sigma, "color": gauss["color"]},
    ]
