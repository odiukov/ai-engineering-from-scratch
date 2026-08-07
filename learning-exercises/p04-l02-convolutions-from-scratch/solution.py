"""
Свёртки с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def conv_output_size(size, kernel, padding=0, stride=1):
    """Пространственный размер выхода свёртки: floor((size - kernel + 2p)/s) + 1.

    conv_output_size(32, 3)                      ->  30   (valid, без padding)
    conv_output_size(32, 3, padding=1)           ->  32   (same padding)
    conv_output_size(32, 3, padding=1, stride=2) ->  16   (downsample вдвое)
    conv_output_size(32, 2, stride=2)            ->  16   (pool 2x2)

    Формулу придётся считать десятки раз на каждую архитектуру, поэтому она
    заслуживает отдельной функции: ошибка в ней всплывёт не здесь, а на
    несходящихся формах через десять слоёв.

    Деление целочисленное, floor: хвост входа, который не покрывается
    последним окном, просто отбрасывается.
    """
    return (size - kernel + 2 * padding) // stride + 1


def conv_params(c_in, c_out, kernel, bias=True):
    """Число обучаемых параметров свёрточного слоя.

    conv_params(3, 64, 3)              ->  1792   (64*3*3*3 + 64)
    conv_params(3, 64, 3, bias=False)  ->  1728

    Веса имеют форму (C_out, C_in, K, K), смещений ровно C_out — по одному
    на выходной канал, а не на пиксель: в этом и состоит parameter sharing.

    Полносвязный слой на той же картинке 224x224x3 потребовал бы 150528
    весов на ОДИН нейрон. Сравни с 1792 на целых 64 карты признаков.
    """
    total = c_out * c_in * kernel * kernel
    return total + c_out if bias else total


def pad2d(x, p):
    """Обрамить двумерную сетку p нулями со всех сторон.

    pad2d([[1]], 1)  ->  [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pad2d([[1]], 0)  ->  [[1]]

    Без padding каждая свёртка съедает по (K-1)/2 пикселя с каждой стороны.
    Стопка из двадцати слоёв превратит 224 в 184 и сломает residual-связи,
    которым нужны совпадающие формы.

    Тип нуля — int; если на входе float, тесты это не различают, но помни,
    что numpy сохранил бы dtype входа.
    """
    if p == 0:
        return [list(row) for row in x]
    w = len(x[0])
    top = [[0] * (w + 2 * p) for _ in range(p)]
    middle = [[0] * p + list(row) + [0] * p for row in x]
    return top + middle + [[0] * (w + 2 * p) for _ in range(p)]


def conv2d(x, kernel, stride=1, padding=0):
    """Свёртка одного канала: двумерная сетка x и ядро kernel.

    Возвращает двумерную сетку размера conv_output_size по каждой оси.

    conv2d([[1, 2], [3, 4]], [[1]])            ->  [[1, 2], [3, 4]]
    conv2d([[1, 2], [3, 4]], [[1, 1], [1, 1]]) ->  [[10]]

    В каждой позиции — сумма поэлементных произведений окна и ядра. Ядро НЕ
    переворачивается: то, что все называют convolution, математически есть
    cross-correlation. Перевернёшь — Sobel начнёт давать противоположный знак.

    Дельта-ядро [[0,0,0],[0,1,0],[0,0,0]] с padding=1 обязано вернуть вход
    без изменений — самая быстрая проверка, что индексы не съехали.
    """
    kh = len(kernel)
    kw = len(kernel[0])
    xp = pad2d(x, padding)
    h_out = conv_output_size(len(x), kh, padding, stride)
    w_out = conv_output_size(len(x[0]), kw, padding, stride)

    out = []
    for i in range(h_out):
        row = []
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            acc = 0.0
            for a in range(kh):
                xrow = xp[hs + a]
                krow = kernel[a]
                for b in range(kw):
                    acc += xrow[ws + b] * krow[b]
            row.append(acc)
        out.append(row)
    return out


def conv2d_multichannel(x, weight, bias=None, stride=1, padding=0):
    """Свёртка по всем каналам сразу.

    x       — (C_in, H, W), вложенные списки;
    weight  — (C_out, C_in, KH, KW);
    bias    — список длины C_out или None;
    выход   — (C_out, H_out, W_out).

    Один выходной канал = сумма свёрток входных каналов со своими срезами
    ядра плюс смещение. Складываем ПО каналам, а не конкатенируем: канальная
    ось входа исчезает, её место занимает C_out.

    conv2d_multichannel([[[1, 2]], [[10, 20]]], [[[[1]], [[1]]]])  ->  [[[11, 22]]]

    Считай через conv2d — переписывать четверной цикл заново незачем.
    """
    c_out = len(weight)
    c_in = len(x)
    out = []
    for oc in range(c_out):
        # аккумулируем карту признаков по входным каналам
        acc = None
        for ic in range(c_in):
            part = conv2d(x[ic], weight[oc][ic], stride, padding)
            if acc is None:
                acc = part
            else:
                for i in range(len(acc)):
                    for j in range(len(acc[0])):
                        acc[i][j] += part[i][j]
        if bias is not None:
            for i in range(len(acc)):
                for j in range(len(acc[0])):
                    acc[i][j] += bias[oc]
        out.append(acc)
    return out


def im2col(x, kh, kw, stride=1, padding=0):
    """Разложить все окна входа по столбцам матрицы.

    Возвращает кортеж (cols, h_out, w_out), где cols — матрица размера
    (C_in * kh * kw) x (h_out * w_out).

    Порядок внутри столбца: сначала канал, потом строка ядра, потом столбец
    ядра — тот же, что даёт weight.reshape(C_out, -1) у ядра. Столбцы идут
    по позициям выхода построчно.

    cols, h, w = im2col([[[1, 2], [3, 4]]], 2, 2)
    cols  ->  [[1], [2], [3], [4]]
    (h, w) ->  (1, 1)

    Зачем: GPU не любит четверные циклы, зато отлично умеет одно большое
    матричное умножение. im2col превращает свёртку ровно в него — на этом
    построены все быстрые реализации conv.

    Цена — память: каждое перекрытие окон копируется, и матрица получается
    в K*K раз больше входа.
    """
    c_in = len(x)
    h_out = conv_output_size(len(x[0]), kh, padding, stride)
    w_out = conv_output_size(len(x[0][0]), kw, padding, stride)
    padded = [pad2d(plane, padding) for plane in x]

    cols = [[0.0] * (h_out * w_out) for _ in range(c_in * kh * kw)]
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            row = 0
            for c in range(c_in):
                for a in range(kh):
                    for b in range(kw):
                        cols[row][col] = padded[c][hs + a][ws + b]
                        row += 1
            col += 1
    return cols, h_out, w_out


def conv2d_im2col(x, weight, bias=None, stride=1, padding=0):
    """Та же свёртка, что conv2d_multichannel, но через im2col и матмул.

    Ядро разворачивается в матрицу (C_out, C_in*KH*KW), умножается на cols,
    результат раскладывается обратно в (C_out, H_out, W_out).

    Результат обязан совпасть с conv2d_multichannel до порядка сложения:
    это и есть тест на то, что порядок осей в im2col выбран верно.

    Аналог того, что делает torch.nn.functional.conv2d под капотом
    (плюс тайлинг кэша и Winograd, которых тут нет).
    """
    c_out = len(weight)
    cols, h_out, w_out = im2col(x, len(weight[0][0]), len(weight[0][0][0]), stride, padding)
    # разворачиваем ядро в строку: порядок обязан совпасть с порядком в столбце
    flat_w = [[v for plane in weight[oc] for row in plane for v in row] for oc in range(c_out)]

    out = []
    for oc in range(c_out):
        wrow = flat_w[oc]
        values = []
        for col in range(h_out * w_out):
            acc = 0.0
            for r in range(len(wrow)):
                acc += wrow[r] * cols[r][col]
            if bias is not None:
                acc += bias[oc]
            values.append(acc)
        out.append([values[i * w_out : (i + 1) * w_out] for i in range(h_out)])
    return out


def receptive_field(layers):
    """Размер рецептивного поля после стопки слоёв.

    layers — список пар (kernel, stride) в порядке от входа к выходу.

    receptive_field([(3, 1), (3, 1)])          ->  5
    receptive_field([(3, 1), (3, 1), (3, 1)])  ->  7
    receptive_field([(3, 2), (3, 1)])          ->  7

    Рекуррентно: rf = 1, jump = 1, и на каждом слое
        rf += (kernel - 1) * jump
        jump *= stride

    Для стопки K x K со stride 1 это сворачивается в 1 + L * (K - 1) — то
    самое «две 3x3 видят столько же, сколько одна 5x5», из-за которого VGG
    отказалась от больших ядер.

    Осторожно: jump умножается ПОСЛЕ прибавки, иначе первый слой посчитается
    с чужим шагом.
    """
    rf = 1
    jump = 1
    for kernel, stride in layers:
        rf += (kernel - 1) * jump
        jump *= stride
    return rf
