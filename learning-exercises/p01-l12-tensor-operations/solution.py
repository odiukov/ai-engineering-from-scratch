"""
Операции с тензорами — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
from itertools import product


def strides(shape):
    """Шаги по памяти для формы shape в row-major порядке.

    strides((3, 4))     ->  (4, 1)    (строка — 4 элемента, столбец — 1)
    strides((2, 3, 4))  ->  (12, 4, 1)
    strides(())         ->  ()

    Шаг по оси — сколько элементов плоского списка нужно перепрыгнуть,
    чтобы сдвинуться на единицу вдоль этой оси. Последняя ось всегда 1,
    остальные считаются справа налево: stride[i] = stride[i+1] * shape[i+1].

    Тензор не хранит вложенные списки. Он хранит один плоский список и
    форму; шаги — это весь мост между ними.
    """
    if not shape:
        return ()
    out = [1] * len(shape)
    # справа налево: каждый следующий шаг больше предыдущего в shape[i+1] раз
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return tuple(out)


def flat_index(shape, index):
    """Позиция многомерного индекса в плоском списке.

    flat_index((3, 4), (0, 0))        ->  0
    flat_index((3, 4), (1, 2))        ->  6      (1*4 + 2*1)
    flat_index((2, 3, 4), (1, 2, 3))  ->  23

    Это скалярное произведение индекса на шаги. Именно так любая библиотека
    тензоров превращает t[i][j][k] в один поход в память.
    """
    return sum(i * s for i, s in zip(index, strides(shape)))


def reshape(shape, new_shape):
    """Новая форма для того же плоского списка. Данные НЕ двигаются.

    reshape((2, 6), (3, 4))    ->  (3, 4)
    reshape((2, 6), (-1, 3))   ->  (4, 3)     (-1 = «посчитай сам»)
    reshape((2, 6), (5, 5))    ->  ValueError (12 элементов не лягут в 25)

    Реформа законна ровно тогда, когда произведение размеров совпадает.
    Ровно одна ось может быть -1: её размер выводится делением.

    Несовпадение числа элементов — ValueError, две минус-единицы — тоже.
    Других отрицательных размеров у формы не бывает, они тоже дают ValueError.
    """
    if any(d < 0 for d in shape):
        raise ValueError("исходная форма не может содержать отрицательные размеры")
    if any(d < -1 for d in new_shape):
        raise ValueError("только -1 может обозначать выводимый размер")
    total = math.prod(shape)
    holes = [d for d in new_shape if d == -1]
    if len(holes) > 1:
        raise ValueError("минус-единица может быть только одна")
    if holes:
        known = math.prod(d for d in new_shape if d != -1)
        if known == 0 or total % known != 0:
            raise ValueError(f"{total} элементов не делятся на {known}")
        new_shape = tuple(total // known if d == -1 else d for d in new_shape)
    else:
        new_shape = tuple(new_shape)
    if math.prod(new_shape) != total:
        raise ValueError(f"{total} элементов не лягут в форму {new_shape}")
    return new_shape


def permute(data, shape, order):
    """Переставить оси. Возвращает (новые данные, новая форма).

    permute([0, 1, 2, 3, 4, 5], (2, 3), (1, 0))
        ->  ([0, 3, 1, 4, 2, 5], (3, 2))

    order — куда какая ось едет: order[k] это номер СТАРОЙ оси, которая
    станет k-й новой. Транспонирование матрицы — это order=(1, 0), перевод
    NCHW в NHWC — order=(0, 2, 3, 1).

    В настоящих библиотеках перестановка не копирует данные, а лишь меняет
    шаги местами, и тензор становится «нессплошным» (non-contiguous). Здесь
    мы честно материализуем результат: так виднее, что порядок элементов
    в памяти действительно другой.
    """
    new_shape = tuple(shape[a] for a in order)
    out = []
    # product обходит индексы ровно в row-major порядке — том же, в каком
    # элементы лягут в выходной список
    for idx in product(*(range(d) for d in new_shape)):
        source = [0] * len(shape)
        for new_axis, old_axis in enumerate(order):
            source[old_axis] = idx[new_axis]
        out.append(data[flat_index(shape, source)])
    return out, new_shape


def broadcast_shapes(shape_a, shape_b):
    """Форма результата поэлементной операции над двумя формами.

    broadcast_shapes((3, 1), (1, 4))           ->  (3, 4)
    broadcast_shapes((8, 1, 6, 1), (7, 1, 5))  ->  (8, 7, 6, 5)
    broadcast_shapes((3,), (4,))               ->  ValueError

    Правила ровно три:
      1. формы выравниваются СПРАВА, короткая дополняется единицами слева;
      2. на каждой оси размеры совместимы, если они равны или один из них 1;
      3. в результат идёт больший из двух.

    Выравнивание справа — самая частая ловушка: (B, T, D) и (D,) сходятся,
    а (B, T, D) и (B,) нет, хотя на глаз второе выглядит логичнее.
    """
    width = max(len(shape_a), len(shape_b))
    a = (1,) * (width - len(shape_a)) + tuple(shape_a)
    b = (1,) * (width - len(shape_b)) + tuple(shape_b)
    out = []
    for da, db in zip(a, b):
        if da != db and da != 1 and db != 1:
            raise ValueError(f"формы {tuple(shape_a)} и {tuple(shape_b)} несовместимы")
        out.append(max(da, db))
    return tuple(out)


def broadcast_to(data, shape, target):
    """Растянуть тензор до формы target. Возвращает плоский список.

    broadcast_to([1, 2, 3], (3, 1), (3, 2))  ->  [1, 1, 2, 2, 3, 3]
    broadcast_to([1, 2], (2,), (3, 2))       ->  [1, 2, 1, 2, 1, 2]
    broadcast_to([1, 2, 3], (3,), (5,))      ->  ValueError

    Ось размера 1 повторяется столько раз, сколько нужно; ось нужного
    размера копируется как есть. Растянуть ось размера 3 до 5 нельзя.

    Никакой новой информации растяжение не создаёт: это просто способ
    прибавить bias формы (D,) к батчу (B, T, D), не заводя B*T копий bias.
    """
    padded = (1,) * (len(target) - len(shape)) + tuple(shape)
    if broadcast_shapes(padded, target) != tuple(target):
        raise ValueError(f"форма {tuple(shape)} не растягивается до {tuple(target)}")
    out = []
    for idx in product(*(range(d) for d in target)):
        # по растянутой оси всегда берём нулевой элемент — он там один
        source = [0 if d == 1 else i for d, i in zip(padded, idx)]
        out.append(data[flat_index(padded, source)])
    return out


def add(data_a, shape_a, data_b, shape_b):
    """Поэлементное сложение с broadcasting. Возвращает (данные, форма).

    add([1, 2], (2,), [10, 20], (2,))         ->  ([11, 22], (2,))
    add([1, 2, 3], (3, 1), [10, 20], (1, 2))  ->  ([11, 21, 12, 22, 13, 23], (3, 2))

    Сначала считается общая форма, потом ОБА тензора растягиваются до неё,
    и только потом идёт сложение. Порядок важен: складывать «как получится»
    без выравнивания — верный способ тихо просуммировать не те оси.
    """
    target = broadcast_shapes(shape_a, shape_b)
    left = broadcast_to(data_a, shape_a, target)
    right = broadcast_to(data_b, shape_b, target)
    return [x + y for x, y in zip(left, right)], target


def reduce_sum(data, shape, axis):
    """Схлопнуть одну ось суммированием. Возвращает (данные, форма).

    reduce_sum([1, 2, 3, 4, 5, 6], (2, 3), 0)  ->  ([5, 7, 9], (3,))
    reduce_sum([1, 2, 3, 4, 5, 6], (2, 3), 1)  ->  ([6, 15], (2,))
    reduce_sum([1, 2, 3], (3,), 0)             ->  ([6], ())

    Ось исчезает из формы, ранг падает на единицу. Так работает mean-pooling
    по длине последовательности: (B, T, D) с axis=1 даёт (B, D).

    Номер оси — не «строки или столбцы», а позиция в кортеже формы.
    Перепутанный axis не падает с ошибкой, а тихо считает не то.
    """
    new_shape = tuple(d for i, d in enumerate(shape) if i != axis)
    out = [0] * math.prod(new_shape)
    for position, idx in enumerate(product(*(range(d) for d in shape))):
        # все позиции, отличающиеся только по axis, попадают в одну ячейку
        target = [v for i, v in enumerate(idx) if i != axis]
        out[flat_index(new_shape, target)] += data[position]
    return out, new_shape
