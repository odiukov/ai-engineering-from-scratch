"""
Transfusion: авторегрессия по тексту и диффузия по картинке — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Модель урока: одна последовательность, в которой текстовые токены — обычные
неотрицательные целые id, а картинка — блок непрерывных патчей между тегами
IMG_OPEN и IMG_CLOSE. Сам transformer тут не нужен: весь смысл Transfusion в
трёх вещах, и все три можно собрать на голом Python —

  * block-triangular attention mask (causal по тексту, bidirectional внутри
    картинки);
  * два лосса на одном backbone: NTP по тексту и flow matching по патчам;
  * счёт forward-проходов на генерацию — там, где Chameleon платит по проходу
    за каждый патч, Transfusion платит по проходу за denoise step.

Только стандартная библиотека. Ничего случайного здесь нет вовсе, так что
тесты воспроизводимы по построению.
"""

# Разметка последовательности. Отрицательные значения выбраны так, чтобы
# не сталкиваться с настоящими id токенов из BPE-словаря.
IMG_OPEN = -1
IMG_CLOSE = -2
PATCH = -3


def find_image_blocks(tokens):
    """Границы блоков непрерывных патчей: список пар (start, end), end не включён.

    find_image_blocks([5, IMG_OPEN, PATCH, PATCH, IMG_CLOSE, 6])  ->  [(2, 4)]
    find_image_blocks([5, 6])                                     ->  []

    Сами теги IMG_OPEN и IMG_CLOSE в блок НЕ входят: это дискретные токены,
    они живут по текстовым правилам внимания. Внутрь попадают только патчи.

    Ловушка: незакрытый IMG_OPEN, IMG_CLOSE без пары и вложенный IMG_OPEN —
    это битая последовательность, а не «блок до конца строки». Бросай
    ValueError, иначе маска молча получится неправильной, а отладить её
    потом невозможно.
    """
    blocks = []
    start = None
    for i, tok in enumerate(tokens):
        if tok == IMG_OPEN:
            if start is not None:
                raise ValueError(f"вложенный IMG_OPEN на позиции {i}")
            start = i + 1
        elif tok == IMG_CLOSE:
            if start is None:
                raise ValueError(f"IMG_CLOSE без пары на позиции {i}")
            blocks.append((start, i))
            start = None
    if start is not None:
        raise ValueError("IMG_OPEN не закрыт до конца последовательности")
    return blocks


def build_mask(tokens):
    """Block-triangular attention mask: матрица n x n из нулей и единиц.

    M[i][j] == 1 значит «позиция i имеет право смотреть на позицию j».

    build_mask([7, 8])  ->  [[1, 0],
                             [1, 1]]

    Четыре правила урока, ровно в этом виде:
      * текст -> текст:       можно, если j <= i          (causal)
      * патч  -> патч:        можно, если тот же блок     (bidirectional)
      * текст -> патч:        можно, если j < i           (картинки в прошлом)
      * патч  -> текст:       можно, если j < начала своего блока

    Теги IMG_OPEN/IMG_CLOSE считаются текстом.

    Обрати внимание на третье и четвёртое правило: патч НЕ видит свой
    собственный IMG_CLOSE и вообще ничего после блока, а вот текст после
    картинки видит все её патчи. Из-за этого маска несимметрична везде,
    кроме внутренностей одного блока.
    """
    blocks = find_image_blocks(tokens)
    n = len(tokens)

    # block_of[i] — номер картинки, которой принадлежит позиция, или None
    # для текста. Считаем один раз: иначе внутренний цикл станет кубическим.
    block_of = [None] * n
    for b, (start, end) in enumerate(blocks):
        for i in range(start, end):
            block_of[i] = b

    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        bi = block_of[i]
        for j in range(n):
            bj = block_of[j]
            if bi is None and bj is None:
                allowed = j <= i
            elif bi is not None and bj is not None:
                allowed = bi == bj
            elif bi is None:
                allowed = j < i
            else:
                allowed = j < blocks[bi][0]
            mask[i][j] = 1 if allowed else 0
    return mask


def flow_interpolate(x0, eps, t):
    """Точка на прямой из данных в шум: xt = (1 - t) * x0 + t * eps.

    flow_interpolate([0.0, 1.0], [1.0, 3.0], 0.0)  ->  [0.0, 1.0]
    flow_interpolate([0.0, 1.0], [1.0, 3.0], 1.0)  ->  [1.0, 3.0]
    flow_interpolate([0.0, 1.0], [1.0, 3.0], 0.5)  ->  [0.5, 2.0]

    В flow matching никакого хитрого noise schedule нет: между чистым патчем
    и шумом просто линейная интерполяция. Отсюда и «rectified flow» — путь
    прямой.

    Ловушка: t вне [0, 1] это не «сильнее зашумить», а выход за отрезок.
    Бросай ValueError.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t должно быть в [0, 1], получено {t}")
    if len(x0) != len(eps):
        raise ValueError(f"разная длина: {len(x0)} и {len(eps)}")
    return [(1.0 - t) * a + t * b for a, b in zip(x0, eps)]


def flow_target(x0, eps):
    """Целевое velocity field: eps - x0. Куда двигаться от данных к шуму.

    flow_target([0.0, 1.0], [1.0, 3.0])  ->  [1.0, 2.0]
    flow_target([2.0], [2.0])            ->  [0.0]

    Именно это, а не сам шум, предсказывает сеть в v-параметризации. Заметь:
    от t цель не зависит вовсе — прямая одна и та же на всём пути, у неё
    постоянная скорость.
    """
    if len(x0) != len(eps):
        raise ValueError(f"разная длина: {len(x0)} и {len(eps)}")
    return [b - a for a, b in zip(x0, eps)]


def flow_loss(pred, x0, eps):
    """Диффузионный лосс патча: MSE между предсказанным и целевым velocity.

    flow_loss([1.0, 2.0], [0.0, 1.0], [1.0, 3.0])  ->  0.0   (угадали ровно)
    flow_loss([0.0, 0.0], [0.0, 1.0], [1.0, 3.0])  ->  2.5

    Разбор второго примера: цель [1.0, 2.0], ошибки 1 и 2,
    (1 + 4) / 2 = 2.5.

    Делим на длину, а не суммируем: иначе величина лосса зависела бы от
    размера патча, и веса двух лоссов пришлось бы перетюнивать при каждой
    смене разрешения.
    """
    target = flow_target(x0, eps)
    if len(pred) != len(target):
        raise ValueError(f"разная длина: {len(pred)} и {len(target)}")
    if not pred:
        raise ValueError("пустой патч: MSE не определён")
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)


def flow_loss_grad(pred, x0, eps):
    """Аналитический градиент flow_loss по предсказанию.

    flow_loss_grad([0.0, 0.0], [0.0, 1.0], [1.0, 3.0])  ->  [-1.0, -2.0]

    d/dpred_k среднего квадрата ошибки = 2 * (pred_k - target_k) / n.

    Проверь себя численной центральной разностью — расхождение больше 1e-6
    почти всегда значит, что потеряли двойку или деление на n.
    """
    target = flow_target(x0, eps)
    if len(pred) != len(target):
        raise ValueError(f"разная длина: {len(pred)} и {len(target)}")
    if not pred:
        raise ValueError("пустой патч: градиент не определён")
    n = len(pred)
    return [2.0 * (p - t) / n for p, t in zip(pred, target)]


def balanced_weights(text_loss, image_loss):
    """Веса двух лоссов, уравнивающие их вклад. Вернуть кортеж (w_text, w_img).

    balanced_weights(2.0, 20.0)  ->  (1.0, 0.1)
    balanced_weights(5.0, 5.0)   ->  (1.0, 1.0)

    Нормировка: w_text всегда 1.0, подгоняется только w_img. Иначе решений
    бесконечно много и сравнивать прогоны между собой невозможно.

    Это упражнение 1 урока: диффузионный лосс живёт на другой числовой шкале,
    чем cross-entropy. Если веса не выровнять, один head задавит другой и
    вторая модальность просто не выучится.

    Ловушка: нулевой или отрицательный лосс — не «идеально обучились», а
    сломанный расчёт. Бросай ValueError, деление на ноль тут недопустимо.
    """
    if text_loss <= 0:
        raise ValueError(f"text_loss должен быть положительным, получено {text_loss}")
    if image_loss <= 0:
        raise ValueError(f"image_loss должен быть положительным, получено {image_loss}")
    return (1.0, text_loss / image_loss)


def generation_forward_passes(n_text, n_patches, n_denoise_steps):
    """Сколько проходов transformer стоит генерация текста и одной картинки.

    generation_forward_passes(50, 256, 20)  ->  70    (Transfusion)
    generation_forward_passes(50, 256, 0)   ->  306   (Chameleon-режим)

    Текст авторегрессивный всегда: n_text проходов, по одному на токен.
    А дальше развилка:
      * n_denoise_steps > 0 — патчи денойзятся ПАРАЛЛЕЛЬНО, все сразу, и
        цена не зависит от их количества: ровно n_denoise_steps проходов;
      * n_denoise_steps == 0 — режим дискретных токенов (Chameleon, Emu3):
        каждый патч предсказывается отдельно, n_patches проходов.

    Вот и весь выигрыш Transfusion на инференсе: 20 вместо 256.

    Отрицательные аргументы — ValueError.
    """
    if n_text < 0 or n_patches < 0 or n_denoise_steps < 0:
        raise ValueError("количества проходов не могут быть отрицательными")
    if n_denoise_steps == 0:
        return n_text + n_patches
    return n_text + n_denoise_steps
