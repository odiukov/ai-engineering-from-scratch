"""
Instance-сегментация и Mask R-CNN — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def box_iou(a, b):
    """IoU двух боксов в формате (x1, y1, x2, y2).

    box_iou((0, 0, 2, 2), (0, 0, 2, 2))  ->  1.0
    box_iou((0, 0, 2, 2), (2, 2, 4, 4))  ->  0.0   (касаются углом, пересечения нет)
    box_iou((0, 0, 4, 4), (2, 2, 6, 6))  ->  0.142857...  (4 / 28)

    Ловушка: ширина пересечения бывает отрицательной. Обрежь её нулём
    ПЕРЕД умножением, иначе два непересекающихся бокса дадут положительную
    «площадь» из произведения двух минусов.

    IoU — валюта детекции: по нему работает NMS, по нему считается mAP,
    по нему же (только на масках, а не на боксах) считается mask AP.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # два минуса дают плюс — вот почему обрезка нулём идёт до умножения
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms(boxes, scores, iou_threshold=0.7):
    """Non-maximum suppression. Вернуть индексы выживших боксов.

    Алгоритм: берём бокс с максимальным score, кладём в ответ, выкидываем
    все оставшиеся, у которых IoU с ним БОЛЬШЕ порога. Повторяем.

    nms([(0, 0, 2, 2), (0, 0, 2, 2)], [0.9, 0.1], 0.5)          ->  [0]
    nms([(0, 0, 2, 2), (5, 5, 7, 7)], [0.1, 0.9], 0.5)          ->  [1, 0]

    Индексы возвращаются в порядке убывания score, а не в исходном.
    Порог строгий (> порога подавляем, ровно порог оставляем) — так делает
    torchvision.ops.nms. RPN в Mask R-CNN гоняет NMS с порогом 0.7.
    """
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        best = order[0]
        keep.append(best)
        order = [i for i in order[1:] if box_iou(boxes[best], boxes[i]) <= iou_threshold]
    return keep


def decode_box_delta(anchor, delta):
    """Превратить якорь + предсказанную поправку в бокс. Так работает RPN.

    anchor = (x1, y1, x2, y2), delta = (dx, dy, dw, dh).

    cx' = cx + dx * w      w' = w * exp(dw)
    cy' = cy + dy * h      h' = h * exp(dh)

    decode_box_delta((0, 0, 10, 10), (0, 0, 0, 0))    ->  (0.0, 0.0, 10.0, 10.0)
    decode_box_delta((0, 0, 10, 10), (0.1, 0, 0, 0))  ->  (1.0, 0.0, 11.0, 10.0)

    Почему сдвиг умножается на размер якоря, а масштаб идёт через exp:
    сеть предсказывает величины порядка единицы независимо от того, якорь
    это 16 пикселей или 512. И exp физически не даёт получить
    отрицательную ширину, какой бы бред сеть ни выдала.
    """
    x1, y1, x2, y2 = anchor
    dx, dy, dw, dh = delta
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    new_cx = cx + dx * w
    new_cy = cy + dy * h
    new_w = w * math.exp(dw)
    new_h = h * math.exp(dh)
    return (
        new_cx - new_w / 2.0,
        new_cy - new_h / 2.0,
        new_cx + new_w / 2.0,
        new_cy + new_h / 2.0,
    )


def bilinear_sample(feature, y, x):
    """Билинейная выборка одного значения из карты признаков H x W.

    feature[i][j] — значение в ЦЕНТРЕ пикселя с целыми координатами (i, j).
    Координаты за краем карты прижимаются к краю (clamp), а не обнуляются.

    f = [[0.0, 1.0],
         [2.0, 3.0]]
    bilinear_sample(f, 0.0, 0.0)  ->  0.0
    bilinear_sample(f, 0.0, 0.5)  ->  0.5    (ровно между 0 и 1)
    bilinear_sample(f, 0.5, 0.5)  ->  1.5    (среднее всех четырёх)
    bilinear_sample(f, -5, -5)    ->  0.0    (clamp в угол)

    Порядок аргументов — (y, x), как индексация feature[y][x], а не как
    координаты бокса (x, y). Перепутать их — классика.

    Это сердце RoIAlign: ни одного округления, значит градиент течёт во
    все четыре соседних пикселя.
    """
    h = len(feature)
    w = len(feature[0])
    # clamp ДО floor: тогда веса на границе получаются корректными сами собой
    y = min(max(y, 0.0), h - 1.0)
    x = min(max(x, 0.0), w - 1.0)
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)
    wy = y - y0
    wx = x - x0
    top = feature[y0][x0] * (1 - wx) + feature[y0][x1] * wx
    bottom = feature[y1][x0] * (1 - wx) + feature[y1][x1] * wx
    return top * (1 - wy) + bottom * wy


def roi_align(feature, box, output_size=7, spatial_scale=1.0):
    """RoIAlign: вырезать из карты признаков сетку output_size x output_size.

    box задан в координатах ИСХОДНОГО изображения, spatial_scale — обратная
    величина stride карты признаков (1/16 для stride 16).

    Шаги:
      1. перевести бокс в координаты карты: c * spatial_scale - 0.5
         (минус полпикселя — переход от «угол пикселя» к «центр пикселя»);
      2. разбить бокс на output_size^2 одинаковых ячеек;
      3. взять bilinear_sample ровно в центре каждой ячейки.

    Никаких округлений ни на одном шаге.

    f = [[0.0, 1.0, 2.0, 3.0]] * 4          # значение равно номеру столбца
    roi_align(f, (0, 0, 4, 4), 2, 1.0)  ->  [[0.5, 2.5], [0.5, 2.5]]
    roi_align([[5.0] * 4] * 4, (0, 0, 4, 4), 2, 1.0)  ->  [[5.0, 5.0], [5.0, 5.0]]

    Соответствует torchvision.ops.roi_align(..., sampling_ratio=1,
    aligned=True). Замена RoIPool на RoIAlign дала +3-4 пункта mask AP на
    COCO бесплатно.
    """
    x1 = box[0] * spatial_scale - 0.5
    y1 = box[1] * spatial_scale - 0.5
    x2 = box[2] * spatial_scale - 0.5
    y2 = box[3] * spatial_scale - 0.5
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size
    out = []
    for i in range(output_size):
        cy = y1 + bin_h * (i + 0.5)
        row = [bilinear_sample(feature, cy, x1 + bin_w * (j + 0.5)) for j in range(output_size)]
        out.append(row)
    return out


def roi_pool(feature, box, output_size=7, spatial_scale=1.0):
    """RoIPool — как делали до 2017 года. Нужен, чтобы увидеть, что он ломает.

    Отличий от roi_align два, и оба — округления:
      1. координаты бокса округляются до целых (round);
      2. границы ячеек тоже округляются (floor для начала, ceil для конца);
      3. внутри ячейки берётся МАКСИМУМ, а не билинейная выборка.

    f = [[0.0, 1.0, 2.0, 3.0]] * 4
    roi_pool(f, (0, 0, 4, 4), 2, 1.0)  ->  [[1.0, 3.0], [1.0, 3.0]]

    Сравни с roi_align на тех же данных: 0.5 и 2.5 против 1.0 и 3.0. Max
    систематически завышает, а округление смещает окно. На stride 32 эта
    ошибка — целый пиксель карты признаков, то есть 32 пикселя картинки.

    Пустая ячейка (после округления в неё не попало ни одного пикселя)
    даёт 0.0 — так же делает оригинальный Fast R-CNN.
    """
    h = len(feature)
    w = len(feature[0])
    x1 = int(round(box[0] * spatial_scale))
    y1 = int(round(box[1] * spatial_scale))
    x2 = int(round(box[2] * spatial_scale))
    y2 = int(round(box[3] * spatial_scale))
    roi_w = max(x2 - x1, 1)
    roi_h = max(y2 - y1, 1)
    bin_w = roi_w / output_size
    bin_h = roi_h / output_size
    out = []
    for i in range(output_size):
        row = []
        hstart = max(0, min(h, y1 + int(math.floor(i * bin_h))))
        hend = max(0, min(h, y1 + int(math.ceil((i + 1) * bin_h))))
        for j in range(output_size):
            wstart = max(0, min(w, x1 + int(math.floor(j * bin_w))))
            wend = max(0, min(w, x1 + int(math.ceil((j + 1) * bin_w))))
            values = [
                feature[yy][xx] for yy in range(hstart, hend) for xx in range(wstart, wend)
            ]
            row.append(max(values) if values else 0.0)
        out.append(row)
    return out


def paste_mask(mask, box, image_h, image_w, threshold=0.5):
    """Растянуть маску головы (28x28 вероятностей) на бокс в полном кадре.

    mask  — маленькая сетка вероятностей в [0, 1];
    box   — (x1, y1, x2, y2) в пикселях кадра;
    ответ — бинарная сетка image_h x image_w из 0 и 1.

    Пиксель кадра с центром (px + 0.5, py + 0.5) переводится в координату
    маски и читается через bilinear_sample; 1 ставится там, где значение
    СТРОГО больше threshold. Вне бокса всегда 0.

    paste_mask([[1.0]], (0, 0, 2, 2), 2, 2)  ->  [[1, 1], [1, 1]]
    paste_mask([[1.0]], (0, 0, 1, 1), 2, 2)  ->  [[1, 0], [0, 0]]
    paste_mask([[0.2]], (0, 0, 2, 2), 2, 2)  ->  [[0, 0], [0, 0]]

    Это последний шаг инференса Mask R-CNN. torchvision делает его внутри и
    отдаёт уже готовые маски формы (N, 1, H, W) — порог 0.5 на твоей совести.
    """
    mh = len(mask)
    mw = len(mask[0])
    x1, y1, x2, y2 = box
    # защита от вырожденного бокса: деление на ноль вместо предсказания
    span_x = max(x2 - x1, 1e-9)
    span_y = max(y2 - y1, 1e-9)
    out = [[0] * image_w for _ in range(image_h)]
    py_lo = max(0, int(math.floor(y1)))
    py_hi = min(image_h, int(math.ceil(y2)))
    px_lo = max(0, int(math.floor(x1)))
    px_hi = min(image_w, int(math.ceil(x2)))
    for py in range(py_lo, py_hi):
        v = (py + 0.5 - y1) / span_y * mh - 0.5
        for px in range(px_lo, px_hi):
            u = (px + 0.5 - x1) / span_x * mw - 0.5
            if bilinear_sample(mask, v, u) > threshold:
                out[py][px] = 1
    return out


def mask_iou(a, b):
    """IoU двух бинарных масок одинакового размера. Метрика mask AP.

    mask_iou([[1, 1], [0, 0]], [[1, 1], [0, 0]])  ->  1.0
    mask_iou([[1, 1], [0, 0]], [[1, 0], [0, 0]])  ->  0.5
    mask_iou([[0, 0]], [[0, 0]])                  ->  0.0  (пустое объединение)

    Тот же самый IoU, что и у боксов, только считается по пикселям силуэта.
    Именно подмена box IoU на mask IoU превращает mAP детекции в mask AP —
    и именно поэтому две цифры расходятся, когда бокс точный, а силуэт нет.
    """
    inter = 0
    union = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            if va and vb:
                inter += 1
            if va or vb:
                union += 1
    if union == 0:
        return 0.0
    return inter / union
