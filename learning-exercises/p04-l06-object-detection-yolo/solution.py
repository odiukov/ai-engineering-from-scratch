"""
Детекция объектов — YOLO — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Бокс везде — кортеж (x1, y1, x2, y2) в пикселях, левый верхний и правый
нижний углы. Формат (cx, cy, w, h) встречается не реже, и перепутать их —
самая частая ошибка в детекции.
"""

import math


def sigmoid(x):
    """Сигмоида: любое число -> (0, 1). Аналог torch.sigmoid.

    sigmoid(0.0)   ->  0.5
    sigmoid(2.0)   ->  примерно 0.8808

    В decode она держит центр бокса внутри своей ячейки сетки: что бы сеть
    ни выдала, sigmoid(tx) лежит между 0 и 1.

    Ловушка — переполнение на больших по модулю аргументах: math.exp(1000)
    бросает OverflowError. Формула 1/(1+exp(-x)) безопасна при x >= 0,
    а exp(x)/(1+exp(x)) — при x < 0. Выбирай ветку по знаку.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # здесь x < 0, значит exp(x) < 1 и переполниться нечему
    return e / (1.0 + e)


def iou(a, b):
    """Intersection over Union двух боксов: площадь пересечения / площадь объединения.

    iou((0, 0, 2, 2), (0, 0, 2, 2))  ->  1.0
    iou((0, 0, 1, 1), (5, 5, 6, 6))  ->  0.0
    iou((0, 0, 2, 2), (1, 1, 3, 3))  ->  примерно 0.1429  (1 / 7)

    Пересечение считается по каждой оси отдельно:
    min(правых краёв) - max(левых краёв). Отрицательную ширину обязательно
    обрезать нулём — иначе два непересекающихся бокса дадут положительное
    произведение двух минусов и IoU из воздуха.

    Вырожденный бокс нулевой площади не должен ронять функцию делением на
    ноль: объединение тогда тоже ноль, а разумный ответ — 0.0.

    IoU решает в детекции всё: по нему предсказание засчитывают в true
    positive и по нему же NMS выбрасывает дубли.
    """
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:  # не пересекаются или касаются краями
        return 0.0
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression: индексы боксов, которые надо оставить.

    Жадный алгоритм: берём самый уверенный бокс, выбрасываем все,
    у которых IoU с ним больше порога, повторяем на остатке.

    nms([(0,0,2,2), (0,0,2,2), (9,9,10,10)], [0.9, 0.8, 0.7])  ->  [0, 2]

    Возвращаются индексы в ИСХОДНОМ списке, в порядке убывания score.

    Порог сравнивается строго: бокс с IoU ровно iou_threshold остаётся.
    Типичное значение 0.45.

    Зачем: свёрточная сеть предсказывает один и тот же объект из нескольких
    соседних ячеек. Без NMS каждый объект приходит по десять раз, precision
    падает в пол, а на картинке гирлянда рамок.
    """
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    while order:
        best = order[0]
        keep.append(best)
        # остаток: только те, кто недостаточно пересекается с победителем
        order = [i for i in order[1:] if iou(boxes[best], boxes[i]) <= iou_threshold]
    return keep


def decode_box(t, cell_x, cell_y, stride, anchor):
    """Сырой выход сети (tx, ty, tw, th) -> бокс (x1, y1, x2, y2) в пикселях.

    cx = (sigmoid(tx) + cell_x) * stride,  w = anchor_w * exp(tw)

    decode_box((0.0, 0.0, 0.0, 0.0), 3, 4, 32, (30, 60))
        ->  центр (112.0, 144.0), размер ровно (30, 60)

    sigmoid держит центр внутри своей ячейки, exp даёт положительную ширину
    при любом tw и позволяет масштабировать её от анкера в обе стороны.
    Именно поэтому сеть регрессирует логарифм отношения, а не саму ширину:
    у логарифма симметричная шкала, «в два раза меньше» стоит столько же,
    сколько «в два раза больше».

    Один и тот же decode стоит в каждой версии YOLO начиная с v2.
    """
    tx, ty, tw, th = t
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor[0] * math.exp(tw)
    h = anchor[1] * math.exp(th)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def encode_box(box, cell_x, cell_y, stride, anchor):
    """Бокс в пикселях -> цель обучения (tx, ty, tw, th). Обратная к decode_box.

    encode_box((97, 114, 127, 174), 3, 4, 32, (30, 60))  ->  примерно нули

    Обратная к sigmoid — логит: log(p / (1 - p)). Без него encode и decode
    не будут обратными друг другу, и сеть станет учиться на целях, которые
    после decode дают не тот бокс. Проверяется одним прогоном туда-обратно.

    Смещение центра внутри ячейки может выйти ровно 0 или 1 — на границе
    логит даёт бесконечность. Поджимай значение маленьким eps.

    Ширина кодируется как log(w / anchor_w): деление, а не вычитание,
    потому что decode делает exp и умножение.
    """
    eps = 1e-9
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    w = box[2] - box[0]
    h = box[3] - box[1]
    # смещение внутри ячейки, поджатое так, чтобы логит остался конечным
    px = min(max(cx / stride - cell_x, eps), 1.0 - eps)
    py = min(max(cy / stride - cell_y, eps), 1.0 - eps)
    tx = math.log(px / (1.0 - px))
    ty = math.log(py / (1.0 - py))
    tw = math.log(w / anchor[0])
    th = math.log(h / anchor[1])
    return (tx, ty, tw, th)


def best_anchor(box_wh, anchors):
    """Индекс анкера, чья ФОРМА ближе всего к боксу. Совмещение по центрам.

    box_wh — (ширина, высота) объекта, anchors — список таких же пар.

    best_anchor((32, 64), [(30, 60), (200, 380)])  ->  0
    best_anchor((190, 400), [(30, 60), (200, 380)])  ->  1

    Позиция не участвует: анкер — это только форма, а за положение отвечает
    ячейка сетки. Поэтому IoU считается для боксов, приложенных к одному
    центру, и сводится к min(w, aw) * min(h, ah) в числителе.

    Это назначение из YOLOv2/v3. Версии с v5 уточняют его (task-aligned
    matching, динамический k), но идея «выбери ближайший по форме приор и
    предскажи поправку» та же.
    """
    bw, bh = box_wh
    best_i, best_v = 0, -1.0
    for i, (aw, ah) in enumerate(anchors):
        inter = min(bw, aw) * min(bh, ah)
        union = bw * bh + aw * ah - inter
        v = inter / union if union > 0 else 0.0
        if v > best_v:  # строгое сравнение: при равенстве побеждает первый
            best_i, best_v = i, v
    return best_i


def yolo_loss(
    pred,
    target,
    has_obj,
    lambda_coord=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_cls=1.0,
):
    """Три слагаемых лосса YOLO. Возвращает словарь с частями и суммой.

    pred и target — списки слотов (ячейка x анкер), каждый слот это список
    [tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]. У pred числа сырые
    (логиты), у target — готовые значения: obj это 0 или 1, классы one-hot.
    has_obj — список флагов той же длины.

    Ключи результата: "box", "obj_pos", "obj_neg", "cls", "total".

    Правило, ради которого всё и пишется: слоты БЕЗ объекта дают вклад
    только в objectness. Ни координаты, ни классы пустых ячеек в лосс не
    входят вообще — их «правильное» значение не определено, и штраф за них
    учил бы сеть ерунде.

    box — сумма квадратов ошибок, objectness и классы — binary cross entropy
    от логитов. Устойчивая формула BCE (иначе exp(-1000) переполнится):

        bce(z, y) = max(z, 0) - z * y + log(1 + exp(-|z|))

    total = lambda_coord*box + lambda_obj*obj_pos
          + lambda_noobj*obj_neg + lambda_cls*cls

    lambda_noobj маленькая (0.5), потому что пустых слотов в сотни раз
    больше, чем занятых, и без этого веса лосс превратится в одно длинное
    «здесь ничего нет».
    """

    def bce(z, y):
        # log-sum-exp трюк: ветка exp(-|z|) не переполняется ни при каком z
        return max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))

    box = obj_pos = obj_neg = cls = 0.0
    for p, t, flag in zip(pred, target, has_obj):
        if flag:
            box += sum((a - b) ** 2 for a, b in zip(p[:4], t[:4]))
            obj_pos += bce(p[4], t[4])
            cls += sum(bce(a, b) for a, b in zip(p[5:], t[5:]))
        else:
            obj_neg += bce(p[4], t[4])
    total = (
        lambda_coord * box
        + lambda_obj * obj_pos
        + lambda_noobj * obj_neg
        + lambda_cls * cls
    )
    return {
        "box": box,
        "obj_pos": obj_pos,
        "obj_neg": obj_neg,
        "cls": cls,
        "total": total,
    }


def precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Precision и recall детектора. Словарь с "tp", "fp", "fn", и обеими метриками.

    pred_boxes уже отсортированы по убыванию уверенности. Каждый настоящий
    объект можно засчитать только ОДИН раз: второе предсказание того же
    объекта — это false positive, а не второй true positive.

    precision_recall([(0,0,2,2)], [(0,0,2,2)])
        ->  precision 1.0, recall 1.0
    precision_recall([(0,0,2,2), (0,0,2,2)], [(0,0,2,2)])
        ->  precision 0.5, recall 1.0

    Жадное сопоставление: идём по предсказаниям сверху вниз, каждое
    забирает свободный ground truth с наибольшим IoU, если тот дотягивает
    до порога.

    При пустом знаменателе возвращай 0.0, а не деление на ноль: детектор,
    который не выдал ни одного бокса, встречается чаще, чем хотелось бы.

    Именно правило «один объект — одно попадание» делает NMS обязательным,
    а не косметическим.
    """
    matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pb in pred_boxes:
        best_i, best_v = -1, 0.0
        for i, gb in enumerate(gt_boxes):
            if matched[i]:  # объект уже занят более уверенным предсказанием
                continue
            v = iou(pb, gb)
            if v > best_v:
                best_i, best_v = i, v
        if best_i >= 0 and best_v >= iou_threshold:
            matched[best_i] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
