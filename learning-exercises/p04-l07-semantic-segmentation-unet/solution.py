"""
Семантическая сегментация и U-Net — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(logits):
    """Softmax по классам для ОДНОГО пикселя: логиты -> вероятности.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([2.0, 1.0, 0.0])  ->  [0.6652..., 0.2447..., 0.0900...]

    Ловушка: наивный math.exp(1000) падает с OverflowError. Вычти максимум
    из всех логитов перед exp — отношение экспонент от этого не меняется,
    а переполнения не будет.

    В сегментации softmax применяется в каждой из H*W позиций отдельно:
    выход сети формы (C, H, W) превращается в C карт вероятностей.
    """
    # сдвиг на максимум — единственный способ пережить логиты порядка 1000,
    # а они появляются, как только сеть становится уверенной
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def pixel_accuracy(preds, targets):
    """Доля пикселей, класс которых угадан. preds и targets — сетки H x W из int.

    pixel_accuracy([[0, 1], [1, 1]], [[0, 1], [1, 1]])  ->  1.0
    pixel_accuracy([[0, 0], [0, 0]], [[0, 0], [0, 1]])  ->  0.75

    Метрика дешёвая и обманчивая: если 99% кадра — фон, предсказание
    «везде фон» даёт 0.99 и нулевую пользу. Ради этого и существует IoU.
    """
    hit = 0
    total = 0
    for row_p, row_t in zip(preds, targets):
        for p, t in zip(row_p, row_t):
            total += 1
            if p == t:
                hit += 1
    return hit / total


def pixel_cross_entropy(logits, targets):
    """Кросс-энтропия, усреднённая по всем пикселям.

    logits  — сетка H x W, в каждой позиции список из C логитов: logits[h][w][c].
    targets — сетка H x W из целых номеров классов: targets[h][w].

    pixel_cross_entropy([[[0.0, 0.0]]], [[0]])        ->  0.6931...  (= -log 0.5)
    pixel_cross_entropy([[[100.0, 0.0]]], [[0]])      ->  0.0        (угадал уверенно)

    Формула в каждой позиции: -log( softmax(logits[h][w])[ targets[h][w] ] ).
    Это ровно то, что делает F.cross_entropy на тензорах (N, C, H, W) и
    (N, H, W) — никаких reshape ему не нужно.

    Ловушка: log(0). Softmax никогда не вернёт математический ноль, но при
    очень уверенном ошибочном предсказании вернёт денормализованный ноль.
    Подстрахуйся нижней границей вроде 1e-12.
    """
    total = 0.0
    count = 0
    for row_logits, row_targets in zip(logits, targets):
        for pixel_logits, cls in zip(row_logits, row_targets):
            p = softmax(pixel_logits)[cls]
            total += -math.log(max(p, 1e-12))
            count += 1
    return total / count


def dice_coefficient(pred_mask, true_mask, eps=1e-6):
    """Коэффициент Дайса двух БИНАРНЫХ масок H x W (значения 0 или 1).

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    dice_coefficient([[1, 1], [0, 0]], [[1, 1], [0, 0]])  ->  1.0
    dice_coefficient([[1, 0], [0, 0]], [[0, 1], [0, 0]])  ->  0.0
    dice_coefficient([[1, 1], [0, 0]], [[1, 0], [0, 0]])  ->  0.666...

    eps нужен ровно для одного случая: обе маски пустые. Без него это
    деление 0/0, с ним ответ 1.0 — «оба согласились, что класса тут нет».
    """
    inter = 0.0
    sum_p = 0.0
    sum_t = 0.0
    for row_p, row_t in zip(pred_mask, true_mask):
        for p, t in zip(row_p, row_t):
            inter += p * t
            sum_p += p
            sum_t += t
    return (2.0 * inter + eps) / (sum_p + sum_t + eps)


def dice_loss(logits, targets, num_classes, eps=1e-6):
    """Dice-лосс по мягким вероятностям, макро-усреднение по классам.

    Формат logits и targets — как в pixel_cross_entropy.

    Для каждого класса c: берём карту вероятностей p_c (это softmax, а не
    argmax — иначе нечего дифференцировать) и бинарную маску истины y_c.
      dice_c = (2 * sum(p_c * y_c) + eps) / (sum(p_c) + sum(y_c) + eps)
    Возвращаем 1 - среднее(dice_c).

    dice_loss([[[100.0, 0.0]]], [[0]], 2)  ->  ~0.0   (идеальное попадание)
    dice_loss([[[0.0, 0.0]]], [[0]], 2)    ->  0.5    (полная неуверенность)

    Зачем это в AI: кросс-энтропия считает каждый пиксель равным, поэтому на
    опухоли в 1% кадра она согласна проиграть. Dice — это отношение, и
    дисбаланс классов на него не влияет.
    """
    # накапливаем три суммы на класс за один проход по сетке
    inter = [0.0] * num_classes
    sum_p = [0.0] * num_classes
    sum_t = [0.0] * num_classes
    for row_logits, row_targets in zip(logits, targets):
        for pixel_logits, cls in zip(row_logits, row_targets):
            probs = softmax(pixel_logits)
            for c in range(num_classes):
                inter[c] += probs[c] * (1.0 if c == cls else 0.0)
                sum_p[c] += probs[c]
            sum_t[cls] += 1.0
    dices = [
        (2.0 * inter[c] + eps) / (sum_p[c] + sum_t[c] + eps) for c in range(num_classes)
    ]
    return 1.0 - sum(dices) / num_classes


def combined_loss(logits, targets, num_classes, lam=1.0):
    """Боевой лосс сегментации: CE + lam * Dice.

    Возвращает кортеж (total, parts), где parts — словарь
    {"ce": <кросс-энтропия>, "dice": <dice-лосс>}.

    total = ce + lam * dice
    lam = 0 выключает Dice и оставляет чистую кросс-энтропию.

    Кросс-энтропия даёт устойчивый градиент в начале обучения, Dice
    дотягивает форму маски в конце. Эта пара — дефолт медицинской
    сегментации и её тяжело побить на любых несбалансированных данных.
    """
    ce = pixel_cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce, "dice": dc}


def iou_per_class(preds, targets, num_classes):
    """IoU для каждого класса. preds и targets — сетки H x W из int.

    IoU(c) = |пиксели, где оба = c| / |пиксели, где хотя бы один = c|

    Класс, которого нет ни в предсказании, ни в истине, даёт union = 0.
    Для него верни None — это НЕ ноль. Ноль означал бы «промахнулись
    полностью», а класса просто не было в кадре.

    iou_per_class([[0, 1]], [[0, 1]], 2)  ->  [1.0, 1.0]
    iou_per_class([[0, 0]], [[0, 1]], 2)  ->  [0.5, 0.0]
    iou_per_class([[0, 0]], [[0, 0]], 3)  ->  [1.0, None, None]
    """
    inter = [0] * num_classes
    union = [0] * num_classes
    for row_p, row_t in zip(preds, targets):
        for p, t in zip(row_p, row_t):
            if p == t:
                inter[p] += 1
                union[p] += 1
            else:
                union[p] += 1
                union[t] += 1
    return [
        (inter[c] / union[c]) if union[c] > 0 else None for c in range(num_classes)
    ]


def mean_iou(ious):
    """mIoU: среднее по классам, которые вообще присутствовали (не None).

    mean_iou([1.0, 0.5, None])  ->  0.75
    mean_iou([None, None])      ->  None

    Ловушка: посчитать None как ноль. Тогда пустой класс утащит метрику
    вниз, и модель будет выглядеть хуже, чем она есть.

    И помни про обратную сторону: mIoU 0.78 может прятать класс на 0.15
    среди девяти классов по 0.85. Отчитывайся IoU по классам, не только mIoU.
    """
    present = [v for v in ious if v is not None]
    if not present:
        return None
    return sum(present) / len(present)
