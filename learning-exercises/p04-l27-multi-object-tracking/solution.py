"""
Многообъектный трекинг и память видео — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import itertools


def iou(a, b):
    """IoU двух боксов: площадь пересечения делить на площадь объединения.

    Бокс — кортеж (x1, y1, x2, y2), левый верхний и правый нижний углы.

    iou((0, 0, 2, 2), (0, 0, 2, 2))  ->  1.0
    iou((0, 0, 2, 2), (5, 5, 7, 7))  ->  0.0
    iou((0, 0, 2, 2), (1, 0, 3, 2))  ->  0.3333333333333333

    Ловушка: у непересекающихся боксов ширина пересечения ОТРИЦАТЕЛЬНА. Если
    не обрезать её нулём, получится «отрицательная площадь» и IoU больше
    единицы. Прижимай ширину и высоту к нулю до умножения.

    Соответствует torchvision.ops.box_iou для одной пары.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    # max(0, ...) до умножения: два отрицательных множителя дали бы
    # положительную «площадь» у боксов, которые вообще не пересекаются
    inter = max(0.0, inter_w) * max(0.0, inter_h)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_matrix(a_boxes, b_boxes):
    """Матрица IoU: строки — боксы из a_boxes, столбцы — из b_boxes.

    iou_matrix([(0, 0, 2, 2)], [(0, 0, 2, 2), (5, 5, 7, 7)])  ->  [[1.0, 0.0]]
    iou_matrix([], [(0, 0, 1, 1)])                            ->  []

    В трекинге строки это обычно треки (где объект был), столбцы — детекции
    (что нашёл детектор на новом кадре).
    """
    return [[iou(a, b) for b in b_boxes] for a in a_boxes]


def optimal_assignment(cost):
    """Венгерская задача: назначить строки столбцам с минимальной суммой цен.

    cost — прямоугольная матрица M x N. Вернуть список пар (строка, столбец),
    отсортированный по строке, длиной min(M, N). Каждая строка и каждый столбец
    встречаются не больше одного раза.

    optimal_assignment([[1.0, 2.0], [2.0, 1.0]])  ->  [(0, 0), (1, 1)]
    optimal_assignment([[1.0, 2.0], [1.0, 9.0]])  ->  [(0, 1), (1, 0)]
    optimal_assignment([])                        ->  []

    Второй пример — вся суть. Жадный алгоритм схватит самую дешёвую клетку
    (0,0) = 1 и останется с 9, итого 10. Оптимум берёт 2 + 1 = 3.

    Настоящая венгерка работает за O(n^3), тут хватит честного перебора
    перестановок. Чтобы перебор не взорвался, при max(M, N) > 8 бросай
    ValueError — это ограничение упражнения, а не алгоритма.

    Соответствует scipy.optimize.linear_sum_assignment.
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []
    if max(rows, cols) > 8:
        raise ValueError("brute-force assignment is limited to 8x8, use a real Hungarian solver")
    if rows <= cols:
        # перебираем, какие столбцы достаются строкам 0..rows-1
        best = min(itertools.permutations(range(cols), rows),
                   key=lambda perm: sum(cost[r][c] for r, c in enumerate(perm)))
        return [(r, c) for r, c in enumerate(best)]
    # строк больше — перебираем, какие строки достаются столбцам, и разворачиваем
    best = min(itertools.permutations(range(rows), cols),
               key=lambda perm: sum(cost[r][c] for c, r in enumerate(perm)))
    return sorted((r, c) for c, r in enumerate(best))


def associate(track_boxes, det_boxes, iou_threshold=0.3):
    """Сопоставить треки и детекции по IoU. Ядро tracking-by-detection.

    Цена пары = 1 - IoU. Решаем задачу назначения, потом выбрасываем пары, у
    которых IoU оказался ниже порога — оптимум обязан кого-то с кем-то связать,
    но связь с IoU=0 это не совпадение, а случайность.

    Вернуть кортеж (matches, unmatched_tracks, unmatched_dets), где matches —
    список пар (индекс трека, индекс детекции), а два других — списки индексов.

    associate([(0, 0, 2, 2)], [(0, 0, 2, 2)])              ->  ([(0, 0)], [], [])
    associate([(0, 0, 2, 2)], [(9, 9, 11, 11)])            ->  ([], [0], [0])
    associate([], [(0, 0, 2, 2)])                          ->  ([], [], [0])

    Ловушка: пустой список треков или детекций. Матрица нулевого размера ломает
    перебор — обработай этот случай до него.
    """
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    ious = iou_matrix(track_boxes, det_boxes)
    cost = [[1.0 - value for value in row] for row in ious]
    matches = [(r, c) for r, c in optimal_assignment(cost) if ious[r][c] >= iou_threshold]
    matched_tracks = {r for r, _ in matches}
    matched_dets = {c for _, c in matches}
    return (matches,
            [i for i in range(len(track_boxes)) if i not in matched_tracks],
            [j for j in range(len(det_boxes)) if j not in matched_dets])


def update_tracks(tracks, detections, frame, next_id, iou_threshold=0.3, max_age=5):
    """Один шаг трекера: обновить, родить, состарить. Вернуть (треки, next_id).

    Трек — словарь {"id": int, "bbox": кортеж, "last_frame": int, "hits": int}.

    Что делаем:
      * совпавшие треки получают новый bbox, last_frame = frame, hits + 1;
      * каждая несовпавшая детекция рождает трек с id = next_id, next_id + 1;
      * трек, у которого frame - last_frame > max_age, удаляется.

    update_tracks([], [(0, 0, 2, 2)], 0, 1)
        ->  ([{"id": 1, "bbox": (0, 0, 2, 2), "last_frame": 0, "hits": 1}], 2)

    Ключевое свойство: трек, пропустивший кадр-другой, id НЕ меняет — он просто
    стареет и ждёт свою детекцию. Ради этого max_age и существует.

    Ловушка: не мутируй входной список tracks и его словари. Трекер часто
    гоняют на нескольких гипотезах параллельно, общий изменяемый стейт всё
    испортит.
    """
    working = [dict(t) for t in tracks]
    matches, _, unmatched_dets = associate([t["bbox"] for t in working],
                                           list(detections), iou_threshold)
    for track_idx, det_idx in matches:
        working[track_idx]["bbox"] = detections[det_idx]
        working[track_idx]["last_frame"] = frame
        working[track_idx]["hits"] += 1
    for det_idx in unmatched_dets:
        working.append({"id": next_id, "bbox": detections[det_idx],
                        "last_frame": frame, "hits": 1})
        next_id += 1
    # старение в самом конце: только что рождённые треки удалиться не могут
    alive = [t for t in working if frame - t["last_frame"] <= max_age]
    return alive, next_id


def run_tracker(frames, iou_threshold=0.3, max_age=5):
    """Прогнать трекер по всему видео. Вернуть по списку (id, bbox) на кадр.

    frames — список кадров, кадр — список боксов-детекций.
    Внутри каждого кадра результат отсортирован по id.

    run_tracker([[(0, 0, 2, 2)], [(0, 0, 2, 2)]])
        ->  [[(1, (0, 0, 2, 2))], [(1, (0, 0, 2, 2))]]

    Три объекта, едущие по прямым, обязаны сохранить свои id на всех кадрах.
    Именно так и проверяют трекер: не «красиво ли выглядит», а «сколько раз
    id перескочил».
    """
    tracks, next_id = [], 1
    out = []
    for frame, detections in enumerate(frames):
        tracks, next_id = update_tracks(tracks, detections, frame, next_id,
                                        iou_threshold, max_age)
        out.append(sorted((t["id"], t["bbox"]) for t in tracks))
    return out


def count_id_switches(tracks_per_frame, gt_per_frame, iou_threshold=0.5):
    """Сколько раз объект из ground truth сменил присвоенный ему id трека.

    tracks_per_frame — по кадрам список пар (track_id, bbox), предсказание.
    gt_per_frame     — по кадрам список пар (gt_id, bbox), истина.

    На каждом кадре для каждого gt-объекта ищем трек с максимальным IoU. Если
    IoU выше порога, запоминаем назначение. Если у этого gt-объекта РАНЬШЕ был
    другой track_id — засчитываем переключение.

    count_id_switches([[(1, (0, 0, 2, 2))], [(1, (0, 0, 2, 2))]],
                      [[(7, (0, 0, 2, 2))], [(7, (0, 0, 2, 2))]])  ->  0

    Упрощённый родственник IDF1: настоящие MOTA/IDF1/HOTA живут в
    py-motmetrics и TrackEval.

    Ловушка: пропуск кадра переключением НЕ является. Если на кадре объекта не
    нашлось, прошлое назначение просто сохраняется до следующей встречи.
    """
    previous = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        ious = iou_matrix([box for _, box in gts], [box for _, box in tracks])
        for g_idx, (gt_id, _) in enumerate(gts):
            row = ious[g_idx]
            best = max(range(len(row)), key=lambda j: row[j])
            if row[best] > iou_threshold:
                track_id = tracks[best][0]
                if gt_id in previous and previous[gt_id] != track_id:
                    switches += 1
                previous[gt_id] = track_id
    return switches


def mota(num_fn, num_fp, num_switches, num_gt):
    """MOTA: 1 - (FN + FP + переключения) / число объектов в ground truth.

    mota(0, 0, 0, 100)   ->  1.0
    mota(10, 0, 0, 100)  ->  0.9
    mota(50, 60, 10, 100) -> -0.2

    Обрати внимание: MOTA бывает ОТРИЦАТЕЛЬНОЙ. Это не ошибка — если ложных
    срабатываний больше, чем объектов, метрика честно уходит в минус.

    Все три ошибки входят с одинаковым весом, поэтому MOTA смешивает качество
    детекции с качеством ассоциации. Когда важны именно id, смотрят IDF1, а для
    общего сравнения — HOTA, которая разложена на DetA и AssA.

    Ловушка: num_gt = 0 — делить не на что, брось ValueError.
    """
    if num_gt <= 0:
        raise ValueError("MOTA is undefined without ground truth objects")
    return 1.0 - (num_fn + num_fp + num_switches) / num_gt
