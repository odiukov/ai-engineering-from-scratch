"""
Полный vision-pipeline: капстоун — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import statistics


def validate_box(box):
    """Проверить контракт бокса и вернуть его как кортеж из четырёх float.

    validate_box([10, 20, 30, 40])   ->  (10.0, 20.0, 30.0, 40.0)
    validate_box((0, 0, 0, 0))       ->  (0.0, 0.0, 0.0, 0.0)  (пустой бокс легален)

    Контракт: ровно четыре числа в порядке (x1, y1, x2, y2), абсолютные
    пиксели, x1 <= x2 и y1 <= y2. Всё остальное -> ValueError.

    Ради чего это: детекторы отдают боксы в двух несовместимых форматах —
    (x1, y1, x2, y2) и (cx, cy, w, h). Подставив второй вместо первого, ты
    получишь не падение, а молча пустые кропы и классификации на пустоте.
    Проверка на границе превращает тихую порчу данных в громкую ошибку.
    Это ровно то, что в уроке делает Pydantic.
    """
    if len(box) != 4:
        raise ValueError(f"box must have 4 coordinates, got {len(box)}")
    x1, y1, x2, y2 = (float(v) for v in box)
    # порядок координат — единственное, что отличает (x1,y1,x2,y2) от (cx,cy,w,h)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"box must be (x1, y1, x2, y2) with x1 <= x2, got {tuple(box)}")
    return (x1, y1, x2, y2)


def validate_detection(box, score, class_id):
    """Собрать одну проверенную детекцию: dict с box, score, class_id.

    validate_detection([0, 0, 10, 10], 0.9, 3)
        ->  {"box": (0.0, 0.0, 10.0, 10.0), "score": 0.9, "class_id": 3}

    Контракт: score лежит в [0, 1] включительно, class_id — неотрицательное
    целое. Нарушение -> ValueError.

    Границы включительны специально: детекторы регулярно отдают ровно 1.0
    после softmax, и падать на этом нельзя.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative, got {class_id}")
    return {"box": validate_box(box), "score": float(score), "class_id": int(class_id)}


def clamp_box(box, width, height):
    """Зажать бокс в границы картинки [0, width] x [0, height].

    clamp_box((-5, -5, 50, 50), 100, 100)     ->  (0.0, 0.0, 50.0, 50.0)
    clamp_box((10, 10, 999, 999), 100, 80)    ->  (10.0, 10.0, 100.0, 80.0)
    clamp_box((200, 200, 300, 300), 100, 100) ->  (100.0, 100.0, 100.0, 100.0)

    Детектор не обязан держаться внутри картинки: регрессия боксов свободно
    вылезает за край. Срез numpy/torch по отрицательному индексу не упадёт —
    он молча возьмёт кусок с другой стороны. Поэтому зажимаем до кропа.

    Третий пример показывает нормальный исход для бокса целиком за кадром:
    он схлопывается в нулевую площадь, а не превращается в ошибку.
    """
    x1, y1, x2, y2 = validate_box(box)
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    # зажатие по одной координате не может нарушить x1 <= x2: min/max монотонны
    return (x1, y1, x2, y2)


def is_classifiable(box, min_crop):
    """Достаточно ли бокс велик, чтобы отправлять кроп в классификатор.

    is_classifiable((0, 0, 64, 64), 32)   ->  True
    is_classifiable((0, 0, 64, 8), 32)    ->  False   (низкий, но широкий)
    is_classifiable((0, 0, 32, 32), 32)   ->  True    (граница включительна)

    Оба измерения должны быть не меньше min_crop. Кроп 3x200, растянутый до
    224x224, — это не изображение объекта, а полоса шума; классификатор выдаст
    на нём уверенную чушь. Дешевле пропустить детекцию, чем врать про неё.
    """
    x1, y1, x2, y2 = validate_box(box)
    return (x2 - x1) >= min_crop and (y2 - y1) >= min_crop


def select_crops(boxes, width, height, min_crop):
    """Подготовить боксы к кропу: вернуть (clamped_boxes, valid_indices).

    clamped_boxes — ВСЕ боксы, зажатые в кадр (по одному на вход).
    valid_indices — индексы тех из них, что прошли min_crop, по возрастанию.

    select_crops([(0, 0, 50, 50), (0, 0, 4, 4)], 100, 100, 32)
        ->  ([(0.0, 0.0, 50.0, 50.0), (0.0, 0.0, 4.0, 4.0)], [0])
    select_crops([], 100, 100, 32)  ->  ([], [])

    Почему индексы возвращаются отдельно, а мелкие боксы не выбрасываются:
    детекции уходят в ответ ВСЕ, а в классификатор — только часть. Как только
    два списка разъезжаются по длине, начинается самый живучий баг пайплайна —
    классификация приклеивается не к своей детекции. valid_indices — это мост
    между двумя нумерациями.

    Пустой список детекций — штатный исход, а не ошибка.
    """
    clamped = [clamp_box(b, width, height) for b in boxes]
    valid = [i for i, b in enumerate(clamped) if is_classifiable(b, min_crop)]
    return clamped, valid


def attach_classifications(valid_indices, preds, class_names):
    """Пришить предсказания классификатора к их исходным детекциям.

    preds — список пар (class_id, score) в том же порядке, что valid_indices.

    attach_classifications([0, 2], [(1, 0.8), (0, 0.6)], ["cat", "dog"])
        ->  [{"detection_index": 0, "class_id": 1, "class_name": "dog", "score": 0.8},
             {"detection_index": 2, "class_id": 0, "class_name": "cat", "score": 0.6}]

    Ловушка ровно здесь: длины valid_indices и preds обязаны совпадать. Если
    классификатор вернул больше или меньше, чем ты отправил, соединять их
    по zip нельзя — zip молча обрежет по короткому, и часть детекций получит
    чужой класс. Несовпадение длин -> ValueError.

    class_id вне диапазона class_names -> ValueError: пустая строка в ответе
    хуже, чем отказ.
    """
    if len(valid_indices) != len(preds):
        raise ValueError(
            f"got {len(preds)} predictions for {len(valid_indices)} crops"
        )
    out = []
    for det_index, (class_id, score) in zip(valid_indices, preds):
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"class_id {class_id} is outside the label map")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        out.append(
            {
                "detection_index": int(det_index),
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "score": float(score),
            }
        )
    return out


def build_result(image_id, detections, classifications, inference_ms):
    """Собрать итоговый ответ сервиса и проверить его целостность.

    build_result("demo", [], [], 12.5)
        ->  {"image_id": "demo", "detections": [], "classifications": [],
             "inference_ms": 12.5}

    Проверяется главное свойство ответа: каждый detection_index указывает на
    существующую детекцию. Ссылка за пределы списка -> ValueError.

    Пустой список детекций — это валидный ответ 200, а не ошибка: на картинке
    может не быть ни одного объекта. Отрицательное inference_ms -> ValueError,
    такое значение означает сломанный замер времени.
    """
    if inference_ms < 0:
        raise ValueError(f"inference_ms must be non-negative, got {inference_ms}")
    for c in classifications:
        idx = c["detection_index"]
        if not 0 <= idx < len(detections):
            raise ValueError(
                f"classification points at detection {idx}, "
                f"but there are only {len(detections)} detections"
            )
    return {
        "image_id": str(image_id),
        "detections": list(detections),
        "classifications": list(classifications),
        "inference_ms": float(inference_ms),
    }


def bottleneck_stage(stage_times):
    """Найти самую дорогую стадию пайплайна: вернуть (имя, доля от общего).

    Вход — dict {имя стадии: список замеров в мс}. Стадия сравнивается по
    МЕДИАНЕ, доля считается от суммы медиан всех стадий.

    bottleneck_stage({"preprocess": [3.0, 3.0], "detect": [400.0, 400.0],
                      "classify": [97.0, 97.0]})
        ->  ("detect", 0.8)

    Медиана, а не среднее: один случайный выброс не должен переназначать
    приоритет оптимизации. Пустой вход или пустая стадия -> ValueError.

    Зачем: оптимизировать пайплайн можно только по порядку. На CPU детектор
    обычно съедает 70-90% времени, но после переезда на GPU лидером внезапно
    становится препроцессинг — декодирование JPEG и ресайз никуда не уехали.
    """
    if not stage_times:
        raise ValueError("no stages to compare")
    medians = {}
    for name, times in stage_times.items():
        if not times:
            raise ValueError(f"stage {name!r} has no measurements")
        medians[name] = statistics.median(times)
    total = sum(medians.values())
    if total <= 0:
        raise ValueError("total pipeline time is zero, nothing to profile")
    # max по значению; при равенстве медиан имя выбирается детерминированно
    worst = max(sorted(medians), key=lambda n: medians[n])
    return worst, medians[worst] / total
