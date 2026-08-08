"""
SAM 3 и open-vocabulary сегментация — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re


def split_concepts(sentence):
    """Разобрать фразу пользователя на короткие concept-промпты для SAM 3.

    Разделители: запятая, точка с запятой, амперсанд и ОТДЕЛЬНЫЕ слова
    "and" и "or". Куски обрезаются по краям, пустые выбрасываются.

    split_concepts("cats, dogs and balloons")  ->  ["cats", "dogs", "balloons"]
    split_concepts("yellow school bus")        ->  ["yellow school bus"]
    split_concepts("sandwich")                 ->  ["sandwich"]

    Ловушка в последнем примере: "sandwich" содержит подстроку "and". Наивная
    проверка `if "and" in sentence` разрежет слово пополам и пошлёт в модель
    "s" и "wich". Разделителем считается только отдельное СЛОВО.

    Аналог: SAM 3 берёт один concept за один forward pass, готового
    сплиттера в API нет — эта функция и есть та граница, где текст
    пользователя превращается в список запросов к модели.
    """
    # \b по краям: "and"/"or" режут только как отдельные слова, а не как
    # подстроки внутри sandwich / orange / brand
    parts = re.split(r"[,;&]|\band\b|\bor\b", sentence)
    return [p.strip() for p in parts if p.strip()]


def rle_encode(mask):
    """Сжать бинарную маску в run-length строку вида "0x100;1x50".

    Маска — список строк из нулей и единиц, обход построчно. Каждый run
    записывается как "значениеxдлина", runs разделены точкой с запятой.

    rle_encode([[0, 0, 1]])          ->  "0x2;1x1"
    rle_encode([[1, 1], [1, 1]])     ->  "1x4"

    Пустая маска сериализуется в пустую строку, как в коде урока. Значения
    кроме 0 и 1 дают ValueError.

    Зачем: SAM 3 возвращает маски в полном разрешении, и на сотне инстансов
    ответ сервиса раздувается до мегабайт. RLE сжимает его в килобайты, и
    формат одинаков у SAM 2, SAM 3 и Grounded SAM 2.
    """
    flat = [value for row in mask for value in row]
    if not flat:
        return ""
    if any(value not in (0, 1) for value in flat):
        raise ValueError("mask must contain only 0 and 1")

    runs = []
    prev, count = flat[0], 0
    for value in flat:
        if value == prev:
            count += 1
        else:
            runs.append((prev, count))
            prev, count = value, 1
    runs.append((prev, count))          # последний run закрывается вручную
    return ";".join(f"{value}x{length}" for value, length in runs)


def rle_decode(rle, width):
    """Развернуть RLE обратно в двумерную маску заданной ширины.

    rle_decode("0x2;1x1", 3)  ->  [[0, 0, 1]]
    rle_decode("1x4", 2)      ->  [[1, 1], [1, 1]]

    Если суммарная длина не делится на width — ValueError: значит RLE и
    ширина от разных масок, и молча дорисовывать хвост нельзя.

    Точная обратная к rle_encode: rle_decode(rle_encode(m), len(m[0])) == m.
    """
    if width < 1:
        raise ValueError("width must be positive")

    flat = []
    for chunk in rle.split(";"):
        value, length = chunk.split("x")
        flat.extend([int(value)] * int(length))
    if len(flat) % width:
        raise ValueError("decoded length is not divisible by width")
    # нарезка на строки срезами: собирать построчно в цикле дольше и
    # ничего не добавляет к читаемости
    return [flat[i : i + width] for i in range(0, len(flat), width)]


def mask_area(rle):
    """Площадь маски (число единиц) прямо из RLE, без разворачивания.

    mask_area("0x2;1x1")          ->  1
    mask_area("1x4;0x10;1x6")     ->  10

    Ловушка: соблазн сделать sum(sum(row) for row in rle_decode(...)). Это
    даст тот же ответ, но развернёт мегабайтную маску ради одного числа.
    Длины уже лежат в строке — их достаточно сложить.

    Зачем: площадь нужна на каждом шаге пост-обработки — отсеять мусорные
    инстансы в пару пикселей, отсортировать по величине, посчитать покрытие.
    """
    total = 0
    for chunk in rle.split(";"):
        value, length = chunk.split("x")
        if int(value) == 1:
            total += int(length)
    return total


def mask_to_box(mask):
    """Габаритный прямоугольник маски: (x1, y1, x2, y2), границы включительно.

    mask_to_box([[0, 0], [0, 1]])  ->  (1, 1, 1, 1)
    mask_to_box([[0, 0], [0, 0]])  ->  None

    Пустая маска даёт None, а не (0, 0, 0, 0): нулевой бокс в левом верхнем
    углу неотличим от настоящего объекта размером в один пиксель.

    Зачем: SAM 3 отдаёт и маски, и боксы, но после любой пост-обработки
    (склейки, эрозии, фильтрации) бокс приходится пересчитывать самому.
    """
    xs, ys = [], []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def mask_iou(mask_a, mask_b):
    """IoU двух бинарных масок: пересечение делить на объединение.

    mask_iou([[1, 0]], [[1, 0]])  ->  1.0
    mask_iou([[1, 0]], [[0, 1]])  ->  0.0

    Две пустые маски дают 0.0, а не деление на ноль и не 1.0: «оба ничего
    не нашли» — это не совпадение предсказания с разметкой.

    Разные размеры — ValueError. Сравнивать маску 32x32 с маской 64x64
    бессмысленно, а zip молча обрежет по короткой и покажет красивое число.

    Зачем: это метрика качества сегментации. Дообучил SAM 3 на своих
    классах — меряешь прирост именно в mask IoU.
    """
    if len(mask_a) != len(mask_b):
        raise ValueError("masks must have the same height")
    intersection = 0
    union = 0
    for row_a, row_b in zip(mask_a, mask_b):
        if len(row_a) != len(row_b):
            raise ValueError("masks must have the same width")
        for a, b in zip(row_a, row_b):
            if a and b:
                intersection += 1
            if a or b:
                union += 1
    if union == 0:
        return 0.0
    return intersection / union


def presence_gate(detections, presence_score, threshold=0.5):
    """Presence head: если концепта нет на картинке, выкинуть ВСЕ детекции.

    presence_gate([{"score": 0.99}], 0.9)  ->  [{"score": 0.99}]
    presence_gate([{"score": 0.99}], 0.1)  ->  []

    Ловушка: порог применяется к presence_score, а НЕ к score отдельных
    инстансов. Уверенная детекция с score 0.99 обязана исчезнуть, если
    presence head сказал «этого объекта тут нет». В этом и весь смысл
    развязки «есть ли оно?» и «где оно?»: локализатор всегда найдёт что-то
    похожее, а presence head режет ложные срабатывания целиком.

    Список возвращается новый — входной трогать нельзя.
    """
    if presence_score < threshold:
        return []
    return list(detections)


def merge_concept_results(per_concept):
    """Склеить результаты нескольких concept-промптов в один список детекций.

    Вход: словарь concept -> список детекций (словари с полями box, score,
    mask_rle). На выходе плоский список копий, у каждой проставлены
    "concept" и "instance_id". instance_id уникален ВНУТРИ концепта и
    выдаётся по убыванию score, начиная с 0. Итоговый список отсортирован
    по убыванию score, ничьи разводятся по имени концепта.

    merge_concept_results({"cat": [{"score": 0.4}, {"score": 0.9}]})
        ->  [{"score": 0.9, "concept": "cat", "instance_id": 0},
             {"score": 0.4, "concept": "cat", "instance_id": 1}]
    merge_concept_results({})  ->  []

    Ловушка: instance_id нумеруется заново для каждого концепта. Сквозная
    нумерация выглядит аккуратнее, но ломает трекинг: id инстанса должен
    отвечать на вопрос «какая это по счёту кошка», а не «какая это по счёту
    строка в ответе».

    Входные словари не мутируются — кладём копии.

    Зачем: SAM 3 принимает один концепт за проход. Мульти-концептный запрос
    это цикл, и склейка его результатов — работа вызывающей стороны.
    """
    merged = []
    for concept, detections in per_concept.items():
        # сортируем внутри концепта по убыванию score: instance_id 0 должен
        # достаться самой уверенной детекции, а не первой попавшейся
        ordered = sorted(detections, key=lambda d: -d["score"])
        for instance_id, detection in enumerate(ordered):
            copy = dict(detection)
            copy["concept"] = concept
            copy["instance_id"] = instance_id
            merged.append(copy)
    # ничья по score разводится именем концепта и id, иначе порядок
    # зависел бы от порядка ключей словаря и тесты бы плавали
    merged.sort(key=lambda d: (-d["score"], d["concept"], d["instance_id"]))
    return merged
