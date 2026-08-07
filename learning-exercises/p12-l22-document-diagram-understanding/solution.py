"""
Понимание документов и диаграмм — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Стеки из «рецепта 2026» урока.
STACKS = (
    "ocr-pipeline",          # чистая печать в огромном объёме, дёшево за страницу
    "nougat+vlm",            # научные статьи: формулы через Nougat, картинки через VLM
    "vlm-native",            # смесь печати, рукописи и форм
    "ocr-pipeline+crosscheck",  # регуляторка: OCR как источник истины, VLM как проверка
)

# Признаки проекта, которые понимает pick_stack. Всё остальное — опечатка.
PROFILE_KEYS = frozenset({"pages_per_day", "handwriting", "math", "regulated"})


def normalize_bbox(bbox, page_w, page_h, scale=1000):
    """Перевод рамки в систему координат LayoutLM: целые 0..scale.

    bbox — (x0, y0, x1, y1) в пикселях страницы.

    normalize_bbox((100, 50, 300, 80), 1000, 1000)  ->  (100, 50, 300, 80)
    normalize_bbox((100, 50, 300, 80), 2000, 1000)  ->  (50, 50, 150, 80)

    Зачем нормализация: LayoutLM учит эмбеддинг на каждую из 1001 позиций
    по каждой оси. Скан 300 DPI и скан 150 DPI одной и той же страницы
    обязаны дать ОДИН И ТОТ ЖЕ поток bbox, иначе модель придётся учить
    заново под каждое разрешение.

    Ловушка: результат надо обрезать в [0, scale]. OCR регулярно выдаёт
    рамки, чуть вылезающие за край страницы, а эмбеддинга на позицию 1004
    в модели нет.
    """
    x0, y0, x1, y1 = bbox
    # деление на размер страницы, а не на константу: именно оно и делает
    # координаты независимыми от DPI
    out = (
        int(round(x0 / page_w * scale)),
        int(round(y0 / page_h * scale)),
        int(round(x1 / page_w * scale)),
        int(round(y1 / page_h * scale)),
    )
    return tuple(min(scale, max(0, v)) for v in out)


def iou(box_a, box_b):
    """Intersection over Union двух рамок: от 0.0 (не пересекаются) до 1.0.

    iou((0, 0, 2, 2), (0, 0, 2, 2))  ->  1.0
    iou((0, 0, 2, 2), (1, 0, 3, 2))  ->  0.333...
    iou((0, 0, 1, 1), (5, 5, 6, 6))  ->  0.0

    Нужна для гибридной схемы из урока: OCR-пайплайн и VLM выдали свои
    рамки полей, и надо понять, говорят ли они об одном и том же поле.

    Ловушка: ширина пересечения может выйти отрицательной. Отрицательную
    площадь надо занулить, иначе непересекающиеся рамки дадут
    положительный IoU из двух минусов.
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    # max(0, ...) на каждой оси отдельно — иначе два отрицательных
    # перекрытия перемножатся в положительную площадь
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    if union <= 0:
        return 0.0
    return inter / union


def reading_order(tokens, line_height):
    """Разложить токены страницы в порядке чтения: сверху вниз, слева направо.

    Токен — кортеж (text, (x0, y0, x1, y1)). Возвращает НОВЫЙ список токенов.

    reading_order([("b", (300, 10, 350, 40)), ("a", (100, 10, 150, 40))], 50)
        ->  [("a", ...), ("b", ...)]

    Строка определяется полосой высотой line_height: токены с y0 в одной
    полосе считаются одной строкой и сортируются по x0.

    Ловушка: сортировать просто по (y0, x0) нельзя. Базовые линии соседних
    слов почти всегда расходятся на пару пикселей, и «Total» окажется
    строкой ниже, чем «$1,245» справа от него.
    """
    # сортировка по кортежу (номер полосы, x0): один проход, O(n log n),
    # и вход остаётся нетронутым, потому что sorted возвращает новый список
    return sorted(tokens, key=lambda t: (t[1][1] // line_height, t[1][0]))


def layoutlm_input(tokens, page_w, page_h, patch_grid=(16, 16)):
    """Три входных потока LayoutLMv3: текст, нормализованные рамки, патчи.

    Возвращает словарь с ключами "text", "bbox", "n_patches".

    layoutlm_input([("Total", (400, 400, 500, 430))], 1000, 1000)
        ->  {"text": ["Total"], "bbox": [(400, 400, 500, 430)], "n_patches": 256}

    Ровно эти три потока и маскируются совместно при обучении: маскируют
    слово, маскируют патч, маскируют координату. Отсюда и берётся
    понимание, что «Total» внизу справа — это итог, а не сноска.

    Патчи от текста не зависят: их всегда patch_grid[0] * patch_grid[1],
    даже если OCR не нашёл ни одного слова.
    """
    return {
        "text": [text for text, _ in tokens],
        "bbox": [normalize_bbox(box, page_w, page_h) for _, box in tokens],
        "n_patches": patch_grid[0] * patch_grid[1],
    }


def donut_serialize(record):
    """Разметка Donut: словарь полей -> строка тегов, которую модель генерирует.

    donut_serialize({"total": "1245"})  ->  "<s_total>1245</s_total>"
    donut_serialize({})                 ->  ""

    Формат ровно такой: <s_KEY>VALUE</s_KEY>, поля идут подряд без
    разделителей, в порядке ключей словаря.

    Смысл: Donut не делает OCR и не возвращает координат. Он сразу
    генерирует целевую структуру — то есть учится решать задачу, а не
    восстанавливать текст.
    """
    return "".join(f"<s_{k}>{v}</s_{k}>" for k, v in record.items())


def donut_parse(markup):
    """Разбор разметки Donut обратно в словарь. Обратная к donut_serialize.

    donut_parse("<s_total>1245</s_total>")  ->  {"total": "1245"}
    donut_parse("")                         ->  {}

    Порядок полей сохраняется. На сломанной разметке (нет закрывающего
    тега, теги вложены не так) — ValueError: в проде это сигнал, что модель
    сорвалась в галлюцинацию, и тихо вернуть половину полей нельзя.

    Значения могут содержать что угодно, кроме символа "<".
    """
    out = {}
    pos = 0
    while pos < len(markup):
        if markup[pos] != "<":
            raise ValueError(f"expected a tag at position {pos}")
        end = markup.find(">", pos)
        if end == -1 or not markup.startswith("<s_", pos):
            raise ValueError(f"broken opening tag at position {pos}")
        key = markup[pos + 3:end]
        closing = f"</s_{key}>"
        close_at = markup.find(closing, end + 1)
        if close_at == -1:
            raise ValueError(f"no closing tag for {key!r}")
        out[key] = markup[end + 1:close_at]
        pos = close_at + len(closing)
    return out


def anyres_tokens(width, height, tile_px=336, tokens_per_tile=576, thumbnail=True):
    """Сколько визуальных токенов стоит страница при AnyRes-тайлинге.

    anyres_tokens(336, 336)   ->  1152   (один тайл + миниатюра)
    anyres_tokens(337, 336)   ->  1728   (влез второй тайл — целиком)
    anyres_tokens(672, 672)   ->  2880

    Страница режется на тайлы tile_px x tile_px, каждый стоит
    tokens_per_tile токенов, плюс одна миниатюра всей страницы, чтобы
    модель видела общий макет.

    Ловушка: число тайлов округляется ВВЕРХ. Лишний пиксель по ширине
    стоит целого тайла — поэтому ресайз страницы под кратный размер
    экономит реальные деньги.

    Отсюда и берётся цена страницы: A4 при 300 DPI это ~2500x3500, то есть
    8 x 11 тайлов, почти 52 тысячи токенов на страницу.
    """
    tiles = math.ceil(width / tile_px) * math.ceil(height / tile_px)
    return tiles * tokens_per_tile + (tokens_per_tile if thumbnail else 0)


def pick_stack(profile):
    """Выбор стека document AI по профилю проекта. Возвращает элемент STACKS.

    pick_stack({"regulated": True})              ->  "ocr-pipeline+crosscheck"
    pick_stack({"math": True})                   ->  "nougat+vlm"
    pick_stack({"pages_per_day": 10_000_000})    ->  "ocr-pipeline"
    pick_stack({})                               ->  "vlm-native"

    Приоритет правил из урока, сверху вниз:
      1. regulated — нужен воспроизводимый OCR как источник истины;
      2. math      — Nougat учили на LaTeX, VLM без такой цели врёт формулы;
      3. handwriting — тут выигрывает VLM-native;
      4. от миллиона страниц в сутки чистой печати — OCR-пайплайн дешевле;
      5. иначе VLM-native.

    Неизвестный ключ профиля — ValueError. Опечатка "regulatd" молча
    выберет vlm-native и уронит аудит.
    """
    unknown = set(profile) - PROFILE_KEYS
    if unknown:
        raise ValueError(f"unknown profile keys: {sorted(unknown)}")
    if profile.get("regulated"):
        return "ocr-pipeline+crosscheck"
    if profile.get("math"):
        return "nougat+vlm"
    if profile.get("handwriting"):
        return "vlm-native"
    if profile.get("pages_per_day", 0) >= 1_000_000:
        return "ocr-pipeline"
    return "vlm-native"
