"""
Семейство Qwen-VL и видео с динамическим FPS — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json
import math

# FPS, из которых Qwen2.5-VL разрешает выбирать при семплировании видео.
FPS_CHOICES = (1, 2, 4, 8)

# Желаемый FPS по уровню движения в кадре. Теннисный розыгрыш и запись
# экрана — это разные требования к временному разрешению.
MOTION_FPS = {"low": 1, "medium": 2, "high": 8}


def rope_frequencies(dim, base=10000.0):
    """Частоты RoPE: theta_i = base^(-2i/dim) для каждой пары координат.

    rope_frequencies(4)  ->  [1.0, 0.01]      (base^0 и base^-0.5)
    rope_frequencies(2)  ->  [1.0]

    Пар ровно dim // 2: RoPE вращает координаты парами, как точки на
    плоскости. Нечётный dim — ValueError, пару не из чего собрать.

    Низкие i дают быстрые частоты (различают соседние позиции), высокие —
    медленные (различают далёкие). Это и есть «таблицы позиций больше нет».
    """
    if dim < 2 or dim % 2:
        raise ValueError(f"dim должен быть чётным и >= 2, получено {dim}")
    return [base ** (-2.0 * i / dim) for i in range(dim // 2)]


def rotate_pairs(vec, position, freqs):
    """Повернуть каждую пару координат вектора на угол position * theta_i.

    rotate_pairs([1.0, 0.0], 0, [1.0])  ->  [1.0, 0.0]   (нулевая позиция
                                                          ничего не меняет)

    Формула на пару (x, y) с углом a = position * theta_i:
        x' = x * cos(a) - y * sin(a)
        y' = x * sin(a) + y * cos(a)

    Это чистый поворот, поэтому длина вектора не меняется — RoPE не
    масштабирует активации, только крутит их. И главное свойство: скалярное
    произведение повёрнутых q и k зависит только от РАЗНОСТИ позиций.

    Длина vec обязана быть ровно 2 * len(freqs), иначе ValueError.
    """
    if len(vec) != 2 * len(freqs):
        raise ValueError(f"нужно {2 * len(freqs)} координат, дано {len(vec)}")
    out = []
    for i, theta in enumerate(freqs):
        angle = position * theta
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y = vec[2 * i], vec[2 * i + 1]
        out.append(x * cos_a - y * sin_a)
        out.append(x * sin_a + y * cos_a)
    return out


def mrope_positions(sequence):
    """Позиции (t, h, w) для упакованной последовательности разных модальностей.

    sequence — список описаний кусков:
        ("text", n)                 n текстовых токенов
        ("image", rows, cols)       патчи одной картинки, обход по строкам
        ("video", frames, rows, cols)

    mrope_positions([("text", 2), ("image", 2, 2)])
        ->  [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]
    mrope_positions([("video", 2, 1, 2)])
        ->  [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]

    Правила: текстовый токен двигает временную ось на 1 и не имеет
    пространственных координат. Вся картинка стоит в ОДНОМ моменте времени
    и занимает решётку (h, w). Каждый кадр видео двигает время на 1.

    Курсор времени общий на всю последовательность — иначе текст после
    картинки начал бы отсчёт заново, и модель не поняла бы, что было раньше.

    Неизвестный тип куска — ValueError.
    """
    positions = []
    cursor = 0
    for chunk in sequence:
        kind = chunk[0]
        if kind == "text":
            n = chunk[1]
            for i in range(n):
                positions.append((cursor + i, 0, 0))
            cursor += n
        elif kind == "image":
            rows, cols = chunk[1], chunk[2]
            for r in range(rows):
                for c in range(cols):
                    positions.append((cursor, r, c))
            cursor += 1
        elif kind == "video":
            frames, rows, cols = chunk[1], chunk[2], chunk[3]
            for f in range(frames):
                for r in range(rows):
                    for c in range(cols):
                        positions.append((cursor + f, r, c))
            cursor += frames
        else:
            raise ValueError(f"неизвестный тип куска {kind!r}")
    return positions


def mrope_rotate(vec, position, base=10000.0):
    """M-RoPE: разрезать вектор на три полосы и повернуть каждую своей осью.

    position — тройка (t, h, w) из mrope_positions.

    mrope_rotate([1.0] * 6, (0, 0, 0))  ->  [1.0] * 6   (нулевая позиция)

    Полосы идут подряд: первая треть — время, вторая — высота, третья —
    ширина. Каждая крутится через rotate_pairs со своей позицией.

    Отсюда следует полезное: у текстового токена h = w = 0, значит две
    последние полосы остаются нетронутыми и M-RoPE вырождается в обычный
    RoPE. Один код на текст, картинку и видео, без ветвлений.

    len(vec) обязана делиться на 6: три полосы, и каждая — из пар.
    """
    if len(vec) < 6 or len(vec) % 6:
        raise ValueError(f"длина вектора должна делиться на 6, дано {len(vec)}")
    band = len(vec) // 3
    freqs = rope_frequencies(band, base)
    out = []
    for k in range(3):
        chunk = vec[k * band : (k + 1) * band]
        out.extend(rotate_pairs(chunk, position[k], freqs))
    return out


def pick_fps(duration_s, budget_tokens, tokens_per_frame, motion="low", allowed=FPS_CHOICES):
    """Выбрать FPS: как можно выше, но чтобы влезло в бюджет токенов.

    Потолок из бюджета: fps_max = budget / (duration * tokens_per_frame).
    Потолок из содержания: MOTION_FPS[motion].
    Берём наибольшее значение из allowed, не превышающее оба потолка.

    pick_fps(60, 19440, 81, "high")  ->  4     (fps_max ровно 4.0)
    pick_fps(60, 19440, 81, "low")   ->  1     (движения мало, экономим)
    pick_fps(600, 1000, 81, "high")  ->  None  (не влезает даже 1 FPS)

    None — честный ответ «этот бюджет не тянет это видео»: надо резать
    длительность или tokens_per_frame, а не молча отдавать 1 FPS.

    Неизвестный motion — ValueError.
    """
    if motion not in MOTION_FPS:
        raise ValueError(f"неизвестный уровень движения {motion!r}")
    if duration_s <= 0 or tokens_per_frame < 1:
        raise ValueError("duration_s > 0 и tokens_per_frame >= 1")
    fps_max = budget_tokens / (duration_s * tokens_per_frame)
    ceiling = min(fps_max, MOTION_FPS[motion])
    fits = [f for f in allowed if f <= ceiling + 1e-9]
    return max(fits) if fits else None


def frame_timestamps(duration_s, fps):
    """Метки времени равномерного семплирования видео, в секундах.

    frame_timestamps(2.0, 2)  ->  [0.0, 0.5, 1.0, 1.5]
    frame_timestamps(1.0, 1)  ->  [0.0]
    frame_timestamps(0.3, 1)  ->  [0.0]

    Именно эти числа Qwen2.5-VL вставляет в поток как <time>t</time>: модель
    видит абсолютные секунды, а не индексы кадров, и потому может ответить
    «на 0:04 кот прыгнул», а не «на четвёртом кадре».

    Кадров floor(duration * fps), но не меньше одного: клип короче кадра —
    это всё ещё клип, а пустой список ниже по конвейеру ломает всё.

    Неположительные duration_s или fps — ValueError.
    """
    if duration_s <= 0 or fps <= 0:
        raise ValueError("duration_s и fps должны быть положительными")
    # +1e-9 гасит случай 3.0 * 1 == 2.9999999999999996 при дробных duration
    count = max(1, int(duration_s * fps + 1e-9))
    return [k / fps for k in range(count)]


def parse_tool_call(text, required=("tool",)):
    """Вытащить из ответа модели первый JSON-объект и проверить обязательные ключи.

    parse_tool_call('{"tool": "click", "coords": [380, 220]}')
        ->  {"tool": "click", "coords": [380, 220]}
    parse_tool_call('Конечно!\\n```json\\n{"tool": "scroll"}\\n```')
        ->  {"tool": "scroll"}

    Ловушка: искать закрывающую скобку через text.rfind("}") или считать
    скобки без учёта строк — неверно. В '{"tool": "type", "text": "{}"}'
    фигурные скобки живут внутри строкового литерала и вложенность не
    меняют. Нужен посимвольный проход с флагом «мы внутри строки».

    Нет объекта, битый JSON, отсутствует обязательный ключ, coords не пара
    чисел — всё это ValueError. Ради этого Qwen2.5-VL и учили выдавать JSON:
    ошибку видно сразу, а не через regex, который «почти сработал».
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON-объекта")
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("незакрытый JSON-объект")
    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise ValueError(f"битый JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("ожидался объект, а не массив или скаляр")
    for key in required:
        if key not in obj:
            raise ValueError(f"нет обязательного ключа {key!r}")
    if "coords" in obj:
        coords = obj["coords"]
        if not isinstance(coords, list) or len(coords) != 2:
            raise ValueError("coords должен быть парой чисел")
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in coords):
            raise ValueError("coords должен быть парой чисел")
    return obj
