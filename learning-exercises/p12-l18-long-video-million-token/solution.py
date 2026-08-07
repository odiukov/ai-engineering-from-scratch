"""
Длинное видео и контекст на миллион токенов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок сравнивает четыре пути к часовому видео, и каждый из них здесь
превращается в число:

  * brute context — сколько токенов вообще стоит ролик и что из этого влезает
    в заданный контекст;
  * token compression (Video-XL) — один summary-токен на клип и во сколько
    раз это дешевле;
  * ring attention (LWM) — как последовательность режется по устройствам;
  * agentic retrieval (VideoAgent) — стоимость трёх выдернутых клипов против
    стоимости всего ролика.

Плюс needle-in-a-haystack: кривая recall по позиции иглы и один воспроизводимый
прогон. Всё случайное принимает rng параметром — глобальный random здесь
запрещён, иначе тест не повторить.

Только стандартная библиотека.
"""

import math


def token_budget(duration_s, fps, tokens_per_frame):
    """Сколько визуальных токенов стоит ролик.

    token_budget(1800, 1, 81)  ->  145800   (полчаса при 1 FPS и 3x3 pooling)
    token_budget(7200, 1, 81)  ->  583200   (двухчасовой фильм)

    Кадры считаем целыми: неполный кадр в конце ролика не существует.

    Эти два числа и есть весь сюжет урока. 145k влезает в открытые модели
    впритык, 583k — уже никуда, и дальше приходится выбирать между сжатием
    и retrieval.
    """
    if duration_s < 0 or fps <= 0 or tokens_per_frame <= 0:
        raise ValueError("длительность, fps и tokens_per_frame должны быть положительными")
    return int(duration_s * fps) * tokens_per_frame


def max_duration(context_limit, fps, tokens_per_frame):
    """Сколько секунд видео влезает в контекст. Обратная задача к token_budget.

    max_duration(145800, 1, 81)  ->  1800.0
    max_duration(50, 1, 81)      ->  0.0    (не влезает даже один кадр)

    Считаем через целое число кадров: половина кадра в контекст не кладётся.

    Ноль — законный ответ, а не ошибка. Именно так выглядит попытка засунуть
    2 FPS в контекст 32k.
    """
    if context_limit < 0 or fps <= 0 or tokens_per_frame <= 0:
        raise ValueError("контекст, fps и tokens_per_frame должны быть положительными")
    frames = context_limit // tokens_per_frame
    return frames / fps


def summary_token_budget(duration_s, fps, frames_per_clip):
    """Путь Video-XL: один summary-токен на клип из frames_per_clip кадров.

    summary_token_budget(1800, 1, 16)  ->  113
    summary_token_budget(10, 1, 16)    ->  1

    Неполный клип в конце всё равно даёт свой токен — округляем вверх,
    иначе хвост ролика просто пропадёт.

    113 токенов на полчаса против 145800 — вот за что платят потерей
    точной привязки ко времени.
    """
    if duration_s < 0 or fps <= 0 or frames_per_clip <= 0:
        raise ValueError("длительность, fps и frames_per_clip должны быть положительными")
    frames = int(duration_s * fps)
    return math.ceil(frames / frames_per_clip)


def compression_gain(raw_tokens, compressed_tokens):
    """Во сколько раз сжатое представление дешевле сырого.

    compression_gain(145800, 145)  ->  1005.51...
    compression_gain(100, 100)     ->  1.0

    Число больше 1.0 — выигрыш, ровно 1.0 — сжатия нет, меньше 1.0 —
    «сжатие», которое сделало хуже (бывает на коротких роликах, где
    служебные токены перевешивают).

    Ноль или отрицательное во втором аргументе — ValueError.
    """
    if raw_tokens < 0:
        raise ValueError(f"raw_tokens не может быть отрицательным, получено {raw_tokens}")
    if compressed_tokens <= 0:
        raise ValueError(
            f"compressed_tokens должен быть положительным, получено {compressed_tokens}"
        )
    return raw_tokens / compressed_tokens


def ring_chunk(seq_len, devices):
    """Как ring attention делит последовательность по устройствам.

    ring_chunk(10, 4)  ->  [3, 3, 2, 2]
    ring_chunk(8, 1)   ->  [8]

    Остаток раскидываем по первым устройствам, чтобы куски отличались не
    больше чем на единицу: самое большое устройство и определяет пиковую
    память, ради этого всё и затевалось.

    Заметь два свойства: сумма кусков равна всей последовательности (ни один
    токен не потерян), а размер куска падает с числом устройств. Отсюда и
    линейная по памяти масштабируемость LWM — при том, что полное внимание
    по всей последовательности сохраняется за счёт вращения кусков по кольцу.

    devices <= 0 — ValueError.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len не может быть отрицательным, получено {seq_len}")
    if devices <= 0:
        raise ValueError(f"нужно хотя бы одно устройство, получено {devices}")
    base, extra = divmod(seq_len, devices)
    return [base + 1 if i < extra else base for i in range(devices)]


def recall_at(curve, position):
    """Ожидаемый recall иглы, стоящей в доле position от начала ролика.

    curve — список пар (порог, recall), пороги строго возрастают. Берём
    recall первого порога, который не меньше position.

    recall_at([(0.1, 0.98), (0.5, 0.90), (1.0, 0.85)], 0.05)  ->  0.98
    recall_at([(0.1, 0.98), (0.5, 0.90), (1.0, 0.85)], 0.5)   ->  0.90
    recall_at([(0.1, 0.98), (0.5, 0.90), (1.0, 0.85)], 0.99)  ->  0.85

    Ловушка: неотсортированная кривая молча даст неправильный ответ, потому
    что первый подходящий порог окажется не тем. ValueError.
    position вне [0, 1] — тоже ValueError: это доля, а не секунды.
    """
    if not curve:
        raise ValueError("пустая кривая recall")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position должен быть в [0, 1], получено {position}")
    thresholds = [t for t, _ in curve]
    if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("пороги кривой должны строго возрастать")
    for threshold, value in curve:
        if position <= threshold:
            return value
    return curve[-1][1]


def needle_trial(duration_s, curve, rng):
    """Один прогон needle-in-a-haystack: воткнуть иглу и оценить шанс её найти.

    Возвращает словарь с ключами 'needle_time', 'position', 'recall'.

    rng — random.Random. Момент иглы берётся как rng.uniform(0, duration_s),
    ровно один бросок за вызов: два прогона с одинаковым seed обязаны
    совпасть до последнего знака, иначе бенчмарк ничего не измеряет.

    needle_trial(100.0, [(1.0, 0.9)], random.Random(0))['recall']  ->  0.9

    position — доля от начала ролика, её и подают в recall_at.
    """
    if duration_s <= 0:
        raise ValueError(f"длительность должна быть положительной, получено {duration_s}")
    needle_time = rng.uniform(0.0, duration_s)
    position = needle_time / duration_s
    return {
        "needle_time": needle_time,
        "position": position,
        "recall": recall_at(curve, position),
    }


def pick_strategy(duration_minutes, query_kind):
    """Какой путь выбрать под длительность и тип вопроса.

    pick_strategy(10, "general")   ->  'brute'
    pick_strategy(30, "general")   ->  'compression'
    pick_strategy(120, "specific") ->  'agentic'
    pick_strategy(120, "general")  ->  'brute'

    Логика урока:
      * меньше 15 минут — весь ролик влезает, брать нативный контекст;
      * 15..60 минут — сжимать (LongVILA, Video-XL);
      * больше часа и вопрос про конкретный момент — retrieval: выдернуть
        три клипа дешевле, чем закодировать два часа;
      * больше часа и вопрос общий («о чём ролик») — retrieval не поможет,
        нужен один проход по всему, то есть снова brute.

    Обрати внимание, что brute появляется и слева, и справа. Это не ошибка:
    на коротком ролике он дёшев, на длинном общем вопросе — безальтернативен.

    Неизвестный тип вопроса — ValueError.
    """
    if query_kind not in ("general", "specific"):
        raise ValueError(f"неизвестный тип вопроса {query_kind!r}")
    if duration_minutes <= 0:
        raise ValueError(
            f"длительность должна быть положительной, получено {duration_minutes}"
        )
    if duration_minutes < 15:
        return "brute"
    if duration_minutes <= 60:
        return "compression"
    return "agentic" if query_kind == "specific" else "brute"
