"""
Генерация видео — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def position_embedding(pos, dim=8):
    """Синусоидальное вложение позиции: список длины dim.

    position_embedding(0, 4)  ->  [0.0, 1.0, 0.0, 1.0]

    Пары (sin, cos) с частотами 1 / 10000^(i / (half - 1)), half = dim // 2.
    Ровно то же, что позиционное кодирование в трансформере.

    Для видео позиций три: время, y и x. Здесь мы делаем одномерный случай —
    время. В настоящем DiT три таких вложения склеивают в одно.

    Ловушка: при dim = 2 знаменатель half - 1 обнуляется. Защити его.
    """
    out = []
    half = dim // 2
    for i in range(half):
        freq = 1.0 / (10000 ** (i / max(half - 1, 1)))
        out.append(math.sin(pos * freq))
        out.append(math.cos(pos * freq))
    return out[:dim]


def patchify(video, patch_t):
    """Разбить клип на патчи по patch_t кадров. Вернуть список списков.

    patchify([1, 2, 3, 4], 2)  ->  [[1, 2], [3, 4]]
    patchify([1, 2, 3, 4], 1)  ->  [[1], [2], [3], [4]]

    Патчи — это токены видео-DiT. У Sora-подобных моделей patch_t = 1 или 2:
    один токен на кадр или на пару кадров. Десять секунд 1080p дают
    20-100 тысяч токенов.

    Если длина клипа не делится на patch_t, честно брось ValueError: молча
    отбросить хвостовые кадры — это потерянные данные, которые потом никто не
    найдёт.
    """
    if patch_t <= 0:
        raise ValueError("patch_t must be positive")
    if len(video) % patch_t != 0:
        raise ValueError("clip length must be divisible by patch_t")
    return [list(video[i:i + patch_t]) for i in range(0, len(video), patch_t)]


def patch_tokens(video, patch_t, pos_dim=4):
    """Токены для DiT: значения патча плюс вложение его позиции во времени.

    patch_tokens([1.0, 2.0], 1, 2)
        ->  [[1.0, 0.0, 1.0], [2.0, 0.8414709848078965, 0.5403023058681398]]

    Длина токена — patch_t + pos_dim, число токенов — len(video) // patch_t.

    Без позиционного вложения attention видит МНОЖЕСТВО патчей, а не
    последовательность: два одинаковых кадра станут неразличимы, и порядок
    времени потеряется. Это прямой путь к мерцанию.
    """
    return [
        patch + position_embedding(k, pos_dim)
        for k, patch in enumerate(patchify(video, patch_t))
    ]


def attention_pairs(n_frames, n_positions, factorized=False):
    """Сколько пар «запрос-ключ» считает attention. Это и есть цена в FLOPs.

    attention_pairs(8, 16)        ->  16384   (полное 3-D attention: (8*16)^2)
    attention_pairs(8, 16, True)  ->  3072    (8 * 16^2 + 16 * 8^2)

    Полное 3-D: каждый токен смотрит на все, то есть (n_frames * n_positions)^2.
    Факторизованное: сначала пространственное внутри каждого кадра
    (n_frames * n_positions^2), потом временное по одной пространственной
    позиции через все кадры (n_positions * n_frames^2).

    Именно из этой арифметики берётся «полное 3-D attention в 16-100 раз
    дороже». Считай пары, а не гадай.
    """
    if factorized:
        spatial = n_frames * n_positions * n_positions
        temporal = n_positions * n_frames * n_frames
        return spatial + temporal
    tokens = n_frames * n_positions
    return tokens * tokens


def frame_deltas(video):
    """Модули покадровых разностей: список длины len(video) - 1.

    frame_deltas([1.0, 1.5, 1.0])  ->  [0.5, 0.5]
    frame_deltas([2.0, 2.0])       ->  [0.0]

    Модуль, а не разность со знаком: нас интересует величина скачка, а не
    направление движения. Плавный подъём и плавный спуск одинаково хороши.
    """
    return [abs(video[i + 1] - video[i]) for i in range(len(video) - 1)]


def flicker_score(video):
    """Средняя покадровая разность — численная мера мерцания.

    flicker_score([1.0, 1.5, 1.0])  ->  0.5
    flicker_score([2.0, 2.0, 2.0])  ->  0.0

    Чем меньше, тем плавнее движение. Это самая грубая метрика temporal
    coherence, но она уже отличает совместное сэмплирование от покадрового.
    """
    deltas = frame_deltas(video)
    return sum(deltas) / len(deltas)


def sample_frames(n_frames, rng, coupling=0.0):
    """Игрушечный клип из n_frames кадров с заданной силой связи между кадрами.

    Один общий шум на весь клип плюс свой шум у каждого кадра:
        frame_i = coupling * shared + sqrt(1 - coupling^2) * own_i

    sample_frames(4, rng, coupling=1.0)  ->  четыре ОДИНАКОВЫХ значения
    sample_frames(4, rng, coupling=0.0)  ->  четыре независимых значения

    coupling = 0 — это тот самый flicker baseline: покадровая диффузия, где
    шум каждого кадра свой. coupling = 1 — идеальная (и скучная) когерентность.

    Множитель sqrt(1 - coupling^2) подобран так, чтобы дисперсия ОТДЕЛЬНОГО
    кадра осталась единичной при любом coupling. Иначе связь нельзя было бы
    сравнивать: клип просто становился бы тише или громче.

    Общий шум тяни ОДИН раз до цикла — в этом и смысл слова «общий».
    """
    shared = rng.gauss(0.0, 1.0)
    own_scale = math.sqrt(max(0.0, 1.0 - coupling * coupling))
    return [coupling * shared + own_scale * rng.gauss(0.0, 1.0) for _ in range(n_frames)]


def condition_on_first_frame(video, first_value):
    """Image-to-video: прибить нулевой кадр к first_value, движение сохранить.

    condition_on_first_frame([1.0, 1.5, 3.0], 10.0)  ->  [10.0, 10.5, 12.0]
    condition_on_first_frame([2.0, 3.0], 2.0)        ->  [2.0, 3.0]   (уже на месте)

    Сдвиг один и тот же для всех кадров, поэтому все покадровые разности
    остаются прежними: движение то же, стартовая точка новая.

    Это и есть режим I2V у WAN, Kling и Runway: даёшь картинку — получаешь
    видео, которое с неё начинается. Условие распространяется на весь клип
    именно потому, что кадры связаны, а не независимы.

    Вход не мутируем.
    """
    offset = first_value - video[0]
    return [v + offset for v in video]
