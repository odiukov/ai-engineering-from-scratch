"""
Video-language модели: временные токены и grounding — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Видео не стопка фотографий, и весь урок про это. Три инженерных сюжета,
каждый считается на голом Python:

  * сколько кадров брать и откуда — uniform против dynamic FPS, где сигналом
    служит frame differencing;
  * сколько токенов стоит кадр после pooling — от этого зависит, влезет ли
    ролик в контекст вообще;
  * как измерить temporal grounding — IoU по отрезкам времени и recall по
    порогу, плюс разбор <time>-разметки, которую модель выдаёт.

Отдельно проверяется тезис TMRoPE: позиции по абсолютному времени сохраняют
неравномерность сэмплирования, позиции по индексу кадра её теряют.

Только стандартная библиотека.
"""

import re


def frame_difference(frames):
    """Покадровая «сила движения»: среднее модуля разности с предыдущим кадром.

    Кадр — плоский список чисел (яркости пикселей). У первого кадра предыдущего
    нет, поэтому его движение считаем нулевым.

    frame_difference([[0.0, 0.0], [1.0, 1.0]])  ->  [0.0, 1.0]
    frame_difference([[5.0], [5.0], [5.0]])     ->  [0.0, 0.0, 0.0]

    Это дешёвый заменитель optical flow: именно такой сигнал и подают в
    dynamic-FPS сэмплер, чтобы он сгущал кадры там, где что-то происходит.

    Ловушка: кадры разной длины — ValueError. Молча обрезать по zip значит
    получить правдоподобное число из сравнения разных кусков картинки.
    """
    if not frames:
        raise ValueError("нужен хотя бы один кадр")
    width = len(frames[0])
    if width == 0:
        raise ValueError("пустой кадр")
    motion = [0.0]
    for prev, cur in zip(frames, frames[1:]):
        if len(prev) != width or len(cur) != width:
            raise ValueError("все кадры должны быть одного размера")
        motion.append(sum(abs(a - b) for a, b in zip(prev, cur)) / width)
    return motion


def uniform_sample(duration, n):
    """n равномерных отметок времени по ролику длиной duration секунд.

    Берём центры равных корзин, а не границы: крайний кадр в нуле секунд
    почти всегда чёрный, а кадр ровно в конце может не существовать.

    uniform_sample(10.0, 2)  ->  [2.5, 7.5]
    uniform_sample(10.0, 1)  ->  [5.0]

    n <= 0 или неположительная длительность — ValueError.
    """
    if duration <= 0:
        raise ValueError(f"длительность должна быть положительной, получено {duration}")
    if n <= 0:
        raise ValueError(f"нужен хотя бы один кадр, получено {n}")
    step = duration / n
    return [step * (i + 0.5) for i in range(n)]


def dynamic_sample(motion, budget, fps_cap):
    """Отметки времени, сгущённые там, где больше движения.

    motion — по одному числу на секунду ролика (выход frame_difference,
    усреднённый посекундно). budget — сколько кадров всего хотим взять.
    fps_cap — потолок кадров на одну секунду.

    dynamic_sample([0.0, 1.0], 2, 4)  ->  [0.5, 1.5]

    Разбор: минимум один кадр на секунду применяется, когда budget хватает
    на все секунды. Здесь две секунды и два кадра, поэтому каждая получает по
    одному. Остаток распределяется пропорционально движению с детерминированным
    разруливанием округлений. Итог никогда не превышает budget, а fps_cap ограничивает
    каждую секунду.

    Ловушка: суммарное движение может быть нулевым (статичная камера).
    Делить на ноль нельзя — в этом случае возвращаем обычный uniform.
    """
    if not motion:
        raise ValueError("нужна хотя бы одна секунда")
    if any(m < 0 for m in motion):
        raise ValueError("движение не может быть отрицательным")
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or not isinstance(fps_cap, int)
        or isinstance(fps_cap, bool)
        or budget <= 0
        or fps_cap <= 0
    ):
        raise ValueError("budget и fps_cap должны быть положительными целыми")

    seconds = len(motion)
    target = min(budget, seconds * fps_cap)
    if sum(motion) == 0:
        return uniform_sample(float(seconds), target)

    minimum = 1 if target >= seconds else 0
    counts = [minimum] * seconds
    remaining = target - minimum * seconds

    # Метод наибольших остатков с повторным распределением после того,
    # как бурная секунда упёрлась в fps_cap.
    while remaining:
        active = [i for i, count in enumerate(counts) if count < fps_cap]
        weights = [motion[i] for i in active]
        if sum(weights) == 0:
            weights = [1.0] * len(active)
        weight_total = sum(weights)
        shares = {i: remaining * weight / weight_total for i, weight in zip(active, weights)}

        allocated = 0
        for i in active:
            take = min(fps_cap - counts[i], int(shares[i]))
            counts[i] += take
            allocated += take
        remaining -= allocated
        if remaining and allocated == 0:
            ranked = sorted(active, key=lambda i: (-shares[i], i))
            for i in ranked[:remaining]:
                counts[i] += 1
            remaining -= min(remaining, len(ranked))

    times = []
    for second, count in enumerate(counts):
        for j in range(count):
            times.append(second + (j + 0.5) / count)
    return times


def pooled_tokens(grid_side, pool):
    """Сколько токенов остаётся от кадра после pooling pool x pool.

    grid_side — сторона сетки патчей. У Qwen2.5-VL это 27 (27 * 27 = 729),
    у ViT на 336 пикселей — 24 (576).

    pooled_tokens(24, 3)  ->  64    (576 -> 64, тот самый sweet spot урока)
    pooled_tokens(27, 3)  ->  81
    pooled_tokens(24, 1)  ->  576   (pooling выключен)

    Неполные корзины отбрасываются: делим сторону нацело. Поэтому pool
    больше стороны — ValueError, ноль токенов на кадр это не «сильное
    сжатие», а потерянный кадр.
    """
    if grid_side <= 0:
        raise ValueError(f"сторона сетки должна быть положительной, получено {grid_side}")
    if pool <= 0:
        raise ValueError(f"pool должен быть положительным, получено {pool}")
    if pool > grid_side:
        raise ValueError(f"pool {pool} больше стороны сетки {grid_side}")
    return (grid_side // pool) ** 2


def temporal_iou(a_start, a_end, b_start, b_end):
    """IoU двух временных отрезков: пересечение делить на объединение.

    temporal_iou(0.0, 2.0, 0.0, 2.0)  ->  1.0
    temporal_iou(0.0, 2.0, 1.0, 3.0)  ->  0.333...   (1 / 3)
    temporal_iou(0.0, 1.0, 5.0, 6.0)  ->  0.0

    Объединение считаем как len(a) + len(b) - пересечение, а не как разброс
    от самого раннего начала до самого позднего конца: для непересекающихся
    отрезков разброс включил бы пустой промежуток между ними.

    Отрезок нулевой или отрицательной длины — ValueError. Событие, которое
    длится ноль секунд, ломает IoU: знаменатель обнуляется.
    """
    if a_end <= a_start:
        raise ValueError(f"пустой отрезок a: [{a_start}, {a_end}]")
    if b_end <= b_start:
        raise ValueError(f"пустой отрезок b: [{b_start}, {b_end}]")
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union


def grounding_recall(predictions, truths, tol_iou=0.3):
    """Доля событий из truths, для которых нашлось предсказание с IoU >= tol_iou.

    И predictions, и truths — списки троек (имя, начало, конец).

    grounding_recall([("jump", 4.1, 4.7)], [("jump", 4.0, 4.5)], 0.3)  ->  1.0
    grounding_recall([("turn", 4.1, 4.7)], [("jump", 4.0, 4.5)], 0.3)  ->  0.0

    Сопоставляем только события с одинаковым именем: угадать время, но
    назвать не то действие — это не попадание.

    Пустой список truths — ValueError: делить не на что, а вернуть 1.0
    значило бы отрапортовать идеальный recall на пустом бенчмарке.
    tol_iou вне [0, 1] — тоже ValueError.
    """
    if not truths:
        raise ValueError("пустой ground truth: recall не определён")
    if not 0.0 <= tol_iou <= 1.0:
        raise ValueError(f"tol_iou должен быть в [0, 1], получено {tol_iou}")
    hits = 0
    for name, gt_start, gt_end in truths:
        best = 0.0
        for p_name, p_start, p_end in predictions:
            if p_name != name:
                continue
            best = max(best, temporal_iou(p_start, p_end, gt_start, gt_end))
        if best >= tol_iou:
            hits += 1
    return hits / len(truths)


def position_ids(times, mode):
    """Позиции визуальных токенов: 'index' — номер кадра, 'time' — секунды.

    position_ids([0.0, 0.5, 4.0], "index")  ->  [0, 1, 2]
    position_ids([0.0, 0.5, 4.0], "time")   ->  [0.0, 0.5, 4.0]

    Разница между двумя режимами и есть вклад TMRoPE. При равномерном
    сэмплировании они несут одно и то же. При dynamic FPS режим 'index'
    сообщает модели, что кадры шли через равные промежутки, — а они не шли,
    и вопрос «на какой секунде кот прыгнул» становится неотвечаемым.

    Неизвестный режим — ValueError.
    """
    if mode == "index":
        return list(range(len(times)))
    if mode == "time":
        return [float(t) for t in times]
    raise ValueError(f"неизвестный режим {mode!r}")


def parse_time_tokens(text):
    """Достать секунды из token-based разметки <time>...</time>.

    parse_time_tokens("The cat jumps at <time>4.2</time>")       ->  [4.2]
    parse_time_tokens("<time>1.0</time> then <time>3.5</time>")  ->  [1.0, 3.5]
    parse_time_tokens("no events here")                          ->  []

    Порядок сохраняем как в тексте: он несёт порядок событий.

    Ловушки: нечисловое содержимое тега и отрицательное время — ValueError.
    Обе ошибки в свободном тексте модели встречаются регулярно, и пропустить
    их дальше в пайплайн значит получить NaN в метрике вместо разбора.
    """
    out = []
    for raw in re.findall(r"<time>(.*?)</time>", text):
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"нечисловая отметка времени {raw!r}") from None
        if value < 0:
            raise ValueError(f"отрицательная отметка времени {value}")
        out.append(value)
    return out
