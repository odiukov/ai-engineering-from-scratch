"""
Emu3: генерация картинок и видео обычным next-token — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Температуры, рекомендованные в статье Emu3: восприятие честнее при 1.0,
# генерация чище при 0.8.
RECOMMENDED_TEMPERATURE = {"perception": 1.0, "generation": 0.8}

# Рабочий диапазон веса classifier-free guidance.
GUIDANCE_RANGE = (3.0, 7.0)


def image_tokens(width, height, reduction):
    """Сколько VQ-токенов уйдёт на картинку при заданном сжатии стороны.

    image_tokens(512, 512, 8)    ->  4096    (столько тратит Emu3)
    image_tokens(1024, 1024, 8)  ->  16384

    Зависимость КВАДРАТИЧНАЯ по стороне: удвоил разрешение — учетверил
    число токенов, а с ним и время авторегрессионной генерации. Отсюда и
    ответ на «почему Emu3 не делает 4K в один проход».

    Сторона, не кратная reduction, — ValueError: дробного числа токенов не
    бывает, а тихое округление вниз обрежет край картинки.
    """
    if width <= 0 or height <= 0 or reduction <= 0:
        raise ValueError("размеры и reduction должны быть положительными")
    if width % reduction or height % reduction:
        raise ValueError(f"{width}x{height} не кратно reduction={reduction}")
    return (width // reduction) * (height // reduction)


def video_tokens(width, height, frames, spatial_reduction, temporal_reduction):
    """Токены 3D-VQ: один код покрывает кубик пикселей, а не квадратик.

    video_tokens(256, 256, 32, 4, 4)  ->  32768

    Разбор примера из урока: 256/4 = 64 по каждой пространственной оси,
    32/4 = 8 по времени, итого 64 * 64 * 8.

    Именно поэтому патч у Emu3 4x4x4, а не 8x8x1: соседние кадры почти
    одинаковы, и сжимать время выгоднее, чем добивать пространство.

    Некратное число кадров — ValueError, как и у картинок.
    """
    if frames <= 0 or temporal_reduction <= 0:
        raise ValueError("frames и temporal_reduction должны быть положительными")
    if frames % temporal_reduction:
        raise ValueError(f"{frames} кадров не кратно {temporal_reduction}")
    per_frame_group = image_tokens(width, height, spatial_reduction)
    return per_frame_group * (frames // temporal_reduction)


def frames_in_clip(duration_s, fps):
    """Сколько кадров в клипе указанной длины.

    frames_in_clip(4.0, 8)  ->  32

    Отдельная функция ради одного: длительность в секундах и число кадров
    легко перепутать местами при подсчёте токенов, а ошибка в 8 раз в
    бюджете обнаружится только на счёте за GPU.

    Неположительные аргументы — ValueError.
    """
    if duration_s <= 0 or fps <= 0:
        raise ValueError("duration_s и fps должны быть положительными")
    # +1e-9 гасит 4.0 * 8 == 31.999999999999996 при дробной длительности
    return int(duration_s * fps + 1e-9)


def generation_seconds(n_tokens, tokens_per_second):
    """Время авторегрессионной генерации в секундах.

    generation_seconds(4096, 30)  ->  136.53   (то самое «две минуты на
                                                картинку 512x512»)

    Здесь и живёт главный минус подхода: SDXL рисует ту же картинку за
    2-5 секунд, потому что диффузия делает 30 шагов по всему полотну, а
    не 4096 последовательных предсказаний.

    Неположительная скорость — ValueError.
    """
    if n_tokens < 0:
        raise ValueError("n_tokens не может быть отрицательным")
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second должен быть положительным")
    return n_tokens / tokens_per_second


def cfg_logits(cond, uncond, guidance):
    """Classifier-free guidance: развести условные и безусловные логиты.

    Формула: uncond + guidance * (cond - uncond).

    cfg_logits([2.0, 0.0], [1.0, 1.0], 1.0)  ->  [2.0, 0.0]   (чистый cond)
    cfg_logits([2.0, 0.0], [1.0, 1.0], 0.0)  ->  [1.0, 1.0]   (чистый uncond)
    cfg_logits([2.0, 0.0], [1.0, 1.0], 3.0)  ->  [4.0, -2.0]

    Вес 1.0 — это ровно «без guidance». Всё, что больше, растягивает
    разницу между «с подписью» и «без подписи»: картинка сильнее слушается
    промпта, но теряет разнообразие. Emu3 работает в диапазоне 3-7.

    Разная длина списков — ValueError: складывать логиты разных словарей
    бессмысленно, а zip молча обрежет длинный и спрячет баг.
    """
    if len(cond) != len(uncond):
        raise ValueError(f"длины не совпадают: {len(cond)} и {len(uncond)}")
    return [u + guidance * (c - u) for c, u in zip(cond, uncond)]


def softmax(logits, temperature=1.0):
    """Логиты в вероятности с температурой.

    softmax([0.0, 0.0])            ->  [0.5, 0.5]
    softmax([0.0, 1.0], 0.01)      ->  примерно [0.0, 1.0]

    Ловушка: math.exp(1000) — это OverflowError. Вычти максимум логита
    перед экспонентой: softmax от сдвинутых логитов тот же самый, а
    переполнения больше нет. Это не микрооптимизация, без этого функция
    просто падает на реальных логитах длинного словаря.

    Температура 0 или меньше — ValueError. Ноль хочется трактовать как
    «всегда argmax», но тогда функция перестаёт возвращать распределение.
    """
    if not logits:
        raise ValueError("пустые логиты")
    if temperature <= 0:
        raise ValueError("температура должна быть положительной")
    scaled = [x / temperature for x in logits]
    shift = max(scaled)
    exps = [math.exp(x - shift) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def sample_token(logits, rng, temperature=1.0):
    """Вытянуть индекс следующего токена из распределения. rng — random.Random.

    sample_token([0.0, 100.0], random.Random(0))  ->  1

    Метод обратной функции распределения: одно rng.random(), потом идём по
    накопленным вероятностям.

    rng приходит аргументом: генерация с одним seed обязана давать одну и
    ту же картинку, иначе ни отладить, ни сравнить два guidance нельзя.
    """
    probabilities = softmax(logits, temperature)
    u = rng.random()
    acc = 0.0
    for index, p in enumerate(probabilities):
        acc += p
        if u < acc:
            return index
    # сюда попадаем только из-за накопленной ошибки float, когда acc чуть
    # меньше единицы, а u оказался в этом зазоре
    return len(probabilities) - 1


def sample_image_tokens(
    n,
    cond,
    uncond,
    rng,
    guidance=GUIDANCE_RANGE[0],
    temperature=RECOMMENDED_TEMPERATURE["generation"],
):
    """Сгенерировать n токенов картинки: сначала CFG, потом семплирование.

    Это скелет Emu3-Gen. Настоящая модель пересчитывает логиты после
    каждого токена, здесь распределение фиксировано — так виден вклад
    guidance и температуры в чистом виде, без влияния контекста.

    sample_image_tokens(3, [2.0, 0.0], [1.0, 1.0], random.Random(0))
        ->  список из трёх индексов

    Чем больше guidance, тем чаще в выдаче тот токен, который предпочитают
    условные логиты, — ровно то, что видно глазом как «сильнее слушается
    промпта».

    Отрицательное n — ValueError.
    """
    if n < 0:
        raise ValueError("n не может быть отрицательным")
    mixed = cfg_logits(cond, uncond, guidance)
    return [sample_token(mixed, rng, temperature) for _ in range(n)]
