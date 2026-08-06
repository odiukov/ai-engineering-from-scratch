"""
Латентная диффузия и Stable Diffusion — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def encode(x, scaling_factor=0.18215):
    """Первая стадия: перевод пикселей в латент. Здесь — линейная игрушка.

    encode(10.0)       ->  1.8215
    encode(4.0, 0.5)   ->  2.0

    Настоящий энкодер VAE — свёрточная сеть, но её выход всегда домножают на
    scaling_factor чекпоинта (у SD 1.x это 0.18215). Смысл множителя: привести
    латенты к единичной дисперсии, потому что расписание шума DDPM рассчитано
    именно на такой масштаб.

    В диффузорах это `latents = vae.encode(image).latent_dist.sample()
    * vae.config.scaling_factor` — одна строка, которую тут собираем руками.
    """
    return x * scaling_factor


def decode(z, scaling_factor=0.18215):
    """Обратная сторона первой стадии: из латента обратно в пиксели.

    decode(1.8215)     ->  10.0
    decode(2.0, 0.5)   ->  4.0

    Ловушка урока: множитель обязан совпасть с тем, что был при encode.
    Латенты SDXL, SD3 и Flux несовместимы именно поэтому — у каждого чекпоинта
    свой VAE и свой scaling_factor.
    """
    return z / scaling_factor


def latent_compression_ratio(height, width, channels, downsample, latent_channels):
    """Во сколько раз латент меньше картинки по числу чисел.

    latent_compression_ratio(512, 512, 3, 8, 4)   ->  48.0   (это SD 1.5)
    latent_compression_ratio(64, 64, 4, 1, 4)     ->  1.0    (сжатия нет)

    Пикселей: height * width * channels.
    Латента: (height // downsample) * (width // downsample) * latent_channels.

    Ровно из этого числа растёт вся идея Rombach: U-Net гоняет не 786432
    значения, а 16384. Обрати внимание, что downsample входит в КВАДРАТЕ —
    он режет обе пространственные оси.
    """
    pixels = height * width * channels
    latent = (height // downsample) * (width // downsample) * latent_channels
    return pixels / latent


def softmax(scores):
    """Softmax: список чисел -> список весов, сумма которых равна 1.

    softmax([0.0, 0.0])      ->  [0.5, 0.5]
    softmax([1000.0, 1000.0]) ->  [0.5, 0.5]

    Ловушка: наивный math.exp(1000) даёт OverflowError. Вычти максимум перед
    экспонентой — результат не изменится (softmax инвариантен к сдвигу),
    а переполнения не будет.
    """
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def cross_attention(query, keys, values):
    """Cross-attention: единственный канал, по которому текст влияет на картинку.

    query  — вектор длины d (признак картинки)
    keys   — список из n векторов длины d (токены текста)
    values — список из n векторов (те же токены, но в «значенческом» виде)

    cross_attention([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [[0.0], [4.0]])  ->  [2.0]

    Считай так: score_i = dot(query, keys[i]) / sqrt(d), веса = softmax(scores),
    ответ = взвешенная сумма values.

    Деление на sqrt(d) обязательно: без него при больших d скоры разъезжаются,
    softmax насыщается и градиенты умирают.

    Это и есть та строка h = h + CrossAttention(Q=h, K=V=text_embed), которая
    отличает class-conditional диффузию от Stable Diffusion.
    """
    d = len(query)
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(d) for key in keys]
    weights = softmax(scores)
    out_dim = len(values[0])
    # аккумулируем по столбцам: выход — выпуклая комбинация values
    return [sum(w * v[j] for w, v in zip(weights, values)) for j in range(out_dim)]


def drop_label_for_cfg(label, null_label, p_drop, rng):
    """Обучающий трюк CFG: с вероятностью p_drop подменить метку на null.

    drop_label_for_cfg(1, 2, 0.0, rng)  ->  1   (никогда не роняем)
    drop_label_for_cfg(1, 2, 1.0, rng)  ->  2   (роняем всегда)

    Стандартное p_drop — 0.1. Без этого dropout'а у модели просто не будет
    безусловного предсказания eps_uncond, и CFG на инференсе не из чего собрать.

    Ловушка: сравнение обязано быть rng.random() < p_drop. При p_drop = 0.0
    строгое «меньше» гарантирует, что метку не уронят никогда.
    """
    return null_label if rng.random() < p_drop else label


def classifier_free_guidance(eps_cond, eps_uncond, w):
    """Смешивание условного и безусловного предсказаний шума.

    eps_cfg = (1 + w) * eps_cond - w * eps_uncond

    classifier_free_guidance([1.0], [0.0], 0.0)  ->  [1.0]
    classifier_free_guidance([1.0], [0.0], 3.0)  ->  [4.0]

    Это экстраполяция, а не интерполяция: результат уезжает ДАЛЬШЕ условного
    предсказания в сторону от безусловного. Отсюда и пересыщенные картинки
    при больших w.

    Соответствие с диффузорами: их guidance_scale = 1 + w. То есть
    guidance_scale = 1 (то есть w = 0) — это ровно чистое условное
    предсказание, гайденса нет. Рабочий диапазон guidance_scale 4-8.
    """
    return [(1.0 + w) * c - w * u for c, u in zip(eps_cond, eps_uncond)]
