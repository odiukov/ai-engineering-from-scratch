"""
Stable Diffusion — архитектура и дообучение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def latent_compression_factor(image_shape, latent_shape):
    """Во сколько раз латент меньше картинки. Форма — кортеж (C, H, W).

    latent_compression_factor((3, 512, 512), (4, 64, 64))  ->  48.0
    latent_compression_factor((3, 64, 64), (3, 64, 64))    ->  1.0

    Это ровно то число, ради которого придумали latent diffusion: U-Net
    вместо 786432 значений видит 16384, и вся арифметика обучения и
    сэмплирования дешевеет во столько же раз.

    Считай произведение всех измерений, а не только сторон: каналов у
    латента 4, а у картинки 3, и это меняет ответ.
    """
    pixels = 1
    for dim in image_shape:
        pixels *= dim
    latents = 1
    for dim in latent_shape:
        latents *= dim
    return pixels / latents


def scale_latents(latents, factor=0.18215, inverse=False):
    """Домножить латенты на масштабный коэффициент VAE (или поделить обратно).

    Это та самая строка `latents * 0.18215` из каждого пайплайна Stable
    Diffusion: сырой выход VAE-энкодера имеет разброс порядка 5, а U-Net
    обучали на данных с разбросом около 1.

    scale_latents([10.0], 0.5)                ->  [5.0]
    scale_latents([5.0], 0.5, inverse=True)   ->  [10.0]

    Ловушка: перед декодером нужно поделить обратно, иначе картинка выйдет
    выцветшей и серой. Прямой и обратный вызов обязаны давать исходный
    список.
    """
    if inverse:
        return [v / factor for v in latents]
    return [v * factor for v in latents]


def softmax(scores):
    """Превратить список чисел в распределение: положительные, сумма равна 1.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([1.0, 1.0, 1.0])  ->  [1/3, 1/3, 1/3]
    softmax([0.0, 100.0])     ->  примерно [0.0, 1.0]

    Ловушка: math.exp(1000) — OverflowError. Вычти максимум перед
    экспонентой: softmax(s) = softmax(s - max s), результат тот же, а
    переполнения нет. Это и делает torch.softmax внутри.
    """
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def cross_attention(query, keys, values):
    """Один запрос смотрит на все токены текста. Возвращает вектор-ответ.

    query — вектор длины d (латентный токен картинки).
    keys   — список векторов длины d (токены промпта).
    values — список векторов той же длины, что и ответ.

        scores  = dot(query, key_i) / sqrt(d)
        weights = softmax(scores)
        out     = sum weights_i * value_i

    cross_attention([0.0], [[0.0], [0.0]], [[1.0], [3.0]])  ->  [2.0]
    cross_attention([10.0], [[1.0], [-1.0]], [[1.0], [3.0]])  ->  почти [1.0]

    Деление на sqrt(d) — не украшение: без него при большой размерности
    скоры разъезжаются, softmax насыщается и градиент умирает.

    Именно так текст попадает в U-Net: на каждом уровне разрешения латентные
    токены задают вопросы токенам промпта и подмешивают ответ к себе.
    """
    d = len(query)
    scale = math.sqrt(d)
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    weights = softmax(scores)
    out = [0.0] * len(values[0])
    for w, value in zip(weights, values):
        for i, v in enumerate(value):
            out[i] += w * v
    return out


def classifier_free_guidance(eps_uncond, eps_cond, scale):
    """Смешать безусловное и условное предсказание шума.

        eps = eps_uncond + scale * (eps_cond - eps_uncond)

    classifier_free_guidance([0.0], [1.0], 0.0)  ->  [0.0]   (промпт не влияет)
    classifier_free_guidance([0.0], [1.0], 1.0)  ->  [1.0]   (обычное условное)
    classifier_free_guidance([0.0], [1.0], 7.5)  ->  [7.5]   (дефолт SD)

    Обрати внимание: при scale > 1 это не интерполяция, а ЭКСТРАПОЛЯЦИЯ —
    результат уезжает дальше условного предсказания. Отсюда и характерные
    пережжённые картинки на guidance_scale 15 и выше.

    Плата за scale — разнообразие: чем сильнее тянем к промпту, тем меньше
    вариантов выдаёт модель на разных сидах.
    """
    return [u + scale * (c - u) for u, c in zip(eps_uncond, eps_cond)]


def img2img_timesteps(num_steps, strength):
    """Какие шаги расписания реально прогоняются в img2img. Список индексов.

    Работаем с конца: чем меньше strength, тем позже входим в цепочку и тем
    меньше успеваем изменить.

        сколько шагов = min(int(num_steps * strength), num_steps)
        список = [k-1, k-2, ..., 0]

    img2img_timesteps(50, 1.0)  ->  [49, 48, ..., 0]   (входим с чистого шума)
    img2img_timesteps(50, 0.6)  ->  [29, 28, ..., 0]   (30 шагов)
    img2img_timesteps(50, 0.0)  ->  []                 (картинка не изменится)

    strength = 1.0 полностью игнорирует входную картинку: шума добавили
    столько, что от неё ничего не осталось. Отсюда и рабочий диапазон
    0.5-0.7 для смены стиля с сохранением композиции.
    """
    count = min(int(num_steps * strength), num_steps)
    return list(range(count - 1, -1, -1))


def inpaint_blend(denoised, original, mask):
    """Собрать латент inpainting: маска говорит, где брать новое.

    mask — числа от 0 до 1 той же длины. 1 — перерисовать, 0 — сохранить.

        out_i = mask_i * denoised_i + (1 - mask_i) * original_i

    inpaint_blend([9.0], [1.0], [0.0])   ->  [1.0]   (маска пустая — не трогаем)
    inpaint_blend([9.0], [1.0], [1.0])   ->  [9.0]
    inpaint_blend([9.0], [1.0], [0.5])   ->  [5.0]   (мягкий край маски)

    Это делается на КАЖДОМ шаге сэмплирования, а не один раз в конце: иначе
    сгенерированная область не согласуется с сохранённым фоном по швам.

    Ловушка: входные списки менять нельзя, оригинал понадобится на следующем
    шаге ещё раз.
    """
    return [m * d + (1.0 - m) * o for d, o, m in zip(denoised, original, mask)]


def lora_update(W, A, B, alpha=1.0):
    """Матрица весов с приклеенным LoRA-адаптером: W + alpha * (A @ B).

    W — матрица d_in x d_out (список строк), A — d_in x r, B — r x d_out.
    Возвращает НОВУЮ матрицу, W остаётся нетронутой.

    lora_update([[1.0, 1.0]], [[1.0]], [[2.0, 0.0]])            ->  [[3.0, 1.0]]
    lora_update([[1.0, 1.0]], [[1.0]], [[2.0, 0.0]], alpha=0.0) ->  [[1.0, 1.0]]

    Смысл в размере: вместо d_in * d_out обучаемых чисел получаем
    r * (d_in + d_out). Для слоя 1024x1024 и r = 8 это 16384 вместо 1048576 —
    в 64 раза меньше. Поэтому адаптер весит 10-50 МБ, а не 4 ГБ.

    Поправка обязана иметь ранг не выше r: она не может изменить веса
    как угодно, и это ограничение — не баг, а то, что защищает базовую
    модель от разрушения при дообучении на двадцати картинках.
    """
    rank = len(B)
    out = []
    for i, row in enumerate(W):
        new_row = []
        for j, w in enumerate(row):
            # delta[i][j] = sum_k A[i][k] * B[k][j]
            delta = sum(A[i][k] * B[k][j] for k in range(rank))
            new_row.append(w + alpha * delta)
        out.append(new_row)
    return out
