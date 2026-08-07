"""
InternVL3: нативное мультимодальное предобучение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Корпус InternVL3: текста больше всех, видео — тонкой струйкой.
NATIVE_MIX = {"text": 0.40, "interleaved": 0.35, "caption": 0.20, "video": 0.05}

# Сколько визуальных токенов стоит каждый ярус разрешения у ViR.
TIER_TOKENS = {"low": 256, "medium": 576, "high": 2048}

# Границы детализации между ярусами: detail < 0.4 — низкий, < 0.7 — средний.
VIR_THRESHOLDS = (0.4, 0.7)


def normalize_mix(mix):
    """Привести доли корпуса к сумме 1.0, сохранив пропорции.

    normalize_mix({"text": 2, "video": 2})  ->  {"text": 0.5, "video": 0.5}
    normalize_mix({"text": 40, "video": 10})
        ->  {"text": 0.8, "video": 0.2}

    Пропорции важнее абсолютных чисел: «40 к 35 к 20 к 5» и «8 к 7 к 4 к 1»
    описывают один и тот же корпус.

    Пустой словарь, отрицательная доля или нулевая сумма — ValueError.
    Нулевая сумма особенно коварна: без проверки получится деление на ноль
    посреди подготовки данных, за час до старта прогона.
    """
    if not mix:
        raise ValueError("пустой состав корпуса")
    if any(w < 0 for w in mix.values()):
        raise ValueError("отрицательная доля")
    total = sum(mix.values())
    if total == 0:
        raise ValueError("сумма долей равна нулю")
    return {name: w / total for name, w in mix.items()}


def sample_modalities(mix, n, rng):
    """Набрать n модальностей по долям корпуса. rng — экземпляр random.Random.

    Именно так собирается батч при нативном предобучении: текст, interleaved,
    caption и video тянут из одного распределения и попадают в один лосс.

    sample_modalities({"text": 1.0}, 3, random.Random(0))
        ->  ["text", "text", "text"]

    rng приходит аргументом, а не берётся из глобального random: два прогона
    обучения с одним seed обязаны увидеть одинаковый порядок данных.

    Модальность с нулевой долей не встретится ни разу. Порядок обхода —
    по отсортированным именам, чтобы результат не зависел от того, в каком
    порядке ключи попали в словарь.

    Отрицательное n — ValueError.
    """
    if n < 0:
        raise ValueError("n не может быть отрицательным")
    weights = normalize_mix(mix)
    names = sorted(weights)
    # накопленные вероятности: один rng.random() на образец, поиск линейный —
    # модальностей единицы, бинарный поиск здесь только запутает
    cumulative = []
    acc = 0.0
    for name in names:
        acc += weights[name]
        cumulative.append(acc)
    out = []
    for _ in range(n):
        u = rng.random()
        chosen = names[-1]  # страховка от накопленной ошибки: acc может быть 0.9999
        for name, edge in zip(names, cumulative):
            if u < edge:
                chosen = name
                break
        out.append(chosen)
    return out


def route_resolution(detail, thresholds=VIR_THRESHOLDS):
    """ViR: по оценке детализации запроса выбрать ярус разрешения.

    route_resolution(0.1)  ->  "low"
    route_resolution(0.5)  ->  "medium"
    route_resolution(0.9)  ->  "high"

    Границы включаются в ВЕРХНИЙ ярус: detail ровно 0.4 даёт "medium".
    Правило «меньше порога» вместо «меньше или равно» выбрано затем, чтобы
    сомнительный случай уходил в более подробный ярус: недокодировать
    картинку дороже, чем потратить лишние токены.

    detail вне [0, 1] — ValueError.
    """
    if not 0.0 <= detail <= 1.0:
        raise ValueError(f"detail должен быть в [0, 1], получено {detail}")
    low_edge, medium_edge = thresholds
    if detail < low_edge:
        return "low"
    if detail < medium_edge:
        return "medium"
    return "high"


def routed_tokens(details, tier_tokens=TIER_TOKENS, thresholds=VIR_THRESHOLDS):
    """Средняя цена запроса в визуальных токенах после маршрутизации ViR.

    routed_tokens([0.1, 0.9])  ->  1152.0   то есть (256 + 2048) / 2

    Пустой список — ValueError: среднее по нулю запросов не определено, а
    вернуть 0.0 значит соврать, что трафик бесплатный.
    """
    if not details:
        raise ValueError("пустой список запросов")
    total = sum(tier_tokens[route_resolution(d, thresholds)] for d in details)
    return total / len(details)


def routing_speedup(details, tier_tokens=TIER_TOKENS, thresholds=VIR_THRESHOLDS):
    """Во сколько раз ViR дешевле, чем кодировать всё в максимальном разрешении.

    routing_speedup([0.1, 0.1, 0.9])  ->  2.4

    База сравнения — самый дорогой ярус на каждый запрос. Результат не
    может быть меньше 1.0: маршрутизатор в худшем случае отправит всё
    наверх и сравняется с базой, но проиграть ей он не способен.
    """
    baseline = max(tier_tokens.values())
    return baseline / routed_tokens(details, tier_tokens, thresholds)


def dvd_speedup(encoder_ms, llm_ms):
    """Выигрыш DvD: энкодер и LLM на разных GPU вместо одной.

    На одной GPU стадии идут последовательно: encoder_ms + llm_ms.
    На двух они конвейеризуются, и такт задаёт медленная: max(...).

    dvd_speedup(50, 50)   ->  2.0    (идеальный баланс)
    dvd_speedup(10, 90)   ->  1.111  (энкодер простаивает)
    dvd_speedup(1, 999)   ->  ~1.0   (разделять нечего)

    Отсюда ответ на вопрос «когда DvD не помогает»: когда стадии сильно
    несбалансированы. Потолок ускорения — ровно 2, и только при равенстве.

    Неположительное время любой стадии — ValueError.
    """
    if encoder_ms <= 0 or llm_ms <= 0:
        raise ValueError("время стадии должно быть положительным")
    return (encoder_ms + llm_ms) / max(encoder_ms, llm_ms)


def alignment_debt(text_before, text_after, vision_gain):
    """Цена мультимодальности в баллах текстовых бенчмарков на балл зрения.

    alignment_debt(80.0, 74.0, 12.0)  ->  0.5   (потеряли 6 текста за 12 зрения)
    alignment_debt(80.0, 80.0, 12.0)  ->  0.0   (долга нет)

    Это и есть alignment debt в измеримом виде: post-hoc модель проседает
    на GSM8K и MMLU, нативная — почти нет. Сравнивать модели по абсолютной
    просадке нельзя, её надо нормировать на то, сколько зрения куплено.

    Отрицательный результат возможен и означает, что текстовые скоры
    выросли — так бывает у нативного предобучения.

    Нулевой или отрицательный vision_gain — ValueError: делить просадку не
    на что, и «купили зрение» тут просто не про эту модель.
    """
    if vision_gain <= 0:
        raise ValueError("vision_gain должен быть положительным")
    return (text_before - text_after) / vision_gain
