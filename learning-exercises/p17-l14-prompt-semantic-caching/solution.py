"""
Кэш промптов и семантический кэш: два слоя и их цена — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих API:

    cosine, nearest_entry   <-  векторный поиск в Redis VSS / Qdrant / GPTCache
    semantic_lookup         <-  L1: слой ПЕРЕД вызовом LLM, порог 0.95+
    run_semantic_cache      <-  прогон трафика через L1 с записью промахов
    cache_stats             <-  дашборд «hit rate» в Portkey / Helicone Cache
    common_prefix_tokens    <-  то, что провайдер реально может переиспользовать
    l2_request_cost         <-  Anthropic cache_control: 1.25x за запись с TTL
                                5 минут, 2x за час, чтение ~10x дешевле входа;
                                у OpenAI то же автоматически от 1024 токенов
    parallel_wave_cost      <-  анти-паттерн параллелизации: N запросов
                                прилетают раньше, чем встала первая запись

Ключевое отличие двух слоёв. L2 (кэш префикса у провайдера) НЕ МОЖЕТ отдать
неверный ответ: совпадение точное, побайтовое. L1 (семантический кэш) может,
и именно поэтому здесь векторы настоящие, а не хеш строки. Порог 0.85 отдаёт
чужой ответ и делает это молча — тесты это фиксируют, а не прячут.

Цены — снимок рейт-карты Claude Sonnet-класса на 2026-04. Они дрейфуют;
проверяй по живой странице, прежде чем считать бюджет.
"""

import math

# $/M токенов. Вход, чтение из кэша префикса, выход.
PRICE_INPUT = 3.00
PRICE_CACHED_READ = 0.30
PRICE_OUTPUT = 15.00

# Премия за запись в кэш префикса: ключ — TTL.
WRITE_MULTIPLIER = {"5min": 1.25, "1hr": 2.00}


class CacheError(Exception):
    """Кэш спросили о том, чего он посчитать не может.

    Свой класс, а не ValueError и не RuntimeError: заготовка бросает
    NotImplementedError, а он наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """


def cosine(a, b):
    """Косинус угла между двумя векторами: мера «про одно ли это».

    cosine([1, 0], [0, 1])   ->  0.0    (перпендикулярны, ничего общего)
    cosine([1, 0], [2, 0])   ->  1.0    (то же направление, длина не важна)
    cosine([3, 4], [6, 8])   ->  1.0

    Ловушка: нулевой вектор. Угол к нему не определён, деление на ноль даст
    ZeroDivisionError или nan — оба варианта отравят порог. Бросай CacheError.
    Разная длина векторов — тоже CacheError: это разные модели эмбеддингов.

    Именно косинус, а не равенство строк, делает кэш семантическим. Кэш на
    хеше строки не отдаст неверный ответ никогда — и попадёт в 2% случаев.
    """
    if len(a) != len(b):
        raise CacheError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise CacheError("zero vector has no direction")
    return dot / (na * nb)


def nearest_entry(entries, vector):
    """Ближайшая запись кэша к вектору: (запись, близость).

    entries — список dict с ключами "vector" и "answer".

    nearest_entry([], v)                  ->  (None, -1.0)
    nearest_entry([e1, e2], e1["vector"])  ->  (e1, 1.0)

    Ловушка: на пустом кэше вернуть None и близость -1.0, а не бросать и не
    возвращать 0.0. Ноль — это «перпендикулярно», то есть осмысленная
    близость, и порог 0.0 на пустом кэше выдал бы попадание в пустоту.

    При равной близости выигрывает более ранняя запись: линейный поиск
    сравнивает строго больше. В настоящем ANN-индексе порядок другой, но
    воспроизводимость важнее.
    """
    best = None
    best_sim = -1.0
    for entry in entries:
        sim = cosine(entry["vector"], vector)
        if sim > best_sim:
            best_sim = sim
            best = entry
    return (best, best_sim)


def semantic_lookup(entries, vector, threshold):
    """Поиск в L1: (запись, близость), если близость >= threshold, иначе (None, близость).

    Близость возвращаем всегда — по ней настраивают порог, и без неё
    непонятно, промах был «далеко» или «чуть-чуть не дотянул».

    semantic_lookup([e], e["vector"], 0.95)   ->  (e, 1.0)
    semantic_lookup([e], far_vector, 0.95)    ->  (None, 0.70)

    Порог — единственная ручка между экономией и враньём. Чем ниже, тем чаще
    попадания и тем чаще ответ не про то, что спросили.
    """
    entry, sim = nearest_entry(entries, vector)
    if entry is not None and sim >= threshold:
        return (entry, sim)
    return (None, sim)


def run_semantic_cache(queries, threshold):
    """Прогнать поток запросов через L1 и записать, что получил пользователь.

    queries — список dict {"vector": [...], "answer": "..."}, где answer это
    ЧЕСТНЫЙ ответ на этот запрос (то, что вернула бы LLM).

    Правило: попали в кэш — отдаём сохранённый ответ и LLM не зовём; промах —
    зовём LLM, отдаём честный ответ и кладём запрос в кэш.

    Возвращает список записей:
      {"served": "cache"|"llm", "answer", "expected", "correct", "similarity"}

    Ловушка: класть в кэш надо ТОЛЬКО промахи. Если писать и попадания тоже,
    кэш засорится дублями и близость перестанет отражать разнообразие.

    "correct" — то, ради чего этот прогон и нужен: на низком пороге кэш
    отдаёт ответ соседа, пользователь получает чужой текст, и ни один
    счётчик hit rate этого не покажет.
    """
    entries = []
    records = []
    for q in queries:
        entry, sim = semantic_lookup(entries, q["vector"], threshold)
        if entry is not None:
            answer = entry["answer"]
            served = "cache"
        else:
            # промах: «зовём LLM» и запоминаем пару вектор-ответ
            answer = q["answer"]
            served = "llm"
            entries.append({"vector": list(q["vector"]), "answer": q["answer"]})
        records.append({
            "served": served,
            "answer": answer,
            "expected": q["answer"],
            "correct": answer == q["answer"],
            "similarity": sim,
        })
    return records


def cache_stats(records):
    """Свернуть записи прогона в отчёт.

    Возвращает dict:
      total, hits, misses, llm_calls,
      hit_rate       — hits / total, 0.0 на пустом входе,
      false_hits     — попадания, отдавшие ЧУЖОЙ ответ,
      false_hit_rate — false_hits / hits, 0.0 если попаданий не было.

    cache_stats([])["hit_rate"]  ->  0.0

    Ловушка со знаменателем: false_hit_rate считается от ПОПАДАНИЙ, а не от
    всех запросов. Вендорские «95% accuracy» — это про попадания. Кэш с 5%
    hit rate и 20% ложных выглядит в отчёте «от всех запросов» как 1% ошибок
    и кажется безобидным.
    """
    total = len(records)
    hits = sum(1 for r in records if r["served"] == "cache")
    false_hits = sum(1 for r in records if r["served"] == "cache" and not r["correct"])
    return {
        "total": total,
        "hits": hits,
        "misses": total - hits,
        "llm_calls": total - hits,
        "hit_rate": hits / total if total else 0.0,
        "false_hits": false_hits,
        "false_hit_rate": false_hits / hits if hits else 0.0,
    }


def common_prefix_tokens(a, b):
    """Сколько ведущих токенов совпадает у двух промптов.

    common_prefix_tokens(["Ты", "ассистент", "."], ["Ты", "ассистент", "!"])  ->  2
    common_prefix_tokens(["14:32", "Ты"], ["14:33", "Ты"])                    ->  0

    Второй пример — весь анти-паттерн динамического содержимого в одной
    строке: время в начале промпта обнуляет общий префикс, и провайдеру
    нечего переиспользовать, даже если дальше идут четыре тысячи одинаковых
    токенов. Тот же текст со временем В КОНЦЕ кэшируется целиком.

    Провайдер сравнивает префикс побайтово, а не по смыслу: L1 и L2 — разные
    механизмы, и «почти совпало» здесь не считается.
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def l2_request_cost(prefix_tokens, dynamic_tokens, output_tokens, ttl, prefix_cached):
    """Цена одного запроса при кэше префикса у провайдера, в долларах.

    Префикс в кэше — платим цену чтения. Не в кэше — платим запись с премией
    за TTL. Динамический хвост и выход всегда по базовой цене.

    l2_request_cost(4000, 200, 200, "5min", False)  ->  0.0186   (холодный)
    l2_request_cost(4000, 200, 200, "5min", True)   ->  0.0048   (тёплый)

    Разбор холодного: 4000/1e6 * 3.00 * 1.25 = 0.015 за запись, плюс
    200/1e6 * 3.00 = 0.0006 за хвост, плюс 200/1e6 * 15.00 = 0.003 за выход.

    Ловушка: неизвестный ttl — CacheError, а не «возьмём 1.0». Молчаливая
    единица занизит счёт ровно на премию за запись, то есть на то, что мы
    и считаем.
    """
    if ttl not in WRITE_MULTIPLIER:
        raise CacheError(f"unknown ttl {ttl!r}, expected one of {tuple(WRITE_MULTIPLIER)}")
    if prefix_cached:
        prefix_cost = prefix_tokens / 1e6 * PRICE_CACHED_READ
    else:
        prefix_cost = prefix_tokens / 1e6 * PRICE_INPUT * WRITE_MULTIPLIER[ttl]
    return (
        prefix_cost
        + dynamic_tokens / 1e6 * PRICE_INPUT
        + output_tokens / 1e6 * PRICE_OUTPUT
    )


def parallel_wave_cost(n, prefix_tokens, dynamic_tokens, output_tokens, ttl, serialize_first):
    """Цена волны из n запросов с общим префиксом.

    serialize_first=False — все n прилетают одновременно, запись первого ещё
    не встала, каждый платит премию за запись. Это и есть анти-паттерн
    параллелизации.

    serialize_first=True — первый идёт один, остальные n-1 читают готовый
    префикс.

    parallel_wave_cost(10, 4000, 50, 50, "1hr", False)  ->  0.249
    parallel_wave_cost(10, 4000, 50, 50, "1hr", True)   ->  0.0438

    То есть исправление стоит одной задержки на первом вызове и экономит
    почти шестикратно. Ловушка: n <= 0 — CacheError, «волна из нуля
    запросов» это ошибка вызывающего, а не бесплатный запрос.
    """
    if n <= 0:
        raise CacheError(f"wave size must be positive, got {n}")
    if not serialize_first:
        return n * l2_request_cost(prefix_tokens, dynamic_tokens, output_tokens, ttl, False)
    first = l2_request_cost(prefix_tokens, dynamic_tokens, output_tokens, ttl, False)
    rest = l2_request_cost(prefix_tokens, dynamic_tokens, output_tokens, ttl, True)
    return first + (n - 1) * rest
