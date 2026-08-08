"""
Многоуровневый KV-кэш: GPU, CPU, диск и цена попадания — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих системах:

    recompute_ms      <-  повторный prefill того же префикса: то, что движок
                          делает, когда KV из HBM выселили
    restore_ms        <-  подъём KV с уровня через Connector API vLLM 0.9.0+:
                          host RAM, потом Ceph/S3
    effective_hit_ms  <-  трезвый движок: если поднять дороже, чем посчитать
                          заново, он считает заново. Попадание в кэш и
                          выигрыш от кэша — разные вещи
    make_cache        <-  иерархия LMCache: HBM -> CPU DRAM -> диск
    cache_put         <-  вытеснение при нехватке HBM, каскадом вниз
    serve_request     <-  один запрос: попадание, промоушен обратно наверх
    run_workload      <-  бенчмарк 16xH100 из урока в миниатюре: где LMCache
                          начинает окупаться и где он бесполезен

Ни vLLM, ни LMCache, ни сети: иерархия кэша — это словари с ёмкостью и
политикой вытеснения, и она моделируется честно. Никаких sleep, всё время —
числа.

Числа — снимок H100-класса на 2026, они дрейфуют.
"""

import math

# 80 слоёв * 2 (K/V) * 8 KV-голов * 128 * 1 байт FP8.
KV_BYTES_PER_TOKEN = 163_840

# Скорость повторного prefill на GPU, токенов в секунду.
PREFILL_TPS = 8_000.0

# Уровни строго сверху вниз. Порядок важен: вытеснение идёт по нему.
TIERS = ("gpu", "cpu", "disk")

# Полоса чтения уровня, ГБ/с. У GPU бесконечность: KV уже лежит в HBM,
# везти нечего.
TIER_GBPS = {"gpu": math.inf, "cpu": 50.0, "disk": 1.5}

# Постоянная часть чтения: обращение к аллокатору, сетевой хоп до LMCache,
# поиск объекта в Ceph.
TIER_SETUP_MS = {"gpu": 0.0, "cpu": 0.5, "disk": 40.0}


class CacheError(Exception):
    """Кэш спрошен о невозможном: чужой уровень, отрицательная ёмкость.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """


def recompute_ms(tokens, prefill_tps=PREFILL_TPS):
    """Сколько стоит посчитать префикс заново. Опорная цена для всего урока.

    recompute_ms(4000)  ->  500.0
    recompute_ms(1000)  ->  125.0

    Это та работа, которую кэш пытается не делать: 0.125 мс на токен.
    Всё остальное в уроке сравнивается с этим числом.

    CacheError на отрицательные токены и неположительную скорость.
    """
    if tokens < 0:
        raise CacheError(f"tokens must be non-negative, got {tokens}")
    if prefill_tps <= 0:
        raise CacheError(f"prefill_tps must be positive, got {prefill_tps}")
    return tokens / prefill_tps * 1000.0


def restore_ms(tokens, tier, bytes_per_token=KV_BYTES_PER_TOKEN):
    """Сколько стоит поднять KV с уровня в HBM.

    restore_ms(4000, "gpu")   ->  0.0     (уже в HBM, поднимать нечего)
    restore_ms(4000, "cpu")   ->  13.6072 (0.5 + 655.36 МБ по 50 ГБ/с)
    restore_ms(4000, "disk")  ->  примерно 476.91  (40 + 655.36 МБ по 1.5 ГБ/с)

    Сравни с recompute_ms(4000) = 500.0: CPU дешевле пересчёта примерно в 37 раз,
    диск — всего в 1.05 раза, и то не всегда.

    Ловушка: у диска огромное ПОСТОЯННОЕ слагаемое в 40 мс. На коротком
    префиксе везти почти нечего, а хоп до Ceph стоит столько же, сколько
    пересчёт 320 токенов. Из-за этого у диска есть длина, ниже которой он
    бесполезен, — см. effective_hit_ms.

    CacheError на уровень не из TIERS и на отрицательные токены.
    """
    if tier not in TIERS:
        raise CacheError(f"unknown tier {tier!r}")
    if tokens < 0:
        raise CacheError(f"tokens must be non-negative, got {tokens}")
    seconds = tokens * bytes_per_token / (TIER_GBPS[tier] * 1e9)
    return TIER_SETUP_MS[tier] + seconds * 1000.0


def effective_hit_ms(tokens, tier):
    """Во что реально обойдётся попадание: минимум из подъёма и пересчёта.

    effective_hit_ms(4000, "disk")  ->  примерно 476.91  (подъём дешевле)
    effective_hit_ms(500, "disk")   ->  62.5             (пересчёт дешевле!)
    effective_hit_ms(4000, "cpu")   ->  13.6072

    Диск обгоняет пересчёт начиная с 2536 токенов:
    40 / (0.125 - 0.109227) = 2535.93.
    Ниже этой длины попадание на диске есть, а выигрыша нет — движок молча
    считает заново. Это уточняет правило урока про короткие контексты:
    transfer time > re-prefill, только с точно посчитанной границей.

    Ловушка мышления: «hit rate 90%» ничего не говорит о выигрыше, пока не
    сказано, на каком уровне эти попадания и какой длины контексты. Попадание
    на диске на коротком префиксе экономит ровно ноль.
    """
    return min(restore_ms(tokens, tier), recompute_ms(tokens))


def make_cache(gpu_tokens, cpu_tokens, disk_tokens):
    """Пустая трёхуровневая иерархия с ёмкостями в ТОКЕНАХ.

    make_cache(100_000, 500_000, 0)["capacity"]["disk"]  ->  0
    make_cache(0, 0, 10_000)                             ->  только диск

    Ёмкость в токенах, а не в записях: в KV-кэше место занимает контекст, и
    один запрос на 32К токенов вытеснит тридцать запросов по 1К.

    Нулевая ёмкость — законная конфигурация: так выключают уровень. Кэш из
    одного GPU-уровня — это обычный vLLM без LMCache, и с ним надо уметь
    сравнивать.

    CacheError на отрицательную ёмкость.
    """
    capacity = {"gpu": gpu_tokens, "cpu": cpu_tokens, "disk": disk_tokens}
    for tier, size in capacity.items():
        if size < 0:
            raise CacheError(f"tier {tier}: capacity must be non-negative, got {size}")
    # словарь на уровень: ключ -> размер в токенах. Порядок вставки = порядок
    # LRU, самый старый первым — питоновский dict это гарантирует.
    return {"capacity": capacity, "tiers": {tier: {} for tier in TIERS}}


def cache_lookup(cache, key):
    """На каком уровне лежит префикс. None — нигде. Кэш не меняет.

    cache_lookup(make_cache(1000, 0, 0), "doc")  ->  None

    Отдельная функция, а не флажок внутри serve_request, потому что роутер из
    урока 11 задаёт этот вопрос ДО того, как решит, на какой движок послать
    запрос, и ответ не должен ничего сдвигать в LRU.
    """
    for tier in TIERS:
        if key in cache["tiers"][tier]:
            return tier
    return None


def cache_put(cache, key, tokens):
    """Положить префикс на самый верхний подходящий уровень. Возвращает НОВЫЙ кэш.

    Правила:
      * ключ сначала удаляется отовсюду — повторная запись это промоушен,
        а не второй экземпляр;
      * запись идёт на первый сверху уровень, чья ЁМКОСТЬ её вмещает;
      * не хватает свободного места — вытесняем самую старую запись уровня,
        и она падает на следующий уровень, где всё повторяется;
      * упавшее с диска пропадает.

    cache_put(make_cache(2000, 0, 0), "a", 1000)["tiers"]["gpu"]  ->  {'a': 1000}
    cache_put(make_cache(0, 0, 5000), "a", 1000)["tiers"]["disk"] ->  {'a': 1000}
    cache_put(make_cache(100, 100, 100), "a", 9999)               ->  нигде

    Ловушка первая: вытеснение КАСКАДНОЕ. Вытолкнутое с GPU не исчезает, оно
    ложится на CPU и уже там может вытолкнуть кого-то на диск. Реализация,
    которая просто выбрасывает жертву, покажет отличный hit rate на GPU и
    никакого выигрыша на длинном прогоне.

    Ловушка вторая: запись, которая длиннее ёмкости уровня, не помещается туда
    НИКОГДА — её нельзя пытаться впихнуть, вытеснив весь уровень. Иначе один
    запрос с контекстом на 128К вымоет весь кэш ради самого себя и не влезет.

    CacheError на неположительный размер записи.
    """
    if tokens <= 0:
        raise CacheError(f"entry {key!r}: tokens must be positive, got {tokens}")
    new = {
        "capacity": dict(cache["capacity"]),
        "tiers": {tier: dict(cache["tiers"][tier]) for tier in TIERS},
    }
    for tier in TIERS:
        new["tiers"][tier].pop(key, None)

    # очередь «кого куда пристроить»: (ключ, размер, с какого уровня искать)
    pending = [(key, tokens, 0)]
    while pending:
        k, n, level = pending.pop(0)
        # пропускаем уровни, которые не вмещают запись даже пустыми
        while level < len(TIERS) and n > new["capacity"][TIERS[level]]:
            level += 1
        if level == len(TIERS):
            continue                      # не влезает никуда — теряем
        tier = TIERS[level]
        used = sum(new["tiers"][tier].values())
        while used + n > new["capacity"][tier]:
            # первый ключ словаря — самый давно вставленный, он же LRU
            victim = next(iter(new["tiers"][tier]))
            victim_tokens = new["tiers"][tier].pop(victim)
            used -= victim_tokens
            # жертва приходит вниз как свежая: её только что тронули
            # вытеснением. Класть её в начало нижнего уровня значило бы
            # выбрасывать её при первом же всплеске.
            pending.append((victim, victim_tokens, level + 1))
        new["tiers"][tier][k] = n
    return new


def serve_request(cache, key, tokens):
    """Обслужить один запрос. Возвращает пару (новый кэш, отчёт).

    Отчёт: key, tier (None при промахе), hit, cost_ms, baseline_ms, saved_ms.

    Промах:     cost_ms = recompute_ms(tokens), saved_ms = 0.0
    Попадание:  cost_ms = effective_hit_ms(tokens, tier)

    После любого исхода префикс кладётся обратно наверх — это промоушен, из-за
    которого горячие контексты сами всплывают в HBM, а холодные тонут.

    Ловушка: saved_ms бывает НУЛЁМ при попадании. Короткий контекст на диске
    поднимать дороже, чем посчитать заново, и движок его считает. Отчёт,
    который считает такие попадания экономией, врёт про пользу LMCache ровно
    там, где урок предупреждает не включать его.
    """
    tier = cache_lookup(cache, key)
    baseline = recompute_ms(tokens)
    cost = baseline if tier is None else effective_hit_ms(tokens, tier)
    report = {
        "key": key,
        "tier": tier,
        "hit": tier is not None,
        "cost_ms": cost,
        "baseline_ms": baseline,
        "saved_ms": baseline - cost,
    }
    return cache_put(cache, key, tokens), report


def run_workload(cache, requests):
    """Прогнать поток запросов через иерархию. Возвращает (новый кэш, отчёт).

    requests — список пар (key, tokens).

    Отчёт:
      requests, misses,
      hits_by_tier   — счётчик попаданий по всем уровням, нули включены,
      hit_rate       — доля попаданий,
      wasted_hits    — попадания, сэкономившие ровно ноль,
      total_ms       — что заплатили,
      baseline_ms    — что заплатили бы вообще без кэша,
      saved_ms/pct   — экономия.

    run_workload(make_cache(0, 0, 0), [])["saved_pct"]  ->  0.0

    Главное свойство, ради которого урок: нижние уровни дают прирост ТОЛЬКО
    когда KV не влезает в HBM. Пока GPU-уровень вмещает рабочее множество, все
    попадания приходят на gpu, а cpu и disk стоят пустые — включать LMCache
    здесь не за чем. Стоит ужать GPU-уровень, и та же смесь запросов начинает
    жить на CPU. Обе стороны проверены тестами.

    Разница hit_rate и saved_pct — вторая половина сюжета. Диск поднимает
    hit_rate, но на коротких контекстах не приносит ни миллисекунды: смотри
    wasted_hits.
    """
    hits_by_tier = {tier: 0 for tier in TIERS}
    misses = wasted = 0
    total = baseline = 0.0
    for key, tokens in requests:
        cache, report = serve_request(cache, key, tokens)
        total += report["cost_ms"]
        baseline += report["baseline_ms"]
        if report["hit"]:
            hits_by_tier[report["tier"]] += 1
            wasted += int(report["saved_ms"] == 0.0)
        else:
            misses += 1
    n = len(requests)
    hits = sum(hits_by_tier.values())
    return cache, {
        "requests": n,
        "misses": misses,
        "hits_by_tier": hits_by_tier,
        "hit_rate": hits / n if n else 0.0,
        "wasted_hits": wasted,
        "total_ms": total,
        "baseline_ms": baseline,
        "saved_ms": baseline - total,
        "saved_pct": (baseline - total) / baseline * 100 if baseline else 0.0,
    }
