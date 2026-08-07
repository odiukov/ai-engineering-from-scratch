"""
Мультирегиональный инференс и локальность KV-кэша — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import math

# TTFT на промпте в 2k токенов, Llama 3.3 70B FP8 на H100 (числа из урока).
CACHE_HIT_MS = 80.0
CACHE_MISS_MS = 800.0

REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1")

# RTT между регионами в миллисекундах: время туда-обратно, не в одну сторону.
RTT_MS = {
    ("us-east-1", "us-west-2"): 65.0,
    ("us-east-1", "eu-west-1"): 75.0,
    ("us-east-1", "ap-southeast-1"): 220.0,
    ("us-west-2", "eu-west-1"): 130.0,
    ("us-west-2", "ap-southeast-1"): 170.0,
    ("eu-west-1", "ap-southeast-1"): 250.0,
}

# Минимальный DR-манифест: 32% провалов восстановления — это не веса,
# а вот эти файлы вокруг них.
REQUIRED_DR_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "quantize_config.json",
    "chat_template.jinja",
    "vllm_config.yaml",
    "deployment.yaml",
)

ROUTING_STRATEGIES = ("round_robin", "cache_aware", "cache_aware_regional")


class UnknownRegionError(Exception):
    """В таблице RTT нет такой пары регионов.

    Свой класс, а не KeyError и не RuntimeError: NotImplementedError сам
    наследуется от RuntimeError, и тест на RuntimeError позеленел бы на
    пустой заготовке.
    """


class NoEligibleReplicaError(Exception):
    """Роутеру некуда отправить запрос: все реплики отсечены ограничениями."""


def prefix_key(tokens, prefix_len):
    """Ключ префикс-кэша: устойчивый хеш первых prefix_len токенов.

    prefix_key([1, 2, 3], 2) == prefix_key([1, 2, 9, 9], 2)   ->  True
    prefix_key([1, 2, 3], 2) == prefix_key([1, 5], 2)         ->  False

    Список короче prefix_len хешируется целиком.

    Ловушка: встроенный hash() для строк солится при каждом запуске Python
    (PYTHONHASHSEED), поэтому роутер на нём будет по-разному раскладывать
    один и тот же трафик между перезапусками. Нужен именно устойчивый хеш —
    hashlib.
    """
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    head = tuple(tokens[:prefix_len])
    raw = ",".join(str(t) for t in head).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def rtt_ms(a, b, table=RTT_MS):
    """RTT между регионами. Внутри одного региона — ноль.

    rtt_ms("us-east-1", "us-east-1")     ->  0.0
    rtt_ms("us-east-1", "eu-west-1")     ->  75.0
    rtt_ms("eu-west-1", "us-east-1")     ->  75.0   (таблица симметрична)

    Пары в таблице записаны один раз, поэтому проверять надо оба порядка.
    Незнакомая пара — UnknownRegionError: подставлять «ну пусть 200»
    нельзя, на этом числе принимается решение о маршруте.
    """
    if a == b:
        return 0.0
    if (a, b) in table:
        return table[(a, b)]
    if (b, a) in table:
        return table[(b, a)]
    raise UnknownRegionError(f"no RTT between {a} and {b}")


def expected_ttft_ms(cache_hit, origin, target_region, table=RTT_MS):
    """Ожидаемый TTFT: prefill (с кэшем или без) плюс сеть.

    expected_ttft_ms(True, "us-east-1", "us-east-1")   ->  80.0
    expected_ttft_ms(False, "us-east-1", "us-east-1")  ->  800.0
    expected_ttft_ms(True, "us-east-1", "eu-west-1")   ->  155.0

    Здесь и живёт главный вывод урока: попадание в кэш экономит 720 мс, а
    перелёт через Атлантику стоит 75. Считать надо СУММУ, а не выбирать
    ближайший регион и не гнаться за кэшем любой ценой.
    """
    prefill = CACHE_HIT_MS if cache_hit else CACHE_MISS_MS
    return prefill + rtt_ms(origin, target_region, table)


def route_round_robin(index, replicas):
    """Круговая раскладка: вернуть ИНДЕКС реплики номер index по модулю.

    route_round_robin(0, replicas)  ->  0
    route_round_robin(5, replicas)  ->  5 % len(replicas)

    Оптимально для stateless-сервисов и вредно для инференса: KV-кэш живёт
    в конкретной реплике, а этот роутер про него не знает.

    Индекс, а не сама реплика: так же устроен и route_cache_aware, и обе
    стратегии подставляются в симуляцию без переходников.

    Пустой список реплик — NoEligibleReplicaError.
    """
    if not replicas:
        raise NoEligibleReplicaError("no replicas to route to")
    return index % len(replicas)


def route_cache_aware(prefix, origin, replicas, caches, table=RTT_MS,
                      residency_bound=False):
    """Выбрать реплику с наименьшим ожидаемым TTFT.

    caches — список списков ключей, параллельный replicas: что сейчас лежит
    в префикс-кэше каждой реплики.
    residency_bound=True запрещает выпускать запрос за пределы региона
    происхождения (GDPR важнее TTFT).

    Возвращает индекс выбранной реплики. Если после фильтра по residency не
    осталось реплик — NoEligibleReplicaError.

    Роутер сравнивает именно expected_ttft_ms, поэтому дальний регион с
    кэшем выигрывает у соседнего без кэша, а очень дальний (RTT больше
    720 мс) проигрывает даже локальному промаху.

    Равные кандидаты (типичный случай: холодный префикс, все реплики
    региона одинаково пусты) разводятся по хешу префикса, а не по порядку в
    списке. Иначе все промахи свалятся на первую реплику, она затрёт свой
    кэш, и cache-aware роутер выродится в один горячий узел.
    """
    costs = []
    for i, replica in enumerate(replicas):
        if residency_bound and replica["region"] != origin:
            continue
        costs.append((expected_ttft_ms(prefix in caches[i], origin,
                                       replica["region"], table), i))
    if not costs:
        raise NoEligibleReplicaError(f"no replica available for origin {origin}")
    best_cost = min(cost for cost, _ in costs)
    candidates = [i for cost, i in costs if cost == best_cost]
    if len(candidates) == 1:
        return candidates[0]
    spread = int(hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:8], 16)
    return candidates[spread % len(candidates)]


def percentile(values, p):
    """Перцентиль по методу nearest-rank, p в процентах.

    percentile([1, 2, 3, 4, 5], 50)   ->  3
    percentile([1, 2, 3, 4, 5], 100)  ->  5

    Без интерполяции: перцентиль — это реально наблюдавшееся значение.
    Пустая выборка — ValueError.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0 <= p <= 100:
        raise ValueError("p must be a percentage between 0 and 100")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100.0 * len(ordered)) - 1)]


def simulate(requests, replicas, strategy, cache_capacity=8, prefix_len=512,
             table=RTT_MS):
    """Прогнать трафик через роутер и собрать статистику.

    requests — список словарей {"origin": регион, "tokens": список токенов}.
    strategy — одна из ROUTING_STRATEGIES.

    Возвращает {"hit_rate", "mean_ttft", "p50_ttft", "p99_ttft", "cross_region"}.

    Кэш реплики — список ключей с вытеснением по LRU: попадание двигает ключ
    в конец, промах дописывает его и выкидывает самый старый. Сам список
    replicas не мутируется, состояние кэшей живёт внутри симуляции — иначе
    два прогона подряд давали бы разные числа.

    Незнакомая стратегия — ValueError.
    """
    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    caches = [[] for _ in replicas]
    hits = 0
    cross = 0
    ttfts = []
    for i, request in enumerate(requests):
        prefix = prefix_key(request["tokens"], prefix_len)
        origin = request["origin"]
        if strategy == "round_robin":
            chosen = route_round_robin(i, replicas)
        else:
            chosen = route_cache_aware(
                prefix, origin, replicas, caches, table,
                residency_bound=(strategy == "cache_aware_regional"),
            )
        cache = caches[chosen]
        hit = prefix in cache
        ttfts.append(expected_ttft_ms(hit, origin, replicas[chosen]["region"], table))
        if hit:
            hits += 1
            cache.remove(prefix)     # LRU: свежее обращение — в конец списка
        cache.append(prefix)
        while len(cache) > cache_capacity:
            cache.pop(0)
        if replicas[chosen]["region"] != origin:
            cross += 1
    return {
        "hit_rate": hits / len(requests),
        "mean_ttft": sum(ttfts) / len(ttfts),
        "p50_ttft": percentile(ttfts, 50),
        "p99_ttft": percentile(ttfts, 99),
        "cross_region": cross,
    }


def dr_manifest_gaps(backed_up, required=REQUIRED_DR_FILES):
    """Чего не хватает в бэкапе, чтобы поднять модель в другом регионе.

    dr_manifest_gaps(["model.safetensors"])  ->  список из шести файлов
    dr_manifest_gaps(REQUIRED_DR_FILES)      ->  []

    Возвращает отсортированный список недостающих имён.

    32% провалов DR у LLM — это не потерянные веса, а забытый tokenizer.json
    и конфиг квантования: реплика просто отказывается стартовать.
    """
    present = set(backed_up)
    return sorted(name for name in required if name not in present)
