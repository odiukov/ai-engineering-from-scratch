"""
Мультирегиональный инференс и локальность KV-кэша

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l11-multi-region-kv-locality
Разбор:  /check-code p17-l11-multi-region-kv-locality
"""

import hashlib
import math

CACHE_HIT_MS = 80.0
CACHE_MISS_MS = 800.0
REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1")
RTT_MS = {
    ("us-east-1", "us-west-2"): 65.0,
    ("us-east-1", "eu-west-1"): 75.0,
    ("us-east-1", "ap-southeast-1"): 220.0,
    ("us-west-2", "eu-west-1"): 130.0,
    ("us-west-2", "ap-southeast-1"): 170.0,
    ("eu-west-1", "ap-southeast-1"): 250.0,
}
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
    pass


class NoEligibleReplicaError(Exception):
    """Роутеру некуда отправить запрос: все реплики отсечены ограничениями."""
    pass


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
    raise NotImplementedError


def rtt_ms(a, b, table=RTT_MS):
    """RTT между регионами. Внутри одного региона — ноль.

    rtt_ms("us-east-1", "us-east-1")     ->  0.0
    rtt_ms("us-east-1", "eu-west-1")     ->  75.0
    rtt_ms("eu-west-1", "us-east-1")     ->  75.0   (таблица симметрична)

    Пары в таблице записаны один раз, поэтому проверять надо оба порядка.
    Незнакомая пара — UnknownRegionError: подставлять «ну пусть 200»
    нельзя, на этом числе принимается решение о маршруте.
    """
    raise NotImplementedError


def expected_ttft_ms(cache_hit, origin, target_region, table=RTT_MS):
    """Ожидаемый TTFT: prefill (с кэшем или без) плюс сеть.

    expected_ttft_ms(True, "us-east-1", "us-east-1")   ->  80.0
    expected_ttft_ms(False, "us-east-1", "us-east-1")  ->  800.0
    expected_ttft_ms(True, "us-east-1", "eu-west-1")   ->  155.0

    Здесь и живёт главный вывод урока: попадание в кэш экономит 720 мс, а
    перелёт через Атлантику стоит 75. Считать надо СУММУ, а не выбирать
    ближайший регион и не гнаться за кэшем любой ценой.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def percentile(values, p):
    """Перцентиль по методу nearest-rank, p в процентах.

    percentile([1, 2, 3, 4, 5], 50)   ->  3
    percentile([1, 2, 3, 4, 5], 100)  ->  5

    Без интерполяции: перцентиль — это реально наблюдавшееся значение.
    Пустая выборка — ValueError.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def dr_manifest_gaps(backed_up, required=REQUIRED_DR_FILES):
    """Чего не хватает в бэкапе, чтобы поднять модель в другом регионе.

    dr_manifest_gaps(["model.safetensors"])  ->  список из шести файлов
    dr_manifest_gaps(REQUIRED_DR_FILES)      ->  []

    Возвращает отсортированный список недостающих имён.

    32% провалов DR у LLM — это не потерянные веса, а забытый tokenizer.json
    и конфиг квантования: реплика просто отказывается стартовать.
    """
    raise NotImplementedError
