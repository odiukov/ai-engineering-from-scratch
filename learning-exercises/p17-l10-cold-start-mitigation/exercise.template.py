"""
Холодный старт serverless-LLM и как его лечить

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l10-cold-start-mitigation
Разбор:  /check-code p17-l10-cold-start-mitigation
"""

import math

PHASES_70B = {
    "node provision": 50.0,
    "image pull": 180.0,
    "weights to HBM": 75.0,
    "engine init": 20.0,
    "first forward": 3.0,
}
MITIGATIONS = {
    # Bottlerocket: образ уже лежит на локальном NVMe — тянуть нечего.
    "pre_seeded": {"image pull": 0.0},
    # Run:ai Model Streamer: загрузка весов идёт внахлёст с инициализацией.
    "streamer": {"weights to HBM": 0.5},
    # Modal-style снапшот HBM: весов грузить не надо, движок уже прогрет.
    # Ноду всё равно кто-то должен выдать, поэтому node provision не трогаем.
    "gpu_snapshot": {"image pull": 0.0, "weights to HBM": 0.0, "engine init": 0.1},
}
HOURS_IN_MONTH = 24 * 30


class UnknownMitigationError(Exception):
    """Запрошена мера, которой нет в MITIGATIONS.

    Свой класс, а не KeyError и не RuntimeError: NotImplementedError сам
    является RuntimeError, и тест на RuntimeError позеленел бы на пустой
    заготовке, ничего не проверив.
    """
    pass


def weights_load_seconds(model_gb, read_gb_s):
    """Сколько секунд веса едут с диска в HBM.

    weights_load_seconds(140.0, 7.0)   ->  20.0   (70B в BF16 с NVMe 7 GB/s)
    weights_load_seconds(35.0, 7.0)    ->   5.0   (та же модель в INT4)

    Это чистая арифметика «объём делить на скорость», и именно она объясняет,
    почему квантование сокращает не только счёт за память, но и холодный
    старт.

    Нулевая скорость чтения — ValueError, а не бесконечность.
    """
    raise NotImplementedError


def cold_start_seconds(phases, mitigations=()):
    """Суммарный холодный старт после применения мер.

    phases — словарь «фаза -> секунды» (см. PHASES_70B), mitigations —
    имена из MITIGATIONS. Множители разных мер к одной фазе перемножаются.

    cold_start_seconds(PHASES_70B)                    ->  328.0
    cold_start_seconds(PHASES_70B, ["pre_seeded"])    ->  148.0
    cold_start_seconds(PHASES_70B, ["gpu_snapshot"])  ->   55.0

    Незнакомое имя меры — UnknownMitigationError. Молча игнорировать опечатку
    нельзя: план по SLA будет посчитан по несуществующей оптимизации.
    """
    raise NotImplementedError


def mitigation_savings(phases, mitigations):
    """Сколько секунд сэкономил стек мер относительно голого старта.

    mitigation_savings(PHASES_70B, ["pre_seeded"])  ->  180.0

    Считается через cold_start_seconds, а не отдельной формулой: две формулы
    одного и того же обязательно разъедутся.
    """
    raise NotImplementedError


def ready_at(started_at, cold_seconds):
    """Момент, когда стартовавшая реплика начнёт отвечать.

    ready_at(0.0, 328.0)    ->  328.0
    ready_at(600.0, 15.0)   ->  615.0

    Время в этом уроке всегда приходит аргументом. Никакого time.time():
    модель должна давать один и тот же ответ через год после запуска.
    """
    raise NotImplementedError


def available_replicas(ready_times, now):
    """Сколько реплик уже готовы к моменту now.

    available_replicas([0.0, 328.0, 700.0], 400.0)  ->  2
    available_replicas([328.0], 328.0)              ->  1   (ровно готова)

    Момент готовности включается: реплика, у которой ready_at == now, уже
    принимает трафик.
    """
    raise NotImplementedError


def simulate_arrivals(arrivals, capacity_per_replica, warm_pool, cold_seconds,
                      slot_seconds, start_time=0.0):
    """Прогнать профиль нагрузки и посчитать долю холодных запросов.

    arrivals — сколько запросов пришло в каждом слоте времени.
    Модель: в слоте i (момент start_time + i*slot_seconds) обслуживают только
    уже готовые реплики. Запросы сверх их ёмкости считаются холодными — они
    ждут, пока поднимется новая реплика. Реплики поднимаются по мере
    надобности и обратно не гасятся.

    Вернуть {"total", "cold", "cold_share", "peak_replicas"}.

    simulate_arrivals([10, 10], 10, 1, 300.0, 60.0)["cold"]  ->  0
    simulate_arrivals([10, 10], 10, 0, 300.0, 60.0)["cold"]  ->  20

    Ровно это и есть цена scale-to-zero: пока реплика греется, запросы либо
    ждут минутами, либо отваливаются по таймауту.
    """
    raise NotImplementedError


def warm_pool_monthly_cost(warm_pool, usd_per_gpu_hour, hours=HOURS_IN_MONTH):
    """Во что обходится тёплый пул за месяц простоя.

    warm_pool_monthly_cost(1, 4.50)  ->  3240.0
    warm_pool_monthly_cost(5, 4.50)  ->  16200.0

    Реплика тарифицируется круглосуточно независимо от того, позвал её
    кто-нибудь или нет. Пять продуктов по одной тёплой реплике — это
    5 * 720 = 3600 GPU-часов в месяц.
    """
    raise NotImplementedError


def min_warm_pool_for(target_cold_share, arrivals, capacity_per_replica,
                      cold_seconds, slot_seconds, start_time=0.0):
    """Наименьший тёплый пул, при котором доля холодных запросов не выше цели.

    min_warm_pool_for(0.0, [10, 10], 10, 300.0, 60.0)   ->  1
    min_warm_pool_for(0.5, [10, 10], 10, 300.0, 60.0)   ->  1

    Перебор снизу вверх до пика спроса: пул размером с пик убирает холодные
    старты полностью, всё, что меньше, оставляет хвост.

    Так и выглядит выбор из урока: платить за простаивающие GPU или мириться
    с холодным хвостом. Функция считает цену первого варианта в репликах.
    """
    raise NotImplementedError
