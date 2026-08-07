"""
GPU-автоскейлинг в Kubernetes: сигнал, окно стабилизации и gang scheduling — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Ниже этой утилизации политика WhenEmptyOrUnderutilized считает узел
# «недогруженным» и гасит его — вместе с работающими на нём запросами.
UNDERUTILIZED_THRESHOLD = 50.0

# Политики консолидации Karpenter из урока.
POLICIES = ("WhenEmpty", "WhenEmptyOrUnderutilized")


class EmptyDeployment(Exception):
    """Реплик ноль: метрику «на реплику» считать не от чего.

    Свой класс, а не ZeroDivisionError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, поэтому `pytest.raises(RuntimeError)` прошёл бы
    зелёным на пустой заготовке и ничего бы не проверил.
    """


class GangSchedulingFailure(Exception):
    """Все N подов разместить нельзя, значит не размещаем ни одного.

    У экземпляра есть атрибут `stranded` — сколько GPU занял бы обычный
    планировщик и оставил ждать недостающую. Это и есть цена ловушки «7 из 8».
    """


def duty_cycle_util(active_requests, replicas, saturating_concurrency):
    """Duty-cycle утилизация GPU в процентах — тот самый DCGM_FI_DEV_GPU_UTIL.

    duty_cycle_util(2, 1, 4)    ->  50.0
    duty_cycle_util(10, 1, 4)   ->  100.0
    duty_cycle_util(100, 1, 4)  ->  100.0   <- то же число при десятикратной нагрузке!

    Метрика отвечает на вопрос «GPU была занята в момент замера?», а не «чем
    именно и сколько её просили». Выше saturating_concurrency одновременных
    запросов она упирается в 100 и перестаёт нести информацию — поэтому HPA
    на этом сигнале слеп ровно там, где нужен.

    Реплик ноль — EmptyDeployment.
    """
    if replicas <= 0:
        raise EmptyDeployment("no replicas to measure")
    if saturating_concurrency <= 0:
        raise ValueError("saturating_concurrency must be positive")
    capacity = replicas * saturating_concurrency
    # min: выше насыщения метрика физически не растёт — в этом вся её беда
    return min(100.0, 100.0 * active_requests / capacity)


def queue_depth_per_replica(queue_len, replicas):
    """Глубина очереди на реплику — правильный сигнал для prefill-bound масштабирования.

    queue_depth_per_replica(10, 1)   ->  10.0
    queue_depth_per_replica(100, 1)  ->  100.0   <- в отличие от duty cycle, растёт
    queue_depth_per_replica(10, 5)   ->  2.0

    Очередь не насыщается: сколько запросов ждёт, столько и видно. Именно
    поэтому Dynamo Planner и llm-d WVA скейлят по ней, а не по DCGM.

    Реплик ноль — EmptyDeployment.
    """
    if replicas <= 0:
        raise EmptyDeployment("no replicas to divide by")
    return queue_len / replicas


def desired_replicas(current_replicas, metric, target, min_replicas, max_replicas):
    """Формула HPA: сколько реплик нужно, чтобы метрика вернулась к target.

    desired_replicas(2, 40.0, 10.0, 1, 16)  ->  8    (метрика вчетверо выше цели)
    desired_replicas(4, 0.0, 10.0, 1, 16)   ->  1    (упёрлись в нижнюю границу)
    desired_replicas(2, 400.0, 10.0, 1, 16) ->  16   (упёрлись в верхнюю)

    Формула: ceil(current * metric / target), затем зажать в [min, max].
    Округление ВВЕРХ — принципиально: недобрать реплику значит остаться выше
    цели и на следующем тике повторить то же решение.

    Упрощение относительно настоящего HPA: у него есть мёртвая зона ±10%
    вокруг target, здесь её нет — окно стабилизации мы моделируем отдельно.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if min_replicas < 1 or max_replicas < min_replicas:
        raise ValueError("bad replica bounds")
    want = math.ceil(current_replicas * metric / target)
    return max(min_replicas, min(max_replicas, want))


def stabilize(desired_history, window):
    """Окно стабилизации: берём максимум желаемого за последние window тиков.

    stabilize([4, 1], 1)        ->  1    (окна нет — верим последнему замеру)
    stabilize([4, 1], 3)        ->  4    (помним недавний пик, вниз не спешим)
    stabilize([1, 1, 4], 3)     ->  4    (вверх реагируем сразу)

    Так же устроен stabilizationWindowSeconds в Kubernetes: масштабирование
    ВНИЗ откладывается на окно, ВВЕРХ происходит немедленно. Максимум даёт
    ровно такую асимметрию одной строкой.

    История короче окна — берём всё, что есть. window < 1 — ValueError.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if not desired_history:
        raise ValueError("empty history")
    return max(desired_history[-window:])


def run_autoscaler(load_series, target_per_replica, window, min_replicas, max_replicas):
    """Прогнать контроллер по ряду нагрузки. Вернуть список числа реплик по тикам.

    run_autoscaler([40, 4, 40, 4], 10.0, 1, 1, 16)  ->  [4, 1, 4, 1]
    run_autoscaler([40, 4, 40, 4], 10.0, 3, 1, 16)  ->  [4, 4, 4, 4]

    Один и тот же пилообразный ряд: без окна контроллер флаппит на каждом
    тике, с окном в три тика держит уровень. Реплики не бесплатны — каждое
    поднятие это провижининг узла и загрузка 70B весов, минуты.

    Порядок внутри тика: замерить очередь на реплику, посчитать желаемое
    число реплик, положить его в историю, применить окно стабилизации.
    """
    replicas = min_replicas
    raw_history = []
    out = []
    for load in load_series:
        metric = queue_depth_per_replica(load, replicas)
        want = desired_replicas(replicas, metric, target_per_replica,
                                min_replicas, max_replicas)
        raw_history.append(want)
        # окно применяем к СЫРОМУ желаемому, а не к уже сглаженному:
        # иначе максимум залипнет навсегда и вниз мы не поедем никогда
        replicas = stabilize(raw_history, window)
        out.append(replicas)
    return out


def count_scale_events(series):
    """Сколько раз число реплик изменилось — мера флаппинга.

    count_scale_events([4, 1, 4, 1])  ->  3
    count_scale_events([4, 4, 4, 4])  ->  0
    count_scale_events([1])           ->  0

    Каждое событие вверх — минуты провижининга; каждое вниз — выброшенный
    прогретый узел. Ряд без событий на пилообразной нагрузке и есть цель
    окна стабилизации.
    """
    return sum(1 for a, b in zip(series, series[1:]) if a != b)


def gang_schedule(free_gpus_per_node, needed_gpus):
    """Разместить needed_gpus по принципу «все или ни одного».

    Вернуть словарь узел -> сколько GPU занято.

    gang_schedule({"n1": 4, "n2": 4}, 8)  ->  {"n1": 4, "n2": 4}
    gang_schedule({"n1": 8, "n2": 4}, 8)  ->  {"n1": 8}          (одним узлом лучше)
    gang_schedule({"n1": 4, "n2": 3}, 8)  ->  GangSchedulingFailure(stranded=7)

    Третий пример — ловушка «7 из 8» из урока: обычный планировщик занял бы
    все семь свободных GPU, а восьмую ждал бы минуты провижининга, и всё это
    время семь GPU жгли бы деньги впустую. Gang scheduling не занимает
    ничего и говорит об этом сразу.

    У исключения должен быть атрибут `stranded` — сколько GPU было бы занято
    впустую. Заводи объект, ставь атрибут, потом raise.

    Узлы перебираем от самого свободного к менее свободному (при равенстве —
    по имени): чем меньше узлов задето, тем лучше топология для NVLink.
    """
    if needed_gpus < 0:
        raise ValueError("needed_gpus must not be negative")
    total_free = sum(free_gpus_per_node.values())
    if total_free < needed_gpus:
        err = GangSchedulingFailure(
            f"need {needed_gpus} GPUs, only {total_free} free"
        )
        err.stranded = total_free
        raise err
    # сначала самые ёмкие узлы: -free по убыванию, имя по возрастанию —
    # так результат не зависит от порядка ключей в словаре
    order = sorted(free_gpus_per_node, key=lambda n: (-free_gpus_per_node[n], n))
    plan = {}
    left = needed_gpus
    for node in order:
        if left == 0:
            break
        take = min(left, free_gpus_per_node[node])
        if take > 0:
            plan[node] = take
            left -= take
    return plan


def consolidation_plan(nodes, policy, now, consolidate_after):
    """Какие узлы гасит консолидация Karpenter и сколько запросов при этом умрёт.

    Вернуть {"terminate": [имена по алфавиту], "evicted_requests": int}.

    Узел описывается словарём:
      running_requests — сколько запросов на нём сейчас считается
      empty_since      — момент, с которого узел пуст (None, если не пуст)
      utilization      — утилизация в процентах

    consolidation_plan({"n1": {"running_requests": 0, "empty_since": 0.0,
                              "utilization": 0.0}}, "WhenEmpty", 7200.0, 3600.0)
        ->  {"terminate": ["n1"], "evicted_requests": 0}

    WhenEmpty гасит только по-настоящему пустые узлы и только если они
    простояли пустыми не меньше consolidate_after.

    WhenEmptyOrUnderutilized гасит ещё и недогруженные (utilization ниже
    UNDERUTILIZED_THRESHOLD) — НЕ глядя на работающие запросы. Это и есть
    дефолт Karpenter, который выбивает inference-поды посреди генерации.

    Время подаётся параметром now. Неизвестная политика — ValueError.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown consolidation policy: {policy}")
    terminate = []
    for name in sorted(nodes):
        node = nodes[name]
        empty_long_enough = (
            node["running_requests"] == 0
            and node["empty_since"] is not None
            and now - node["empty_since"] >= consolidate_after
        )
        underutilized = (
            policy == "WhenEmptyOrUnderutilized"
            and node["utilization"] < UNDERUTILIZED_THRESHOLD
        )
        if empty_long_enough or underutilized:
            terminate.append(name)
    evicted = sum(nodes[n]["running_requests"] for n in terminate)
    return {"terminate": terminate, "evicted_requests": evicted}
