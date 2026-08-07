"""
GPU-автоскейлинг в Kubernetes: сигнал, окно стабилизации и gang scheduling

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l03-gpu-autoscaling-kubernetes
Разбор:  /check-code p17-l03-gpu-autoscaling-kubernetes
"""

import math

UNDERUTILIZED_THRESHOLD = 50.0
POLICIES = ("WhenEmpty", "WhenEmptyOrUnderutilized")


class EmptyDeployment(Exception):
    """Реплик ноль: метрику «на реплику» считать не от чего.

    Свой класс, а не ZeroDivisionError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, поэтому `pytest.raises(RuntimeError)` прошёл бы
    зелёным на пустой заготовке и ничего бы не проверил.
    """
    pass


class GangSchedulingFailure(Exception):
    """Все N подов разместить нельзя, значит не размещаем ни одного.

    У экземпляра есть атрибут `stranded` — сколько GPU занял бы обычный
    планировщик и оставил ждать недостающую. Это и есть цена ловушки «7 из 8».
    """
    pass


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
    raise NotImplementedError


def queue_depth_per_replica(queue_len, replicas):
    """Глубина очереди на реплику — правильный сигнал для prefill-bound масштабирования.

    queue_depth_per_replica(10, 1)   ->  10.0
    queue_depth_per_replica(100, 1)  ->  100.0   <- в отличие от duty cycle, растёт
    queue_depth_per_replica(10, 5)   ->  2.0

    Очередь не насыщается: сколько запросов ждёт, столько и видно. Именно
    поэтому Dynamo Planner и llm-d WVA скейлят по ней, а не по DCGM.

    Реплик ноль — EmptyDeployment.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def count_scale_events(series):
    """Сколько раз число реплик изменилось — мера флаппинга.

    count_scale_events([4, 1, 4, 1])  ->  3
    count_scale_events([4, 4, 4, 4])  ->  0
    count_scale_events([1])           ->  0

    Каждое событие вверх — минуты провижининга; каждое вниз — выброшенный
    прогретый узел. Ряд без событий на пилообразной нагрузке и есть цель
    окна стабилизации.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
