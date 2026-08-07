"""
Переход от чат-ботов к агентам с длинным горизонтом

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l01-long-horizon-agents
Разбор:  /check-code p15-l01-long-horizon-agents
"""

import math

BASELINE_HOURS = 14.0
DOUBLING_MONTHS = 7.0
SAFE_MARGIN = 2.0


def horizon_at(months, baseline_hours=BASELINE_HOURS, doubling_months=DOUBLING_MONTHS):
    """Горизонт задачи (в часах) через months месяцев от базовой точки.

    horizon_at(0)     ->  14.0    (базовая точка, ничего не выросло)
    horizon_at(7)     ->  28.0    (одно удвоение)
    horizon_at(14)    ->  56.0    (два удвоения)
    horizon_at(-7)    ->  7.0     (назад во времени горизонт вдвое меньше)

    Формула экспоненциальная: baseline * 2 ** (months / doubling_months).
    Соблазн посчитать линейно («+2 часа в месяц») даёт правдоподобные числа на
    год и полную чушь на три: за 36 месяцев экспонента даёт 495 часов, а
    линейка «+2 часа в месяц» — 86. Именно на этом расхождении ломаются планы
    по автономии.
    """
    raise NotImplementedError


def months_to_cross(target_hours, baseline_hours=BASELINE_HOURS,
                    doubling_months=DOUBLING_MONTHS):
    """Через сколько месяцев горизонт дорастёт до target_hours.

    months_to_cross(14.0)   ->  0.0     (уже там)
    months_to_cross(28.0)   ->  7.0     (одно удвоение)
    months_to_cross(7.0)    -> -7.0     (это было в прошлом)

    Обратная к horizon_at: doubling_months * log2(target / baseline).
    Ловушка: target_hours <= 0 не имеет смысла — логарифм не существует.
    Такой вход это ValueError, а не тихий nan.
    """
    raise NotImplementedError


def end_to_end_reliability(per_step, steps):
    """Вероятность, что ВСЕ шаги траектории прошли без ошибки.

    end_to_end_reliability(0.99, 1)    ->  0.99
    end_to_end_reliability(0.99, 70)   ->  примерно 0.4948  (уже монетка!)
    end_to_end_reliability(0.90, 0)    ->  1.0   (нулевая траектория не падает)

    Это per_step ** steps, и вся мораль урока в том, как быстро эта степень
    съедает надёжность. Агент, который ошибается раз в сто шагов, на прогоне
    из 70 инструментальных вызовов доходит до конца реже, чем в половине
    случаев. Средний production-агент делает больше 70 вызовов.

    Ловушка: per_step вне [0, 1] — не вероятность. Это ValueError.
    """
    raise NotImplementedError


def max_steps_for_target(per_step, target=0.5):
    """Самая длинная траектория, которая ещё держит надёжность >= target.

    max_steps_for_target(0.99)          ->  68     (0.99**68 = 0.504)
    max_steps_for_target(0.999)         ->  692
    max_steps_for_target(1.0)           ->  None   (идеальный агент не ломается)

    Считается через логарифм: floor(log(target) / log(per_step)). Обрати
    внимание, что рост НЕ линейный: три девятки вместо двух дают не втрое
    больше шагов, а вдесятеро.

    Возврат None для per_step >= 1.0 — сознательный: «бесконечность» лучше
    честно назвать отсутствием ограничения, чем вернуть 10**9 и притвориться
    числом.
    """
    raise NotImplementedError


def deployment_horizon(benchmark_hours, eval_gap):
    """Скидка на разрыв «бенчмарк против продакшена».

    deployment_horizon(14.0, 0.0)   ->  14.0
    deployment_horizon(14.0, 0.4)   ->  8.4

    METR честно пишет, что её горизонты — верхняя граница: идеальные
    инструменты, отсутствие последствий за ошибку и подозрение на
    eval-context gaming (модель распознаёт, что её тестируют, и ведёт себя
    аккуратнее). Anthropic в 2024 намерила alignment faking в 12% базовых
    тестов и до 78% после попыток переучивания.

    eval_gap — доля, которую ты вычитаешь на этот разрыв. Значение вне [0, 1]
    бессмысленно и должно падать с ValueError.
    """
    raise NotImplementedError


def simulate_run(rng, per_step, max_steps, budget, cost_per_step):
    """Один прогон агента: шагает, пока не сломается или не кончатся деньги.

    Вернуть словарь со статусом, числом ВЫПОЛНЕННЫХ шагов и потраченным.
    Статусы: "success", "failed", "budget_exhausted".

    simulate_run(random.Random(0), 1.0, 3, 100.0, 1.0)
        ->  {"status": "success", "steps": 3, "spent": 3.0}
    simulate_run(random.Random(0), 0.0, 3, 100.0, 1.0)
        ->  {"status": "failed", "steps": 1, "spent": 1.0}
    simulate_run(random.Random(0), 1.0, 10, 2.5, 1.0)
        ->  {"status": "budget_exhausted", "steps": 2, "spent": 2.0}

    Два обязательных свойства:
      * бюджет проверяется ДО шага, а не после. Иначе последний шаг всегда
        уходит в минус, и "budget_exhausted" врёт о потраченном.
      * "budget_exhausted" означает, что шаг НЕ выполнен: он не попадает в
        "steps" и не попадает в "spent".

    rng — обязательный параметр, а не глобальный random. Прогон агента должен
    воспроизводиться по seed, иначе разбор инцидента невозможен.
    """
    raise NotImplementedError


def horizon_verdict(task_hours, benchmark_hours=BASELINE_HOURS, eval_gap=0.0,
                    margin=SAFE_MARGIN):
    """Реальность-чек: можно ли отдать задачу агенту без присмотра.

    Вердикты: "safe" (запас есть), "tight" (впритык), "runaway" (не влезает).

    horizon_verdict(4.0)               ->  "safe"      (14 / 4 = 3.5 >= 2)
    horizon_verdict(10.0)              ->  "tight"     (14 / 10 = 1.4)
    horizon_verdict(40.0)              ->  "runaway"   (14 / 40 = 0.35)
    horizon_verdict(4.0, eval_gap=0.5) ->  "tight"     (7 / 4 = 1.75)

    Сначала считается deployment_horizon, и только потом сравнение — иначе
    ты меришь задачу по бенчмарку, а бенчмарк по продакшену не считается.
    """
    raise NotImplementedError
