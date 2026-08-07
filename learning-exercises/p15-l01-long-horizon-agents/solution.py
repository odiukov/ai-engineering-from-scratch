"""
Переход от чат-ботов к агентам с длинным горизонтом — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок про две величины, которые растут в разные стороны:

  * горизонт задачи (METR Time Horizon) удваивается примерно раз в 7 месяцев;
  * надёжность прогона падает экспоненциально с числом шагов траектории.

Здесь мы собираем руками обе кривые и «реальность-чек» поверх них: покрывает
ли сегодняшний горизонт задачу, которую ты хочешь отдать агенту без присмотра.
Никакой сети и никаких LLM — только арифметика и один воспроизводимый rng.
"""

import math

# Оценка METR Time Horizon 1.1 (январь 2026) для Claude Opus 4.6: 14 часов
# экспертной работы при надёжности 50%. Фит удвоения — около 7 месяцев.
BASELINE_HOURS = 14.0
DOUBLING_MONTHS = 7.0

# Ниже этого отношения «горизонт / длительность задачи» запускать агента без
# присмотра нельзя: у прогона нет запаса на неудачные ветки.
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
    return baseline_hours * 2.0 ** (months / doubling_months)


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
    if target_hours <= 0 or baseline_hours <= 0:
        raise ValueError("горизонт измеряется в положительных часах")
    return doubling_months * math.log2(target_hours / baseline_hours)


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
    if not 0.0 <= per_step <= 1.0:
        raise ValueError("per_step — вероятность, она лежит в [0, 1]")
    if steps < 0:
        raise ValueError("число шагов не бывает отрицательным")
    return per_step ** steps


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
    if not 0.0 <= per_step <= 1.0:
        raise ValueError("per_step — вероятность, она лежит в [0, 1]")
    if not 0.0 < target <= 1.0:
        raise ValueError("target — вероятность в (0, 1]")
    if per_step >= 1.0:
        return None
    if per_step == 0.0:
        return 0
    # log(per_step) отрицателен, поэтому деление переворачивает неравенство
    return math.floor(math.log(target) / math.log(per_step))


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
    if not 0.0 <= eval_gap <= 1.0:
        raise ValueError("eval_gap — доля, она лежит в [0, 1]")
    if benchmark_hours < 0:
        raise ValueError("горизонт не бывает отрицательным")
    return benchmark_hours * (1.0 - eval_gap)


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
    if cost_per_step < 0:
        raise ValueError("шаг не может стоить отрицательно")
    spent = 0.0
    for step in range(1, max_steps + 1):
        # деньги списываются до действия — так же, как их резервирует биллинг
        if spent + cost_per_step > budget:
            return {"status": "budget_exhausted", "steps": step - 1, "spent": spent}
        spent += cost_per_step
        if rng.random() >= per_step:
            return {"status": "failed", "steps": step, "spent": spent}
    return {"status": "success", "steps": max_steps, "spent": spent}


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
    if task_hours <= 0:
        raise ValueError("длительность задачи положительна")
    effective = deployment_horizon(benchmark_hours, eval_gap)
    ratio = effective / task_hours
    if ratio >= margin:
        return "safe"
    if ratio >= 1.0:
        return "tight"
    return "runaway"
