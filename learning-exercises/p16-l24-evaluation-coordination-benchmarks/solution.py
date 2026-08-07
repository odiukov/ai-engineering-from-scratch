"""
Оценка и бенчмарки координации — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Разрыв между «виденным» и отложенным сплитом, после которого результат
# считается подозрительным на контаминацию. Ровно то, что отличает
# SWE-bench Verified от Pro: ~70% против ~23% у тех же моделей.
CONTAMINATION_THRESHOLD = 0.1

# z для двустороннего 95% доверительного интервала при нормальном приближении.
Z_95 = 1.96


def accuracy(results):
    """Доля успешных задач.

    accuracy([True, True, False, False])  ->  0.5
    accuracy([])                          ->  0.0

    Самая честная и самая бедная метрика: всё или ничего за задачу.
    Именно её MARBLE и дополняет вехами — см. milestone_score.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def milestone_score(achieved, weights=None):
    """Взвешенная доля достигнутых вех (частичный зачёт в духе MARBLE).

    milestone_score([True, True, False, False])  ->  0.5
    milestone_score([True, False], [3.0, 1.0])   ->  0.75
    milestone_score([])                          ->  0.0

    weights=None означает равные веса. Нормировать надо на СУММУ весов, а не
    считать, что она равна единице: веса в бенчмарках задают в «очках».

    Смысл вех: система, дошедшая до 4 из 5 шагов и упавшая на последнем,
    и система, не начавшая работу, — это не одинаковый ноль.
    """
    if not achieved:
        return 0.0
    if weights is None:
        weights = [1.0] * len(achieved)
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(w for ok, w in zip(achieved, weights) if ok) / total


def lift_over_random(score, n_options):
    """Насколько результат оторвался от случайного угадывания.

    Базовая линия — 1/n_options. Возвращаем нормированный отрыв:
    (score - baseline) / (1 - baseline). Ноль — уровень случайности,
    единица — идеал, минус — хуже монетки.

    lift_over_random(0.25, 4)   ->  0.0
    lift_over_random(0.625, 4)  ->  0.5
    lift_over_random(1.0, 4)    ->  1.0

    Ради этой строки COMMA и стоит в уроке: фронтирные модели на
    агент-агентной координации не отрываются от случайной базы. Без явной
    случайной базы в отчёте цифра 0.25 выглядит «каким-то результатом».

    n_options < 2 — ValueError: при одном варианте база равна единице и
    делить не на что.
    """
    if n_options < 2:
        raise ValueError("случайная база не определена при n_options < 2")
    baseline = 1.0 / n_options
    return (score - baseline) / (1.0 - baseline)


def coordination_gain(team, solos):
    """Вклад именно координации: команда минус ЛУЧШИЙ одиночка.

    team — оценки команды по задачам, solos[i] — оценки i-го агента, если бы
    он работал один. Вернуть mean(team) - max_i mean(solos[i]).

    coordination_gain([1, 1, 0, 0], [[1, 1, 0, 0], [0, 0, 0, 0]])  ->  0.0
    coordination_gain([1, 1, 1, 1], [[1, 1, 0, 0], [0, 0, 1, 1]])  ->  0.5
    coordination_gain([0, 0, 0, 0], [[1, 1, 1, 1]])                ->  -1.0

    Вычитается максимум, а не среднее по агентам. Сравнение со средним
    одиночкой — самый частый способ показать несуществующий выигрыш:
    команда из сильного и двух слабых обгонит среднее просто так.

    Отрицательное значение — coordination tax. MedAgentBoard находит его
    регулярно: на многих задачах мультиагентность проигрывает одной LLM.
    """
    if not solos:
        raise ValueError("нет одиночных прогонов для сравнения")
    team_mean = sum(team) / len(team) if team else 0.0
    best_solo = max(
        (sum(runs) / len(runs) if runs else 0.0) for runs in solos
    )
    return team_mean - best_solo


def cost_per_milestone(tokens, milestone, price_per_1k):
    """Стоимость одной достигнутой вехи в деньгах.

    cost_per_milestone(20000, 0.5, 0.01)  ->  0.4
    cost_per_milestone(20000, 1.0, 0.01)  ->  0.2
    cost_per_milestone(20000, 0.0, 0.01)  ->  inf

    Ноль вех — бесконечная цена за веху, а не нулевая: система не сделала
    ничего, но токены сожгла.

    Пункт 6 чеклиста. Решение на 90% при 20-кратной цене — это бизнес-выбор,
    а не заявка на способности, и без этой колонки его не видно.
    """
    cost = tokens / 1000.0 * price_per_1k
    if milestone <= 0:
        return math.inf
    return cost / milestone


def contamination_gap(seen, held):
    """Разрыв точности между «виденным» сплитом и отложенным.

    contamination_gap([True, True, True, True], [True, False, False, False])
        ->  0.75
    contamination_gap([True, False], [True, False])  ->  0.0

    Большой положительный разрыв — сигнал, что бенчмарк утёк в обучающий
    корпус. Отрицательный разрыв ничего не говорит о контаминации, это
    просто шум или разная сложность сплитов.

    Каноничная величина этого разрыва в 2026-м — SWE-bench Verified против
    Pro: 70%+ против ~23% у тех же моделей.
    """
    return accuracy(seen) - accuracy(held)


def mean_confidence_interval(scores, z=Z_95):
    """Среднее и полуширина доверительного интервала (нормальное приближение).

    Вернуть (среднее, полуширина), где полуширина = z * s / sqrt(n),
    s — выборочное стандартное отклонение с делением на n-1.

    mean_confidence_interval([0.5, 0.5, 0.5, 0.5])  ->  (0.5, 0.0)
    mean_confidence_interval([0.0, 1.0])            ->  (0.5, 0.98)
    mean_confidence_interval([0.7])                 ->  (0.7, inf)

    Один прогон не даёт никакого интервала — полуширина inf, а не ноль.
    Пункт 4 чеклиста: фронтирные модели шумные, и одиночный прогон вводит
    в заблуждение чаще, чем помогает.

    Делим на n-1, а не на n: тут мы оцениваем неопределённость среднего,
    а не описываем ровно эту выборку.
    """
    n = len(scores)
    if n == 0:
        return (0.0, math.inf)
    mean = sum(scores) / n
    if n < 2:
        return (mean, math.inf)
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return (mean, z * math.sqrt(variance) / math.sqrt(n))


def scorecard(system, contamination_threshold=CONTAMINATION_THRESHOLD):
    """Карточка результатов системы: собрать все метрики урока в один словарь.

    system — словарь с ключами: seen, held, milestones, milestone_weights,
    tokens, price_per_1k, n_options, team, solos.

    Ключи результата: accuracy, milestone, lift_over_random,
    coordination_gain, cost_per_milestone, contamination_gap, contaminated,
    confidence_interval.

    Точность считается по ОТЛОЖЕННОМУ сплиту: seen нужен только для того,
    чтобы измерить разрыв. Публиковать число с виденного сплита как итог —
    ровно та ошибка, ради которой существует SWE-bench Pro.
    """
    held = system["held"]
    acc = accuracy(held)
    milestone = milestone_score(system["milestones"], system.get("milestone_weights"))
    gap = contamination_gap(system["seen"], held)
    return {
        "accuracy": acc,
        "milestone": milestone,
        "lift_over_random": lift_over_random(acc, system["n_options"]),
        "coordination_gain": coordination_gain(system["team"], system["solos"]),
        "cost_per_milestone": cost_per_milestone(
            system["tokens"], milestone, system["price_per_1k"]
        ),
        "contamination_gap": gap,
        "contaminated": gap > contamination_threshold,
        "confidence_interval": mean_confidence_interval([float(r) for r in held]),
    }
