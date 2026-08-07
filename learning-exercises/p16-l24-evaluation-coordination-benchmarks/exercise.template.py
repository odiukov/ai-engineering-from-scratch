"""
Оценка и бенчмарки координации

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l24-evaluation-coordination-benchmarks
Разбор:  /check-code p16-l24-evaluation-coordination-benchmarks
"""

import math

CONTAMINATION_THRESHOLD = 0.1
Z_95 = 1.96


def accuracy(results):
    """Доля успешных задач.

    accuracy([True, True, False, False])  ->  0.5
    accuracy([])                          ->  0.0

    Самая честная и самая бедная метрика: всё или ничего за задачу.
    Именно её MARBLE и дополняет вехами — см. milestone_score.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
