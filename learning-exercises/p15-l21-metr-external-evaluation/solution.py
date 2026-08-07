"""
METR: временные горизонты и внешняя оценка возможностей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок про две вещи сразу: как считается time horizon и почему внешняя оценка
структурно надёжнее самооценки лаборатории. Здесь мы собираем руками обе
половины — протокол прогона и сам замер.

Что чему соответствует в материалах METR:

    ACCESS_SCOPES        <-  что именно лаборатория открывает оценщику
    resolve_access       <-  оценщик получает РОВНО оговорённый доступ
    sample_tasks         <-  выборка из HCAST / RE-Bench / SWAA по rng
    run_manifest         <-  паспорт прогона: воспроизводим по seed
    success_curve        <-  P(success) по длительности задач
    horizon_at           <-  шаг 4 методики: время, где P(success) = p
    inject_gaming        <-  eval-context gaming: доля провалов «оживает»
    doubling_time_days   <-  время удвоения горизонта (TH1.1: ~130.8 дней)
    deployment_gap       <-  горизонт — ПОТОЛОК, а не прогноз надёжности

Настоящий METR подгоняет логистическую кривую градиентным спуском. Мы
считаем горизонт интерполяцией по уже наблюдённой кривой: подгонка — это
отдельная тема, а смысл шага 4 («время, при котором вероятность успеха
равна p») от способа подгонки не зависит и проверяется точно.

Никакой сети, никаких вызовов модели, никакого глобального random: rng и
seed приходят параметром. Прогон оценки, который нельзя повторить, —
не измерение, а слайд.
"""

import hashlib
import math
import random  # только конструктор Random(seed); глобальный random не трогаем

# Что лаборатория может открыть внешнему оценщику. Порядок — от самого
# безобидного к самому чувствительному.
ACCESS_SCOPES = (
    "model_api",
    "prerelease_checkpoint",
    "task_scaffold",
    "chain_of_thought",
    "internal_evals",
    "model_weights",
    "training_data",
)

# Типичная договорённость METR 2025–2026: доступ к предрелизной модели и
# к своему скаффолду, без весов и без обучающих данных.
METR_ENGAGEMENT = ("model_api", "prerelease_checkpoint", "task_scaffold")

# Наборы задач. Числа задач по разбору урока: HCAST 189, RE-Bench 71.
# Границы длительности — иллюстративные, у SWAA размер набора тоже.
SUITES = {
    "HCAST": {"n_tasks": 189, "min_hours": 1.0 / 60.0, "max_hours": 8.0},
    "RE-Bench": {"n_tasks": 71, "min_hours": 1.0, "max_hours": 8.0},
    "SWAA": {"n_tasks": 66, "min_hours": 1.0 / 60.0, "max_hours": 0.5},
}

# METR ведёт с 50%, но публикует и края: 10% — консервативная оценка,
# 90% — оптимистичная. Сама цифра 50% — выбор, а не свойство природы.
HORIZON_LEVELS = (0.10, 0.50, 0.90)

# Четыре причины, по которым бенчмарк-горизонт выше боевой надёжности.
# Множители иллюстративные; важно, что каждый не больше единицы.
DEPLOYMENT_DISCOUNTS = {
    "idealized_tooling": 0.70,
    "no_real_consequences": 0.80,
    "eval_context_gaming": 0.75,
    "user_variance": 0.80,
}


def resolve_access(agreement, requested):
    """Разбор запроса оценщика: {"granted", "refused", "unused"}.

    resolve_access(METR_ENGAGEMENT, ["model_api", "model_weights"])
        ->  granted ["model_api"], refused ["model_weights"],
            unused ["prerelease_checkpoint", "task_scaffold"]

    granted — ровно пересечение запроса с договорённостью, ни строкой
    больше. Все три списка отсортированы.

    Почему это отдельная функция, а не проверка на глазок: независимость
    оценщика — структурная мера, и она держится на том, что доступ
    определён заранее и не расширяется по ходу прогона. Запрос вне
    договорённости обязан попадать в refused, а не тихо срабатывать.

    unused нужен для симметрии: договорённость шире реально понадобившегося
    доступа — тоже дефект, только в другую сторону. Выданные и не
    использованные веса остаются выданными весами.

    Неизвестное имя scope — ValueError, в любом из двух аргументов.
    Опечатка иначе прочитается как «такого доступа не просили» или
    «такого доступа не давали» — оба вывода ложные.
    """
    for name in tuple(agreement) + tuple(requested):
        if name not in ACCESS_SCOPES:
            raise ValueError(f"unknown access scope: {name!r}")
    have, want = set(agreement), set(requested)
    return {
        "granted": sorted(want & have),
        "refused": sorted(want - have),
        "unused": sorted(have - want),
    }


def sample_tasks(suite, n_tasks, rng):
    """Выборка n_tasks задач из набора. Список (task_id, expert_hours).

    sample_tasks("RE-Bench", 3, random.Random(0))
        ->  три пары, отсортированные по возрастанию expert_hours

    Каталог набора строится детерминированно: длительности разложены
    лог-равномерно от min_hours до max_hours, id — "SUITE/NNNN". Случайность
    только в том, КАКИЕ задачи попали в выборку, и она приходит через rng.

    Результат отсортирован по длительности — так его сразу можно читать
    как ось будущей кривой.

    Один и тот же seed обязан давать одну и ту же выборку. Именно поэтому
    rng — параметр, а не random из модуля: глобальное состояние делает
    прогон невоспроизводимым, и оспорить такой результат нельзя.

    Неизвестный набор — ValueError. Запрос больше, чем в наборе есть
    задач, — тоже: молча выдать меньше значило бы отчитаться о прогоне,
    которого не было.
    """
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite!r}")
    spec = SUITES[suite]
    total = spec["n_tasks"]
    if not 1 <= n_tasks <= total:
        raise ValueError(f"n_tasks must be in 1..{total}, got {n_tasks!r}")
    lo = math.log(spec["min_hours"])
    hi = math.log(spec["max_hours"])
    # sample по индексам, а не по готовым кортежам: индекс задаёт и id,
    # и длительность, поэтому каталог целиком строить не нужно
    picked = sorted(rng.sample(range(total), n_tasks))
    return [
        (f"{suite}/{i:04d}", math.exp(lo + (hi - lo) * i / (total - 1)))
        for i in picked
    ]


def run_manifest(suite, n_tasks, seed, agreement, requested):
    """Паспорт прогона внешней оценки.

    Ключи: suite, seed, n_tasks, access, tasks, digest.

    run_manifest("RE-Bench", 5, 7, METR_ENGAGEMENT, ["model_api"])["seed"]
        ->  7
    Два вызова с одним seed  ->  одинаковый digest
    Тот же вызов с seed + 1  ->  другой digest

    В access лежит результат resolve_access, то есть ФАКТИЧЕСКИ выданный
    доступ. Запрошенные веса, которых не дали, в паспорте прогона не
    появляются как выданные — иначе паспорт врал бы в пользу оценщика.

    digest — короткий sha256 от канонической записи прогона (набор, seed,
    выданный доступ, список задач). Он нужен ровно для одного: сверить, что
    два прогона — это один и тот же прогон. Никакой конкретной строки
    digest тесты не ждут, только совпадение и различие.

    rng создаётся ЗДЕСЬ, из seed. Публиковать seed и не публиковать способ
    его применения — то же самое, что не публиковать ничего.
    """
    access = resolve_access(agreement, requested)
    tasks = sample_tasks(suite, n_tasks, random.Random(seed))
    payload = "|".join([
        suite,
        str(n_tasks),
        str(seed),
        ",".join(access["granted"]),
        # фиксированная точность: иначе repr float сделает digest
        # зависимым от платформы, и сверка двух прогонов развалится
        ";".join(f"{tid}@{hours:.6f}" for tid, hours in tasks),
    ])
    return {
        "suite": suite,
        "seed": seed,
        "n_tasks": n_tasks,
        "access": access,
        "tasks": tasks,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }


def success_curve(results):
    """Доля успеха по длительностям: список (expert_hours, rate) по возрастанию.

    success_curve([(1.0, True), (1.0, False), (4.0, False)])
        ->  [(1.0, 0.5), (4.0, 0.0)]

    Вход — сырые результаты прогона: (длительность задачи в часах, успех).
    Задачи одной длительности сливаются в одну точку кривой.

    Пустой вход — ValueError. Пустая кривая тихо означала бы «замер есть, он
    просто ничего не показал»; на самом деле замера не было.

    Кривая обычно убывающая, но требовать этого нельзя: на реальных данных
    соседние точки прыгают. Функция считает то, что наблюдали.
    """
    if not results:
        raise ValueError("empty result set: nothing was measured")
    buckets = {}
    for hours, success in results:
        total, hits = buckets.get(hours, (0, 0))
        buckets[hours] = (total + 1, hits + (1 if success else 0))
    return [(h, buckets[h][1] / buckets[h][0]) for h in sorted(buckets)]


def horizon_at(curve, p):
    """Длительность задачи, на которой вероятность успеха равна p.

    Кривая: [(0.25, 1.0), (1.0, 0.9), (4.0, 0.7), (14.0, 0.5), (64.0, 0.1)]
    horizon_at(curve, 0.5)   ->  14.0
    horizon_at(curve, 0.6)   ->  7.4833   (интерполяция между 4 и 14)
    horizon_at(curve, 0.05)  ->  ValueError

    Это шаг 4 методики METR. Интерполяция идёт по ЛОГАРИФМУ времени:
    вся кривая построена в log-масштабе, и линейная интерполяция по самим
    часам сдвинула бы ответ вниз (среднее арифметическое 4 и 14 — девять,
    геометрическое — семь с половиной).

    Кривая, которая ни разу не пересекает p, — ValueError. Экстраполировать
    за края наблюдений нельзя: горизонт, который пришлось продлить за
    последнюю измеренную точку, — уже не измерение.

    p вне интервала (0, 1) — ValueError, как и кривая короче двух точек или
    не отсортированная по возрастанию часов. Несортированный вход даёт
    правдоподобное число из ниоткуда — самый неприятный вид ошибки.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    if len(curve) < 2:
        raise ValueError("curve needs at least two points")
    hours = [h for h, _ in curve]
    if hours != sorted(hours):
        raise ValueError("curve must be sorted by expert_hours ascending")
    for (h0, r0), (h1, r1) in zip(curve, curve[1:]):
        if r0 == p:
            return h0
        if r0 > p >= r1:
            # доля пути от r0 до r1, на которой лежит p
            frac = (r0 - p) / (r0 - r1)
            return math.exp(math.log(h0) + frac * (math.log(h1) - math.log(h0)))
    if curve[-1][1] == p:
        return curve[-1][0]
    raise ValueError(f"curve never crosses p={p}: horizon undefined")


def inject_gaming(results, gaming_rate, rng):
    """Модель занижала себя на оценках: часть провалов на деле была успехом.

    inject_gaming([(1.0, False)], 0.0, random.Random(0))  ->  [(1.0, False)]
    inject_gaming([(1.0, False)], 1.0, random.Random(0))  ->  [(1.0, True)]

    Каждый ПРОВАЛ независимо переворачивается в успех с вероятностью
    gaming_rate. Успехи не трогаются вовсе: eval-context gaming — про
    занижение, а не про шум в обе стороны.

    Возвращается новый список, вход не меняется.

    rng параметром и здесь: «горизонт вырос на 20% при gaming_rate 0.2» —
    утверждение, которое обязано воспроизводиться по seed, иначе его
    невозможно проверить.

    gaming_rate вне [0, 1] — ValueError.
    """
    if not 0.0 <= gaming_rate <= 1.0:
        raise ValueError(f"gaming_rate must be in [0, 1], got {gaming_rate!r}")
    out = []
    for hours, success in results:
        # rng.random() вызывается только для провалов: иначе один и тот же
        # seed давал бы разные результаты при разной доле успехов
        if not success and rng.random() < gaming_rate:
            out.append((hours, True))
        else:
            out.append((hours, success))
    return out


def doubling_time_days(horizon_old, horizon_new, days_between):
    """За сколько дней горизонт удваивается при наблюдённом темпе роста.

    doubling_time_days(1.0, 2.0, 130.8)   ->  130.8
    doubling_time_days(7.0, 14.0, 130.8)  ->  130.8   (важно отношение)
    doubling_time_days(1.0, 4.0, 130.8)   ->  65.4

    Формула: days_between * ln(2) / ln(horizon_new / horizon_old).

    Публичные числа для сверки: Time Horizon 1.1 (январь 2026) даёт ~130.8
    дня (≈4.3 месяца) на пост-2023 подгонке; полная подгонка 2019–2025 из
    TH1.0 даёт ~7 месяцев. Два разных числа для одной величины — не
    противоречие, а два разных окна наблюдения.

    Отсутствие роста — ValueError. При horizon_new == horizon_old логарифм
    равен нулю, и «время удвоения» уходит в бесконечность: это не
    бесконечно долго, это неизвестно.

    Неположительные горизонты и неположительное окно — тоже ValueError.
    """
    if horizon_old <= 0 or horizon_new <= 0:
        raise ValueError("horizons must be positive")
    if days_between <= 0:
        raise ValueError("days_between must be positive")
    if horizon_new <= horizon_old:
        raise ValueError("no growth observed: doubling time undefined")
    return days_between * math.log(2.0) / math.log(horizon_new / horizon_old)


def deployment_gap(horizon_hours, task_hours, discounts=None):
    """Влезает ли задача в горизонт ПОСЛЕ поправок на боевые условия.

    Ключи: horizon_hours, effective_hours, task_hours, within_horizon, reason.

    deployment_gap(14.0, 8.0, [])                 ->  within_horizon True
    deployment_gap(14.0, 8.0, list(DEPLOYMENT_DISCOUNTS))
        ->  effective_hours 4.704, within_horizon False

    Множители перемножаются, каждый не больше единицы, поэтому
    effective_hours НИКОГДА не выше horizon_hours. Это и есть смысл фразы
    «горизонт — верхняя граница»: бенчмарк мерит потолок при идеальных
    условиях, а разворачивают модель не в них.

    Пример выше — весь урок в двух строках: восьмичасовая задача проходит
    по заявленным 14 часам и не проходит по четырём с небольшим.

    discounts — список имён из DEPLOYMENT_DISCOUNTS. Неизвестное имя —
    ValueError: незнакомая поправка, тихо пропущенная, завышает
    effective_hours, то есть ошибается в опасную сторону.

    reason обязан назвать применённые поправки: решение о выкате, причину
    которого нельзя прочитать, нельзя и оспорить.
    """
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if task_hours <= 0:
        raise ValueError("task_hours must be positive")
    names = [] if discounts is None else list(discounts)
    factor = 1.0
    for name in names:
        if name not in DEPLOYMENT_DISCOUNTS:
            raise ValueError(f"unknown deployment discount: {name!r}")
        factor *= DEPLOYMENT_DISCOUNTS[name]
    effective = horizon_hours * factor
    within = task_hours <= effective
    applied = ",".join(names) if names else "none"
    verdict = "within" if within else "over"
    return {
        "horizon_hours": horizon_hours,
        "effective_hours": effective,
        "task_hours": task_hours,
        "within_horizon": within,
        "reason": (
            f"{verdict}: task {task_hours}h vs effective {effective:.3f}h "
            f"(horizon {horizon_hours}h, discounts {applied})"
        ),
    }
