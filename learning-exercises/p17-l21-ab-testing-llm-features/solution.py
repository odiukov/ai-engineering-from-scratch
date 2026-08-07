"""
A/B-тесты LLM-фич: мощность, интервалы, значимость — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Statsig и GrowthBook считают всё это за тебя. Здесь мы пишем статистику
руками — scipy в окружении нет, и это к лучшему: пока не посчитаешь
интервал Уилсона сам, «доверительный интервал» остаётся словом.
Соответствие настоящему продукту:

    normal_cdf, z_quantile  <-  scipy.stats.norm.cdf / .ppf
    sample_size             <-  калькулятор мощности перед запуском теста
    wilson_interval         <-  доверительный интервал доли в отчёте
    proportion_test         <-  frequentist engine GrowthBook
    srm_check               <-  SRM check (sample ratio mismatch)
    benjamini_hochberg      <-  BH-коррекция при множественных сравнениях
    run_experiment          <-  сам прогон: фиксированная выборка или
                                подглядывание (peeking)

Случайность есть только в run_experiment и приходит объектом rng
параметром. Глобальный random запрещён: два прогона теста обязаны совпасть
до последнего знака.
"""

import math

# Множитель размера выборки на LLM-недетерминизм. Урок: одинаковые входы
# дают неодинаковые выходы, эффективный размер выборки ниже номинального,
# запас +30-50%.
NONDETERMINISM_BUFFER = 1.4


def normal_cdf(z):
    """Функция распределения стандартной нормали: P(Z <= z).

    normal_cdf(0.0)   ->  0.5
    normal_cdf(1.96)  ->  примерно 0.975
    normal_cdf(-40.0) ->  примерно 0.0

    Считается через math.erf: Phi(z) = (1 + erf(z / sqrt(2))) / 2.
    Своя аппроксимация полиномом здесь не нужна — erf уже в стандартной
    библиотеке и точнее любой таблицы из учебника.

    Отсюда берутся все p-value в уроке: p = 2 * (1 - Phi(|z|)).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def z_quantile(p):
    """Обратная к normal_cdf: такое z, что normal_cdf(z) == p.

    z_quantile(0.5)    ->  0.0
    z_quantile(0.975)  ->  примерно 1.95996  (то самое 1.96 из формул)
    z_quantile(0.8)    ->  примерно 0.84162  (z_beta для мощности 80%)

    Считается двоичным поиском по normal_cdf на отрезке [-40, 40]: функция
    строго возрастает, значит поиск сходится. Точность полсотни итераций
    уже выше, чем нужно любому отчёту.

    Односторонний квантиль. Для двустороннего критерия с уровнем alpha
    бери z_quantile(1 - alpha / 2) — это и есть 1.96 при alpha = 0.05.

    p вне (0, 1) — ValueError: квантиль нуля и единицы бесконечны, а молча
    вернуть 40 значило бы отдать в отчёт правдоподобное на вид число.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be strictly inside (0, 1), got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sample_size(p_baseline, relative_lift, alpha=0.05, power=0.80, buffer=NONDETERMINISM_BUFFER):
    """Сколько наблюдений НА КАЖДУЮ ветку нужно, чтобы увидеть эффект.

    sample_size(0.03, 0.05)              ->  291114
    sample_size(0.03, 0.05, buffer=1.0)  ->  207938
    sample_size(0.03, 0.50, buffer=1.0)  ->  2518

    Формула для двух долей:

        p_b = p_baseline * (1 + relative_lift)
        p_bar = (p_baseline + p_b) / 2
        n = (z_a * sqrt(2 * p_bar * (1 - p_bar))
             + z_b * sqrt(p_a*(1-p_a) + p_b*(1-p_b)))^2 / (p_b - p_a)^2

    где z_a = z_quantile(1 - alpha/2) (двусторонний), z_b = z_quantile(power).

    Смотри на знаменатель: он квадратичный. Эффект вдвое меньше — выборка
    вчетверо больше. Ровно поэтому «померим лифт в полпроцента на нашем
    трафике» почти всегда означает «не померим».

    buffer — запас на LLM-недетерминизм (урок: ×1.3-1.5). Одинаковый промпт
    даёт разные ответы, наблюдения не совсем независимы, эффективный размер
    выборки ниже номинального.

    Округление вверх: 207937.4 наблюдения не бывает, а вниз округлять —
    занижать мощность.

    relative_lift == 0 — ValueError: нулевой эффект требует бесконечной
    выборки, и деление на ноль здесь честнее любого числа.
    """
    if relative_lift == 0:
        raise ValueError("relative_lift must not be zero: zero effect needs infinite n")
    if not 0.0 < p_baseline < 1.0:
        raise ValueError(f"p_baseline must be inside (0, 1), got {p_baseline}")
    p_a = p_baseline
    p_b = p_baseline * (1.0 + relative_lift)
    if not 0.0 < p_b < 1.0:
        raise ValueError(f"lifted rate must stay inside (0, 1), got {p_b}")
    z_a = z_quantile(1.0 - alpha / 2.0)
    z_b = z_quantile(power)
    p_bar = (p_a + p_b) / 2.0
    numerator = (
        z_a * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_b * math.sqrt(p_a * (1.0 - p_a) + p_b * (1.0 - p_b))
    ) ** 2
    return math.ceil(buffer * numerator / (p_b - p_a) ** 2)


def wilson_interval(successes, n, alpha=0.05):
    """Доверительный интервал доли по Уилсону. Вернуть (lo, hi).

    wilson_interval(45, 50)   ->  примерно (0.7864, 0.9565)
    wilson_interval(0, 10)    ->  примерно (0.0, 0.2775)
    wilson_interval(10, 10)   ->  примерно (0.7225, 1.0)

    Формула (z = z_quantile(1 - alpha/2), p = successes / n):

        center = (p + z^2/(2n)) / (1 + z^2/n)
        half   = z / (1 + z^2/n) * sqrt(p*(1-p)/n + z^2/(4n^2))

    Почему не наивное p +- z*sqrt(p(1-p)/n): на 45 из 50 наивная формула
    даёт (0.8168, 0.9832), на 10 из 10 — (1.0, 1.0), то есть «уверены на
    100%» по десяти наблюдениям. Уилсон в этих же случаях честно тянет
    интервал к середине и не вылезает за [0, 1].

    Интервал НЕ симметричен вокруг p — центр смещён к 0.5. Это не ошибка,
    это и есть поправка Уилсона.

    n <= 0 или successes вне [0, n] — ValueError.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = z_quantile(1.0 - alpha / 2.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    # клипы нужны только против арифметики на краях: на 10 из 10 центр плюс
    # половина вылезает за единицу на 1e-16
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_test(successes_a, n_a, successes_b, n_b, alpha=0.05):
    """Сравнить две доли: z-критерий с объединённой оценкой дисперсии.

    Вернуть {"p_a", "p_b", "effect", "z", "p_value", "significant"}.

    proportion_test(300, 10000, 360, 10000)
        ->  effect 0.006, z примерно 2.375, p_value примерно 0.0175, significant True
    proportion_test(300, 10000, 305, 10000)
        ->  significant False

    effect — абсолютная разница p_b - p_a. Относительный лифт из неё
    получается делением на p_a, но в отчёте всегда хранят абсолютный:
    относительный при малом p_a скачет как ему вздумается.

    Объединённая доля p = (sa + sb) / (na + nb) используется в
    ЗНАМЕНАТЕЛЕ, потому что нулевая гипотеза говорит «доли равны». Считать
    стандартную ошибку по раздельным долям — другой критерий, и он завышает
    значимость на малых выборках.

    p_value двусторонний: 2 * (1 - Phi(|z|)). Односторонний вдвое меньше и
    выглядит убедительнее — ровно поэтому его так любят в презентациях.

    Нулевая стандартная ошибка (обе ветки без единого события) — z = 0,
    p_value = 1.0: различать нечего.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both arms need at least one observation")
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        z, p_value = 0.0, 1.0
    else:
        z = (p_b - p_a) / se
        p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return {
        "p_a": p_a,
        "p_b": p_b,
        "effect": p_b - p_a,
        "z": z,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def srm_check(n_a, n_b, expected_share=0.5, alpha=0.001):
    """SRM: разъехалось ли фактическое деление трафика с задуманным.

    Вернуть {"observed_share", "z", "p_value", "srm"}.

    srm_check(47, 53)      ->  srm False (на сотне наблюдений 47/53 — это шум)
    srm_check(4700, 5300)  ->  srm True  (та же пропорция, но на десяти
                                          тысячах она уже невозможна случайно)

    z = (n_a - N * share) / sqrt(N * share * (1 - share)), p — двусторонний.

    Урок говорит: «if 50/50 split delivers 47/53, something is broken». Это
    верно только про достаточно большой N, и вся суть проверки именно в
    этом. Пропорция сама по себе ничего не значит — значит пропорция вместе
    с объёмом.

    alpha здесь 0.001, а не 0.05, намеренно: SRM-проверка идёт на каждом
    эксперименте и на каждом срезе, и при 5% половина дашборда была бы
    красной без всякой поломки.

    srm True означает «механизм назначения сломан, результатам верить
    нельзя» — эксперимент останавливают, а не интерпретируют.
    """
    total = n_a + n_b
    if total <= 0:
        raise ValueError("srm_check needs at least one observation")
    if not 0.0 < expected_share < 1.0:
        raise ValueError(f"expected_share must be inside (0, 1), got {expected_share}")
    expected_a = total * expected_share
    sigma = math.sqrt(total * expected_share * (1.0 - expected_share))
    z = (n_a - expected_a) / sigma
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return {
        "observed_share": n_a / total,
        "z": z,
        "p_value": p_value,
        "srm": p_value < alpha,
    }


def benjamini_hochberg(p_values, fdr=0.05):
    """BH-коррекция: какие из множества тестов признать значимыми.

    Вернуть кортеж True/False В ПОРЯДКЕ ВХОДА.

    benjamini_hochberg([0.001, 0.04, 0.9])   ->  (True, False, False)
    benjamini_hochberg([0.001, 0.02, 0.03])  ->  (True, True, True)
    benjamini_hochberg([])                   ->  ()

    Процедура: отсортировать p по возрастанию, найти НАИБОЛЬШИЙ ранг k
    (нумерация с 1), при котором p_(k) <= k/m * fdr, и отвергнуть все
    гипотезы с рангом до k включительно.

    Ловушка в слове «наибольший». Наивный обход «идём по возрастанию и
    останавливаемся на первом нарушении» даёт другой ответ: в примере
    [0.001, 0.04, 0.9] порог для ранга 2 равен 0.0333, и 0.04 его не
    проходит — но если бы третье p было 0.045, наивный обход остановился бы
    на ранге 1, а BH отверг бы все три (0.045 <= 3/3 * 0.05).

    Зачем это в A/B: двадцать тестов на уровне 95% дают один ложный
    положительный просто так. Бонферрони делит alpha на число тестов и режет
    мощность; BH контролирует долю ложных открытий и мягче.
    """
    p_values = list(p_values)
    m = len(p_values)
    if m == 0:
        return ()
    if not all(0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be inside [0, 1]")
    order = sorted(range(m), key=lambda i: p_values[i])
    max_rank = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= rank / m * fdr:
            max_rank = rank
    rejected = [False] * m
    for rank, index in enumerate(order, start=1):
        if rank <= max_rank:
            rejected[index] = True
    return tuple(rejected)


def run_experiment(p_a, p_b, n_total, rng, peek_every=None, alpha=0.05, min_per_arm=50):
    """Прогнать эксперимент: фиксированная выборка либо подглядывание.

    rng — random.Random(seed). Каждое наблюдение: монетка на ветку, затем
    монетка на конверсию с вероятностью ветки.

    Вернуть {"n_a", "n_b", "p_a", "p_b", "effect", "p_value",
             "significant", "stopped_at", "looks"}.

    run_experiment(0.10, 0.10, 4000, random.Random(0))
        ->  stopped_at None (дошли до конца), significant False
    run_experiment(0.10, 0.10, 4000, random.Random(0), peek_every=200)
        ->  stopped_at 1000 и significant True — на РОВНО ТЕХ ЖЕ данных,
            где никакого эффекта нет

    peek_every=None — честный фиксированный горизонт: критерий считается
    один раз, в конце.

    peek_every=k — то, что делает половина команд: смотреть на дашборд
    каждые k наблюдений и останавливаться, как только загорелось зелёным.
    Это и есть peeking. Каждый взгляд — ещё один шанс поймать шум, и
    настоящая вероятность ложного срабатывания у двадцати взглядов сильно
    выше заявленных 5%. Дополнительно эффект у остановленных тестов
    систематически ЗАВЫШЕН: останавливаются те прогоны, где шум оказался в
    нужную сторону.

    Лечится не силой воли, а математикой: последовательные критерии
    (mSPRT, доверительные последовательности Ховарда) поднимают порог с
    ростом числа взглядов. Statsig и GrowthBook возят их из коробки.

    min_per_arm — не подглядывать, пока в ветке слишком мало данных: на
    десяти наблюдениях нормальное приближение не работает вовсе.
    """
    successes_a = successes_b = 0
    n_a = n_b = 0
    looks = 0
    for i in range(1, n_total + 1):
        if rng.random() < 0.5:
            n_b += 1
            if rng.random() < p_b:
                successes_b += 1
        else:
            n_a += 1
            if rng.random() < p_a:
                successes_a += 1
        if peek_every and i % peek_every == 0 and n_a >= min_per_arm and n_b >= min_per_arm:
            looks += 1
            test = proportion_test(successes_a, n_a, successes_b, n_b, alpha)
            if test["significant"]:
                return {
                    "n_a": n_a, "n_b": n_b,
                    "p_a": test["p_a"], "p_b": test["p_b"],
                    "effect": test["effect"], "p_value": test["p_value"],
                    "significant": True, "stopped_at": i, "looks": looks,
                }
    test = proportion_test(successes_a, n_a, successes_b, n_b, alpha)
    return {
        "n_a": n_a, "n_b": n_b,
        "p_a": test["p_a"], "p_b": test["p_b"],
        "effect": test["effect"], "p_value": test["p_value"],
        "significant": test["significant"], "stopped_at": None, "looks": looks,
    }
