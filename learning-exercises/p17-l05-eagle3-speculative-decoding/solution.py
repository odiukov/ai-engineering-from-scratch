"""
EAGLE-3: спекулятивное декодирование, acceptance rate и цена черновика — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Порог из урока: ниже этого alpha на продовом трафике спекулятивное
# декодирование уходит в минус, и флаг надо гасить, а не тюнить.
ALPHA_GATE = 0.55

# Смешанный трафик из упражнения 2 урока: 70% общего чата с alpha 0.7
# и 30% кода с alpha 0.4. Пары (доля, alpha).
TRAFFIC_MIX = ((0.7, 0.7), (0.3, 0.4))

# Учебный словарь на четыре токена. Целевая модель и три черновика к ней:
# идеальный (совпадает с целью), хороший и откровенно плохой.
TARGET_PROBS = (0.5, 0.3, 0.15, 0.05)
GOOD_DRAFT = (0.45, 0.32, 0.16, 0.07)
BAD_DRAFT = (0.05, 0.05, 0.10, 0.80)


def expected_speedup(alpha, k, verify_overhead):
    """Ожидаемое ускорение спекулятивного декодирования по формуле урока.

    expected_speedup(0.7, 5, 0.1)  ->  4.0909...   (4.5 токена за форвард / 1.1)
    expected_speedup(0.0, 5, 0.1)  ->  0.9090...   <- ни один черновик не принят
    expected_speedup(1.0, 5, 0.1)  ->  5.4545...

    За один форвард целевой модели выходит 1 + k * alpha токенов: один
    гарантированный плюс принятые черновые. Делим на (1 + verify_overhead) —
    цену черновика и переверификации.

    alpha здесь ровно то, что репортит vLLM: принятые черновые токены,
    делённые на ЗАПРОШЕННУЮ длину черновика k, а не на число реально
    проверенных позиций.

    alpha вне [0, 1], k < 1 или отрицательный verify_overhead — ValueError.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    if k < 1:
        raise ValueError("k must be at least 1")
    if verify_overhead < 0:
        raise ValueError("verify_overhead must not be negative")
    return (1.0 + k * alpha) / (1.0 + verify_overhead)


def breakeven_alpha(k, verify_overhead):
    """Acceptance rate, при котором спекулятивное декодирование не даёт ничего.

    breakeven_alpha(5, 0.15)  ->  0.03
    breakeven_alpha(5, 10.0)  ->  2.0    <- больше единицы, значит недостижимо
    breakeven_alpha(8, 0.16)  ->  0.02

    Решаем (1 + k * alpha) / (1 + verify_overhead) = 1 относительно alpha.

    Результат больше 1 означает, что при такой цене черновика окупиться
    нельзя вообще: даже стопроцентное принятие не отобьёт накладные.

    Ловушка урока: сырая формула даёт мизерные 0.03, но на продовой
    конкурентности verify_overhead растёт вместе с батчем, и практический
    порог уползает к ALPHA_GATE. Формула — нижняя граница, а не цель.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if verify_overhead < 0:
        raise ValueError("verify_overhead must not be negative")
    return verify_overhead / k


def blended_alpha(mix):
    """Средневзвешенный acceptance rate по сегментам трафика.

    mix — последовательность пар (доля, alpha). Доли обязаны давать в сумме 1.

    blended_alpha(((0.7, 0.7), (0.3, 0.4)))  ->  0.61
    blended_alpha(((1.0, 0.42),))            ->  0.42

    Средневзвешенное прячет проигрывающий сегмент: смесь может пройти
    гейт, хотя внутри неё есть трафик, на котором спекуляция в минусе.
    Поэтому alpha меряют по сегментам, а не одним числом на весь прод.

    Доли не в сумме 1, отрицательная доля, alpha вне [0, 1] — ValueError.
    """
    if not mix:
        raise ValueError("empty traffic mix")
    total = 0.0
    blended = 0.0
    for share, alpha in mix:
        if share < 0:
            raise ValueError("share must not be negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        total += share
        blended += share * alpha
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"shares must sum to 1, got {total}")
    return blended


def normalize(weights):
    """Превратить неотрицательные веса в распределение вероятностей.

    normalize([1, 3])          ->  [0.25, 0.75]
    normalize([0.4, 0.2, 0.0]) ->  [0.6666..., 0.3333..., 0.0]

    Нули остаются нулями — это важно для residual_distribution: токен, который
    черновик перепредлагает, обязан получить РОВНО ноль, а не малую добавку.

    Все веса нулевые или есть отрицательный — ValueError.
    """
    if not weights:
        raise ValueError("empty weights")
    total = 0.0
    for w in weights:
        if w < 0:
            raise ValueError("weights must not be negative")
        total += w
    if total <= 0:
        raise ValueError("weights sum to zero: nothing to normalize")
    return [w / total for w in weights]


def sample_index(probs, rng):
    """Выбрать индекс по распределению probs. Случайность — только из rng.

    sample_index([0.0, 1.0, 0.0], rng)  ->  1   (при любом rng)

    Классический обратный CDF: тянем rng.random() и идём по накопленной сумме.

    Ловушка: сравнивать надо СТРОГО, `r < acc`. С `r <= acc` токен с нулевой
    вероятностью в начале списка вернётся при r = 0.0 — и спекулятивный вывод
    перестанет совпадать с целевым распределением.

    Глобальный random использовать нельзя: прогон обязан воспроизводиться
    по seed, иначе проверку распределения не написать.
    """
    if not probs:
        raise ValueError("empty probs")
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    # сюда попадаем только из-за накопленной ошибки float на хвосте
    return len(probs) - 1


def residual_distribution(target, draft):
    """Остаточное распределение (p - q)+ / ||(p - q)+|| — из чего сэмплят после отказа.

    residual_distribution([0.5, 0.3, 0.2], [0.1, 0.1, 0.8])
        ->  [0.6666..., 0.3333..., 0.0]

    Вычитание ПОВЕКТОРНОЕ, по каждому токену словаря отдельно, затем
    отрицательные обнуляются и остаток нормируется. Скалярная поправка
    (вычесть одно число из всех) ломает главную гарантию метода: выход
    перестаёт совпадать с распределением целевой модели.

    Токен, который черновик предлагал чаще цели, получает РОВНО ноль: он уже
    «оплачен» на этапе принятия, добавлять его ещё раз — сместить выход.

    Черновик совпадает с целью (остаток пустой) — ValueError: отказа в этом
    случае просто не бывает.

    Длины не совпали — ValueError.
    """
    if len(target) != len(draft):
        raise ValueError("target and draft must share the vocabulary")
    diff = [max(0.0, t - d) for t, d in zip(target, draft)]
    if sum(diff) <= 0:
        raise ValueError("residual is empty: draft already covers the target")
    return normalize(diff)


def speculative_step(target, draft, k, rng):
    """Один цикл draft-verify-reject. Вернуть {"tokens", "accepted", "drafted"}.

    Черновик предлагает k токенов из draft. Целевая модель проверяет их по
    порядку: токен x принимается с вероятностью min(1, target[x] / draft[x]).
    На первом же отказе цепочка обрывается, вместо отвергнутого токена
    выдаётся один сэмпл из residual_distribution — и шаг закончен. Если
    приняты все k, сверху выдаётся бонусный токен прямо из target.

    speculative_step(TARGET_PROBS, TARGET_PROBS, 5, rng)["accepted"]  ->  5
    len(...["tokens"])                                                ->  6

    Отсюда инвариант, который стоит проверить тестом: за шаг выходит РОВНО
    accepted + 1 токен, и в отказе, и в полном принятии. Он и превращается в
    формулу 1 + k * alpha из expected_speedup.

    Упрощение относительно настоящего EAGLE-3: там распределения зависят от
    уже выданного префикса, здесь на каждой позиции они одни и те же. Проверка
    «выход распределён как у цели» от этого не страдает — позиции независимы.

    Все k черновых токенов сэмплятся сразу, до проверки: в железе это один
    проход черновой головы, и отвергнутые позиции всё равно оплачены.
    Поэтому drafted всегда равен k, а не числу проверенных позиций.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(target) != len(draft):
        raise ValueError("target and draft must share the vocabulary")
    proposals = [sample_index(draft, rng) for _ in range(k)]
    tokens = []
    accepted = 0
    for x in proposals:
        q = draft[x]
        # q == 0 недостижимо: sample_index такой токен не вернёт. Оставлено
        # ради явности — деления на ноль здесь быть не может.
        ratio = 1.0 if q <= 0.0 else target[x] / q
        if rng.random() < min(1.0, ratio):
            tokens.append(x)
            accepted += 1
        else:
            # остаток пересчитывается на каждом отказе: в железе он фьюзится
            # в ядро верификации, здесь важна только читаемость
            tokens.append(sample_index(residual_distribution(target, draft), rng))
            break
    else:
        tokens.append(sample_index(target, rng))
    return {"tokens": tokens, "accepted": accepted, "drafted": k}


def run_speculative(target, draft, k, steps, rng, verify_overhead=0.0):
    """Прогнать steps шагов и собрать метрики, которые смотрят в проде.

    Вернуть словарь: drafted, accepted, emitted, acceptance_rate,
    tokens_per_forward, speedup, distribution.

    distribution — эмпирическое распределение всех выданных токенов. Главная
    гарантия метода в том, что оно совпадает с target даже при никудышном
    черновике: плохой черновик стоит скорости, но не качества.

    acceptance_rate — accepted / drafted, ровно то, что делит vLLM. Отсюда
    tokens_per_forward обязан в точности равняться 1 + k * acceptance_rate.

    speedup считается через expected_speedup от измеренного alpha.

    steps < 1 — ValueError.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    counts = [0] * len(target)
    accepted = 0
    emitted = 0
    for _ in range(steps):
        step = speculative_step(target, draft, k, rng)
        accepted += step["accepted"]
        for token in step["tokens"]:
            counts[token] += 1
            emitted += 1
    alpha = accepted / (steps * k)
    return {
        "drafted": steps * k,
        "accepted": accepted,
        "emitted": emitted,
        "acceptance_rate": alpha,
        "tokens_per_forward": emitted / steps,
        "speedup": expected_speedup(alpha, k, verify_overhead),
        "distribution": normalize(counts),
    }
