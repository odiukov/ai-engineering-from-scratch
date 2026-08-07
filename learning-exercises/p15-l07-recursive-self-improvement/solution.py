"""
Рекурсивное самоулучшение: гонка capability и alignment — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def next_cycle(capability, alignment, r_c, r_a):
    """Один цикл RSI: обе метрики умножаются на свой темп роста.

    next_cycle(1.0, 1.0, 1.15, 1.08)  ->  (1.15, 1.08)
    next_cycle(2.0, 2.0, 1.0, 1.0)    ->  (2.0, 2.0)

    capability — насколько система хорошо решает задачу, alignment —
    насколько она всё ещё преследует нужную цель. Темпы r_c и r_a
    независимы, и именно их разница даёт misalignment gap.

    Вернуть кортеж (capability, alignment), а не список: пара тут
    неизменяемая, её удобно распаковывать в цикле.
    """
    return (capability * r_c, alignment * r_a)


def race(cycles, r_c, r_a, start_c=1.0, start_a=1.0):
    """Траектория гонки: список кортежей (cycle, C, A, gap), gap = C - A.

    race(0, 1.15, 1.08)     ->  [(0, 1.0, 1.0, 0.0)]
    race(1, 1.2, 1.1)[1]    ->  (1, 1.2, 1.1, 0.1 с точностью до float)

    Нулевой цикл входит в результат, поэтому длина списка равна cycles + 1.
    Это не придирка: без стартовой точки нельзя проверить, что gap начинался
    с нуля.

    Строй шаг через next_cycle, а не переписывай умножение заново.
    """
    c, a = float(start_c), float(start_a)
    out = [(0, c, a, c - a)]
    for cyc in range(1, cycles + 1):
        c, a = next_cycle(c, a, r_c, r_a)
        out.append((cyc, c, a, c - a))
    return out


def crossing_cycle(trajectory, threshold):
    """Первый цикл, где gap дорос до threshold. Если не дорос — вернуть -1.

    crossing_cycle([(0, 1.0, 1.0, 0.0), (1, 2.0, 1.0, 1.0)], 0.5)  ->  1
    crossing_cycle([(0, 1.0, 1.0, 0.0)], 0.5)                      ->  -1

    Сравнение нестрогое: gap ровно на пороге уже считается пересечением.
    Порог — это «дальше пауза и человек», а не «дальше можно ещё чуть-чуть».

    Ловушка: -1 и 0 — разные ответы. 0 означает «gap был выше порога уже на
    старте», то есть петлю нельзя запускать вовсе.
    """
    for cyc, _c, _a, gap in trajectory:
        if gap >= threshold:
            return cyc
    return -1


def noisy_race(cycles, r_c, r_a, noise_c, noise_a, rng, floor=0.9):
    """Гонка с шумом: темп каждого цикла = rate + rng.gauss(0, noise).

    rng — экземпляр random.Random, а не глобальный модуль random. Иначе
    траектория зависит от того, что кто-то ещё дёргал генератор, и тесты
    перестают быть воспроизводимыми.

    noisy_race(5, 1.1, 1.1, 0.0, 0.0, random.Random(0)) даёт то же самое,
    что race(5, 1.1, 1.1): нулевая сигма — это отсутствие шума.

    floor не даёт темпу упасть слишком низко: одна неудачная выборка не
    должна обнулять систему. Порядок вызовов rng фиксирован — сначала
    capability, потом alignment; поменяешь местами, и seed перестанет
    воспроизводить чужие числа.
    """
    c, a = float(1.0), float(1.0)
    out = [(0, c, a, c - a)]
    for cyc in range(1, cycles + 1):
        # порядок важен: два вызова rng подряд, capability первым
        rc = max(floor, r_c + rng.gauss(0.0, noise_c))
        ra = max(floor, r_a + rng.gauss(0.0, noise_a))
        c, a = next_cycle(c, a, rc, ra)
        out.append((cyc, c, a, c - a))
    return out


def crossing_share(trials, cycles, r_c, r_a, noise_c, noise_a, threshold, rng):
    """Доля прогонов Монте-Карло, в которых gap пересёк threshold.

    Вернуть число от 0.0 до 1.0. При trials <= 0 вернуть 0.0.

    crossing_share(100, 30, 1.15, 1.08, 0.02, 0.03, 1.5, random.Random(1))
        ->  около 1.0: при таком разрыве темпов почти каждый прогон падает
            за порог

    Смысл замера: одна траектория ничего не доказывает, шум мог повезти.
    Доля пересечений — это и есть оценка риска пайплайна.
    """
    if trials <= 0:
        return 0.0
    crossed = 0
    for _ in range(trials):
        traj = noisy_race(cycles, r_c, r_a, noise_c, noise_a, rng)
        if crossing_cycle(traj, threshold) >= 0:
            crossed += 1
    return crossed / trials


def self_improve(propose, score, start, max_cycles, min_gain=0.0):
    """Цикл самоулучшения с измеримым критерием и жёстким потолком итераций.

    propose(system) -> система-преемник; score(system) -> число (чем больше,
    тем лучше). Преемник принимается, только если прирост строго больше
    min_gain.

    Вернуть dict с ключами:
      "system"  — последняя принятая система
      "score"   — её оценка
      "cycles"  — сколько преемников принято
      "history" — список оценок, начиная с оценки start
      "reason"  — "ceiling", если израсходован max_cycles,
                  "no_gain", если очередной преемник не дал прироста

    self_improve(lambda x: x + 1, float, 0, 3)
        ->  {"system": 3, "score": 3.0, "cycles": 3,
             "history": [0.0, 1.0, 2.0, 3.0], "reason": "ceiling"}

    Главное свойство, ради которого это упражнение: потолок останавливает
    петлю ДАЖЕ когда метрика продолжает расти. Без потолка «ещё один цикл,
    ведь стало лучше» — это условие, которое никогда не станет ложным.
    """
    system = start
    current = score(system)
    history = [current]
    cycles = 0
    reason = "ceiling"
    for _ in range(max_cycles):
        candidate = propose(system)
        candidate_score = score(candidate)
        if candidate_score - current <= min_gain:
            # улучшение не заработано — петля не должна крутиться вхолостую
            reason = "no_gain"
            break
        system, current = candidate, candidate_score
        history.append(current)
        cycles += 1
    return {
        "system": system,
        "score": current,
        "cycles": cycles,
        "history": history,
        "reason": reason,
    }


def audit_cycles(cycles, audit_every):
    """Номера циклов, после которых петля обязана остановиться на inter-cycle audit.

    audit_cycles(10, 3)  ->  [3, 6, 9]
    audit_cycles(10, 0)  ->  []   (человека между циклами нет)

    audit_every <= 0 означает «петля закрыта без человека»: быстрее, и это
    ровно тот режим, про который Хассабис спрашивал, можно ли его допускать.
    Пустой список тут — не ошибка, а честная модель такого режима.
    """
    if audit_every <= 0:
        return []
    return [cyc for cyc in range(1, cycles + 1) if cyc % audit_every == 0]
