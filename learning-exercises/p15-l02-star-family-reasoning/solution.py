"""
STaR, V-STaR, Quiet-STaR: самообучение на собственных рассуждениях — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Мы собираем петлю STaR (Zelikman et al., 2022) руками:

    сэмплировать обоснование -> оставить те, что дали ВЕРНЫЙ ответ ->
    дообучить -> повторить

«Модель» здесь — таблица весов по стратегиям рассуждения, «дообучение» —
обновление этой таблицы. Никакой сети, никаких LLM, никакого torch: смысл
урока в том, чтобы увидеть, что именно петля усиливает.

Каждая стратегия описана двумя вероятностями попасть в правильный ответ: на
своём распределении (id) и на чужом (ood), плюс флагом «рассуждение честное».

  sound        честное рассуждение, работает везде;
  shortcut     срез, который угадывает ответ в 40% случаев на своём
               распределении и почти никогда на чужом — тот самый shortcut
               rationale, ради которого написан раздел «Why all three share a
               safety concern»;
  random       угадывание;
  rationalized обоснование, дописанное задним числом под подсказанный ответ
               (STaR-рационализация). Ответ верный по построению, честность
               рассуждения — нет.
"""

STRATEGIES = {
    "sound": {"id": 1.00, "ood": 1.00, "sound": True},
    "shortcut": {"id": 0.40, "ood": 0.05, "sound": False},
    "random": {"id": 0.10, "ood": 0.10, "sound": False},
    "rationalized": {"id": 0.30, "ood": 0.05, "sound": False},
}


def pick_strategy(rng, weights):
    """Выбрать стратегию по таблице весов (рулетка).

    pick_strategy(rng, {"sound": 1.0, "shortcut": 0.0})   ->  "sound"
    pick_strategy(rng, {"sound": 3, "random": 1})         ->  "sound" в ~75%

    Веса НЕ обязаны быть нормированы — делим на их сумму сами.

    Две ловушки:
      * порядок обхода словаря. Он стабилен в Python, но зависит от порядка
        вставки: одна и та же таблица, собранная в другом порядке, даст другую
        последовательность при том же seed. Обходи sorted(weights) — тогда
        результат зависит только от весов и от rng;
      * rng — параметр, а не глобальный random. Иначе воспроизвести раунд
        обучения невозможно.
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("сумма весов должна быть положительной")
    names = sorted(weights)
    threshold = rng.random() * total
    running = 0.0
    for name in names:
        running += weights[name]
        if threshold < running:
            return name
    # добираемся сюда только из-за накопленной погрешности сложения
    return names[-1]


def sample_trace(rng, weights, on_ood=False):
    """Сгенерировать одно обоснование: стратегия, верность ответа, честность.

    Вернуть словарь с ключами "strategy", "answer_correct", "rationale_sound".

    sample_trace(rng, {"sound": 1.0})
        ->  {"strategy": "sound", "answer_correct": True, "rationale_sound": True}
    sample_trace(rng, {"random": 1.0})
        ->  answer_correct True примерно в 10% случаев

    on_ood переключает распределение задач: у shortcut шанс попасть в ответ
    падает с 0.40 до 0.05, у sound не меняется. Это и есть весь сюжет урока —
    отличить одно от другого по обучающей выборке невозможно.
    """
    name = pick_strategy(rng, weights)
    spec = STRATEGIES[name]
    hit_rate = spec["ood"] if on_ood else spec["id"]
    return {
        "strategy": name,
        "answer_correct": rng.random() < hit_rate,
        "rationale_sound": spec["sound"],
    }


def expected_accuracy(weights, on_ood=False):
    """Ожидаемая доля верных ответов для таблицы весов, без сэмплирования.

    expected_accuracy({"sound": 1.0})                    ->  1.0
    expected_accuracy({"sound": 1.0, "random": 1.0})     ->  0.55
    expected_accuracy({"shortcut": 1.0}, on_ood=True)    ->  0.05

    Это средневзвешенная вероятность попадания. Аналитическая формула вместо
    прогона на миллионе задач: и быстрее, и без шума, а свойства петли на ней
    видно точнее.
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("сумма весов должна быть положительной")
    key = "ood" if on_ood else "id"
    return sum(w * STRATEGIES[n][key] for n, w in weights.items()) / total


def star_filter(traces):
    """Фильтр STaR: оставить только обоснования, давшие ВЕРНЫЙ ответ.

    star_filter([{"strategy": "sound", "answer_correct": True,
                  "rationale_sound": True}])   ->  тот же список
    star_filter([{"strategy": "sound", "answer_correct": False,
                  "rationale_sound": True}])   ->  []

    Три строки, в которых сидит вся проблема метода. Критерий — только ответ:
      * честное рассуждение, приведшее к НЕверному ответу, выбрасывается;
      * срез, случайно попавший в ответ, сохраняется и будет усилен.

    Это и называется answer-conditioned gradient. Процессная разметка
    (process reward model, Lightman et al. 2023) — альтернатива, которая
    смотрит на шаги, а не на итог.
    """
    return [t for t in traces if t["answer_correct"]]


def rationalize(traces):
    """STaR-рационализация: из провалов сделать обоснования по подсказке.

    Для каждого обоснования с неверным ответом вернуть новое, где ответ верен
    по построению (мы подсказали его модели), стратегия "rationalized", а
    честность рассуждения — False.

    rationalize([{"strategy": "random", "answer_correct": False,
                  "rationale_sound": False}])
        ->  [{"strategy": "rationalized", "answer_correct": True,
              "rationale_sound": False}]
    rationalize([{"strategy": "sound", "answer_correct": True,
                  "rationale_sound": True}])   ->  []

    Зачем это нужно: задачи, которые модель НИКОГДА не решает, иначе просто
    выпадают из обучения — фильтр не пропускает ни одного примера. На GSM8K
    рационализация дала основную часть прироста с 5.8% до 10.7%.

    Цена: обоснование написано задним числом под известный ответ, поэтому
    честным оно не считается. Ставить rationale_sound=True здесь — значит
    соврать себе в собственном датасете.
    """
    return [
        {"strategy": "rationalized", "answer_correct": True, "rationale_sound": False}
        for t in traces
        if not t["answer_correct"]
    ]


def finetune(weights, traces, alpha=0.6):
    """«Дообучение»: сместить таблицу весов к распределению стратегий в traces.

    Вернуть НОВУЮ нормированную таблицу (сумма весов равна 1).

        new[s] = alpha * доля s в traces  +  (1 - alpha) * прежняя доля s

    finetune({"sound": 1.0, "random": 1.0},
             [{"strategy": "sound", ...}], alpha=1.0)   ->  {"sound": 1.0, "random": 0.0}
    finetune({"sound": 1.0, "random": 1.0}, [...], alpha=0.0)
        ->  {"sound": 0.5, "random": 0.5}   (прежние доли, данные не при чём)

    Настоящий SFT здесь заменён обновлением таблицы: смысл тот же — то, что
    чаще встречается в обучающем наборе, чаще выдаётся моделью.

    Ловушки:
      * alpha=1.0 стирает всё, чего не было в traces. Смесь с прежним
        распределением — то, что не даёт петле схлопнуться за один раунд;
      * пустой traces — не повод делить на ноль. Возвращай нормированную
        копию прежней таблицы;
      * стратегия, впервые появившаяся в traces (например "rationalized"),
        обязана попасть в результат, даже если её не было в исходной таблице.
    """
    prior_total = sum(weights.values())
    if prior_total <= 0:
        raise ValueError("сумма весов должна быть положительной")
    if not traces:
        return {n: w / prior_total for n, w in weights.items()}

    names = set(weights) | {t["strategy"] for t in traces}
    counts = dict.fromkeys(names, 0)
    for t in traces:
        counts[t["strategy"]] += 1

    updated = {
        n: alpha * (counts[n] / len(traces)) + (1 - alpha) * (weights.get(n, 0.0) / prior_total)
        for n in names
    }
    # по построению сумма уже равна 1, нормировка страхует от накопленной
    # погрешности и от alpha вне [0, 1]
    total = sum(updated.values())
    return {n: v / total for n, v in updated.items()}


def star_round(rng, weights, n_samples=400, alpha=0.6, use_rationalization=False):
    """Один раунд STaR: сэмплировать, отфильтровать, (опц.) рационализировать,
    дообучить.

    Вернуть словарь с ключами:
      "weights"         — новая таблица;
      "kept"            — обоснования, ушедшие в обучение;
      "solved_fraction" — доля задач, решённых ДО рационализации.

    star_round(rng, {"sound": 0.2, "shortcut": 0.8})["solved_fraction"]
        ->  примерно 0.52   (0.2 * 1.0 + 0.8 * 0.4)

    Порядок обязателен: solved_fraction считается по исходной выборке. Если
    посчитать его после рационализации, он всегда будет 1.0 — рационализация
    по построению «решает» всё, и метрика перестанет что-либо измерять.

    Свойство, ради которого раунд существует: ожидаемая точность после раунда
    не ниже, чем до (неравенство Коши-Буняковского на весах и вероятностях).
    Заметь, что оно верно только на СВОЁМ распределении. На чужом та же петля
    точность роняет — она усиливает shortcut, который на ood не работает.
    """
    traces = [sample_trace(rng, weights) for _ in range(n_samples)]
    kept = star_filter(traces)
    solved_fraction = len(kept) / n_samples if n_samples else 0.0
    if use_rationalization:
        kept = kept + rationalize(traces)
    return {
        "weights": finetune(weights, kept, alpha),
        "kept": kept,
        "solved_fraction": solved_fraction,
    }


def vstar_select(traces, verifier):
    """V-STaR: выбрать лучшее обоснование из N по оценке верификатора.

    verifier — функция обоснование -> число. Возвращаем само обоснование.

    vstar_select([a, b], lambda t: 1.0 if t["rationale_sound"] else 0.0)  ->  a,
        если честное рассуждение у a

    При РАВНЫХ оценках возвращается первое встреченное. Это не мелочь:
    best-of-N с недетерминированным разрешением ничьих даёт разные ответы на
    одном и том же входе, и воспроизвести отчёт об оценке становится нельзя.
    Сравнение строгое (>), а не нестрогое (>=) — ровно поэтому.

    Пустой список — ValueError: «лучшего из нуля» не существует.

    В V-STaR (Hosseini et al., 2024) верификатор обучается DPO на обоих
    классах — и верных, и неверных обоснованиях. Основной прирост даёт не
    дообучение генератора, а именно этот отбор на инференсе.
    """
    if not traces:
        raise ValueError("нечего выбирать: список обоснований пуст")
    best = traces[0]
    best_score = verifier(best)
    for trace in traces[1:]:
        score = verifier(trace)
        if score > best_score:
            best, best_score = trace, score
    return best
