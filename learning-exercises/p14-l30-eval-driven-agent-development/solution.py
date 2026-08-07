"""
Eval-driven разработка агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Три слоя оценки из урока: готовый бенчмарк, свой офлайн-набор, продакшн.
CASE_LAYERS = ("benchmark", "custom", "online")


def run_case(case, agent):
    """Один eval-кейс: прогнать агента и проверить ожидание.

    case — {"id", "layer", "prompt", "expect"}; expect — подстрока, которая
    ОБЯЗАНА встретиться в ответе. agent(prompt) возвращает строку.

    Вернуть {"id", "layer", "passed", "answer", "error"}.

    run_case({"id": "c1", "layer": "custom", "prompt": "кто автор ReAct?",
              "expect": "arXiv"}, lambda p: "см. arXiv:2210.03629")
      ->  {"id": "c1", "layer": "custom", "passed": True,
           "answer": "см. arXiv:2210.03629", "error": None}

    Упавший агент — это ПРОВАЛ кейса, а не падение прогона: passed=False,
    answer="", error — текст исключения. Иначе один кривой кейс уронит всю
    суиту в CI, и остальные сорок никто не увидит.

    layer не из CASE_LAYERS -> ValueError: кейс без слоя невозможно потом
    сгруппировать, а разбивка по слоям — половина смысла отчёта.
    """
    if case["layer"] not in CASE_LAYERS:
        raise ValueError(f"неизвестный слой {case['layer']!r}, ожидался из {CASE_LAYERS}")
    try:
        answer = agent(case["prompt"])
    except Exception as exc:                    # noqa: BLE001 — падение агента это FAIL
        return {
            "id": case["id"],
            "layer": case["layer"],
            "passed": False,
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "id": case["id"],
        "layer": case["layer"],
        "passed": case["expect"] in answer,
        "answer": answer,
        "error": None,
    }


def run_suite(cases, agent):
    """Вся суита по порядку. Вернуть список результатов run_case.

    run_suite([], agent)  ->  []

    Повторяющиеся id -> ValueError. Два кейса с одним id — тихая беда:
    базовая линия хранится по id, и один из двух молча затрёт другой,
    так что регрессия перестанет быть видна.

    Порядок сохраняем: рядом стоящие кейсы обычно про одну и ту же фичу, и
    читать отчёт вперемешку невозможно.
    """
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"id кейсов должны быть уникальны: {ids}")
    return [run_case(case, agent) for case in cases]


def summarize(results):
    """Сводка прогона: сколько прошло всего и по слоям.

    Вернуть {"total", "passed", "rate", "by_layer": {слой: {"total", "passed"}}}.

    summarize([{"id": "c1", "layer": "custom", "passed": True,
                "answer": "", "error": None}])
      ->  {"total": 1, "passed": 1, "rate": 1.0,
           "by_layer": {"custom": {"total": 1, "passed": 1}}}

    В by_layer попадают только те слои, по которым есть кейсы: пустой слой —
    это не «100% зелёных», это отсутствие проверки, и путать их нельзя.

    Пустой список -> ValueError. Суита из нуля кейсов не доказывает ничего,
    а rate=0.0 или 1.0 на ней врал бы в обе стороны.
    """
    if not results:
        raise ValueError("пустая суита ничего не доказывает")
    by_layer = {}
    for res in results:
        cell = by_layer.setdefault(res["layer"], {"total": 0, "passed": 0})
        cell["total"] += 1
        cell["passed"] += bool(res["passed"])
    passed = sum(1 for res in results if res["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "rate": passed / len(results),
        "by_layer": by_layer,
    }


def detect_regression(baseline, results):
    """Сравнение с последней известной хорошей линией — по КЕЙСАМ, не по среднему.

    baseline — {id: проходил ли}, results — свежий прогон (список run_case).

    Вернуть {"broken", "fixed", "new", "missing"} — списки id, отсортированные.

    detect_regression({"c1": True}, [{"id": "c1", "layer": "custom",
                                      "passed": False, "answer": "", "error": None}])
      ->  {"broken": ["c1"], "fixed": [], "new": [], "missing": []}

    Главное свойство: один починенный кейс и один сломанный дают то же
    среднее, но broken при этом непустой. Средняя доля прохождения —
    худший детектор регрессии из возможных, потому что она умеет
    компенсировать поломку случайным успехом в другом месте.

    missing — кейс из базовой линии, которого в прогоне нет. Это тоже
    регрессия: убрать красный кейс из суиты дешевле, чем починить, поэтому
    так и делают.
    """
    now = {res["id"]: bool(res["passed"]) for res in results}
    return {
        "broken": sorted(i for i, ok in now.items() if baseline.get(i) and not ok),
        "fixed": sorted(i for i, ok in now.items() if ok and baseline.get(i) is False),
        "new": sorted(i for i in now if i not in baseline),
        "missing": sorted(i for i in baseline if i not in now),
    }


def ci_gate(results, baseline, max_rate_drop=0.05):
    """Гейт на merge: пускать ли ветку в main.

    Вернуть {"allowed": bool, "reason": str}.

    ci_gate([], {"c1": True})["allowed"]  ->  False   (нет кейсов — нет доказательств)

    Блокируем в четырёх случаях:
      1. прогон пустой;
      2. есть broken — кейс, который раньше проходил (имя в reason);
      3. есть missing — кейс из базовой линии пропал из прогона;
      4. доля прохождения просела больше чем на max_rate_drop.

    Пункт 2 срабатывает даже когда средняя доля не изменилась — за этим
    гейт и нужен. Пункт 4 ловит другое: новые кейсы, которые сразу красные.

    max_rate_drop < 0 -> ValueError: отрицательный порог требовал бы
    улучшения на каждом PR, и main перестал бы принимать что-либо.
    """
    if max_rate_drop < 0:
        raise ValueError(f"порог просадки не может быть отрицательным: {max_rate_drop}")
    if not results:
        return {"allowed": False, "reason": "нет кейсов"}
    diff = detect_regression(baseline, results)
    if diff["broken"]:
        return {"allowed": False, "reason": f"сломаны кейсы: {diff['broken']}"}
    if diff["missing"]:
        return {"allowed": False, "reason": f"кейсы пропали из суиты: {diff['missing']}"}
    rate = summarize(results)["rate"]
    base_rate = sum(1 for ok in baseline.values() if ok) / len(baseline) if baseline else 0.0
    drop = base_rate - rate
    if drop > max_rate_drop:
        return {"allowed": False, "reason": f"просадка {drop:.1%} > {max_rate_drop:.1%}"}
    return {"allowed": True, "reason": f"rate={rate:.1%} baseline={base_rate:.1%}"}


def evaluator_optimizer(propose, judge, max_rounds=3):
    """Тесная петля Anthropic: предложить -> оценить -> переписать по замечанию.

    propose(feedback) возвращает кандидата; в первом круге feedback = None.
    judge(candidate) возвращает (прошло ли, причина).

    Вернуть {"passed", "rounds", "final", "reason", "history"}.

    evaluator_optimizer(lambda fb: "готово", lambda c: (True, "ok"))
      ->  {"passed": True, "rounds": 1, "final": "готово", "reason": "ok",
           "history": ["готово"]}

    Причина от judge обязана попасть в СЛЕДУЮЩИЙ вызов propose — иначе это
    не петля, а три независимые попытки наугад. Останавливаемся сразу после
    первого успеха: лишний круг стоит денег и может испортить готовый ответ.

    max_rounds < 1 -> ValueError.
    """
    if max_rounds < 1:
        raise ValueError(f"кругов должно быть хотя бы один: {max_rounds}")
    feedback, history, candidate, reason = None, [], "", ""
    for rnd in range(1, max_rounds + 1):
        candidate = propose(feedback)
        history.append(candidate)
        ok, reason = judge(candidate)
        if ok:
            return {"passed": True, "rounds": rnd, "final": candidate,
                    "reason": reason, "history": history}
        feedback = reason                       # замечание уходит в следующий круг
    return {"passed": False, "rounds": max_rounds, "final": candidate,
            "reason": reason, "history": history}


def flaky_cases(runs):
    """Кейсы, которые в одних прогонах зелёные, в других красные.

    runs — список прогонов, каждый {id: проходил ли}.

    flaky_cases([{"c1": True}, {"c1": False}])   ->  ["c1"]
    flaky_cases([{"c1": False}, {"c1": False}])  ->  []
    flaky_cases([{"c1": True}])                  ->  []

    Стабильно красный кейс — не мигание, а регрессия: его ловит
    detect_regression, и лечится он кодом. Мигание лечится другим —
    фиксацией seed и снимком состояния. Один прогон мигание доказать не
    может в принципе: нужно минимум два разных исхода.

    id, отсутствующий в каком-то прогоне, там просто не наблюдался; это не
    считается ни успехом, ни провалом. Ответ отсортирован.
    """
    seen = {}
    for run in runs:
        for cid, ok in run.items():
            seen.setdefault(cid, set()).add(bool(ok))
    return sorted(cid for cid, outcomes in seen.items() if len(outcomes) > 1)


def coverage_gaps(cases, required_topics):
    """Чего суита не проверяет: темы без кейса и слои без кейса.

    У кейса необязательное поле "topics" — кортеж тем, которые он покрывает.

    Вернуть {"uncovered_topics": [...], "empty_layers": [...]}, оба списка
    отсортированы.

    coverage_gaps([{"id": "c1", "layer": "custom", "prompt": "",
                    "expect": "", "topics": ("guardrails",)}],
                  ("guardrails", "memory"))
      ->  {"uncovered_topics": ["memory"],
           "empty_layers": ["benchmark", "online"]}

    Урок требует ровно этого: каждый guardrail и каждое выученное правило
    отображается в кейс, а слоёв ровно три. Пустой слой в отчёте лучше
    выглядящей стопроцентной суиты, которая проверяет только один слой.
    """
    covered = {topic for case in cases for topic in case.get("topics", ())}
    used_layers = {case["layer"] for case in cases}
    return {
        "uncovered_topics": sorted(t for t in required_topics if t not in covered),
        "empty_layers": sorted(layer for layer in CASE_LAYERS if layer not in used_layers),
    }
