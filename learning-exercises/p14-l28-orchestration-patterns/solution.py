"""
Паттерны оркестрации: supervisor, swarm, hierarchical — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Скриптованный «роутер» вместо LLM: слова-признаки трёх намерений.
INTENT_KEYWORDS = {
    "refund": ("refund", "money back", "chargeback", "возврат"),
    "bug": ("crash", "error", "bug", "broken"),
    "sales": ("price", "pricing", "quote", "plan"),
}

# Кто чем занимается. Специалист один на намерение — иначе роутить не к кому.
SPECIALISTS = {"refund": "billing_agent", "bug": "support_agent", "sales": "sales_agent"}

# Кольцо передач в swarm: центрального роутера нет, агент отдаёт задачу
# следующему соседу.
SWARM_RING = ("billing_agent", "support_agent", "sales_agent")

# Двухуровневая иерархия: верхний супервизор знает только команды,
# команда знает своих специалистов.
TEAMS = {"finance": ("refund",), "product": ("bug", "sales")}

# Паттерны, которые действительно про маршрутизацию. Debate из урока 25 сюда
# не входит: это верификация, а не оркестрация, — так и сам урок оговаривает.
PATTERNS = ("supervisor", "swarm", "hierarchical")


def classify(text):
    """Намерение задачи по словам-признакам: refund, bug, sales или unknown.

    classify("my app keeps crashing with an error")  ->  "bug"
    classify("I want a refund")                      ->  "refund"
    classify("привет")                               ->  "unknown"

    Побеждает намерение с наибольшим числом совпавших слов. Ничья решается
    по алфавиту: "refund and bug" -> "bug".

    Детерминированная ничья здесь не косметика. Роутер, который при равном
    счёте отвечает по порядку словаря, начнёт менять маршрут от перестановки
    ключей — и трейсы перестанут воспроизводиться.
    """
    low = text.lower()
    hits = {
        intent: sum(word in low for word in words)
        for intent, words in INTENT_KEYWORDS.items()
    }
    best = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0] if best[1] else "unknown"


def supervisor_route(tasks):
    """Supervisor-worker: центральный роутер раздаёт задачи специалистам.

    Вернуть {"assignments": [имя специалиста или None], "ops": int}.

    supervisor_route(["I want a refund"])
      ->  {"assignments": ["billing_agent"], "ops": 2}
    supervisor_route(["привет"])
      ->  {"assignments": [None], "ops": 1}

    Стоимость: один ход роутера плюс один ход специалиста. Нераспознанная
    задача стоит один ход — роутер посмотрел и никого не позвал.

    Специалисты друг с другом не разговаривают: весь трафик через роутер.
    Отсюда и главное свойство — трейс читается сверху вниз.
    """
    assignments, ops = [], 0
    for task in tasks:
        ops += 1                                # ход роутера
        specialist = SPECIALISTS.get(classify(task))
        assignments.append(specialist)
        if specialist is not None:
            ops += 1                            # ход специалиста
    return {"assignments": assignments, "ops": ops}


def swarm_route(tasks, entry=SWARM_RING[0], max_hops=2):
    """Swarm: роутера нет, агенты передают задачу соседу по кольцу.

    Вернуть {"assignments": [...], "ops": int, "handoffs": [путь по каждой задаче]}.

    swarm_route(["I want a refund"])["handoffs"]  ->  [["billing_agent"]]
    swarm_route(["crash on save"])["handoffs"]    ->  [["billing_agent", "support_agent"]]
    swarm_route(["привет"])["assignments"]        ->  [None]

    Каждый агент, которому задача не подходит, отдаёт её следующему в
    SWARM_RING. Счётчик передач обязателен: без него нераспознанная задача
    крутится по кольцу вечно. Исчерпали max_hops — задача не назначена.

    Плюс: до «своего» специалиста иногда доходит за один ход, дешевле
    супервизора. Минус: единой точки контроля нет, и отладка трейса тяжелее.
    """
    if entry not in SWARM_RING:
        raise ValueError(f"{entry!r} не входит в кольцо {SWARM_RING}")
    assignments, handoffs, ops = [], [], 0
    for task in tasks:
        target = SPECIALISTS.get(classify(task))
        path = [entry]
        while path[-1] != target and len(path) <= max_hops:
            nxt = SWARM_RING[(SWARM_RING.index(path[-1]) + 1) % len(SWARM_RING)]
            path.append(nxt)
        ops += len(path)
        handoffs.append(path)
        assignments.append(path[-1] if path[-1] == target else None)
    return {"assignments": assignments, "ops": ops, "handoffs": handoffs}


def detect_bouncing(handoff_log):
    """Есть ли в журнале передач возврат к предыдущему агенту: A -> B -> A.

    detect_bouncing(["a", "b", "a"])       ->  True
    detect_bouncing(["a", "b", "c"])       ->  False
    detect_bouncing(["a", "a"])            ->  False

    Повтор подряд ("a", "a") — это не пинг-понг, а один агент, продолжающий
    работу. Ловим именно возврат через одного.

    Урок называет это главной болячкой swarm: A отдаёт B, B не понимает и
    отдаёт обратно A. Счётчик хопов такое останавливает, но не объясняет —
    объясняет вот эта проверка.
    """
    return any(
        handoff_log[i] == handoff_log[i + 2] and handoff_log[i] != handoff_log[i + 1]
        for i in range(len(handoff_log) - 2)
    )


def hierarchical_route(tasks):
    """Hierarchical: верхний супервизор -> супервизор команды -> специалист.

    Вернуть {"assignments": [...], "ops": int, "teams": [имя команды или None]}.

    hierarchical_route(["I want a refund"])
      ->  {"assignments": ["billing_agent"], "ops": 3, "teams": ["finance"]}

    Стоимость три хода вместо двух: за лишний уровень платят всегда.
    Оправдано это ровно одним — когда описания всех специалистов перестают
    помещаться в контекст одного супервизора. Три уровня «потому что
    энтерпрайз» урок прямо называет fake hierarchy.
    """
    assignments, teams, ops = [], [], 0
    for task in tasks:
        ops += 1                                # верхний супервизор
        intent = classify(task)
        team = next((name for name, intents in TEAMS.items() if intent in intents), None)
        teams.append(team)
        if team is None:
            assignments.append(None)
            continue
        ops += 2                                # супервизор команды + специалист
        assignments.append(SPECIALISTS[intent])
    return {"assignments": assignments, "ops": ops, "teams": teams}


def run_parallel(tasks, worker, completion_order=None):
    """Параллельный запуск: результаты в порядке ЗАДАЧ, а не завершения.

    completion_order — перестановка индексов, имитирующая, кто закончил
    раньше. None означает порядок задач.

    run_parallel(["a", "b"], str.upper)                       ->  ["A", "B"]
    run_parallel(["a", "b"], str.upper, completion_order=[1, 0])  ->  ["A", "B"]
    run_parallel(["a"], str.upper, completion_order=[0, 0])   ->  ValueError

    Не перестановка индексов -> ValueError.

    Свойство, ради которого функция и существует: параллельная ветка обязана
    давать один и тот же результат при любом порядке завершения. Если код
    делает `results.append(...)` по мере готовности, порядок ответов начнёт
    зависеть от того, какой воркер сегодня быстрее, — и воспроизвести баг
    станет невозможно.
    """
    n = len(tasks)
    order = list(range(n)) if completion_order is None else list(completion_order)
    if sorted(order) != list(range(n)):
        raise ValueError(f"порядок завершения не перестановка индексов: {order}")
    results = [None] * n
    for idx in order:                           # кладём по индексу, не append
        results[idx] = worker(tasks[idx])
    return results


def pick_pattern(specialists, latency_critical=False, accuracy_critical=False,
                 supervisor_context_ok=True):
    """Выбор топологии по порядку решений из урока.

    pick_pattern(1)                                  ->  "single_agent"
    pick_pattern(3)                                  ->  "supervisor"
    pick_pattern(12, supervisor_context_ok=False)    ->  "hierarchical"
    pick_pattern(3, accuracy_critical=True)          ->  "debate"
    pick_pattern(3, latency_critical=True)           ->  "swarm"

    Порядок проверок, сверху вниз:
      1. один специалист — топология не нужна вообще;
      2. контекст супервизора не вмещает описания — hierarchical (это
         жёсткое ограничение, а не предпочтение);
      3. точность важнее стоимости — debate;
      4. задержка важнее ясности рассуждений — swarm;
      5. иначе — supervisor.

    specialists < 1 -> ValueError.

    Смысл функции — сопротивление topology-first мышлению: сначала называем
    ограничение, потом получаем паттерн, а не наоборот.
    """
    if specialists < 1:
        raise ValueError(f"специалистов не может быть меньше одного: {specialists}")
    if specialists == 1:
        return "single_agent"
    if not supervisor_context_ok:
        return "hierarchical"
    if accuracy_critical:
        return "debate"
    if latency_critical:
        return "swarm"
    return "supervisor"


def compare_patterns(tasks):
    """Один и тот же набор задач через все PATTERNS: кто во что обходится.

    Вернуть {паттерн: {"assignments": [...], "ops": int}}.

    compare_patterns(["I want a refund"])["supervisor"]["ops"]     ->  2
    compare_patterns(["I want a refund"])["hierarchical"]["ops"]   ->  3

    Назначения обязаны совпасть у всех трёх — маршрут не должен зависеть от
    топологии. Различаться должна только цена. Ровно это и есть аргумент
    урока: если ответ один, платить за лишние уровни незачем.
    """
    runners = {
        "supervisor": supervisor_route,
        "swarm": swarm_route,
        "hierarchical": hierarchical_route,
    }
    return {
        name: {"assignments": result["assignments"], "ops": result["ops"]}
        for name, result in ((n, runners[n](tasks)) for n in PATTERNS)
    }
