"""
Failure modes: почему агенты ломаются — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Реестр инструментов агента. Всё, чего здесь нет, агент выдумал.
KNOWN_TOOLS = ("search", "read_file", "write_file", "send_email", "list_dir")

# Обязательные аргументы каждого инструмента. Схема — единственный способ
# отличить «вызвал правильный инструмент неправильно» от «вызвал не тот».
TOOL_SCHEMA = {
    "search": ("query",),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "send_email": ("to", "body"),
    "list_dir": ("path",),
}

# Инструменты, которые меняют мир. Только они дают scope creep и только по
# ним есть смысл проверять success hallucination.
EFFECT_TOOLS = ("write_file", "send_email")

# Куда направлен эффект: у файла это path, у письма — to.
TARGET_KEYS = {"write_file": "path", "send_email": "to"}

# Аргументы, в которых лежит адрес назначения. Ограничения проверяются
# по ним, а не по всем подряд: строка поиска "src/" — это не обращение к src/.
ADDRESS_KEYS = ("path", "to")

# Пять модов из полевых отчётов плюс два из обзора галлюцинаций агентов.
FAILURE_MODES = (
    "cascading_error",
    "context_loss",
    "hallucinated_action",
    "repeat_loop",
    "scope_creep",
    "success_hallucination",
    "tool_misuse",
)


def tool_problems(steps):
    """Две беды уровня вызова: несуществующий инструмент и кривые аргументы.

    Вернуть {"unknown": отсортированные имена, "bad_args": отсортированные индексы}.

    tool_problems([{"tool": "search", "args": {"query": "x"}}])
      ->  {"unknown": [], "bad_args": []}
    tool_problems([{"tool": "magic_scan", "args": {}}])
      ->  {"unknown": ["magic_scan"], "bad_args": []}
    tool_problems([{"tool": "read_file", "args": {"file": "a"}}])
      ->  {"unknown": [], "bad_args": [0]}

    Аргументы считаются кривыми, если не хватает обязательного ключа ИЛИ
    есть лишний, которого нет в схеме. Второе не придирка: лишний ключ почти
    всегда означает, что модель перепутала схемы двух инструментов.

    Выдуманный инструмент в bad_args не попадает: схемы у него нет, ругаться
    на аргументы бессмысленно — это уже другой failure mode.
    """
    unknown, bad_args = set(), []
    for i, step in enumerate(steps):
        tool = step["tool"]
        if tool not in KNOWN_TOOLS:
            unknown.add(tool)
            continue
        expected = set(TOOL_SCHEMA[tool])
        if set(step.get("args", {})) != expected:
            bad_args.append(i)
    return {"unknown": sorted(unknown), "bad_args": bad_args}


def first_repeat_index(steps, limit=3):
    """Индекс шага, на котором агент limit раз ПОДРЯД повторил один и тот же вызов.

    first_repeat_index([{"tool": "search", "args": {"query": "x"}}] * 3)  ->  2
    first_repeat_index([{"tool": "search", "args": {"query": "x"}}] * 2)  ->  None

    Ключевое слово — «подряд». Три обращения к search в разные моменты
    длинной сессии это нормальная работа; три обращения без единого другого
    действия между ними — зацикливание, агент не понял, что уже получил
    ответ, и крутит один и тот же шаг.

    Повтор — это совпадение и инструмента, и аргументов. limit < 2 ->
    ValueError: «повтор из одного шага» смысла не имеет.
    """
    if limit < 2:
        raise ValueError(f"повтор начинается с двух шагов, дано limit={limit}")
    run = 1
    for i in range(1, len(steps)):
        same = (
            steps[i]["tool"] == steps[i - 1]["tool"]
            and steps[i].get("args", {}) == steps[i - 1].get("args", {})
        )
        run = run + 1 if same else 1
        if run >= limit:
            return i
    return None


def cascade_radius(steps):
    """Сколько шагов агент успел сделать ПОСЛЕ первой ошибки.

    cascade_radius([{"tool": "search", "args": {}, "status": "error"},
                    {"tool": "read_file", "args": {}},
                    {"tool": "write_file", "args": {}}])   ->  2
    cascade_radius([{"tool": "search", "args": {}}])       ->  0

    Считаем от ПЕРВОЙ ошибки, а не от последней: урок про то, что одна
    ранняя ошибка тянет за собой хвост. Если взять последнюю, радиус всегда
    выйдет маленьким, и самый дорогой каскад окажется невидимым.

    Шаг без ключа "status" считается успешным.
    """
    for i, step in enumerate(steps):
        if step.get("status", "ok") == "error":
            return len(steps) - 1 - i
    return 0


def context_violations(steps, constraints):
    """Индексы шагов, нарушивших ограничение, объявленное в начале сессии.

    constraints — dict с необязательными ключами "forbidden_tools" и
    "forbidden_paths" (проверяются как префиксы пути или адреса).

    context_violations([{"tool": "read_file", "args": {"path": "README.md"}},
                        {"tool": "write_file", "args": {"path": "src/a.py",
                                                        "content": ""}}],
                       {"forbidden_paths": ("src/",)})   ->  [1]
    context_violations([], {"forbidden_tools": ("send_email",)})  ->  []

    Пустой constraints -> пустой список: нечего терять.

    Почему это называется context loss, а не «непослушание»: интересен не
    сам факт нарушения, а его позиция. Нарушение на шаге 1 — агент не понял
    задачу. Нарушение на шаге 30, когда первые 29 ограничение соблюдали, —
    ограничение вытеснилось из контекста.
    """
    forbidden_tools = set(constraints.get("forbidden_tools", ()))
    forbidden_paths = tuple(constraints.get("forbidden_paths", ()))
    hits = []
    for i, step in enumerate(steps):
        if step["tool"] in forbidden_tools:
            hits.append(i)
            continue
        # только адресные аргументы: строка поиска, начинающаяся с "src/",
        # никуда не обращается, и ругаться на неё — ложная тревога
        args = step.get("args", {})
        values = [args[k] for k in ADDRESS_KEYS if isinstance(args.get(k), str)]
        if forbidden_paths and any(v.startswith(forbidden_paths) for v in values):
            hits.append(i)
    return hits


def scope_creep_targets(steps, allowed_targets):
    """Цели, которые агент изменил, хотя его об этом не просили.

    scope_creep_targets([{"tool": "write_file",
                          "args": {"path": "README.md", "content": "x"}},
                         {"tool": "write_file",
                          "args": {"path": "src/a.py", "content": "x"}}],
                        ("README.md",))            ->  ["src/a.py"]
    scope_creep_targets([{"tool": "read_file", "args": {"path": "secret"}}], ())
      ->  []

    Чтение целью не считается: агент имеет право осмотреться. Считаются
    только EFFECT_TOOLS — запись и отправка. Именно так выглядит «создал
    лишний PR» и «отправил лишнее письмо» из урока.
    """
    allowed = set(allowed_targets)
    touched = set()
    for step in steps:
        tool = step["tool"]
        if tool not in EFFECT_TOOLS:
            continue
        target = step.get("args", {}).get(TARGET_KEYS[tool])
        if target is not None and target not in allowed:
            touched.add(target)
    return sorted(touched)


def success_hallucination(trace):
    """Агент отчитался об успехе, но мир не изменился.

    success_hallucination({"steps": [{"tool": "write_file",
                                      "args": {"path": "a", "content": "b"},
                                      "status": "error"}],
                           "claims_success": True, "state_changed": False})
      ->  True
    success_hallucination({"steps": [{"tool": "search", "args": {"query": "x"}}],
                           "claims_success": True, "state_changed": False})
      ->  False

    Второй случай — чистое чтение: успех там и не должен ничего менять.
    Проверка срабатывает, только если агент ПЫТАЛСЯ что-то изменить.

    Урок формулирует это жёстко: агент не отличает «у меня не получилось» от
    «задача невыполнима» и на 400 нередко закрывает цикл фразой об успехе.
    Поймать это можно только повторной пробой состояния, а не текстом ответа.
    """
    attempted = any(step["tool"] in EFFECT_TOOLS for step in trace["steps"])
    return bool(trace.get("claims_success") and attempted and not trace.get("state_changed"))


def tag_trace(trace):
    """Все failure modes одного трейса, отсортированный список меток.

    trace — dict с ключами "steps" и необязательными "constraints",
    "allowed_targets", "claims_success", "state_changed", "repeat_limit".

    tag_trace({"steps": [{"tool": "magic_scan", "args": {}}]})
      ->  ["hallucinated_action"]
    tag_trace({"steps": [{"tool": "search", "args": {"query": "x"}}]})
      ->  []

    Каскадом считается ошибка, после которой было минимум два шага: один
    шаг после ошибки — это обычно корректный retry, а не каскад.

    Меток может быть несколько: один трейс спокойно ломается сразу тремя
    способами, и разделять их важно — чинятся они по-разному.
    """
    steps = trace["steps"]
    problems = tool_problems(steps)
    labels = []
    if problems["unknown"]:
        labels.append("hallucinated_action")
    if problems["bad_args"]:
        labels.append("tool_misuse")
    if first_repeat_index(steps, trace.get("repeat_limit", 3)) is not None:
        labels.append("repeat_loop")
    if cascade_radius(steps) >= 2:
        labels.append("cascading_error")
    if context_violations(steps, trace.get("constraints", {})):
        labels.append("context_loss")
    if scope_creep_targets(steps, trace.get("allowed_targets", ())):
        labels.append("scope_creep")
    if success_hallucination(trace):
        labels.append("success_hallucination")
    return sorted(labels)


def mode_distribution(traces):
    """Сколько трейсов поймано каждым модом: {метка: число трейсов}.

    Считаются трейсы, а не срабатывания: трейс с двумя repeat_loop подряд
    всё равно даёт единицу.

    mode_distribution([{"steps": [{"tool": "magic_scan", "args": {}}]},
                       {"steps": [{"tool": "search", "args": {"query": "x"}}]}])
      ->  {"hallucinated_action": 1}

    Моды, не встретившиеся ни разу, в результат не попадают, ключи
    отсортированы. Это дешёвая замена кластеризации трейсов в Phoenix:
    сначала распределение, потом решение, какой мод чинить первым.
    """
    counts = {}
    for trace in traces:
        for label in tag_trace(trace):
            counts[label] = counts.get(label, 0) + 1
    return {label: counts[label] for label in sorted(counts)}
