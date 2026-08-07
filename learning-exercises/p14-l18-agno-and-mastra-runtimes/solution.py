"""
Продакшн-рантаймы агентов: быстрая инстанциация и типизированные workflow — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками то, что Agno и Mastra дают из коробки:
  * Agno — stateless session-scoped бэкенд: на каждый запрос свежий агент,
    состояние живёт в базе, а не в объекте (`handle_request`);
    заявленные ~2 μs на инстанциацию и ~3.75 KiB памяти на агента
    (`estimate_runtime_cost`).
  * Mastra — Unified Model Router (`route_model`), Zod-типизированные
    инструменты (`typed_tool_call`), Workflows (`run_workflow`),
    composite storage (`split_records`).
  * И выбор между ними (`pick_runtime`).

Ни сети, ни LLM, ни сторонних пакетов: модель — детерминированная заглушка.
"""

# Заявленные Agno целевые показатели (их docs, 2026). Здесь — только
# значения по умолчанию для оценки, ничего не измеряем.
AGNO_INSTANTIATION_US = 2.0
AGNO_MEMORY_KIB = 3.75

# Composite storage: у Mastra каждый вид записей может уходить в свой бэкенд.
RECORD_KINDS = ("memory", "workflows", "observability")


def stub_model(prompt, model_id):
    """Детерминированная заглушка модели: ответ зависит от промпта и модели.

    stub_model("hi", "anthropic/claude-x")  ->  'claude-x:2:209'
    stub_model("hi", "openai/gpt-x")        ->  'gpt-x:2:209'

    Заменяет вызов провайдера. Из model_id берётся часть после последнего "/":
    в ответе видно, куда бы ушёл запрос после роутинга.
    """
    total = sum(ord(ch) for ch in prompt) % 9973
    tag = model_id.split("/")[-1]
    return f"{tag}:{len(prompt)}:{total}"


def route_model(router, requested):
    """Unified Model Router: превратить запрошенное имя в (provider, model).

    router = {"models": {model_id: provider}, "aliases": {alias: model_id}}

    r = {"models": {"anthropic/claude-x": "anthropic"}, "aliases": {"fast": "anthropic/claude-x"}}
    route_model(r, "anthropic/claude-x")  ->  {"provider": "anthropic", "model": "anthropic/claude-x"}
    route_model(r, "fast")                ->  {"provider": "anthropic", "model": "anthropic/claude-x"}
    route_model(r, "gpt-x")               ->  KeyError

    Две ловушки. Первая: алиас, который ведёт на другой алиас, — не «идём
    дальше по цепочке», а ValueError. Цепочки алиасов однажды замкнутся в
    кольцо, и роутер зациклится на проде. Второй прыжок делать нельзя.
    Вторая: неизвестное имя — KeyError, а не «подставим дефолтную модель»:
    молчаливый дефолт означает, что счёт придёт за не ту модель.
    """
    models = router.get("models", {})
    aliases = router.get("aliases", {})

    model_id = requested
    if model_id not in models:
        if model_id not in aliases:
            raise KeyError(requested)
        model_id = aliases[model_id]
        # Ровно один прыжок: если приехали снова в алиас — это ошибка конфига.
        if model_id not in models:
            raise ValueError(f"alias chain: {requested} -> {model_id}")
    return {"provider": models[model_id], "model": model_id}


def typed_tool_call(tool, payload):
    """Вызов инструмента с проверкой схемы — то, что у Mastra делает Zod.

    tool = {"name": str, "schema": {поле: тип}, "handler": callable}

    t = {"name": "search", "schema": {"query": str}, "handler": lambda p: p["query"].upper()}
    typed_tool_call(t, {"query": "ai"})        ->  'AI'
    typed_tool_call(t, {})                     ->  ValueError  (нет поля)
    typed_tool_call(t, {"query": 1})           ->  TypeError   (не тот тип)
    typed_tool_call(t, {"query": "ai", "n": 1}) ->  ValueError  (лишнее поле)

    Ловушка на всю жизнь: isinstance(True, int) == True, поэтому проверка
    через isinstance пропустит булево там, где схема просит число. Сравнивай
    тип точно.

    Зачем в AI: аргументы приходят от модели, то есть от ненадёжного
    источника. Схема — единственное место, где ошибку ещё дёшево поймать.
    """
    schema = tool["schema"]
    for field, expected in schema.items():
        if field not in payload:
            raise ValueError(f"{tool['name']}: missing field {field}")
        # type(...) is expected, а не isinstance: True не должен пройти как int.
        if type(payload[field]) is not expected:
            raise TypeError(f"{tool['name']}: field {field} expects {expected.__name__}")
    extra = sorted(set(payload) - set(schema))
    if extra:
        raise ValueError(f"{tool['name']}: unexpected fields {extra}")
    return tool["handler"](payload)


def handle_request(store, session_id, prompt, agent_factory, router, requested, model=stub_model):
    """Один HTTP-запрос в стиле Agno: свежий агент, состояние — в store.

    agent_factory() вызывается на КАЖДЫЙ запрос и возвращает новый агент
    (dict). Промпт запоминается в agent["seen"], история — в
    store[session_id].

    Возвращает {"answer", "provider", "model", "agent_seen", "history_len"}.

    Два запроса в одну сессию:
      history_len -> 2, затем 4      (состояние копится в store)
      agent_seen  -> ["a"], затем ["b"]  (агент не переживает запрос)

    Ловушка: если тащить состояние в объекте агента, второй запрос уже
    зависит от того, на какой воркер попал первый. Именно поэтому
    рекомендованный путь Agno — stateless session-scoped бэкенд: агент
    одноразовый, память сессии внешняя.
    """
    agent = agent_factory()
    agent.setdefault("seen", []).append(prompt)

    route = route_model(router, requested)

    history = store.setdefault(session_id, [])
    history.append({"role": "user", "content": prompt})
    # Контекст восстанавливается из store, а не из агента — в этом весь смысл.
    context = "|".join(turn["content"] for turn in history)
    answer = model(context, route["model"])
    history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "provider": route["provider"],
        "model": route["model"],
        "agent_seen": list(agent["seen"]),
        "history_len": len(history),
    }


def run_workflow(steps, payload):
    """Mastra Workflow: шаги по очереди, выход одного — вход следующего.

    steps — список пар (имя, функция).

    run_workflow([("inc", lambda x: x + 1), ("dbl", lambda x: x * 2)], 3)
        ->  {"output": 8, "trace": [("inc", 4), ("dbl", 8)], "failed": None}
    run_workflow([("boom", lambda x: 1 / 0), ("after", lambda x: x)], 3)
        ->  {"output": None, "trace": [], "failed": "boom"}
    run_workflow([], 3)
        ->  {"output": 3, "trace": [], "failed": None}

    Ловушка: упавший шаг ОБРЫВАЕТ workflow — следующие шаги не выполняются.
    Иначе шаг после сбоя получит на вход мусор и «починит» падение,
    превратив его в тихо неверный результат.
    """
    trace = []
    current = payload
    for name, fn in steps:
        try:
            current = fn(current)
        except Exception:
            # Имя шага важнее текста исключения: по нему потом ищут в трейсе.
            return {"output": None, "trace": trace, "failed": name}
        trace.append((name, current))
    return {"output": current, "trace": trace, "failed": None}


def split_records(records, routing):
    """Composite storage: разложить записи по бэкендам согласно routing.

    Запись — dict {"kind": <из RECORD_KINDS>, "data": ...}.
    routing — {kind: имя бэкенда}.

    split_records([{"kind": "memory", "data": 1}], {"memory": "pg"})
        ->  {"pg": [{"kind": "memory", "data": 1}]}
    split_records([{"kind": "traces", "data": 1}], {})   ->  ValueError
    split_records([{"kind": "memory", "data": 1}], {})    ->  ValueError

    Ловушка: kind без бэкенда — ValueError, а не «сложим куда-нибудь». Иначе
    observability тихо уедет в базу памяти и раздует её (ClickHouse у Mastra
    рекомендован именно поэтому).

    Порядок записей внутри одного бэкенда сохраняется — по нему потом читают
    хронологию.
    """
    out = {}
    for record in records:
        kind = record["kind"]
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown record kind: {kind}")
        if kind not in routing:
            raise ValueError(f"no backend for kind: {kind}")
        out.setdefault(routing[kind], []).append(record)
    return out


def estimate_runtime_cost(
    n_agents,
    model_call_ms,
    instantiation_us=AGNO_INSTANTIATION_US,
    memory_kib=AGNO_MEMORY_KIB,
):
    """Во что обходятся n агентов: инстанциация, вызовы модели, память.

    Возвращает {"instantiation_ms", "model_ms", "total_ms",
                "instantiation_share", "memory_kib"}.

    estimate_runtime_cost(1000, 0.0)["instantiation_share"]  ->  1.0
    estimate_runtime_cost(1000, 800.0)["instantiation_ms"]    ->  2.0
    estimate_runtime_cost(0, 800.0)["instantiation_share"]    ->  0.0

    Ловушка: n_agents = 0 даёт total_ms = 0, и доля превращается в деление на
    ноль. Считай её нулём.

    Зачем это в AI: ровно этим и проверяется «2 μs — это про мой workload или
    нет». Если один вызов модели идёт 800 ms, доля инстанциации меньше
    тысячной, и выбирать рантайм по ней бессмысленно — это ловушка
    «perf-for-perf's-sake» из урока.
    """
    instantiation_ms = n_agents * instantiation_us / 1000.0
    model_ms = n_agents * model_call_ms
    total_ms = instantiation_ms + model_ms
    share = instantiation_ms / total_ms if total_ms else 0.0
    return {
        "instantiation_ms": instantiation_ms,
        "model_ms": model_ms,
        "total_ms": total_ms,
        "instantiation_share": share,
        "memory_kib": n_agents * memory_kib,
    }


def pick_runtime(profile):
    """Выбрать рантайм по профилю проекта: agno / mastra / langgraph.

    profile — dict {"language": "python"|"typescript",
                    "needs_durable_graph_state": bool}

    pick_runtime({"language": "python", "needs_durable_graph_state": False})     ->  'agno'
    pick_runtime({"language": "typescript", "needs_durable_graph_state": False}) ->  'mastra'
    pick_runtime({"language": "python", "needs_durable_graph_state": True})      ->  'langgraph'
    pick_runtime({"language": "rust", "needs_durable_graph_state": False})       ->  ValueError

    Ловушка в порядке правил: durable graph state перебивает язык. Ни Agno,
    ни Mastra не про долгоживущее состояние графа — если оно нужно, язык
    роли не играет.
    """
    if profile.get("needs_durable_graph_state"):
        return "langgraph"
    language = profile.get("language")
    if language == "python":
        return "agno"
    if language == "typescript":
        return "mastra"
    raise ValueError(f"no runtime for language: {language}")
