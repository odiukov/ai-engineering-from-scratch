"""
Продакшн-рантаймы агентов: быстрая инстанциация и типизированные workflow

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l18-agno-and-mastra-runtimes
Разбор:  /check-code p14-l18-agno-and-mastra-runtimes
"""

AGNO_INSTANTIATION_US = 2.0
AGNO_MEMORY_KIB = 3.75
RECORD_KINDS = ("memory", "workflows", "observability")


def stub_model(prompt, model_id):
    """Детерминированная заглушка модели: ответ зависит от промпта и модели.

    stub_model("hi", "anthropic/claude-x")  ->  'claude-x:2:209'
    stub_model("hi", "openai/gpt-x")        ->  'gpt-x:2:209'

    Заменяет вызов провайдера. Из model_id берётся часть после последнего "/":
    в ответе видно, куда бы ушёл запрос после роутинга.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
