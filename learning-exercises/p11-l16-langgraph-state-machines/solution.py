"""
Агент как машина состояний — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

LangGraph даёт StateGraph, редьюсеры, чекпоинтер и interrupt_before одной
строкой на каждый. Здесь мы пишем этот рантайм руками — на словарях, без
LLM и без сети. «Модель» — это функция, которую передают снаружи; она
возвращает заранее заготовленное сообщение, и весь цикл ReAct становится
воспроизводимым до последнего чекпоинта.

Соответствие настоящей библиотеке:
    add_messages       <-  langgraph.graph.message.add_messages
    merge_state        <-  слияние частичного апдейта узла через редьюсеры
    compile_graph      <-  StateGraph(...).compile()
    route              <-  add_conditional_edges + функция-роутер
    run_graph          <-  app.invoke() с MemorySaver
    resume             <-  Command(resume=...) и путешествие во времени
    build_react_graph  <-  четыре узла из раздела «The ReAct graph»
"""

# Терминальный узел. В LangGraph это langgraph.graph.END.
END = "__end__"


class GraphError(Exception):
    """Граф собран неправильно: висячее ребро, чужой узел, вечный цикл.

    Свой класс, а не RuntimeError, специально: NotImplementedError — тоже
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """


class RecursionLimit(GraphError):
    """Граф крутится дольше отведённых шагов и не дошёл до END."""


def add_messages(old, new):
    """Редьюсер-накопитель для поля messages: приписать новое к старому.

    add_messages([{"role": "user"}], [{"role": "ai"}])
        ->  [{"role": "user"}, {"role": "ai"}]
    add_messages(None, [{"role": "user"}])  ->  [{"role": "user"}]
    add_messages([], {"role": "ai"})        ->  [{"role": "ai"}]

    Одиночное сообщение разрешено передавать без списка.

    Забыть этот редьюсер — самая частая ошибка в LangGraph: по умолчанию
    поле ПЕРЕЗАПИСЫВАЕТСЯ, и второй узел молча стирает половину диалога.

    Возвращает НОВЫЙ список. Дописывать в старый нельзя: на него смотрят
    уже сохранённые чекпоинты, и путешествие во времени сломается.
    """
    if old is None:
        old = []
    if not isinstance(new, list):
        new = [new]
    return list(old) + list(new)


def merge_state(state, update, reducers=None):
    """Применить частичный апдейт узла к состоянию. Вернуть НОВЫЙ словарь.

    merge_state({"n": 1}, {"n": 2})                       ->  {"n": 2}
    merge_state({"n": 1, "k": "x"}, {"n": 2})             ->  {"n": 2, "k": "x"}
    merge_state({"m": [1]}, {"m": [2]}, {"m": add_messages})  ->  {"m": [1, 2]}

    reducers — словарь {поле: функция (старое, новое) -> слитое}. Поля без
    редьюсера просто перезаписываются, как в LangGraph по умолчанию.

    Узел возвращает только те поля, которые менял. Всё остальное обязано
    доехать до следующего узла нетронутым.
    """
    reducers = reducers or {}
    merged = dict(state)
    for key, value in update.items():
        reducer = reducers.get(key)
        # merged.get(key), а не merged[key]: узел вправе завести новое поле
        merged[key] = reducer(merged.get(key), value) if reducer else value
    return merged


def compile_graph(nodes, edges, entry, reducers=None):
    """Проверить топологию и вернуть скомпилированный граф.

    nodes — {имя: функция state -> частичный апдейт}.
    edges — {имя: цель}, где цель это либо строка (статическое ребро), либо
            пара (роутер, {ветка: цель}) для условного.
    entry — имя стартового узла.

    compile_graph({"a": fa}, {"a": END}, "a")
        ->  {"nodes": {...}, "edges": {...}, "entry": "a", "reducers": {}}

    Что ловится ДО запуска, а не в рантайме:
      * entry, которого нет среди узлов;
      * ребро из несуществующего узла или в несуществующий;
      * узел без исходящего ребра — из него некуда идти;
      * цикл из ОДНИХ статических рёбер: выхода нет ни при каком состоянии,
        такой граф не «долго считает», он висит навсегда.

    Цикл через условное ребро законен и нужен — на нём стоит весь ReAct.
    Ограничивает его max_steps в run_graph, а не проверка топологии.

    Всё это GraphError. Граф с ошибкой лучше не собрать, чем повесить прод.
    """
    reducers = dict(reducers or {})
    if entry not in nodes:
        raise GraphError(f"стартовый узел {entry!r} не объявлен")

    targets = {}  # только статические рёбра — по ним ищем гарантированный цикл
    for name, edge in edges.items():
        if name not in nodes:
            raise GraphError(f"ребро из неизвестного узла {name!r}")
        if isinstance(edge, str):
            outs = [edge]
            targets[name] = [edge] if edge != END else []
        else:
            _, mapping = edge
            outs = list(mapping.values())
            targets[name] = []
        for target in outs:
            if target != END and target not in nodes:
                raise GraphError(f"ребро {name!r} ведёт в неизвестный узел {target!r}")

    for name in nodes:
        if name not in edges:
            raise GraphError(f"из узла {name!r} нет исходящего ребра")

    # Обход в глубину тремя цветами: серый узел, встреченный на пути, — цикл.
    # Ищем его ДО запуска, потому что в рантайме он выглядит как зависание.
    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(targets, WHITE)

    def walk(name, path):
        color[name] = GREY
        for nxt in targets.get(name, ()):
            if color.get(nxt) == GREY:
                raise GraphError("статический цикл без выхода: " + " -> ".join([*path, nxt]))
            if color.get(nxt) == WHITE:
                walk(nxt, [*path, nxt])
        color[name] = BLACK

    for name in targets:
        if color[name] == WHITE:
            walk(name, [name])

    return {"nodes": dict(nodes), "edges": dict(edges), "entry": entry, "reducers": reducers}


def route(graph, node, state):
    """Куда идти из узла node при текущем состоянии.

    Для статического ребра ответ не зависит от состояния. Для условного
    вызывается роутер, а его ответ переводится через карту веток.

    route(graph, "agent", {"messages": [...]})  ->  "tools"

    Ветка, которой нет в карте, — GraphError. Молча уйти в END на опечатке
    в имени ветки хуже, чем упасть: агент «отработает» и ничего не сделает.
    """
    edge = graph["edges"][node]
    if isinstance(edge, str):
        return edge
    router, mapping = edge
    branch = router(state)
    if branch not in mapping:
        raise GraphError(f"роутер узла {node!r} вернул неизвестную ветку {branch!r}")
    return mapping[branch]


def run_graph(graph, initial_state, max_steps=25, interrupt_before=(), start=None):
    """Исполнить граф, записывая чекпоинт после каждого перехода.

    Возвращает {"state": итог, "checkpoints": [...], "interrupted": узел или None}.

    Чекпоинт — словарь {"id", "node", "next", "state"}: нулевой снят до
    первого узла, дальше по одному на переход.

        inc = lambda state: {"n": state["n"] + 1}
        graph = compile_graph({"inc": inc}, {"inc": END}, "inc")
        run_graph(graph, {"n": 0})["state"]         ->  {"n": 1}
        len(run_graph(graph, {"n": 0})["checkpoints"])  ->  2

    interrupt_before — имена узлов, ПЕРЕД которыми исполнение замирает.
    Согласование ставят именно перед узлом с побочным эффектом: после
    удаления продакшен-базы согласовывать уже нечего.

    max_steps — предохранитель для циклов через условное ребро. Исчерпан —
    RecursionLimit. Бесконечно крутиться молча агент не имеет права.

    start задаёт узел, отличный от entry: им пользуется resume.
    """
    state = dict(initial_state)
    node = graph["entry"] if start is None else start
    checkpoints = [{"id": 0, "node": None, "next": node, "state": dict(state)}]
    steps = 0
    while node != END:
        if node in interrupt_before:
            return {"state": state, "checkpoints": checkpoints, "interrupted": node}
        if steps >= max_steps:
            raise RecursionLimit(f"граф не дошёл до END за {max_steps} шагов, застрял на {node!r}")
        update = graph["nodes"][node](state)
        state = merge_state(state, update, graph["reducers"])
        nxt = route(graph, node, state)
        # dict(state) — снимок: дальше state заменяется целиком, а не правится
        checkpoints.append({"id": len(checkpoints), "node": node, "next": nxt, "state": dict(state)})
        node = nxt
        steps += 1
    return {"state": state, "checkpoints": checkpoints, "interrupted": None}


def resume(graph, checkpoints, checkpoint_id, update=None, max_steps=25, interrupt_before=()):
    """Продолжить исполнение с сохранённого чекпоинта. Возврат как у run_graph.

    Два применения одного механизма:
      * снять interrupt — продолжить с последнего чекпоинта, добавив в
        состояние решение человека;
      * путешествие во времени — взять чекпоинт из середины и пустить
        ветку заново, ничего не переигрывая до него.

    resume(graph, run["checkpoints"], 2)                  ->  ветка из чекпоинта 2
    resume(graph, run["checkpoints"], 2, {"plan": "b"})   ->  та же ветка, но с правкой

    update проходит через те же редьюсеры, что и обычный апдейт узла.
    Несуществующий checkpoint_id — GraphError.
    """
    cp = next((c for c in checkpoints if c["id"] == checkpoint_id), None)
    if cp is None:
        raise GraphError(f"нет чекпоинта с id {checkpoint_id!r}")
    state = merge_state(cp["state"], update or {}, graph["reducers"])
    return run_graph(graph, state, max_steps, interrupt_before, start=cp["next"])


def build_react_graph(model, tools):
    """Собрать цикл ReAct: узел agent, узел tools и условное ребро между ними.

    model — функция messages -> сообщение ассистента. Сообщение это словарь
    {"role": "assistant", "content": ..., "tool_calls": [...]}, где каждый
    вызов — {"name": имя, "args": {...}}. Без tool_calls ответ считается
    окончательным.
    tools — {имя: функция}, вызывается как fn(**args).

    Топология ровно та, что в уроке:
        agent --(есть tool_calls)--> tools --> agent
        agent --(нет tool_calls)---> END

    Поле messages копится редьюсером add_messages, иначе результат
    инструмента затрёт вопрос пользователя.

    Никакой сети здесь нет и не нужно: LLM — это функция, которую передали
    аргументом, и тест подсовывает вместо неё заранее записанный сценарий.
    """

    def agent(state):
        return {"messages": [model(state["messages"])]}

    def call_tools(state):
        last = state["messages"][-1]
        results = []
        for call in last.get("tool_calls") or ():
            fn = tools[call["name"]]
            results.append(
                {"role": "tool", "name": call["name"], "content": str(fn(**call.get("args", {})))}
            )
        return {"messages": results}

    def should_continue(state):
        return "tools" if state["messages"][-1].get("tool_calls") else "done"

    return compile_graph(
        {"agent": agent, "tools": call_tools},
        {"agent": (should_continue, {"tools": "tools", "done": END}), "tools": "agent"},
        "agent",
        {"messages": add_messages},
    )
