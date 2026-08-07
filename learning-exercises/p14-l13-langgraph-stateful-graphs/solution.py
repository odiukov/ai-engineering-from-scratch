"""
Граф состояний с чекпоинтами — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок про LangGraph, но здесь нет ни langgraph, ни сети, ни LLM. Мы собираем
руками то, что фреймворк даёт одной строкой: типизированное состояние,
узлы-функции, условные рёбра, чекпоинт после каждого узла и точное
возобновление с места остановки.

Соответствие настоящему API:

    merge_update            <-  reducer'ы LangGraph (add_messages и прочие)
    next_node               <-  StateGraph.add_conditional_edges + выбор ветки
    validate_graph          <-  StateGraph.compile(), который ругается на
                                недостижимые узлы и висячие рёбра
    save_checkpoint         <-  Checkpointer.put()  (SqliteSaver, PostgresSaver)
    load_checkpoint         <-  Checkpointer.get_tuple()
    run_graph               <-  CompiledGraph.invoke(state, config)
    resume                  <-  invoke(None, config) после interrupt()
    missing_from_checkpoint <-  диагностика "чекпоинт слишком мал"

Граф — обычный словарь:

    {"entry": "classify",
     "nodes": {"classify": fn, "refund": fn, ...},
     "edges": {"classify": [("refund", cond), ("sales", None)], ...}}

Ребро — пара (куда, условие). Условие None означает безусловный переход.
Состояние — тоже словарь; узел возвращает не состояние целиком, а только
апдейт.
"""

import copy

START = "__start__"
END = "__end__"


def merge_update(state, update):
    """Слить апдейт узла в состояние: НОВЫЙ словарь, старый не трогаем.

    merge_update({"step": 1}, {"route": "bug"})  ->  {"step": 1, "route": "bug"}
    merge_update({"step": 1}, {"step": 2})       ->  {"step": 2}
    merge_update({"messages": ["a"]}, {"messages": ["b"]})
                                                 ->  {"messages": ["a", "b"]}
    merge_update({"step": 1}, None)              ->  {"step": 1}

    Два правила:
      * узел возвращает АПДЕЙТ, а не состояние целиком — ключи, которых в
        апдейте нет, остаются как были (слияние, а не замена);
      * списки склеиваются, а не затираются. Это и есть reducer: история
        сообщений должна расти, иначе после первого же узла от диалога
        ничего не останется.

    Ловушка: state.update(update) вернёт None и испортит состояние, на
    которое уже ссылается сохранённый чекпоинт. Возвращай новый словарь.
    """
    merged = dict(state)
    for key, value in (update or {}).items():
        old = merged.get(key)
        # список + список -> конкатенация; во всех прочих случаях замена.
        # old + value создаёт НОВЫЙ список, поэтому исходное состояние
        # (и чекпоинт, который на него смотрит) остаётся нетронутым
        if isinstance(old, list) and isinstance(value, list):
            merged[key] = old + value
        else:
            merged[key] = value
    return merged


def next_node(graph, current, state):
    """Куда перейти после узла current: первое ребро, чьё условие выполнено.

    Ребро — пара (куда, условие). Условие None срабатывает всегда.

    edges = {"a": [("b", lambda s: s["route"] == "bug"), ("c", None)]}
    next_node(graph, "a", {"route": "bug"})    ->  "b"
    next_node(graph, "a", {"route": "sales"})  ->  "c"
    next_node(graph, "c", {})                  ->  "__end__"  (рёбер нет)

    Порядок объявления важен: перебираем сверху вниз и берём первое
    подходящее. Безусловное ребро в середине списка делает всё, что ниже,
    мёртвым кодом.
    """
    for dst, cond in graph.get("edges", {}).get(current, []):
        if cond is None or cond(state):
            return dst
    return END


def validate_graph(graph):
    """Проверить граф ДО запуска. Вернуть отсортированный список проблем.

    Пустой список — граф можно запускать.

    Что ловим:
      * "no entry node: 'x'"          — точки входа нет среди узлов;
      * "edge to unknown node: 'x'"   — ребро ведёт в несуществующий узел;
      * "unreachable node: 'x'"       — узел недостижим из точки входа;
      * "unconditional cycle: a, b"   — цикл, все рёбра которого безусловны.

    validate_graph({"entry": "a", "nodes": {"a": f}, "edges": {}})  ->  []

    Цикл сам по себе не преступление: цикл агента с условным выходом
    ("думай, пока не готов ответ") — обычный паттерн. Преступление — цикл,
    из которого не ведёт ни одного условного ребра: такой граф крутится
    вечно, и выйти из него нечем.

    END ребром считается корректным адресом, узлом — нет.
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})
    problems = []

    entry = graph.get("entry")
    if entry not in nodes:
        problems.append(f"no entry node: {entry!r}")

    for outs in edges.values():
        for dst, _cond in outs:
            if dst != END and dst not in nodes:
                problems.append(f"edge to unknown node: {dst!r}")

    # достижимость: обычный обход от точки входа, стеком вместо рекурсии —
    # граф агента бывает глубоким, а рекурсия в Python дешёвой не бывает
    seen = set()
    frontier = [entry] if entry in nodes else []
    while frontier:
        node = frontier.pop()
        if node in seen or node not in nodes:
            continue
        seen.add(node)
        frontier.extend(dst for dst, _cond in edges.get(node, []))
    problems.extend(f"unreachable node: {name!r}" for name in nodes if name not in seen)

    # безусловный цикл ищем в подграфе из рёбер с условием None: если по
    # таким рёбрам из узла можно вернуться в него же, рантайм зациклится
    # при любом состоянии
    plain = {src: [d for d, cond in outs if cond is None] for src, outs in edges.items()}
    stuck = []
    for name in nodes:
        reach, frontier = set(), list(plain.get(name, []))
        while frontier:
            node = frontier.pop()
            if node in reach:
                continue
            reach.add(node)
            frontier.extend(plain.get(node, []))
        if name in reach:
            stuck.append(name)
    if stuck:
        problems.append("unconditional cycle: " + ", ".join(sorted(stuck)))

    return sorted(problems)


def save_checkpoint(store, session_id, node, state):
    """Записать состояние после узла. Вернуть номер чекпоинта (с нуля).

    store = {}
    save_checkpoint(store, "s1", "classify", {"step": 1})  ->  0
    store["s1"]  ->  [("classify", {"step": 1})]

    Хранилище — словарь session_id -> список пар (узел, состояние).

    Кладём ГЛУБОКУЮ копию. Иначе узел, который потом допишет что-нибудь в
    state["messages"], задним числом перепишет уже сохранённый чекпоинт, и
    возобновление поедет не с того состояния, которое было на самом деле.
    """
    history = store.setdefault(session_id, [])
    history.append((node, copy.deepcopy(state)))
    return len(history) - 1


def load_checkpoint(store, session_id, index=-1):
    """Достать чекпоинт: пара (узел, состояние). Нет сессии — KeyError.

    load_checkpoint(store, "s1")     ->  ("send", {...})       последний
    load_checkpoint(store, "s1", 0)  ->  ("classify", {...})   первый

    Возвращает копию: человек, который правит загруженное состояние перед
    возобновлением, не должен задним числом менять историю прогона.
    """
    history = store.get(session_id)
    if not history:
        raise KeyError(session_id)
    node, state = history[index]
    return node, copy.deepcopy(state)


def run_graph(graph, state, store, session_id, start_at=None, max_steps=50):
    """Прогнать граф, сохраняя чекпоинт ПОСЛЕ КАЖДОГО узла.

    Узел — функция state -> апдейт. Апдейт с ключом "__pause__" означает
    остановку ради человека: чекпоинт пишется, дальше рантайм не идёт, а
    сам ключ "__pause__" в состояние не попадает.

    Возвращает словарь:
        {"status": "done" | "paused" | "max_steps",
         "state":  итоговое состояние,
         "node":   узел, на котором остановились,
         "reason": причина паузы (или "")}

    run_graph(graph, {"input": "refund"}, {}, "s1")
        ->  {"status": "done", "node": "__end__", ...}

    start_at=START (или None) означает "с точки входа"; любое другое имя —
    начать с него, этим пользуется resume. Неизвестный узел — KeyError, а
    не тихий выход: молчаливо оборванный прогон отлаживать невозможно.
    """
    current = graph.get("entry") if start_at in (None, START) else start_at
    if current is None:
        raise KeyError("entry")
    for _ in range(max_steps):
        if current == END:
            return {"status": "done", "state": state, "node": END, "reason": ""}
        if current not in graph.get("nodes", {}):
            raise KeyError(current)
        # dict(...) — копия апдейта: pop не должен портить словарь,
        # который узел, возможно, держит у себя
        update = dict(graph["nodes"][current](state) or {})
        reason = update.pop("__pause__", "")
        state = merge_update(state, update)
        save_checkpoint(store, session_id, current, state)
        if reason:
            return {"status": "paused", "state": state,
                    "node": current, "reason": reason}
        current = next_node(graph, current, state)
    return {"status": "max_steps", "state": state, "node": current, "reason": ""}


def resume(graph, store, session_id, patch=None, max_steps=50):
    """Продолжить прогон с последнего чекпоинта, НЕ переигрывая сделанное.

    patch — правка состояния человеком, то самое human-in-the-loop:
        resume(graph, store, "s1", {"human_approval": True})

    Узел, на котором остановились, уже отработал и записан в чекпоинт.
    Поэтому продолжаем со СЛЕДУЮЩЕГО узла: заново звать оплату, отправку
    письма или создание тикета нельзя.

    Нет такой сессии — KeyError.
    """
    node, state = load_checkpoint(store, session_id)
    state = merge_update(state, patch)
    # ветку выбираем уже ПОСЛЕ правки: человек мог поменять route руками
    return run_graph(graph, state, store, session_id,
                     start_at=next_node(graph, node, state),
                     max_steps=max_steps)


def missing_from_checkpoint(state, saved):
    """Чего не хватает в чекпоинте: отсортированный список ключей.

    missing_from_checkpoint({"a": 1, "b": 2}, {"a": 1})      ->  ["b"]
    missing_from_checkpoint({"a": 1}, {"a": 9})              ->  ["a"]
    missing_from_checkpoint({"a": 1}, {"a": 1, "extra": 0})  ->  []

    Ключ попадает в список, если его в чекпоинте нет ИЛИ значение
    разошлось. Лишние ключи в чекпоинте не мешают — мешает недостача.

    Это диагностика самой дорогой ошибки урока: сохранили только историю
    сообщений, а состояние инструментов и памяти не сохранили. Прогон
    "возобновился" и поехал с половиной состояния.
    """
    return sorted(k for k, v in state.items() if k not in saved or saved[k] != v)
