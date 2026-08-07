"""
Харнесс как библиотека: подагенты и хранилище сессий — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь собирается руками то, что Claude Agent SDK даёт из коробки:
реестр инструментов с разрешениями (`allowed_tools`), lifecycle-хуки
(`PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`), подагенты с
изолированным контекстом и session store (`append`, `load`,
`list_subkeys`, `delete` с каскадом).

Ни сети, ни LLM, ни сторонних пакетов: модель — детерминированная
заглушка от промпта, инструменты — обычные функции.
"""

# События жизненного цикла, которые перечисляет SDK. Хук, повешенный на
# что-то другое, — почти всегда опечатка в имени, поэтому список закрытый.
HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreCompact",
    "Stop",
    "Notification",
)

# Инструменты, которые SDK регистрирует сам. Нужны только как справочник:
# реестр в упражнении — обычный словарь «имя -> обработчик».
BUILTIN_TOOL_NAMES = ("read_file", "write_file", "list_dir")


def stub_model(prompt):
    """Детерминированная заглушка модели: один и тот же промпт — один ответ.

    stub_model("hi")   ->  'answer[2:209]'
    stub_model("hi ")  ->  'answer[3:241]'
    stub_model("ih")   ->  'answer[2:209]'

    Заменяет вызов Messages API. Свойство, ради которого она такая: ответ
    зависит ТОЛЬКО от промпта, поэтому тест на петлю агента не мигает.
    Обрати внимание на третий пример: перестановка букв ответ не меняет —
    сумма кодов одна и та же. Это не баг заглушки, а напоминание, что
    подпись «по сумме» — не хеш.
    """
    total = sum(ord(ch) for ch in prompt) % 9973
    return f"answer[{len(prompt)}:{total}]"


def select_tools(registry, allowed):
    """Отобрать из реестра инструменты, разрешённые агенту (`allowed_tools`).

    select_tools({"a": f, "b": g}, ["a"])   ->  {"a": f}
    select_tools({"a": f, "b": g}, None)    ->  {"a": f, "b": g}
    select_tools({"a": f}, ["b"])           ->  KeyError

    `allowed=None` — «все инструменты», как в SDK по умолчанию.

    Две ловушки. Первая: имя, которого в реестре нет, — это ошибка
    конфигурации, а не «просто пропустим»; молчаливый пропуск потом выглядит
    как «инструмент есть, но модель его не зовёт». Вторая: возвращать нужно
    НОВЫЙ словарь. Если вернуть сам registry, подагент сможет дописать себе
    инструмент, которого ему не давали, и изоляция превратится в фикцию.
    """
    if allowed is None:
        # dict(...) — поверхностная копия: обработчики те же объекты, но
        # добавление ключа в результат родительский реестр не тронет.
        return dict(registry)
    picked = {}
    for name in allowed:
        if name not in registry:
            raise KeyError(name)
        picked[name] = registry[name]
    return picked


def run_hooks(hooks, event, payload):
    """Прогнать хуки события по порядку; первый deny останавливает цепочку.

    Хук — функция payload -> None (разрешить) либо
    {"decision": "deny", "reason": "..."}.

    run_hooks({}, "PreToolUse", {})                  ->  None
    run_hooks({"PreToolUse": [deny]}, "PreToolUse", {})
                                                     ->  {"reason": "..."}
    run_hooks({}, "PreToolCall", {})                 ->  ValueError

    Событие вне HOOK_EVENTS — ValueError: опечатка в имени иначе даёт хук,
    который просто никогда не срабатывает, и это ловится уже в проде.

    Важное свойство: после первого deny остальные хуки НЕ вызываются —
    решение принято, дальше только лишние побочные эффекты.
    """
    if event not in HOOK_EVENTS:
        raise ValueError(f"unknown hook event: {event}")
    for hook in hooks.get(event, ()):
        verdict = hook(payload)
        if verdict is not None and verdict.get("decision") == "deny":
            # reason нужен вызывающему для журнала; если хук его не дал —
            # подставляем имя события, чтобы в трейсе не было пустоты.
            return {"reason": verdict.get("reason", event)}
    return None


def call_tool(registry, hooks, name, args, journal):
    """Вызов инструмента через ворота хуков; всё записывается в journal.

    Возвращает dict:
      {"ok": True,  "tool": ..., "result": ...}          — отработал
      {"ok": True,  ..., "blocked": reason}              — отработал, но
                                                           PostToolUse
                                                           пометил результат
      {"ok": False, "tool": ..., "error": reason}        — не отработал

    call_tool({"echo": h}, {}, "echo", {"x": 1}, [])
        ->  {"ok": True, "tool": "echo", "result": ...}
    call_tool({}, {}, "write_file", {}, [])
        ->  {"ok": False, "tool": "write_file", "error": "tool_not_allowed"}

    Главная ловушка: PreToolUse — это ворота ДО вызова. Если он сказал deny,
    обработчик не должен быть вызван ни разу, иначе побочный эффект уже
    случился и «запрет» ничего не запретил. PostToolUse, наоборот, отменить
    ничего не может — он только помечает результат.

    Инструмент, которого нет в реестре, — не падение, а отказ: подагенту
    вполне штатно урезали список, и модель об этом узнаёт из ответа.
    """
    if name not in registry:
        journal.append({"event": "blocked", "tool": name, "reason": "tool_not_allowed"})
        return {"ok": False, "tool": name, "error": "tool_not_allowed"}

    deny = run_hooks(hooks, "PreToolUse", {"tool": name, "args": args})
    if deny is not None:
        journal.append({"event": "denied", "tool": name, "reason": deny["reason"]})
        return {"ok": False, "tool": name, "error": deny["reason"]}

    result = registry[name](args)
    journal.append({"event": "called", "tool": name})

    post = run_hooks(hooks, "PostToolUse", {"tool": name, "args": args, "result": result})
    if post is not None:
        # Побочный эффект уже произошёл — честно отдаём результат и метку.
        journal.append({"event": "post_blocked", "tool": name, "reason": post["reason"]})
        return {"ok": True, "tool": name, "result": result, "blocked": post["reason"]}
    return {"ok": True, "tool": name, "result": result}


def session_subkeys(store, session_id):
    """Ключи подагентских сессий данной сессии — SDK-шный `list_subkeys`.

    Соглашение: сессия подагента лежит под ключом "<parent>/<имя>".

    session_subkeys({"s1": [], "s1/a": [], "s2": []}, "s1")  ->  ["s1/a"]
    session_subkeys({"s1": []}, "s1")                        ->  []

    Ловушка: сама сессия в список не входит, а "s10" не является подключом
    "s1" — сравнивать надо по префиксу "s1/", а не по "s1".
    """
    prefix = session_id + "/"
    return sorted(key for key in store if key.startswith(prefix))


def session_delete(store, session_id):
    """Удалить сессию каскадом вместе с сессиями её подагентов.

    Возвращает отсортированный список удалённых ключей.

    store = {"s1": [], "s1/a": [], "s2": []}
    session_delete(store, "s1")  ->  ["s1", "s1/a"];  store == {"s2": []}
    session_delete(store, "нет") ->  []

    Ловушка: без каскада подагентские сессии остаются висеть навсегда —
    ровно тот «session bloat», о котором предупреждает урок. Удаление
    несуществующей сессии — не ошибка, а пустой список.
    """
    removed = session_subkeys(store, session_id)
    if session_id in store:
        removed = [session_id] + removed
    for key in removed:
        del store[key]
    return sorted(removed)


def run_agent(store, session_id, prompt, plan, registry, hooks, model=stub_model):
    """Петля агента: SessionStart, вызовы инструментов по плану, ответ, SessionEnd.

    `plan` — заранее заданная последовательность пар (имя инструмента, args):
    решать, что звать, здесь не нужно, интерес в воротах и в сессии.

    Возвращает {"answer": str, "calls": [...], "journal": [...]}.

    run_agent({}, "s1", "hi", [], {}, {})
        ->  {"answer": "answer[3:333]", "calls": [], "journal": []}
           (модель получает "hi|" — отсюда длина 3)

    В store по ключу session_id копятся ходы: user, по одному tool на каждый
    пункт плана, затем assistant. То есть len(store["s1"]) == len(plan) + 2.

    Ловушка: ответ модели должен зависеть от результатов инструментов, иначе
    вся петля бессмысленна — агент бы ответил и без них.
    """
    run_hooks(hooks, "SessionStart", {"session_id": session_id})

    turns = store.setdefault(session_id, [])
    turns.append({"role": "user", "content": prompt})

    journal = []
    calls = []
    for name, args in plan:
        outcome = call_tool(registry, hooks, name, args, journal)
        calls.append(outcome)
        turns.append({"role": "tool", "content": outcome})

    # Промпт для «модели» склеивается из исходного запроса и того, что
    # вернули инструменты: так ответ детерминированно зависит от прогона.
    observed = "|".join(str(c.get("result", c.get("error"))) for c in calls)
    answer = model(prompt + "|" + observed)
    turns.append({"role": "assistant", "content": answer})

    run_hooks(hooks, "SessionEnd", {"session_id": session_id})
    return {"answer": answer, "calls": calls, "journal": journal}


def spawn_subagents(store, parent_session, tasks, registry, hooks, model=stub_model):
    """Запустить подагентов: у каждого свой контекст и свой список инструментов.

    Задача — dict {"name", "prompt", "plan", "allowed"}; "allowed" по правилам
    select_tools (None — все инструменты родителя).

    Возвращает список {"name", "answer", "calls"} в порядке задач.

    Смысл изоляции контекста: ходы подагента лежат в ЕГО сессии
    "<parent>/<name>", а в родительскую попадает ровно один ход на подагента —
    итоговый ответ. Поэтому контекст оркестратора растёт как число задач, а не
    как их суммарная длина; ради этого подагенты и заводят.

    Ловушка: подагенту нельзя отдавать родительский реестр целиком —
    «allowed» на то и «allowed». Инструмент вне списка обязан вернуть
    tool_not_allowed, а не выполниться.
    """
    parent_turns = store.setdefault(parent_session, [])
    results = []
    for task in tasks:
        sub_id = f"{parent_session}/{task['name']}"
        sub_registry = select_tools(registry, task.get("allowed"))
        run = run_agent(
            store,
            sub_id,
            task["prompt"],
            task.get("plan", ()),
            sub_registry,
            hooks,
            model,
        )
        results.append({"name": task["name"], "answer": run["answer"], "calls": run["calls"]})
        # Наверх поднимается только результат — не транскрипт подагента.
        parent_turns.append(
            {"role": "subagent", "content": {"name": task["name"], "answer": run["answer"]}}
        )
    return results
