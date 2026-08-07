"""
Команды агентов по ролям — роли, задачи, процессы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь руками собирается то, что CrewAI даёт четырьмя примитивами:
  * make_agent  ~  crewai.Agent(role=..., goal=..., backstory=..., tools=[...])
  * make_task   ~  crewai.Task(description=..., expected_output=..., agent=...)
  * run_sequential   ~  Crew(process=Process.sequential).kickoff()
  * run_hierarchical ~  Crew(process=Process.hierarchical, manager_llm=...)
  * run_flow    ~  crewai.flow.Flow с @start и @listen(topic)
  * remember / recall_context  ~  Crew(memory=True): short-term, long-term,
                                  entity и contextual памяти

Ни сети, ни LLM, ни crewai в окружении нет: модель везде приходит
параметром run_agent — детерминированной функцией от промпта. Проверяем мы
раскладку ролей и передачу результата по цепочке, а не качество текста.
"""

# Backstory — это часть промпта, которая уезжает в модель на КАЖДОМ шаге
# каждого агента. Пять агентов по две тысячи слов съедают контекст раньше
# первого вызова инструмента, поэтому реестр такое не принимает.
MAX_BACKSTORY_WORDS = 200


def make_agent(role, goal, backstory, tools=()):
    """Примитив Agent: role + goal + backstory + tools.

    Вернуть словарь с ключами role, goal, backstory, tools (tools — кортеж).

    make_agent("writer", "turn sources into a draft", "editorial voice")["role"]
        ->  'writer'

    ValueError, если:
      * role, goal или backstory пустые (backstory не украшение: он задаёт
        тон и то, когда агент останавливается);
      * в backstory больше MAX_BACKSTORY_WORDS слов;
      * инструмент не вызываемый или у него нет docstring.

    Про docstring: в CrewAI подпись функции становится схемой, а docstring —
    описанием, которое читает модель. Инструмент без docstring модель
    выбирает наугад.
    """
    for name, value in (("role", role), ("goal", goal), ("backstory", backstory)):
        if not value or not value.strip():
            raise ValueError(f"agent {name} must not be empty")
    words = len(backstory.split())
    if words > MAX_BACKSTORY_WORDS:
        raise ValueError(
            f"backstory is {words} words, limit is {MAX_BACKSTORY_WORDS}: prompt bloat"
        )
    for fn in tools:
        if not callable(fn):
            raise ValueError(f"tool {fn!r} is not callable")
        if not (fn.__doc__ or "").strip():
            raise ValueError(f"tool {fn.__name__!r} has no docstring to show the model")
    return {"role": role, "goal": goal, "backstory": backstory, "tools": tuple(tools)}


def make_task(description, expected_output, agent, context=()):
    """Примитив Task: description + expected_output + agent + context.

    Вернуть словарь с этими же ключами (context — кортеж задач).

    ValueError, если description или expected_output пустые, если agent не
    похож на агента (нет ключа role) или если в context попало не задание.

    expected_output — это контракт задачи. Без него следующий агент
    разбирает то, что модель случайно выдала: crew отработал, аудит
    провалился. Это первая из трёх типовых поломок CrewAI.

    context перечисляет задачи ВЫШЕ по цепочке, чьи выходы надо подать на
    вход этой задаче.
    """
    if not description or not description.strip():
        raise ValueError("task description must not be empty")
    if not expected_output or not expected_output.strip():
        raise ValueError("task needs expected_output: it is the audit contract")
    if not isinstance(agent, dict) or "role" not in agent:
        raise ValueError("task must be assigned to an agent")
    for upstream in context:
        if not isinstance(upstream, dict) or "expected_output" not in upstream:
            raise ValueError(f"context item {upstream!r} is not a task")
    return {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
        "context": tuple(context),
    }


def crew_prompt(agent, task, context_outputs=()):
    """Промпт одного шага: всё, что агент увидит перед ответом.

    Строки в фиксированном порядке:
      ROLE / GOAL / BACKSTORY / TASK / EXPECTED OUTPUT / CONTEXT n / TOOLS

    Строка TOOLS появляется только у агента с инструментами, строки
    CONTEXT — только если что-то передали.

    a = make_agent("writer", "draft it", "terse voice")
    t = make_task("write a draft", "3 paragraphs", a)
    crew_prompt(a, t).splitlines()[0]  ->  'ROLE: writer'

    Инструменты показываются как "name — первая строка docstring", по
    алфавиту: стабильный порядок нужен, чтобы промпт попадал в кэш.

    Именно здесь видно цену раздутых backstory: prompt-bloat — это не
    метафора, а длина вот этой строки, умноженная на число шагов.
    """
    lines = [
        f"ROLE: {agent['role']}",
        f"GOAL: {agent['goal']}",
        f"BACKSTORY: {agent['backstory']}",
        f"TASK: {task['description']}",
        f"EXPECTED OUTPUT: {task['expected_output']}",
    ]
    for number, output in enumerate(context_outputs, start=1):
        lines.append(f"CONTEXT {number}: {output}")
    tools = agent.get("tools", ())
    if tools:
        described = sorted(
            f"{fn.__name__} — {fn.__doc__.strip().splitlines()[0]}" for fn in tools
        )
        lines.append("TOOLS: " + "; ".join(described))
    return "\n".join(lines)


def run_sequential(tasks, topic, run_agent):
    """Процесс Sequential: задачи в порядке объявления, выход N — вход N+1.

    run_agent — функция prompt -> output, детерминированная заглушка модели.

    Вернуть список словарей по одному на задачу, в порядке объявления:
      {"role": ..., "prompt": ..., "output": ...}

    Число вызовов модели равно числу задач — это самый дешёвый процесс.

    Если у задачи объявлен context, на вход идут выходы ИМЕННО этих задач;
    если нет — выход предыдущей задачи, а для первой задачи сам topic.

    ValueError на пустом списке задач и на задаче, чей context ссылается
    вперёд: задача из context ещё не выполнялась, подставлять нечего.
    """
    if not tasks:
        raise ValueError("crew needs at least one task")
    trace = []
    previous = topic
    # ключ по id: две задачи с одинаковым описанием — это всё равно две
    # разные задачи, сравнивать их по содержимому нельзя
    done = {}
    for task in tasks:
        if task["context"]:
            for upstream in task["context"]:
                if id(upstream) not in done:
                    raise ValueError(
                        f"context task has not run yet: {upstream['description']!r}"
                    )
            context_outputs = [done[id(t)] for t in task["context"]]
        else:
            context_outputs = [previous]
        prompt = crew_prompt(task["agent"], task, context_outputs)
        output = run_agent(prompt)
        trace.append({"role": task["agent"]["role"], "prompt": prompt, "output": output})
        done[id(task)] = output
        previous = output
    return trace


def run_hierarchical(manager_task, tasks, topic, run_agent, max_rounds=6):
    """Процесс Hierarchical: менеджер каждый раунд выбирает специалиста.

    manager_task — задача менеджера; её агент и есть manager_llm. В context
    менеджеру подаются три строки: ROSTER, DONE и LATEST.

    Ответ менеджера — имя роли или "done".

    Вернуть словарь:
      {"trace": [...], "llm_calls": int, "final": str,
       "done": [роли по порядку], "stop_reason": str}

    Шаг трассы — словарь {"role", "prompt", "pick", "output"}: у шага
    менеджера заполнен pick, у шага специалиста — output.

    llm_calls == 2 * len(done) + 1 для прогона, который менеджер закрыл сам:
    это и есть manager-LLM tax. Пять задач — шесть вызовов вместо пяти, и
    каждый вызов менеджера тащит весь список задач.

    Останов: "done" от менеджера, неизвестная роль, повтор уже сделанной
    роли (иначе петля не кончится) или исчерпанный max_rounds.

    ValueError, если задач нет или две задачи назначены на одну роль:
    вторая была бы недостижима, а менеджер об этом не узнает.
    """
    if not tasks:
        raise ValueError("crew needs at least one task")
    by_role = {}
    for task in tasks:
        role = task["agent"]["role"]
        if role in by_role:
            raise ValueError(f"duplicate specialist role: {role!r}")
        by_role[role] = task

    trace = []
    llm_calls = 0
    current = topic
    done = []
    stop_reason = "budget"
    for _ in range(max_rounds):
        context = (
            "ROSTER: " + ", ".join(sorted(by_role)),
            "DONE: " + (", ".join(done) if done else "none"),
            f"LATEST: {current}",
        )
        manager_prompt = crew_prompt(manager_task["agent"], manager_task, context)
        pick = run_agent(manager_prompt).strip()
        llm_calls += 1
        trace.append(
            {
                "role": manager_task["agent"]["role"],
                "prompt": manager_prompt,
                "pick": pick,
                "output": None,
            }
        )
        if pick == "done":
            stop_reason = "done"
            break
        if pick not in by_role:
            stop_reason = f"unknown pick {pick!r}"
            break
        if pick in done:
            # без этой ветки менеджер, залипший на одной роли, крутится до
            # упора бюджета и платит за каждый раунд дважды
            stop_reason = f"repeated pick {pick!r}"
            break
        task = by_role[pick]
        prompt = crew_prompt(task["agent"], task, (current,))
        output = run_agent(prompt)
        llm_calls += 1
        trace.append({"role": pick, "prompt": prompt, "pick": None, "output": output})
        current = output
        done.append(pick)
    return {
        "trace": trace,
        "llm_calls": llm_calls,
        "final": current,
        "done": done,
        "stop_reason": stop_reason,
    }


def run_flow(start, listeners, payload, max_steps=20):
    """Flow: событийный граф, который принадлежит коду, а не модели.

    start — функция payload -> (topic, output), аналог @start.
    listeners — словарь topic -> функция output -> (topic, output) | None,
    аналог @listen(topic).

    Вернуть трассу: список кортежей (step_name, topic, output). Первый
    элемент всегда ("start", ...).

    Петля кончается, когда на очередной topic нет слушателя или слушатель
    вернул None.

    ValueError, если шагов больше max_steps: значит в графе цикл. Flow, в
    отличие от Crew, обязан завершаться предсказуемо.

    Темы задаёт код, поэтому трасса воспроизводима и её можно
    диффать — ровно то, чего нет у свободного Crew и из-за чего документация
    CrewAI советует начинать с Flow.
    """
    if not callable(start):
        raise ValueError("flow needs a @start step")
    topic, output = start(payload)
    trace = [("start", topic, output)]
    steps = 0
    while topic in listeners:
        if steps >= max_steps:
            raise ValueError(f"flow exceeded max_steps={max_steps}: cycle in the graph?")
        step = listeners[topic]
        result = step(output)
        steps += 1
        if result is None:
            break
        topic, output = result
        trace.append((step.__name__, topic, output))
    return trace


def remember(memory, kind, value, key=None):
    """Записать факт в память. Вернуть тот же (изменённый) словарь memory.

    kind — "short_term", "long_term" или "entity".
    Для "entity" обязателен key: id сущности (клиент, тикет, аккаунт).

    m = remember({}, "long_term", "crew shipped the brief")
    m["long_term"]  ->  ['crew shipped the brief']

    ValueError на неизвестном kind и на entity без key.

    Разница между кучами не в хранилище, а в жизненном цикле: short_term
    стирается в конце прогона, long_term живёт между kickoff-ами, entity
    привязана к сущности и достаётся по ключу, а не по похожести.
    """
    if kind not in ("short_term", "long_term", "entity"):
        raise ValueError(f"unknown memory kind: {kind!r}")
    if kind == "entity":
        if not key:
            raise ValueError("entity memory needs a key")
        memory.setdefault("entity", {}).setdefault(key, []).append(value)
    else:
        memory.setdefault(kind, []).append(value)
    return memory


def recall_context(memory, query, entity_id=None, k=2):
    """Contextual memory: собрать то, что нужно агенту именно сейчас.

    Вернуть словарь с тремя ключами:
      "short_term" — буфер текущего прогона целиком;
      "entity"     — факты по entity_id (пустой список, если id не задан
                     или неизвестен);
      "long_term"  — до k записей с наибольшим пересечением слов с query.

    Записи long_term без общих слов с query НЕ возвращаются: иначе с ростом
    базы выдача становится шумом, а это одна из главных жалоб на
    always-on память.

    При равном пересечении порядок — как записывали.

    recall_context({"long_term": ["brief about agents"]}, "brief")["long_term"]
        ->  ['brief about agents']
    """
    words = set(query.lower().split())
    scored = []
    for index, item in enumerate(memory.get("long_term", [])):
        overlap = len(words & set(item.lower().split()))
        if overlap:
            # минус, чтобы обычная сортировка дала убывание по совпадениям,
            # а index сохранил порядок записи при равном счёте
            scored.append((-overlap, index, item))
    scored.sort()
    entity_facts = []
    if entity_id is not None:
        entity_facts = list(memory.get("entity", {}).get(entity_id, []))
    return {
        "short_term": list(memory.get("short_term", [])),
        "entity": entity_facts,
        "long_term": [item for _, _, item in scored[:k]],
    }
