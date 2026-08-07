"""
OpenAI Agents SDK: хендоффы, guardrails, трейсинг — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь руками собирается то, что SDK даёт пятью примитивами:
  * make_agent        ~  agents.Agent(name=..., instructions=..., tools=[...],
                                      handoffs=[...])
  * handoff_tool_name ~  agents.handoff(agent) -> tool "transfer_to_<name>"
  * run_turn          ~  agents.Runner.run() без guardrails
  * run_guardrails    ~  @input_guardrail / @output_guardrail и их tripwire
  * run_guarded       ~  Runner.run() целиком, с InputGuardrailTripwire-
                         Triggered и OutputGuardrailTripwireTriggered
  * session_prompt    ~  agents.SQLiteSession и параметр session=
  * redact_spans      ~  add_trace_processor(...) с правилами на контент

Ни сети, ни LLM, ни openai в окружении нет: модель приходит параметром
policy — детерминированной функцией от входа. Проверяем мы раскладку
хендоффов, порядок guardrails и форму спанов, а не качество ответа.
"""

# Что именно считается чувствительным в спанах. Правило из урока: контент
# храним снаружи, в спане оставляем только ссылку.
SENSITIVE_ATTRIBUTES = ("output", "result", "args", "text")


def handoff_tool_name(agent_name):
    """Имя инструмента, под которым модель видит хендофф.

    handoff_tool_name("billing")        ->  'transfer_to_billing'
    handoff_tool_name("Billing Agent")  ->  'transfer_to_billing_agent'

    Пробелы и дефисы становятся подчёркиваниями, регистр опускается.

    ValueError, если после нормализации остались символы кроме
    латинских букв, цифр и подчёркивания, или если имя пустое: модель
    вызывает инструмент по имени, и имя с пробелом или кириллицей она
    воспроизвести не сможет.

    Хендофф — это не отдельный примитив в списке инструментов, а обычный
    tool call. Отсюда и всё остальное поведение: модель сама решает, когда
    делегировать.
    """
    slug = agent_name.strip().lower().replace(" ", "_").replace("-", "_")
    if not slug:
        raise ValueError("agent name must not be empty")
    if not slug.isascii() or not all(c.isalnum() or c == "_" for c in slug):
        raise ValueError(f"agent name {agent_name!r} cannot become a tool name")
    return "transfer_to_" + slug


def make_agent(name, instructions, policy, tools=(), handoffs=()):
    """Примитив Agent: name + instructions + policy + tools + handoffs.

    policy — заглушка модели: функция user_input -> решение. Решение это
    словарь одного из трёх видов:
      {"kind": "final",   "text": ...}
      {"kind": "tool",    "tool": имя, "args": {...}}
      {"kind": "handoff", "to": имя агента, "input": необязательно}

    tools — кортеж словарей {"name", "description", "fn"} и необязательного
    "guardrail" (функция args -> (ok, reason)).
    handoffs — кортеж других агентов.

    Вернуть словарь с ключами name, instructions, policy, tools, handoffs.

    ValueError, если:
      * имя не превращается в transfer_to_<...> (см. handoff_tool_name);
      * instructions пустые или policy не вызываемая;
      * у инструмента нет name/description/fn, описание пустое или fn не
        вызываемая;
      * два инструмента с одним именем;
      * два хендоффа на одного агента;
      * имя инструмента совпало с именем хендоффа — модель бы видела два
        разных действия под одним именем.
    """
    handoff_tool_name(name)  # проверка имени: бросит ValueError, если оно негодное
    if not instructions or not instructions.strip():
        raise ValueError("agent instructions must not be empty")
    if not callable(policy):
        raise ValueError("agent policy must be callable")

    tool_names = []
    for tool in tools:
        if not isinstance(tool, dict) or not {"name", "description", "fn"} <= set(tool):
            raise ValueError(f"tool {tool!r} needs name, description and fn")
        if not tool["description"].strip():
            raise ValueError(f"tool {tool['name']!r} needs a description for the model")
        if not callable(tool["fn"]):
            raise ValueError(f"tool {tool['name']!r} fn is not callable")
        if tool["name"] in tool_names:
            raise ValueError(f"duplicate tool name: {tool['name']!r}")
        tool_names.append(tool["name"])

    transfer_names = []
    for target in handoffs:
        if not isinstance(target, dict) or "name" not in target:
            raise ValueError(f"handoff target {target!r} is not an agent")
        transfer = handoff_tool_name(target["name"])
        if transfer in transfer_names:
            raise ValueError(f"duplicate handoff to {target['name']!r}")
        if transfer in tool_names:
            raise ValueError(f"tool name {transfer!r} collides with a handoff")
        transfer_names.append(transfer)

    return {
        "name": name,
        "instructions": instructions,
        "policy": policy,
        "tools": tuple(tools),
        "handoffs": tuple(handoffs),
    }


def visible_tools(agent):
    """Список имён, которые модель видит в этом ходу. Отсортирован.

    Свои функции и хендоффы лежат в ОДНОМ списке: с точки зрения модели
    transfer_to_billing ничем не отличается от issue_refund.

    У агента с инструментом issue_refund и хендоффом на агента billing
    получится ['issue_refund', 'transfer_to_billing']. У агента без того и
    другого — пустой список.

    Сортировка — ради стабильности промпта: переставь инструменты, и кэш
    промптов перестанет попадать.
    """
    names = [tool["name"] for tool in agent["tools"]]
    names += [handoff_tool_name(target["name"]) for target in agent["handoffs"]]
    return sorted(names)


def run_guardrails(guardrails, payload, stage):
    """Прогнать guardrails одной ступени. Вернуть (spans, tripped).

    guardrails — последовательность пар (name, check), где
    check(payload) -> (ok, reason).
    stage — "input", "output" или "tool".

    spans — по одному словарю на КАЖДЫЙ выполненный guardrail:
      {"name": f"{stage}_guardrail.{name}", "attributes": {"passed", "reason"}}
    tripped — None или {"stage", "name", "reason"} по первому срабатыванию.

    run_guardrails((), "hi", "input")  ->  ([], None)

    На первом срабатывании остальные guardrails НЕ запускаются: спанов для
    них тоже нет. Это видно в трейсе и экономит вызовы.

    ValueError на неизвестной ступени: опечатка в stage превратила бы
    output-проверку в невидимую.
    """
    if stage not in ("input", "output", "tool"):
        raise ValueError(f"unknown guardrail stage: {stage!r}")
    spans = []
    for name, check in guardrails:
        ok, reason = check(payload)
        spans.append(
            {
                "name": f"{stage}_guardrail.{name}",
                "attributes": {"passed": ok, "reason": reason},
            }
        )
        if not ok:
            return (spans, {"stage": stage, "name": name, "reason": reason})
    return (spans, None)


def session_prompt(session, user_input, max_turns=None):
    """Собрать вход агента из истории сессии и нового сообщения.

    session — список словарей {"user", "assistant"} в хронологическом
    порядке.

    session_prompt([], "hi")  ->  'USER: hi'
    session_prompt([{"user": "hi", "assistant": "hello"}], "again")
        ->  'USER: hi\\nASSISTANT: hello\\nUSER: again'

    max_turns оставляет только последние max_turns ходов: контекст не
    бесконечен, и старые ходы приходится ронять.

    ValueError, если max_turns не положительный: max_turns=0 отдал бы
    промпт без истории, но втихую — лучше пусть будет видно.
    """
    if max_turns is not None and max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    turns = list(session)
    if max_turns is not None:
        turns = turns[-max_turns:]
    lines = []
    for turn in turns:
        lines.append(f"USER: {turn['user']}")
        lines.append(f"ASSISTANT: {turn['assistant']}")
    lines.append(f"USER: {user_input}")
    return "\n".join(lines)


def run_turn(agent, user_input, max_hops=3, max_steps=12):
    """Цикл одного хода: решения модели, tool calls и хендоффы.

    Вернуть словарь:
      {"output", "spans", "hops", "agent", "llm_calls", "stop_reason"}

    "agent" — имя агента, который дал финальный ответ (после хендоффов это
    уже не тот, с которого начали).

    stop_reason: "final", "hop_limit", "tool_guardrail", "unknown_tool",
    "unknown_handoff", "bad_decision" или "budget".

    Ключевые правила:
      * хендофф без "input" передаёт следующему агенту ТОТ ЖЕ вход —
        исходный запрос теряться не должен;
      * больше max_hops передач не делаем: A -> B -> A -> B иначе не
        кончится, это handoff drift;
      * tool guardrail проверяется ДО вызова fn: не прошёл — инструмент не
        выполняется вообще, никаких побочных эффектов;
      * результат инструмента возвращается модели как наблюдение
        "tool <name> returned: <result>".

    Ошибки (нет такого инструмента, нет такого хендоффа) не бросаются, а
    становятся текстом ответа: цикл агента не должен падать.
    """
    spans = []
    current = agent
    current_input = user_input
    hops = 0
    llm_calls = 0
    output = ""
    stop_reason = "budget"

    for _ in range(max_steps):
        decision = current["policy"](current_input)
        llm_calls += 1
        kind = decision.get("kind")

        if kind == "final":
            output = decision["text"]
            spans.append({"name": f"llm.{current['name']}", "attributes": {"output": output}})
            stop_reason = "final"
            break

        if kind == "tool":
            tool_name = decision["tool"]
            args = decision.get("args", {})
            tool = next((t for t in current["tools"] if t["name"] == tool_name), None)
            if tool is None:
                spans.append(
                    {"name": "tool_error", "attributes": {"tool": tool_name, "passed": False}}
                )
                output = f"error: unknown tool {tool_name!r}"
                stop_reason = "unknown_tool"
                break
            guard = tool.get("guardrail")
            if guard is not None:
                ok, reason = guard(args)
                spans.append(
                    {
                        "name": f"tool_guardrail.{tool_name}",
                        "attributes": {"passed": ok, "reason": reason},
                    }
                )
                if not ok:
                    # выходим ДО tool["fn"](...): побочного эффекта не будет
                    output = f"tool guardrail tripped: {reason}"
                    stop_reason = "tool_guardrail"
                    break
            result = tool["fn"](**args)
            spans.append(
                {"name": f"tool.{tool_name}", "attributes": {"args": args, "result": result}}
            )
            current_input = f"tool {tool_name} returned: {result}"
            continue

        if kind == "handoff":
            target = next(
                (t for t in current["handoffs"] if t["name"] == decision["to"]), None
            )
            if target is None:
                output = f"error: no handoff to {decision['to']!r}"
                stop_reason = "unknown_handoff"
                break
            if hops >= max_hops:
                output = f"handoff budget exhausted after {max_hops} hops"
                stop_reason = "hop_limit"
                break
            spans.append(
                {
                    "name": f"handoff.{handoff_tool_name(target['name'])}",
                    "attributes": {"from": current["name"], "to": target["name"]},
                }
            )
            hops += 1
            current = target
            # ключевая строка: без "input" контекст едет дальше как есть
            current_input = decision.get("input", current_input)
            continue

        output = f"error: unknown policy kind {kind!r}"
        stop_reason = "bad_decision"
        break

    return {
        "output": output,
        "spans": spans,
        "hops": hops,
        "agent": current["name"],
        "llm_calls": llm_calls,
        "stop_reason": stop_reason,
    }


def run_guarded(
    agent,
    user_input,
    input_guardrails=(),
    output_guardrails=(),
    max_hops=3,
    blocking=True,
    session=None,
):
    """Полный прогон: input guardrails -> ход -> output guardrails.

    blocking=True (в SDK это run_in_parallel=False): input-проверка идёт
    ПЕРВОЙ, и на срабатывании основная модель не вызывается вовсе —
    llm_calls == 0.
    blocking=False: основная модель работает параллельно с проверкой, и на
    срабатывании её работа выбрасывается — wasted_llm_calls > 0. Латентность
    ниже, токены сгорели.

    session (если передан) — список словарей {"user", "assistant"}. Вход
    собирается через session_prompt, и удачный ход дописывается в конец.
    Сработавший guardrail в историю НЕ пишется: ответа не было.

    Вернуть словарь:
      {"output", "spans", "tripped", "llm_calls", "wasted_llm_calls",
       "hops", "agent", "stop_reason"}

    tripped — None или {"stage", "name", "reason"}. При срабатывании
    output пустой: наружу отдавать нечего.

    output guardrails всегда «поздние»: ответ уже сгенерирован, поэтому их
    срабатывание сжигает токены при любом режиме.
    """
    prompt = user_input if session is None else session_prompt(session, user_input)
    spans, tripped = run_guardrails(input_guardrails, prompt, "input")

    if tripped is not None and blocking:
        return {
            "output": "",
            "spans": spans,
            "tripped": tripped,
            "llm_calls": 0,
            "wasted_llm_calls": 0,
            "hops": 0,
            "agent": agent["name"],
            "stop_reason": "input_guardrail",
        }

    turn = run_turn(agent, prompt, max_hops=max_hops)
    spans = spans + turn["spans"]

    if tripped is not None:
        return {
            "output": "",
            "spans": spans,
            "tripped": tripped,
            "llm_calls": turn["llm_calls"],
            "wasted_llm_calls": turn["llm_calls"],
            "hops": turn["hops"],
            "agent": turn["agent"],
            "stop_reason": "input_guardrail",
        }

    out_spans, out_tripped = run_guardrails(output_guardrails, turn["output"], "output")
    spans = spans + out_spans
    if out_tripped is not None:
        return {
            "output": "",
            "spans": spans,
            "tripped": out_tripped,
            "llm_calls": turn["llm_calls"],
            "wasted_llm_calls": turn["llm_calls"],
            "hops": turn["hops"],
            "agent": turn["agent"],
            "stop_reason": "output_guardrail",
        }

    if session is not None:
        session.append({"user": user_input, "assistant": turn["output"]})

    return {
        "output": turn["output"],
        "spans": spans,
        "tripped": None,
        "llm_calls": turn["llm_calls"],
        "wasted_llm_calls": 0,
        "hops": turn["hops"],
        "agent": turn["agent"],
        "stop_reason": turn["stop_reason"],
    }


def redact_spans(spans, sensitive=SENSITIVE_ATTRIBUTES):
    """Вынести чувствительный контент из спанов. Вернуть (spans, store).

    Значение каждого атрибута из sensitive заменяется на ссылку "ref:1",
    "ref:2", ... в порядке обхода, а сам контент уезжает в store —
    словарь ссылка -> значение.

    s = [{"name": "llm.billing", "attributes": {"output": "card 4111"}}]
    redact_spans(s)
        ->  ([{"name": "llm.billing", "attributes": {"output": "ref:1"}}],
             {"ref:1": "card 4111"})

    Исходный список НЕ меняется: спаны часто уходят сразу в несколько
    приёмников, и порча их на месте — способ уронить второй приёмник.

    Имена спанов и служебные атрибуты (passed, from, to) остаются как есть:
    без них трейс перестаёт быть читаемым, а секретов в них нет.
    """
    store = {}
    redacted = []
    for span in spans:
        attributes = {}
        for key, value in span["attributes"].items():
            if key in sensitive:
                ref = f"ref:{len(store) + 1}"
                store[ref] = value
                attributes[key] = ref
            else:
                attributes[key] = value
        # новый словарь, а не span.copy() с общим attributes внутри
        redacted.append({"name": span["name"], "attributes": attributes})
    return (redacted, store)
