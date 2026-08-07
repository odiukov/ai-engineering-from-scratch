"""
OpenAI Agents SDK: хендоффы, guardrails, трейсинг

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l16-openai-agents-sdk
Разбор:  /check-code p14-l16-openai-agents-sdk
"""

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
