"""
MCP sampling: сервер просит модель клиента — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Обычный MCP-сервер — тупой исполнитель: получил аргументы, выполнил код,
вернул content. Sampling переворачивает направление: теперь СЕРВЕР просит
клиента прогнать его модель. Так сервер держит у себя алгоритм, не имея ни
одного ключа от LLM. Соответствие настоящему API:

    model_preferences       <-  ModelPreferences из mcp.types
    create_message_request  <-  ctx.session.create_message(...)
    pick_model              <-  колбэк выбора модели на стороне клиента
    sampling_result         <-  CreateMessageResult, который возвращает клиент
    spend_sample            <-  per-session rate limit клиента против loop bomb
    run_sampling_loop       <-  серверный агентный цикл поверх sampling
    review_request          <-  диалог подтверждения (human-in-the-loop)

Модели здесь нет: клиент — это функция, которую мы передаём параметром.
Поэтому цикл детерминирован и тестируется без единого токена.
"""

JSONRPC = "2.0"

SAMPLING_METHOD = "sampling/createMessage"

# Три режима подмешивания контекста. "allServers" протаскивает в запрос
# переписку ЧУЖИХ серверов — с 2025-11-25 это мягко депрекейтнуто.
CONTEXT_MODES = ("none", "thisServer", "allServers")

# Стандартные причины; само поле остаётся открытой строкой.
STANDARD_STOP_REASONS = ("endTurn", "stopSequence", "maxTokens", "toolUse")


class SamplingBudgetExceeded(Exception):
    """Клиент отказался сэмплить: сервер выбрал лимит вызовов."""


def model_preferences(cost, speed, intelligence, hints=()):
    """Приоритеты выбора модели: три независимых числа 0..1.

    model_preferences(0.8, 0.2, 0.6)
        ->  {"costPriority": 0.8, "speedPriority": 0.2,
             "intelligencePriority": 0.6}
    model_preferences(0, 0, 1, hints=["claude-3-5-sonnet"])
        ->  {..., "hints": [{"name": "claude-3-5-sonnet"}]}

    Поля не обязаны суммироваться в 1.0: 0.9/0.9/0.9 валидно и значит,
    что все три характеристики важны. Каждое значение должно лежать в 0..1;
    hints на проводе становятся объектами {"name": ...}.
    """
    priorities = (cost, speed, intelligence)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in priorities):
        raise TypeError("Priorities must be numbers")
    if any(value < 0 or value > 1 for value in priorities):
        raise ValueError("Priorities must be between 0 and 1")
    prefs = {
        "costPriority": cost,
        "speedPriority": speed,
        "intelligencePriority": intelligence,
    }
    # пустой hints не шлём вовсе: пустой список — это обещание, которого нет
    if hints:
        prefs["hints"] = [{"name": name} for name in hints]
    return prefs


def create_message_request(request_id, messages, system_prompt=None, preferences=None,
                           max_tokens=1024, include_context="none", tools=None):
    """Запрос sampling/createMessage от сервера к клиенту.

    create_message_request(42, ["Pick five files"])
        ->  {"jsonrpc": "2.0", "id": 42, "method": "sampling/createMessage",
             "params": {"messages": [{"role": "user",
                                      "content": {"type": "text",
                                                  "text": "Pick five files"}}],
                        "includeContext": "none", "maxTokens": 1024}}

    Голая строка в messages — это сообщение пользователя: разворачиваем её
    в полный объект с типизированным content. Готовые словари пропускаем
    как есть.

    tools (SEP-1577, 2025-11-25) заставляет клиента прогнать полный цикл
    вызова инструментов внутри одного sampling — так сервер получает
    ReAct-петлю чужой моделью. Шейп экспериментальный, SDK его ещё двигают.

    Ловушки:
      * includeContext — одно из трёх значений, опечатка тут молча
        превратится в «сервер просил лишнего»: ValueError;
      * maxTokens обязателен и обязан быть положительным;
      * ключи, которых нет, не шлём как null — их просто нет.
    """
    if include_context not in CONTEXT_MODES:
        raise ValueError(f"includeContext must be one of {CONTEXT_MODES}")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("maxTokens must be a positive integer")

    normalized = [
        {"role": "user", "content": {"type": "text", "text": m}} if isinstance(m, str) else m
        for m in messages
    ]
    params = {
        "messages": normalized,
        "includeContext": include_context,
        "maxTokens": max_tokens,
    }
    if system_prompt is not None:
        params["systemPrompt"] = system_prompt
    if preferences is not None:
        params["modelPreferences"] = preferences
    if tools is not None:
        params["tools"] = tools
    return {"jsonrpc": JSONRPC, "id": request_id, "method": SAMPLING_METHOD, "params": params}


def pick_model(catalog, preferences):
    """Выбрать модель под приоритеты сервера. Решает КЛИЕНТ.

    catalog — [{"name": ..., "cost": 0..1, "speed": 0..1, "intelligence": 0..1}],
    где cost=1 значит «самая дешёвая», а не «самая дорогая».

    pick_model(cat, model_preferences(0, 0, 1))  ->  самая умная
    pick_model(cat, model_preferences(1, 0, 0))  ->  самая дешёвая
    pick_model([], prefs)                        ->  ValueError

    Оценка — скалярное произведение приоритетов на характеристики модели.

    hints — это упорядоченные ПРЕДПОЧТЕНИЯ, а не приказ и не только
    tie-breaker. Имя хинта сопоставляется как подстрока; первый хинт, для
    которого есть кандидаты, задаёт предпочтительную группу. Внутри неё
    побеждает оценка по трём приоритетам. Клиент всё равно делает финальный
    выбор и может сопоставить хинт с эквивалентом другого провайдера.
    """
    if not catalog:
        raise ValueError("Model catalog is empty")
    def score(model):
        return (
            preferences["costPriority"] * model["cost"]
            + preferences["speedPriority"] * model["speed"]
            + preferences["intelligencePriority"] * model["intelligence"]
        )

    candidates = catalog
    for hint in preferences.get("hints", []):
        matched = [m for m in catalog if hint["name"].lower() in m["name"].lower()]
        if matched:
            candidates = matched
            break
    return max(candidates, key=score)["name"]


def sampling_result(request_id, text, model, stop_reason="endTurn"):
    """Ответ клиента на sampling/createMessage.

    sampling_result(42, "done", "claude-3-5-sonnet-20251022")
        ->  {"jsonrpc": "2.0", "id": 42,
             "result": {"role": "assistant",
                        "content": {"type": "text", "text": "done"},
                        "model": "claude-3-5-sonnet-20251022",
                        "stopReason": "endTurn"}}

    Роль всегда "assistant": это ответ модели, других вариантов не бывает.

    Поле model — та модель, которую клиент РЕАЛЬНО взял. Она вполне может
    не совпасть с hints сервера, и сервер обязан это пережить.

    stopReason — открытая строка: кроме стандартных endTurn,
    stopSequence, maxTokens и toolUse клиент может вернуть
    причину своего провайдера. Пустая строка всё ещё ошибка.
    """
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError("stopReason must be a non-empty string")
    return {
        "jsonrpc": JSONRPC,
        "id": request_id,
        "result": {
            "role": "assistant",
            "content": {"type": "text", "text": text},
            "model": model,
            "stopReason": stop_reason,
        },
    }


def spend_sample(budget, key, limit):
    """Списать один вызов sampling из бюджета. Вернуть израсходованное.

    budget — {ключ: сколько уже потрачено}, правится на месте.

    spend_sample({}, "tool:summarize", 5)              ->  1
    spend_sample({"tool:summarize": 5}, "tool:summarize", 5)
        ->  SamplingBudgetExceeded

    Зачем: сервер, зовущий sampling в цикле без выхода, тратит ДЕНЬГИ
    пользователя. Это называется loop bomb, и защищаться от неё обязан
    клиент — сервер тут заинтересованная сторона.

    Ключ — это единица учёта: "tool:<имя>" даёт лимит на вызов инструмента,
    "session:<id>" — общий лимит на пользователя.

    Ловушка: исключение здесь своё, а не RuntimeError. Проверка на
    RuntimeError позеленела бы и на пустой заготовке.
    """
    spent = budget.get(key, 0)
    if spent >= limit:
        raise SamplingBudgetExceeded(f"{key}: sampling limit {limit} reached")
    budget[key] = spent + 1
    return budget[key]


def run_sampling_loop(plan, client, preferences=None, limit=5):
    """Серверный агентный цикл: несколько раундов sampling подряд.

    plan   — список раундов, каждый раунд это список сообщений;
    client — функция запрос -> ответ (то, что вернул бы sampling_result).

    Возвращает {"texts": [...], "rounds": n, "stopReason": ...}.

    Каноничный пример из урока — summarize_repo: раунд первый «выбери пять
    файлов», раунд второй «сложи из них резюме». Ключей от LLM у сервера
    нет вовсе, платит пользователь клиента.

    Цикл обрывается на первом же stopReason, отличном от "endTurn":
    "maxTokens" значит, что ответ обрезан, и продолжать рассуждение поверх
    обрубка бессмысленно.

    Ловушка: id запросов обязаны различаться. Два sampling с одним id — и
    клиент не поймёт, на какой из них он отвечает.
    """
    budget = {}
    texts = []
    stop_reason = "endTurn"
    for index, messages in enumerate(plan):
        # бюджет списываем ДО отправки: отказ не должен стоить денег
        spend_sample(budget, "loop", limit)
        request = create_message_request(index + 1, messages, preferences=preferences)
        result = client(request)["result"]
        texts.append(result["content"]["text"])
        stop_reason = result.get("stopReason", "endTurn")
        if stop_reason != "endTurn":
            break
    return {"texts": texts, "rounds": len(texts), "stopReason": stop_reason}


def review_request(request):
    """Что показать пользователю перед тем, как запускать модель.

    review_request(create_message_request(1, ["Summarize the repo"]))
        ->  {"asks": "Summarize the repo", "maxTokens": 1024,
             "contextMode": "none", "tools": [], "risk": "low"}

    Клиент ОБЯЗАН показать пользователю, о чём сервер просит модель.
    Иначе вредоносный сервер спокойно попросит модель сказать пользователю
    что-нибудь удобное себе, и тот ничего не заметит.

    Высокий риск ставим, когда:
      * includeContext не "none" — в запрос утечёт чужой контекст;
      * есть tools — клиент прогонит целую петлю вызовов, а не один ответ;
      * maxTokens явно велик — значит счёт будет заметным.

    Ловушка: "asks" — текст ПЕРВОГО пользовательского сообщения, а не
    системного промпта. Именно в нём сервер формулирует просьбу.
    """
    params = request["params"]
    asks = ""
    for message in params["messages"]:
        if message["role"] == "user":
            asks = message["content"]["text"]
            break
    tools = [t["name"] for t in params.get("tools", [])]
    context_mode = params.get("includeContext", "none")
    max_tokens = params.get("maxTokens", 0)
    risky = context_mode != "none" or bool(tools) or max_tokens > 4096
    return {
        "asks": asks,
        "maxTokens": max_tokens,
        "contextMode": context_mode,
        "tools": tools,
        "risk": "high" if risky else "low",
    }
