"""
Капстоун — экосистема инструментов целиком — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь собирается вместе всё, что фаза 13 разбирала по частям: закреплённый
манифест описаний (защита от tool poisoning), слияние пространств имён на
шлюзе, OAuth-подобные токены с RBAC, делегирование задачи A2A-агенту с
сохранением непрозрачности и сквозная трасса OTel GenAI поверх всего этого.

Соответствие настоящей системе:

    pin_manifest    <-  манифест хэшей описаний, который CI сверяет на каждом деплое
    verify_pins     <-  сама сверка: описание поменялось — сервер не поднимается
    merge_tools     <-  многосерверный клиент, префикс при коллизии имён
    authorize       <-  проверка scope из access token (Phase 13 · 16)
    emit_span       <-  tracer.start_span из семантики GenAI (Phase 13 · 20)
    delegate_task   <-  A2A tasks/send к writer-агенту (Phase 13 · 19)
    opaque_result   <-  граница непрозрачности A2A
    gateway_call    <-  шлюз: authz -> пин -> спан -> вызов -> аудит
    run_research    <-  оркестратор, весь сценарий "найди и напиши отчёт"
    trace_report    <-  то, что показал бы Jaeger, и валидатор трассы

Ни сети, ни часов: провайдеры инструментов приходят в world, а время и
идентификаторы — через ctx (clock и rng). Иначе сквозную трассу нечем
проверить: у неё каждый раз были бы новые id и новые длительности.

ctx — рабочий контекст одного прогона:
    {"spans": [], "audit": [], "clock": <функция без аргументов -> нс>,
     "rng": <random.Random>}
"""

import copy
import hashlib

SPAN_KINDS = ("INTERNAL", "CLIENT", "SERVER")

# Имена инструментов и skill, вокруг которых крутится сценарий капстоуна.
SEARCH_TOOL = "arxiv_search"
REPORT_TOOL = "generate_report"
WRITER_SKILL = "summarize_papers"

# Разделитель префикса при коллизии имён инструментов на шлюзе. Два
# подчёркивания, а не точка: точка занята в JSON-путях многих SDK.
NAMESPACE_SEP = "__"


def pin_manifest(servers):
    """Посчитать манифест: ключ "<сервер>::<инструмент>" -> sha256 описания.

    pin_manifest({"research": [{"name": "arxiv_search",
                                "description": "Search arXiv."}]})
        ->  {"research::arxiv_search": "<64 hex>"}

    Хэшируется именно ОПИСАНИЕ, а не имя: tool poisoning меняет текст
    description (в него дописывают инструкции для модели), оставляя имя и
    схему нетронутыми. Имя сравнивать бессмысленно — оно не менялось.

    Два одинаковых имени на одном сервере — ValueError: манифест перестал бы
    быть однозначным, и подмену второго из них никто бы не заметил.

    Ключи собираются по отсортированным именам серверов, чтобы манифест был
    воспроизводим и годился для diff в CI.
    """
    manifest = {}
    for server in sorted(servers):
        for tool in servers[server]:
            key = f"{server}::{tool['name']}"
            if key in manifest:
                raise ValueError(f"duplicate tool on one server: {key}")
            manifest[key] = hashlib.sha256(
                tool["description"].encode("utf-8")
            ).hexdigest()
    return manifest


def verify_pins(servers, manifest):
    """Сверить текущие описания с закреплёнными. Список претензий.

    Пустой список — всё совпало, сервер можно поднимать.

    verify_pins(servers, pin_manifest(servers))  ->  []
    verify_pins(servers_с_подменённым_описанием, manifest)
        ->  ["research::arxiv_search: description hash changed"]

    Три сорта расхождения, и все три опасны по-разному:
      * хэш изменился — описание переписали после закрепления. Это и есть
        rug pull: сервер прошёл ревью и подменил себя на следующем запуске;
      * инструмента нет в манифесте — он появился после ревью, и его текст
        никто не читал;
      * инструмент есть в манифесте, но пропал с сервера — не атака, но
        клиент, который на него рассчитывал, сломается молча.

    Претензии отсортированы: список идёт в CI-лог, и он должен быть
    воспроизводимым от прогона к прогону.
    """
    current = pin_manifest(servers)
    problems = []
    for key in sorted(current):
        if key not in manifest:
            problems.append(f"{key}: not in pinned manifest")
        elif current[key] != manifest[key]:
            problems.append(f"{key}: description hash changed")
    problems.extend(
        f"{key}: pinned but missing on the server"
        for key in sorted(manifest)
        if key not in current
    )
    return problems


def merge_tools(servers):
    """Слить инструменты нескольких серверов в одно пространство имён.

    Вернуть (словарь видимое_имя -> запись, кортеж столкнувшихся имён).

    merge_tools({"research": [t_search], "bibliography": [t_bibtex]})
        ->  ({"arxiv_search": {...}, "format_bibtex": {...}}, ())
    merge_tools({"research": [t_search], "bibliography": [t_search_other]})
        ->  ({"research__arxiv_search": {...},
              "bibliography__arxiv_search": {...}}, ("arxiv_search",))

    При коллизии префикс получают ВСЕ участники, а не только опоздавший.
    Иначе имя arxiv_search продолжало бы означать «тот сервер, который
    подключился первым» — и смысл вызова менялся бы от перезапуска к
    перезапуску.

    Серверы обходятся по отсортированным именам: результат не должен
    зависеть от порядка подключения.
    """
    counts = {}
    for server in servers:
        for tool in servers[server]:
            counts[tool["name"]] = counts.get(tool["name"], 0) + 1
    merged = {}
    for server in sorted(servers):
        for tool in servers[server]:
            name = tool["name"]
            exposed = f"{server}{NAMESPACE_SEP}{name}" if counts[name] > 1 else name
            if exposed in merged:
                raise ValueError(f"name collision survived prefixing: {exposed}")
            merged[exposed] = {
                "server": server,
                "tool": name,
                "description": tool["description"],
            }
    collisions = tuple(sorted(name for name, n in counts.items() if n > 1))
    return merged, collisions


def authorize(world, token, tool_name):
    """Решить, можно ли этому токену звать этот инструмент.

    Вернуть {"allow": bool, "user": id или None, "reason": str, "scope": str или None}.

    authorize(world, "tok_alice", "generate_report")
        ->  {"allow": True, "user": "alice", "reason": "ok",
             "scope": "research:write"}
    authorize(world, "tok_bob", "generate_report")
        ->  {"allow": False, "user": "bob", "reason": "insufficient_scope",
             "scope": "research:write"}
    authorize(world, "tok_nobody", "arxiv_search")
        ->  {"allow": False, "user": None, "reason": "unauthenticated", ...}

    Причина отказа возвращается отдельным полем, а не текстом исключения:
    её пишут в аудит, и по ней потом отвечают на вопрос «почему у Боба не
    работает» без чтения логов построчно.

    Инструмент, для которого не объявлен требуемый scope, — отказ, а не
    разрешение. Умолчание «раз не написано, значит можно» — это способ
    выкатить в прод инструмент без охраны.
    """
    user = world["users"].get(token)
    if user is None:
        return {"allow": False, "user": None, "reason": "unauthenticated", "scope": None}
    required = world["required_scopes"].get(tool_name)
    if required is None:
        return {"allow": False, "user": user["id"], "reason": "unknown_tool", "scope": None}
    if required not in set(user["scopes"]):
        return {
            "allow": False,
            "user": user["id"],
            "reason": "insufficient_scope",
            "scope": required,
        }
    return {"allow": True, "user": user["id"], "reason": "ok", "scope": required}


def emit_span(ctx, name, kind, trace_id, parent_id, attrs=None):
    """Открыть спан, положить его в ctx["spans"] и вернуть.

    Время начала берётся из ctx["clock"](), идентификатор — из ctx["rng"].
    endTimeUnixNano остаётся None: закрывает спан вызывающий, когда работа
    действительно закончилась —

        span = emit_span(ctx, "mcp.call", "CLIENT", trace_id, parent, attrs)
        ...
        span["endTimeUnixNano"] = ctx["clock"]()

    emit_span(ctx, "llm.chat", "CLIENT", trace_id, root_id, {...})
        ->  {"name": "llm.chat", "traceId": trace_id, "parentSpanId": root_id,
             "endTimeUnixNano": None, ...}

    trace_id приходит параметром и НЕ рождается внутри: спан, который сам
    себе выдаёт трассу, — главная причина, по которой сквозная трасса
    разваливается на десяток одиночных.
    """
    if kind not in SPAN_KINDS:
        raise ValueError(f"unknown span kind: {kind}")
    span = {
        "name": name,
        "kind": kind,
        "traceId": trace_id,
        "spanId": format(ctx["rng"].getrandbits(64), "016x"),
        "parentSpanId": parent_id,
        "startTimeUnixNano": ctx["clock"](),
        "endTimeUnixNano": None,
        "attributes": dict(attrs or {}),
    }
    ctx["spans"].append(span)
    return span


def delegate_task(task_id, skill_id, payload):
    """A2A-вызов writer-агента. Полная задача, вместе с внутренностями.

    delegate_task("task_1", WRITER_SKILL, {"papers": [p1, p2]})
        ->  {"id": "task_1", "skillId": ..., "state": "completed",
             "artifact": {...}, "_internal": {...}}
    delegate_task("task_1", "no_such_skill", {})
        ->  state "rejected", artifact None

    Ключи с подчёркиванием — внутренняя кухня вызванного агента: его шаги,
    его модель, его собственные вызовы инструментов. Возвращать их наружу
    целиком нельзя, для этого есть opaque_result. Но СЧИТАТЬ их надо здесь:
    вызванный агент видит свои внутренности, он же их и производит.

    Результат детерминирован: ни времени, ни случайности внутри нет, и
    одинаковый payload даёт одинаковый артефакт.
    """
    if skill_id != WRITER_SKILL:
        return {
            "id": task_id,
            "skillId": skill_id,
            "state": "rejected",
            "artifact": None,
            "_internal": {"reason": f"unknown skill: {skill_id}"},
        }
    papers = list(payload.get("papers", ()))
    listing = "; ".join(f"{p['arxiv_id']} {p['title']}" for p in papers)
    return {
        "id": task_id,
        "skillId": skill_id,
        "state": "completed",
        "artifact": {
            "name": "summary",
            "mimeType": "text/markdown",
            "parts": [
                {"kind": "text", "text": f"{len(papers)} papers summarized: {listing}"}
            ],
        },
        "_internal": {
            "steps": ("fetch_pdf", "outline", "draft"),
            "model": "writer-internal-7b",
            "toolCalls": ("pdf_extract", "cite_lookup"),
        },
    }


def opaque_result(task):
    """Срезать с задачи всё внутреннее. То, что оркестратор имеет право видеть.

    opaque_result(delegate_task("t", WRITER_SKILL, {"papers": []}))
        ->  {"id": "t", "skillId": "summarize_papers", "state": "completed",
             "artifact": {...}}          # без "_internal"

    Это и есть граница непрозрачности A2A и главное отличие от MCP: у MCP
    вызов инструмента прозрачен, у A2A вызванный агент показывает состояние
    и артефакт, а рассуждения оставляет себе. На этом и построена
    возможность звать агента конкурента.

    Копия глубокая. Отдать наружу ссылку на артефакт вызванного агента
    значит позволить оркестратору править чужую задачу задним числом — и
    поймать это потом невозможно.
    """
    if not isinstance(task, dict):
        raise TypeError(f"task must be a dict, got {type(task).__name__}")
    return {
        key: copy.deepcopy(value)
        for key, value in task.items()
        if not key.startswith("_")
    }


def gateway_call(world, ctx, token, tool_name, args, trace_id, parent_id):
    """Пройти вызов через шлюз: авторизация, пин, спан, исполнение, аудит.

    Возвращает результат обработчика либо словарь с ключом "error":
        "unauthenticated" | "insufficient_scope" | "unknown_tool" | "hash_mismatch"

    gateway_call(world, ctx, "tok_alice", SEARCH_TOOL, {"query": "a2a"}, tid, pid)
        ->  результат обработчика; в ctx["audit"] запись decision "allow"
    gateway_call(world, ctx, "tok_bob", REPORT_TOOL, {}, tid, pid)
        ->  {"error": "insufficient_scope", "scope": "research:write"}

    Порядок проверок — это и есть defense in depth, и он не переставляется:
      1. кто ты (токен) и можно ли тебе (scope) — до всего остального;
      2. не подменил ли сервер описание — до того, как модель его увидит;
      3. и только потом спан и вызов.

    Отказ тоже пишется в аудит. Журнал, в котором нет отказов, отвечает на
    вопрос «что происходило» ровно наполовину.

    Спан на отказ НЕ открывается: работы не было, а пустой спан в трассе
    выглядит как успешный вызов нулевой длительности.
    """
    decision = authorize(world, token, tool_name)
    if not decision["allow"]:
        ctx["audit"].append(
            {"user": decision["user"], "tool": tool_name, "decision": decision["reason"]}
        )
        return {"error": decision["reason"], "scope": decision["scope"]}

    found = None
    for server in sorted(world["servers"]):
        for tool in world["servers"][server]:
            if tool["name"] == tool_name:
                found = (server, tool)
                break
        if found:
            break
    if found is None:
        ctx["audit"].append(
            {"user": decision["user"], "tool": tool_name, "decision": "unknown_tool"}
        )
        return {"error": "unknown_tool", "scope": None}

    server_name, tool = found
    key = f"{server_name}::{tool_name}"
    # считаем хэш тем же кодом, что и при закреплении: две формулы рано или
    # поздно разъедутся, и сверка начнёт врать
    if world["manifest"].get(key) != pin_manifest({server_name: [tool]})[key]:
        ctx["audit"].append(
            {"user": decision["user"], "tool": tool_name, "decision": "hash_mismatch"}
        )
        return {"error": "hash_mismatch", "tool": key}

    span = emit_span(
        ctx,
        "mcp.call",
        "CLIENT",
        trace_id,
        parent_id,
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": f"call_{format(ctx['rng'].getrandbits(32), '08x')}",
            "mcp.server": server_name,
            "gateway.user": decision["user"],
        },
    )
    result = world["handlers"][tool_name](args)
    span["endTimeUnixNano"] = ctx["clock"]()
    ctx["audit"].append(
        {"user": decision["user"], "tool": tool_name, "decision": "allow"}
    )
    return result


def run_research(world, ctx, token, query):
    """Весь сценарий: найти статьи, делегировать пересказ, собрать отчёт.

    Возвращает {"traceId", "search", "summary", "report"}.

    run_research(world, ctx, "tok_alice", "agent protocol")
        ->  summary — непрозрачный результат writer-агента,
            report — результат MCP-задачи, ctx["spans"] — одна трасса
    run_research(world, ctx, "tok_bob", "agent protocol")
        ->  search отработал (у Боба есть research:read),
            report {"error": "insufficient_scope"}, summary None

    Делегирование A2A происходит ТОЛЬКО если пользователю разрешён отчёт.
    Позвать писателя, а потом упереться в 403 на своём же шлюзе — это
    оплаченная работа, которую некуда деть. Проверка прав идёт до расходов,
    хотя сам gateway_call проверит их ещё раз: доверять решению вызывающего
    шлюз не обязан.

    Все спаны прогона лежат в одной трассе, корень — agent.invoke_agent.
    """
    rng = ctx["rng"]
    trace_id = format(rng.getrandbits(128), "032x")
    root = emit_span(
        ctx,
        "agent.invoke_agent",
        "INTERNAL",
        trace_id,
        None,
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "research-orchestrator",
            "gen_ai.agent.id": "agent_research",
        },
    )

    plan = emit_span(
        ctx,
        "llm.chat",
        "CLIENT",
        trace_id,
        root["spanId"],
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "claude-sonnet",
        },
    )
    plan["endTimeUnixNano"] = ctx["clock"]()

    search = gateway_call(
        world, ctx, token, SEARCH_TOOL, {"query": query}, trace_id, root["spanId"]
    )
    papers = tuple(search.get("papers", ())) if "error" not in search else ()

    summary = None
    if papers and authorize(world, token, REPORT_TOOL)["allow"]:
        a2a = emit_span(
            ctx,
            "a2a.tasks.send",
            "CLIENT",
            trace_id,
            root["spanId"],
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "writer-agent",
                "gen_ai.agent.id": "agent_writer",
                "a2a.skill": WRITER_SKILL,
            },
        )
        task = delegate_task(f"task_{a2a['spanId'][:8]}", WRITER_SKILL, {"papers": papers})
        summary = opaque_result(task)
        a2a["endTimeUnixNano"] = ctx["clock"]()

    report = gateway_call(
        world,
        ctx,
        token,
        REPORT_TOOL,
        {"papers": papers, "summary": summary},
        trace_id,
        root["spanId"],
    )

    final = emit_span(
        ctx,
        "llm.chat",
        "CLIENT",
        trace_id,
        root["spanId"],
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "anthropic",
            "gen_ai.request.model": "claude-sonnet",
        },
    )
    final["endTimeUnixNano"] = ctx["clock"]()
    # корень закрывается последним: он обязан охватывать всё, что было внутри
    root["endTimeUnixNano"] = ctx["clock"]()
    return {"traceId": trace_id, "search": search, "summary": summary, "report": report}


def trace_report(spans):
    """Свести трассу в отчёт и заодно проверить её.

    Возвращает {"traceIds", "roots", "spanCount", "genAiAttributes", "problems"}.

    trace_report(ctx["spans"])
        ->  {"traceIds": ("<32 hex>",), "roots": ("agent.invoke_agent",),
             "spanCount": 6,
             "genAiAttributes": {"gen_ai.operation.name": 6, ...},
             "problems": []}

    genAiAttributes — сколько спанов несут каждый атрибут gen_ai.*. По этой
    таблице сразу видно дыру в инструментации: если gen_ai.tool.name есть
    только у половины execute_tool-спанов, дашборд по инструментам врёт.

    problems пустой — трассу можно экспортировать. Что проверяется:
      * больше одного traceId — прогон развалился на несколько трасс;
      * не ровно один корень — потерянный parentSpanId или лишний корень;
      * незакрытый спан — забытое присваивание endTimeUnixNano;
      * ребёнок начался раньше родителя или закончился позже. Это главная
        смысловая проверка: родитель по определению охватывает всё, что
        произошло внутри него, и нарушение означает, что спан прицепили не
        к тому родителю.
    """
    trace_ids = tuple(sorted({s["traceId"] for s in spans}))
    roots = tuple(s["name"] for s in spans if s["parentSpanId"] is None)
    counts = {}
    for span in spans:
        for key in span["attributes"]:
            if key.startswith("gen_ai."):
                counts[key] = counts.get(key, 0) + 1

    problems = []
    if len(trace_ids) > 1:
        problems.append(f"spans belong to {len(trace_ids)} traces")
    if len(roots) != 1:
        problems.append(f"expected exactly one root span, got {len(roots)}")
    by_id = {s["spanId"]: s for s in spans}
    for span in spans:
        label = span["name"]
        if span["endTimeUnixNano"] is None:
            problems.append(f"{label}: span is not finished")
        parent_id = span["parentSpanId"]
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            problems.append(f"{label}: parent {parent_id} is not in this trace")
            continue
        if span["startTimeUnixNano"] < parent["startTimeUnixNano"]:
            problems.append(f"{label}: starts before parent {parent['name']}")
        if (
            span["endTimeUnixNano"] is not None
            and parent["endTimeUnixNano"] is not None
            and span["endTimeUnixNano"] > parent["endTimeUnixNano"]
        ):
            problems.append(f"{label}: ends after parent {parent['name']}")
    return {
        "traceIds": trace_ids,
        "roots": roots,
        "spanCount": len(spans),
        "genAiAttributes": dict(sorted(counts.items())),
        "problems": problems,
    }
