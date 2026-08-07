"""
Капстоун — экосистема инструментов целиком

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l23-capstone-tool-ecosystem
Разбор:  /check-code p13-l23-capstone-tool-ecosystem
"""

import copy
import hashlib

SPAN_KINDS = ("INTERNAL", "CLIENT", "SERVER")
SEARCH_TOOL = "arxiv_search"
REPORT_TOOL = "generate_report"
WRITER_SKILL = "summarize_papers"
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
