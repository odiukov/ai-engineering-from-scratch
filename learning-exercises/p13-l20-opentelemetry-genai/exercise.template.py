"""
OpenTelemetry GenAI — трассировка вызовов инструментов

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l20-opentelemetry-genai
Разбор:  /check-code p13-l20-opentelemetry-genai
"""

SPAN_KINDS = ("INTERNAL", "CLIENT", "SERVER", "PRODUCER", "CONSUMER")
REQUIRED_ATTRS = {
    "chat": (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    ),
    "embeddings": (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
    ),
    "execute_tool": (
        "gen_ai.operation.name",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
    ),
    "invoke_agent": (
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.agent.id",
    ),
}
CONTENT_EVENTS = (
    "gen_ai.content.prompt",
    "gen_ai.content.completion",
    "gen_ai.content.tool_call",
)
TRACEPARENT_VERSION = "00"
TRACE_ID_HEX_LEN = 32
SPAN_ID_HEX_LEN = 16


def new_span(name, kind, start_ns, rng, parent=None, attrs=None):
    """Создать спан. Дочерний наследует traceId родителя, но не spanId.

    rng — random.Random, из него берутся идентификаторы. Глобальный random
    сюда не годится: два прогона теста обязаны дать одни и те же id.

    root = new_span("agent.invoke_agent", "INTERNAL", 1000, rng)
        ->  {"name": ..., "traceId": <32 hex>, "spanId": <16 hex>,
             "parentSpanId": None, "endTimeUnixNano": None, ...}
    child = new_span("llm.chat", "CLIENT", 1100, rng, parent=root)
        ->  child["traceId"] == root["traceId"]
            child["parentSpanId"] == root["spanId"]

    endTimeUnixNano равен None, пока спан не закрыт. Ноль сюда не годится:
    ноль — законное время, и "не закрыт" от "закрыт в нулевую наносекунду"
    было бы не отличить.

    Неизвестный kind — ValueError. Экспортёр молча выбросит спан с чужим
    видом, и в бэкенде его просто не будет.
    """
    raise NotImplementedError


def finish_span(span, end_ns):
    """Закрыть спан временем end_ns. Вернуть его же.

    finish_span(span, 2000)  ->  span["endTimeUnixNano"] == 2000

    Два отказа, оба ValueError:
      * спан уже закрыт — повторный end() в OTel тихо игнорируется, и потому
        двойное закрытие живёт в коде годами; здесь оно шумит сразу;
      * end_ns меньше начала — часы поехали назад, и длительность выйдет
        отрицательной. В бэкенде такой спан рисуется полосой нулевой длины,
        и найти причину потом почти невозможно.

    Нулевая длительность (end_ns == start_ns) законна: мгновенный шаг.
    """
    raise NotImplementedError


def traceparent(span, sampled=True):
    """Заголовок W3C traceparent для передачи контекста за границу процесса.

    traceparent(span)  ->  "00-<32 hex traceId>-<16 hex spanId>-01"
    traceparent(span, sampled=False)  ->  "...-00"

    Именно эту строку MCP-клиент кладёт в HTTP-заголовок, а для stdio — в
    поле `_meta.traceparent` JSON-RPC-запроса. Сервер её разбирает и
    продолжает ТУ ЖЕ трассу вместо того, чтобы завести свою.

    В заголовок идёт spanId текущего спана: для сервера он станет
    parentSpanId. Родительский id сюда подставлять нельзя — потеряется
    один уровень дерева.
    """
    raise NotImplementedError


def parse_traceparent(header):
    """Разобрать заголовок traceparent. Словарь или ValueError.

    parse_traceparent("00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        ->  {"traceId": "aaaa...", "spanId": "bbbb...", "sampled": True}

    Всё, что не по спецификации, — ValueError, и вот почему по каждому пункту:
      * не четыре поля или чужая версия — заголовок не наш;
      * не hex или неверная длина — id непригоден для корреляции;
      * ВЕРХНИЙ регистр запрещён: спецификация требует нижний, а бэкенды
        сравнивают id как строки, и "AB..." с "ab..." не склеятся в одну
        трассу;
      * traceId или spanId из одних нулей — зарезервированное «нет
        значения»; принять его значит склеить в одну трассу всех, кто
        забыл проставить контекст.
    """
    raise NotImplementedError


def missing_gen_ai_attributes(span):
    """Каких обязательных gen_ai.* атрибутов не хватает спану. Список в порядке semconv.

    Пусть у спана есть только gen_ai.operation.name = "execute_tool".
    missing_gen_ai_attributes(span)
        ->  ["gen_ai.tool.name", "gen_ai.tool.call.id"]

    Полностью заполненный спан даёт [].

    Спан без gen_ai.operation.name — ValueError: по этому атрибуту бэкенд
    решает, как спан вообще читать, и без него список требований не
    определён. Незнакомая операция — тоже ValueError: обычно это опечатка
    ("tool_execute" вместо "execute_tool"), и она стоит целого дашборда.
    """
    raise NotImplementedError


def capture_content_event(span, event_name, content, at_ns, capture_content=False):
    """Добавить событие с содержимым — только если сбор содержимого включён.

    capture_content_event(span, "gen_ai.content.prompt", "hi", 1500)
        ->  False, событий у спана не прибавилось
    capture_content_event(span, "gen_ai.content.prompt", "hi", 1500, True)
        ->  True, span["events"][-1]["attributes"]["content"] == "hi"

    Сбор выключен по умолчанию не из экономии: в prompt лежат персональные
    данные пользователя, и трасса уезжает наружу, в чужой бэкенд. Включают
    его осознанно, через OTEL_SEMCONV_STABILITY_OPT_IN и переменные
    content-capture.

    Имя события проверяется ДО проверки флага. Иначе опечатка в имени
    отлежится в коде до того дня, когда сбор включат в проде.
    """
    raise NotImplementedError


def span_tree(spans):
    """Собрать из плоского списка спанов дерево. Корневой узел или ValueError.

    Узел: {"span": <спан>, "children": [<узел>, ...]}.

    span_tree([root, llm, tool])
        ->  {"span": root, "children": [{"span": llm, "children": []},
                                        {"span": tool, "children": []}]}

    Дети упорядочены по времени начала, при совпадении — по spanId. Порядок
    спанов во входном списке на дерево не влияет: экспортёр отдаёт их в том
    порядке, в каком они закрылись, а закрываются родители последними.

    Четыре причины для ValueError — все означают, что трасса битая и рисовать
    её нельзя: разные traceId в одном списке, повторный spanId, ссылка на
    несуществующего родителя, не ровно один корень.
    """
    raise NotImplementedError


def trace_problems(spans):
    """Проверить трассу перед экспортом. Список претензий, пустой — значит норма.

    В отличие от span_tree ничего не бросает: это линтер, ему нужно
    перечислить ВСЕ беды разом, а не упасть на первой.

    trace_problems([root, child_ending_after_root])
        ->  ["llm.chat: ends after parent agent.invoke_agent"]

    Что проверяется, кроме атрибутов:
      * незакрытый спан — забытый finish_span; в бэкенде он не появится;
      * ребёнок начался раньше родителя или закончился позже него. Это
        главная смысловая проверка трассы: родитель по определению
        охватывает всё, что произошло внутри него. Нарушение означает, что
        спан прицепили не к тому родителю — а по красивой картинке в Jaeger
        это незаметно.
    """
    raise NotImplementedError
