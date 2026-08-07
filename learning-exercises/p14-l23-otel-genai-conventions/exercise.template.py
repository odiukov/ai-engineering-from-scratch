"""
Семантические конвенции OpenTelemetry GenAI

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l23-otel-genai-conventions
Разбор:  /check-code p14-l23-otel-genai-conventions
"""

GEN_AI_OPERATIONS = ("chat", "completion", "create_agent", "invoke_agent", "tool_call")
GEN_AI_PROVIDERS = ("anthropic", "openai", "aws.bedrock", "google.vertex")
CONTENT_MODES = ("off", "inline", "reference")


def describe_span(operation, target=None, remote=False):
    """Имя и kind спана по конвенции GenAI.

    describe_span("invoke_agent", "planner")
        ->  {"name": "invoke_agent planner", "kind": "INTERNAL"}
    describe_span("invoke_agent")
        ->  {"name": "invoke_agent", "kind": "INTERNAL"}
    describe_span("invoke_agent", "planner", remote=True)
        ->  {"name": "invoke_agent planner", "kind": "CLIENT"}
    describe_span("chat", "claude-x")
        ->  {"name": "chat claude-x", "kind": "CLIENT"}

    `target` — это gen_ai.agent.name для агентских спанов,
    gen_ai.request.model для модельных и имя инструмента для tool_call.
    Без него имя спана — просто операция, без хвоста и без пробела.

    Ловушка: remote влияет ТОЛЬКО на invoke_agent. Спаны chat/completion
    всегда CLIENT (это вызов удалённого API), а create_agent и tool_call
    всегда INTERNAL — они происходят внутри процесса, сколько бы ни
    передавали remote=True.
    """
    raise NotImplementedError


def genai_attributes(
    provider,
    operation,
    request_model=None,
    response_model=None,
    agent_name=None,
    data_source_id=None,
):
    """Словарь атрибутов gen_ai.* — только те ключи, значение которых есть.

    genai_attributes("anthropic", "chat", request_model="claude-x")
        ->  {"gen_ai.provider.name": "anthropic",
             "gen_ai.operation.name": "chat",
             "gen_ai.request.model": "claude-x"}

    genai_attributes("openai", "chat", "gpt-x", response_model="gpt-x-0301")
        ->  ... + {"gen_ai.response.model": "gpt-x-0301"}

    genai_attributes("mistral", "chat")  ->  ValueError

    Ловушка: атрибут с пустым значением класть НЕЛЬЗЯ — в бэкенде
    "gen_ai.response.model": None и отсутствие ключа выглядят по-разному, и
    дашборд начинает считать None отдельной моделью. Нет значения — нет ключа.

    Зачем: response.model отличается от request.model, когда провайдер
    отроутил запрос. Без обоих атрибутов регрессию «нас перевели на другую
    ревизию модели» не увидеть.
    """
    raise NotImplementedError


def format_traceparent(trace_id, span_id, sampled=True):
    """Заголовок W3C traceparent: version-traceid-spanid-flags.

    format_traceparent("a" * 32, "b" * 16)
        ->  '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01'
    format_traceparent("a" * 32, "b" * 16, sampled=False)
        ->  '00-...-00'
    format_traceparent("A" * 32, "b" * 16)   ->  ValueError

    Требования W3C, за которые ловят чаще всего: 32 hex-символа на trace id,
    16 на span id, ТОЛЬКО нижний регистр, и ни то, ни другое не может быть
    из одних нулей — все нули означают «идентификатора нет».
    """
    raise NotImplementedError


def continue_trace(header, new_trace_id=None):
    """Создать трейс: либо продолжить входящий traceparent, либо начать новый.

    Возвращает {"trace_id", "spans": [], "stack": [...], "remote_parent"}.

    continue_trace(None, "a" * 32)["trace_id"]      ->  'aaaa...'  (32 символа)
    continue_trace(None, "a" * 32)["remote_parent"] ->  None
    continue_trace("00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        ->  trace_id = "aaaa...", remote_parent = "bbbbbbbbbbbbbbbb"
    continue_trace(None)                            ->  ValueError

    Стек открытых спанов всегда стартует пустым: удалённый родитель живёт в
    отдельном поле remote_parent, потому что закрывать его мы не будем — он
    закроется в чужом процессе.

    Ключевое свойство: trace_id НЕ рождается здесь, когда пришёл заголовок —
    он берётся из него. Сгенерируешь свой — и вызов в дочернем процессе
    окажется отдельным трейсом, а не веткой родительского; именно так
    ломается сквозной трейс через CLI-субпроцесс.

    Версия отличная от "00" — ValueError: разбирать неизвестный формат наугад
    хуже, чем отказаться.
    """
    raise NotImplementedError


def start_span(trace, span_id, name, kind, attributes, start_ns):
    """Открыть спан: родитель — тот, что сейчас на вершине стека трейса.

    trace = continue_trace(None, "a" * 32)
    start_span(trace, "b" * 16, "invoke_agent p", "INTERNAL", {}, 0)["parent_id"]
        ->  None
    затем start_span(trace, "c" * 16, "tool_call s", "INTERNAL", {}, 10)["parent_id"]
        ->  'bbbbbbbbbbbbbbbb'

    Возвращает сам спан: trace_id, span_id, parent_id, name, kind,
    attributes, start_ns, end_ns=None.

    Если стек пуст, а трейс продолжает входящий traceparent, родителем
    становится remote_parent — так ветка из другого процесса прирастает к
    родительскому спану.

    Ловушки. Первая: trace_id спан НЕ придумывает — берёт из трейса; трейс
    без trace_id — ValueError. Вторая: повторный span_id — ValueError, иначе
    дерево склеит два разных спана в один узел. Третья: attributes надо
    скопировать, иначе один словарь окажется общим у нескольких спанов и
    правка одного перепишет остальные.
    """
    raise NotImplementedError


def end_span(trace, span_id, end_ns):
    """Закрыть спан. Закрыть можно только самый внутренний из открытых.

    Возвращает спан с заполненными end_ns и duration_ns.

    Ловушка, ради которой всё это: если разрешить закрывать родителя, пока
    открыт ребёнок, получится спан, который «переживает» родителя — в
    бэкенде это либо отрицательная длительность у ребёнка, либо оторванная
    ветка. Порядок строго LIFO, попытка закрыть не вершину — ValueError.

    Вторая ловушка: end_ns < start_ns — тоже ValueError, а не отрицательная
    длительность. Часы монотонные, время приходит параметром; отрицательная
    длительность означает перепутанные аргументы.
    """
    raise NotImplementedError


def span_tree(trace):
    """Собрать дерево из плоского списка спанов: [{"span": ..., "children": [...]}].

    Корни — спаны без родителя, а также спаны, чей родитель пришёл из
    другого процесса (remote_parent из traceparent): межпроцессный трейс
    обязан выглядеть одним деревом, а не набором обрывков.

    Порядок детей — порядок открытия спанов.

    Ловушки. Первая: незакрытые спаны — ValueError; дерево из середины
    прогона врёт про длительности. Вторая: parent_id, которого нет ни среди
    спанов, ни в remote_parent, — тоже ValueError; это «orphaned tool span»
    из урока, и молча превращать его в корень нельзя, иначе поломка
    контекста никогда не обнаружится.
    """
    raise NotImplementedError


def capture_content(store, span, messages, mode="off"):
    """Контракт content capture: по умолчанию содержимое НЕ попадает в спан.

    Возвращает ссылку (str) в режиме "reference", иначе None.

    capture_content({}, span, ["secret"])                      ->  None,
        и в span["attributes"] не появилось ничего
    capture_content({}, span, ["hi"], mode="inline")           ->  None,
        span["attributes"]["gen_ai.input.messages"] == ["hi"]
    capture_content(store, span, ["secret"], mode="reference") ->  'content-1',
        store["content-1"] == ["secret"], а на спане только ссылка

    Ловушка и смысл: в продовом режиме "reference" на спане не должно быть
    самого текста — ни в одном атрибуте. Трейсы читает вся дежурная смена,
    а в промптах лежат PII и секреты. Содержимое уходит во внешнее
    хранилище, на спане — только идентификатор строки.

    Ссылки нумеруются от размера store, поэтому одинаковые сообщения из
    двух спанов не перезаписывают друг друга.
    """
    raise NotImplementedError
