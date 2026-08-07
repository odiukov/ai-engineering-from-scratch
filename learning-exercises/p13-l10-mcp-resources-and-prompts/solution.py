"""
Ресурсы и промпты MCP: контекст помимо инструментов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Инструментам достаётся девяносто процентов внимания, а два других примитива
решают другие задачи: ресурс отдаёт данные на чтение, промпт отдаёт
многошаговый шаблон. Соответствие настоящему API:

    pick_primitive         <-  решение «tool / resource / prompt» при проектировании
    resource_entry         <-  элемент ответа resources/list
    read_resource          <-  обработчик resources/read (@app.resource в FastMCP)
    expand_template        <-  разбор resourceTemplates (notes://{id})
    resolve                <-  статический ресурс или вычисляемый по шаблону
    subscribe              <-  resources/subscribe и его учёт на сервере
    updated_notifications  <-  рассылка notifications/resources/updated
    render_prompt          <-  обработчик prompts/get (@app.prompt в FastMCP)

Ни файлов, ни сети: хранилище — обычный словарь, подписки — обычный
словарь, а уведомления мы не отправляем, а возвращаем списком.
"""

import base64
import re

JSONRPC = "2.0"

DEFAULT_MIME = "text/plain"

# Плейсхолдер в шаблоне URI: {id}, {tag}. Внутрь "/" не пускаем — иначе
# notes://{id} проглотит notes://a/b/c и шаблон перестанет что-либо значить.
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def pick_primitive(spec):
    """Каким примитивом выставлять возможность: tool, resource или prompt.

    spec — {"mutates": bool, "attachable": bool, "workflow": bool}.

    pick_primitive({"attachable": True})   ->  "resource"
    pick_primitive({"mutates": True})      ->  "tool"
    pick_primitive({"workflow": True})     ->  "prompt"
    pick_primitive({})                     ->  "tool"

    Правило урока словами: если пользователь хочет ПЕРЕИСПОЛЬЗОВАТЬ целый
    многошаговый сценарий — это промпт; если что-то меняется или считается
    по запросу модели — инструмент; если пользователь хочет ПРИЛОЖИТЬ
    данные к разговору — ресурс.

    Приоритет именно такой: workflow > mutates > attachable. Изменяющее
    действие нельзя выставить ресурсом, даже если его результат приятно
    читать: ресурсы по определению только на чтение.

    Ничего не указано — инструмент: это безопасный ответ по умолчанию,
    модель хотя бы сможет им воспользоваться.
    """
    if spec.get("workflow"):
        return "prompt"
    if spec.get("mutates"):
        return "tool"
    if spec.get("attachable"):
        return "resource"
    return "tool"


def resource_entry(uri, name, mime_type=DEFAULT_MIME, description=None):
    """Элемент манифеста resources/list.

    resource_entry("notes://note-1", "MCP overview")
        ->  {"uri": "notes://note-1", "name": "MCP overview",
             "mimeType": "text/plain"}
    resource_entry("file:///a.md", "A", "text/markdown", "Заметка")
        ->  ... плюс "description": "Заметка"
    resource_entry("note-1", "A")  ->  ValueError

    URI обязан нести схему: клиент кеширует и роутит именно по ней, а
    голое "note-1" неотличимо от чужого идентификатора. Схема бывает любая
    — file://, postgres://, notes://, memory:// — но она обязана быть.

    Ловушка: пустое description — не то же самое, что отсутствующее.
    Ключа с None в JSON быть не должно, поле просто опускается.
    """
    if "://" not in uri:
        raise ValueError(f"Resource URI must carry a scheme: {uri}")
    entry = {"uri": uri, "name": name, "mimeType": mime_type}
    if description is not None:
        entry["description"] = description
    return entry


def read_resource(store, uri):
    """Тело ответа resources/read для статического ресурса.

    store — {uri: {"mimeType": ..., "text": "..."}} либо
            {uri: {"mimeType": ..., "data": b"..."}} для двоичного.

    read_resource({"notes://n1": {"text": "hi"}}, "notes://n1")
        ->  {"contents": [{"uri": "notes://n1", "mimeType": "text/plain",
                           "text": "hi"}]}
    read_resource({"img://1": {"mimeType": "image/png", "data": b"\\x89PNG"}}, "img://1")
        ->  {"contents": [{..., "blob": "iVBORw=="}]}   (base64-строка)
    read_resource({}, "notes://ghost")  ->  KeyError

    Двоичное содержимое едет в поле "blob" как base64-СТРОКА: JSON не умеет
    байты. Текстовое — в поле "text".

    Ловушка: в одном content не бывает и "text", и "blob" одновременно.
    Клиент выбирает ветку по наличию ключа, и лишний ключ его ломает.
    """
    entry = store[uri]  # KeyError наружу: диспетчер превратит его в -32602
    content = {"uri": uri, "mimeType": entry.get("mimeType", DEFAULT_MIME)}
    if "data" in entry:
        # base64 отдаём строкой: b64encode возвращает bytes, их не сериализовать
        content["blob"] = base64.b64encode(entry["data"]).decode("ascii")
    else:
        content["text"] = entry["text"]
    return {"contents": [content]}


def expand_template(template, uri):
    """Сопоставить URI с шаблоном. Вернуть параметры или None.

    expand_template("notes://{id}", "notes://note-14")  ->  {"id": "note-14"}
    expand_template("notes://{id}", "files://note-14")  ->  None
    expand_template("notes://{id}", "notes://a/b")      ->  None
    expand_template("db://{table}/{row}", "db://users/7")
        ->  {"table": "users", "row": "7"}

    Ровно это стоит за resourceTemplates: сервер объявляет "notes://{id}",
    хост показывает автодополнение по id, а на чтение приходит конкретный
    URI, который надо разобрать обратно.

    Ловушки:
      * плейсхолдер НЕ должен перепрыгивать через "/" — иначе
        "notes://{id}" совпадёт с чем угодно и шаблон потеряет смысл;
      * литеральная часть шаблона может содержать точки и знаки вопроса,
        которые в регулярном выражении значат совсем другое: экранируй.
    """
    pattern, last = "", 0
    for match in PLACEHOLDER.finditer(template):
        pattern += re.escape(template[last:match.start()])
        # именованная группа: имя параметра приезжает прямо из шаблона
        pattern += f"(?P<{match.group(1)}>[^/]+)"
        last = match.end()
    pattern += re.escape(template[last:])
    found = re.fullmatch(pattern, uri)
    return found.groupdict() if found else None


def resolve(store, templates, uri):
    """Прочитать ресурс: сначала статический, потом вычисляемый по шаблону.

    templates — список {"uriTemplate": "notes://recent/{n}", "read": функция}.
    Функция получает словарь параметров и возвращает {"mimeType":..., "text":...}.

    resolve(store, [], "notes://note-1")        ->  как read_resource
    resolve({}, [recent], "notes://recent/5")   ->  вычисленное содержимое
    resolve({}, [], "notes://ghost")            ->  KeyError

    Статическое хранилище проверяется первым: конкретный URI всегда
    важнее шаблона, который его тоже покрывает.

    Динамический ресурс считается на КАЖДОЕ чтение. Именно поэтому
    "notes://recent" честно отдаёт свежую пятёрку, а не то, что было при
    старте сервера. Обратная сторона: такой URI нельзя кешировать по имени.
    """
    if uri in store:
        return read_resource(store, uri)
    for template in templates:
        params = expand_template(template["uriTemplate"], uri)
        if params is None:
            continue
        entry = template["read"](params)
        content = {"uri": uri, "mimeType": entry.get("mimeType", DEFAULT_MIME)}
        content["text"] = entry["text"]
        return {"contents": [content]}
    raise KeyError(uri)


def subscribe(subscriptions, uri, session_id, on=True):
    """Подписать или отписать сессию от ресурса. Вернуть НОВЫЙ словарь.

    subscriptions — {uri: [session_id, ...]}.

    subscribe({}, "notes://n1", "s1")
        ->  {"notes://n1": ["s1"]}
    subscribe({"notes://n1": ["s1"]}, "notes://n1", "s1")
        ->  {"notes://n1": ["s1"]}          (повтор не плодит дублей)
    subscribe({"notes://n1": ["s1"]}, "notes://n1", "s1", on=False)
        ->  {}                              (пустой URI выкидываем целиком)

    Подписки — это состояние сервера на сессию, и оно обязано быть
    ограниченным: отвалившийся клиент должен исчезать из набора, иначе
    словарь растёт вечно. Поэтому URI без единого подписчика удаляется, а
    не остаётся пустым списком.

    Ловушка: возвращаем новый словарь, входной не трогаем. Правка общего
    состояния на месте — самый частый источник гонок в сервере, который
    обслуживает несколько сессий разом.
    """
    updated = {key: list(value) for key, value in subscriptions.items()}
    listeners = updated.get(uri, [])
    if on:
        if session_id not in listeners:
            listeners.append(session_id)
        updated[uri] = listeners
    else:
        if session_id in listeners:
            listeners.remove(session_id)
        if listeners:
            updated[uri] = listeners
        else:
            updated.pop(uri, None)
    return updated


def updated_notifications(subscriptions, uri):
    """Кому и что разослать, когда ресурс изменился.

    Возвращает список пар (session_id, сообщение).

    updated_notifications({"notes://n1": ["s1", "s2"]}, "notes://n1")
        ->  [("s1", {"jsonrpc": "2.0",
                     "method": "notifications/resources/updated",
                     "params": {"uri": "notes://n1"}}), ("s2", ...)]
    updated_notifications({}, "notes://n1")  ->  []

    Никто не подписан — не шлём ничего. Рассылать «на всякий случай» всем
    сессиям значит заставлять их перечитывать то, что им не нужно.

    Ловушка: это НОТИФИКАЦИЯ. Ключа "id" в ней нет, и ответа на неё ждать
    нельзя — клиент просто перечитает ресурс, когда сочтёт нужным.
    """
    return [
        (
            session_id,
            {
                "jsonrpc": JSONRPC,
                "method": "notifications/resources/updated",
                "params": {"uri": uri},
            },
        )
        for session_id in subscriptions.get(uri, [])
    ]


def render_prompt(prompts, name, arguments):
    """Тело ответа prompts/get: подставить аргументы в шаблон сообщений.

    prompts — {name: {"description": ..., "arguments": [{"name", "required"}],
                      "messages": [{"role", "content": {"type", "text"}}]}}

    render_prompt(p, "review_note", {"note_id": "note-14"})
        ->  {"description": "...", "messages": [... "{note_id}" заменён ...]}
    render_prompt(p, "review_note", {})       ->  ValueError (нет note_id)
    render_prompt(p, "no_such_prompt", {})    ->  KeyError

    Промпт — это контракт между «пользователь нажал /review_note» и «в
    модель ушла вот эта пачка сообщений». Поэтому обязательные аргументы
    проверяются ДО подстановки: пустая дырка в системном сообщении хуже
    честной ошибки.

    Ловушки:
      * str.format здесь не годится: в тексте промпта запросто окажется
        JSON с фигурными скобками, и format упадёт на них;
      * шаблон в реестре править нельзя — второй вызов должен снова
        увидеть "{note_id}", а не результат первой подстановки.
    """
    prompt = prompts[name]  # KeyError наружу: неизвестный промпт
    missing = [
        spec["name"]
        for spec in prompt.get("arguments", [])
        if spec.get("required") and spec["name"] not in arguments
    ]
    if missing:
        raise ValueError(f"Missing required prompt arguments: {', '.join(missing)}")

    messages = []
    for message in prompt["messages"]:
        text = message["content"]["text"]
        for key, value in arguments.items():
            # обычный replace вместо format: фигурные скобки в тексте
            # промпта — норма, а format на них падает
            text = text.replace("{" + key + "}", str(value))
        messages.append({
            "role": message["role"],
            "content": {"type": message["content"].get("type", "text"), "text": text},
        })
    return {"description": prompt["description"], "messages": messages}
