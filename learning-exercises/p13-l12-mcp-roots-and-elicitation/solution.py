"""
Roots и elicitation: границы и вопрос пользователю на лету — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Два клиентских примитива против двух типовых поломок. Roots чинят
захардкоженный путь: клиент объявляет набор URI, дальше которых серверу
ходить нельзя. Elicitation чинит недосказанный аргумент: сервер тормозит
вызов инструмента и спрашивает пользователя формой. Соответствие
настоящему API:

    normalize_root              <-  разбор Root.uri из mcp.types
    within_roots                <-  проверка границы внутри обработчика сервера
    update_roots                <-  реакция на notifications/roots/list_changed
    elicitation_request         <-  ctx.elicit(message=..., schema=...)
    handle_elicitation_response <-  разбор ElicitResult (accept/decline/cancel)
    disambiguate                <-  типовой сценарий «выбери одну из N»
    delete_note                 <-  инструмент, собранный из всего перечисленного

Ни файловой системы, ни диалогов: хранилище — словарь, а пользователь —
функция ask, которую передают параметром. Поэтому всё детерминировано.
"""

import posixpath

JSONRPC = "2.0"

ELICIT_METHOD = "elicitation/create"

# Три исхода диалога: заполнил, закрыл, отменил весь вызов.
ELICIT_ACTIONS = ("accept", "decline", "cancel")

# Форма elicitation ПЛОСКАЯ: вложенные объекты и массивы v1 не поддерживает.
SCALAR_TYPES = ("string", "number", "integer", "boolean")


def normalize_root(uri):
    """Привести URI корня к каноничному виду.

    normalize_root("file:///Users/alice/Notes/")   ->  "file:///Users/alice/Notes"
    normalize_root("file:///Users/alice/../bob")   ->  "file:///Users/bob"
    normalize_root("/Users/alice/Notes")           ->  ValueError

    Корень — это URI, а не путь: без схемы непонятно, файлы это, база или
    вообще чужое пространство имён.

    Ловушки:
      * хвостовой слэш ничего не значит, но ломает сравнение строк:
        снимаем его здесь, один раз, а не в каждой проверке;
      * ".." внутри пути обязан схлопнуться ДО сравнения границ — иначе
        "Notes/../.." окажется «внутри Notes» по префиксу строки.
    """
    if "://" not in uri:
        raise ValueError(f"Root must be a URI with a scheme: {uri}")
    scheme, _, rest = uri.partition("://")
    if not scheme or not rest:
        raise ValueError(f"Root must be a URI with a scheme: {uri}")
    # posixpath.normpath снимает хвостовой слэш, "." и ".." за один проход
    return f"{scheme}://{posixpath.normpath(rest)}"


def within_roots(uri, roots):
    """Лежит ли URI внутри разрешённого набора корней.

    within_roots("file:///Users/alice/Notes/a.md", ["file:///Users/alice/Notes"])
        ->  True
    within_roots("file:///Users/alice/Notes-evil/a.md", ["file:///Users/alice/Notes"])
        ->  False
    within_roots("file:///Users/alice/Notes/a.md", [])
        ->  False

    Roots — это модель согласия пользователя: он разрешил серверу вот эти
    каталоги и никакие другие. Расширить их сервер не может, только сузить.

    Ловушки:
      * сравнение по префиксу СТРОКИ пропускает "Notes-evil" — граница
        проходит по сегментам пути, а не по символам;
      * пустой список корней означает «ничего нельзя», а не «всё можно»:
        клиент, не объявивший roots, согласия не давал.
    """
    target = normalize_root(uri)
    for root in roots:
        normalized = normalize_root(root)
        if target == normalized:
            return True
        # именно "/" в конце: без него "Notes" совпало бы с "Notes-evil"
        if target.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def update_roots(state, roots):
    """Применить новый набор корней. Вернуть новое состояние.

    state — {"roots": [...], "open": [uri, ...]} (открытые сейчас ресурсы).

    update_roots({"roots": [], "open": ["file:///a/x.md"]},
                 ["file:///a"])
        ->  {"roots": ["file:///a"], "open": ["file:///a/x.md"], "evicted": []}

    Именно это надо делать по notifications/roots/list_changed: пользователь
    отобрал каталог, и открытые в нём хендлы обязаны закрыться. Иначе
    сервер продолжит читать то, к чему доступ уже отозван — формально
    «ничего не нарушив», ведь открывал он законно.

    Ловушка: считать выселенными надо ДО подмены набора, но проверять — по
    НОВОМУ набору. Перепутаешь порядок — не выселишь ничего.
    """
    normalized = [normalize_root(root) for root in roots]
    kept, evicted = [], []
    for uri in state.get("open", []):
        (kept if within_roots(uri, normalized) else evicted).append(uri)
    return {"roots": normalized, "open": kept, "evicted": evicted}


def elicitation_request(request_id, message, schema=None, url=None):
    """Запрос elicitation/create: форма или ссылка. Ровно что-то одно.

    elicitation_request(1, "Pick one",
                        schema={"type": "object",
                                "properties": {"note_id": {"type": "string"}},
                                "required": ["note_id"]})
        ->  {"jsonrpc": "2.0", "id": 1, "method": "elicitation/create",
             "params": {"message": "Pick one", "requestedSchema": {...}}}
    elicitation_request(2, "Sign in", url="https://github.com/login/oauth")
        ->  params с ключом "url" вместо "requestedSchema"

    URL-режим — это SEP-1036 (2025-11-25, экспериментальный): клиент
    открывает ссылку в браузере и ждёт возврата. Годится для OAuth и
    оплаты, где форма бессильна. Шейп ответа ещё двигают между SDK.

    Ловушки:
      * форма ПЛОСКАЯ: вложенный объект или массив объектов v1 не умеет,
        SDK такое отвергают;
      * required обязан ссылаться на существующие properties, иначе форма
        не заполнится никогда;
      * ссылка по http уводит пользователя вводить пароль по открытому
        каналу — только https (или localhost для локального колбэка).
    """
    if (schema is None) == (url is None):
        raise ValueError("Pass exactly one of schema or url")

    params = {"message": message}
    if url is not None:
        if not (url.startswith("https://") or url.startswith("http://localhost")):
            raise ValueError(f"Elicitation URL must be https: {url}")
        params["url"] = url
        return {"jsonrpc": JSONRPC, "id": request_id, "method": ELICIT_METHOD, "params": params}

    if schema.get("type") != "object":
        raise ValueError("Elicitation schema must be an object")
    properties = schema.get("properties") or {}
    for name, spec in properties.items():
        if spec.get("type") not in SCALAR_TYPES:
            raise ValueError(f"Elicitation form must be flat: {name} is {spec.get('type')}")
    unknown = [name for name in schema.get("required", []) if name not in properties]
    if unknown:
        raise ValueError(f"required references unknown properties: {', '.join(unknown)}")
    params["requestedSchema"] = schema
    return {"jsonrpc": JSONRPC, "id": request_id, "method": ELICIT_METHOD, "params": params}


def handle_elicitation_response(response, schema=None):
    """Разобрать ответ пользователя на elicitation. Три ветки, не две.

    handle_elicitation_response({"action": "accept",
                                 "content": {"note_id": "note-14"}}, schema)
        ->  {"status": "accepted", "content": {"note_id": "note-14"}}
    handle_elicitation_response({"action": "decline"})
        ->  {"status": "declined", "content": None}
    handle_elicitation_response({"action": "cancel"})
        ->  {"status": "cancelled", "content": None}

    decline — «не хочу отвечать на этот вопрос», cancel — «прекрати весь
    вызов». Сворачивать их в одно «пользователь сказал нет» нельзя: в
    первом случае инструмент вправе пойти другим путём, во втором обязан
    остановиться.

    Ловушка: содержимое формы приходит от КЛИЕНТА, и доверять ему нельзя
    ровно так же, как аргументам инструмента. Значение вне enum или
    пропущенное required — ValueError, а не «ну ладно».
    """
    action = response.get("action")
    if action not in ELICIT_ACTIONS:
        raise ValueError(f"action must be one of {ELICIT_ACTIONS}")
    if action == "decline":
        return {"status": "declined", "content": None}
    if action == "cancel":
        return {"status": "cancelled", "content": None}

    content = response.get("content") or {}
    if schema is not None:
        properties = schema.get("properties") or {}
        for name in schema.get("required", []):
            if name not in content:
                raise ValueError(f"Missing required field: {name}")
        for name, value in content.items():
            spec = properties.get(name)
            if spec is None:
                raise ValueError(f"Unknown field in elicitation content: {name}")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"{name}: {value!r} is not one of {spec['enum']}")
    return {"status": "accepted", "content": content}


def disambiguate(request_id, matches):
    """Спросить пользователя, какой из совпавших объектов он имел в виду.

    matches — список идентификаторов.

    disambiguate(1, ["note-14"])                     ->  None
    disambiguate(1, ["note-3", "note-7", "note-14"]) ->  запрос elicitation
    disambiguate(1, [])                              ->  ValueError

    Один кандидат — диалога НЕТ. Спрашивать «вы точно про этот
    единственный?» на каждый вызов значит превращать elicitation в
    раздражитель, а урок прямо предупреждает: диалог рвёт разговор, в цикле
    его вызывать нельзя.

    Ноль кандидатов — спрашивать не о чем: это ошибка вызова, а не вопрос
    к пользователю.

    Форма: enum из идентификаторов плюс булев confirm — оба обязательны.
    """
    if not matches:
        raise ValueError("Nothing to disambiguate: no matches")
    if len(matches) == 1:
        return None
    schema = {
        "type": "object",
        "properties": {
            "note_id": {"type": "string", "enum": list(matches)},
            "confirm": {"type": "boolean"},
        },
        "required": ["note_id", "confirm"],
    }
    return elicitation_request(
        request_id, f"{len(matches)} notes match; pick one.", schema=schema
    )


def delete_note(store, title, roots, ask):
    """Удалить заметку по заголовку: границы, вопрос пользователю, удаление.

    store — {note_id: {"title": ..., "uri": ...}}, правится на месте;
    ask   — функция запрос -> ответ пользователя (её играет клиент).

    delete_note(store, "TPS report", roots, ask)
        ->  {"deleted": ["note-14"], "status": "accepted"}
    пользователь закрыл форму
        ->  {"deleted": [], "status": "declined"}
    заметка вне корней
        ->  PermissionError

    Порядок именно такой:
      1. нашли кандидатов по заголовку;
      2. один — удаляем молча, несколько — спрашиваем, ноль — ValueError;
      3. проверяем границу roots для ВЫБРАННОЙ заметки;
      4. удаляем.

    Границу проверяем после выбора, а не до: пользователь мог выбрать
    заметку, лежащую вне разрешённых каталогов, и вот это и есть тот
    случай, ради которого roots существуют.

    Ловушка: confirm=False — это отказ, даже когда action="accept".
    Пользователь честно заполнил форму и честно снял галочку.
    """
    matches = sorted(note_id for note_id, note in store.items() if note["title"] == title)
    request = disambiguate(1, matches)

    if request is None:
        chosen = matches[0]
    else:
        answer = handle_elicitation_response(
            ask(request), request["params"]["requestedSchema"]
        )
        if answer["status"] != "accepted":
            return {"deleted": [], "status": answer["status"]}
        if not answer["content"].get("confirm"):
            return {"deleted": [], "status": "declined"}
        chosen = answer["content"]["note_id"]

    if not within_roots(store[chosen]["uri"], roots):
        raise PermissionError(f"{store[chosen]['uri']} is outside the declared roots")
    del store[chosen]
    return {"deleted": [chosen], "status": "accepted"}
