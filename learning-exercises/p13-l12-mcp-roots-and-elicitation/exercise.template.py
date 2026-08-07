"""
Roots и elicitation: границы и вопрос пользователю на лету

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l12-mcp-roots-and-elicitation
Разбор:  /check-code p13-l12-mcp-roots-and-elicitation
"""

import posixpath

JSONRPC = "2.0"
ELICIT_METHOD = "elicitation/create"
ELICIT_ACTIONS = ("accept", "decline", "cancel")
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
