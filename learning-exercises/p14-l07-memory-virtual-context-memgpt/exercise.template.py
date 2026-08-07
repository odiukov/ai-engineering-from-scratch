"""
Виртуальный контекст и подкачка памяти (MemGPT)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l07-memory-virtual-context-memgpt
Разбор:  /check-code p14-l07-memory-virtual-context-memgpt
"""


def core_memory_append(core, section, text):
    """Дописать текст в секцию core memory. Вернуть НОВЫЙ словарь секций.

    core_memory_append({}, "user", "name=ava")
        ->  {"user": "name=ava"}
    core_memory_append({"user": "name=ava"}, "user", "city=Berlin")
        ->  {"user": "name=ava city=Berlin"}

    core memory всегда видна модели — она живёт прямо в промпте. Поэтому
    секция копится через пробел, а не через перевод строки: так дешевле по
    токенам и не ломает разметку промпта.

    Ловушка: вход менять нельзя. core одного агента могут читать несколько
    вызовов сразу, и мутация на месте испортит их всех. Возвращай копию.
    """
    raise NotImplementedError


def core_memory_replace(core, section, old, new):
    """Заменить подстроку в секции core memory. Вернуть НОВЫЙ словарь секций.

    core_memory_replace({"user": "city=Berlin"}, "user", "Berlin", "Lisbon")
        ->  {"user": "city=Lisbon"}
    core_memory_replace({"user": "city=Berlin"}, "user", "Paris", "Lisbon")
        ->  ValueError

    Отсутствие old — именно ValueError, а не тихий no-op. Агент считает, что
    он поправил факт о пользователе; молчаливый отказ означает, что в промпте
    останется устаревшее «city=Berlin», и никто об этом не узнает.
    """
    raise NotImplementedError


def append_message(main, role, text):
    """Добавить сообщение в main context; лишнее вытеснить в evicted (FIFO).

    Вернуть НОВЫЙ main context.

    m = {"core": {}, "messages": [], "evicted": [], "max_messages": 2}
    append_message(m, "user", "a")["messages"]           ->  [("user", "a")]
    После трёх добавлений a, b, c при max_messages=2:
        messages  ->  [("user", "b"), ("user", "c")]
        evicted   ->  [("user", "a")]

    Это page out: контекст кончился, самое старое уезжает «на диск». Порядок
    вытеснения — FIFO, старое первым, иначе восстановить диалог по evicted
    будет нельзя.

    Ловушки:
      * evicted накапливается между вызовами, а не перезаписывается;
      * вход менять нельзя — списки внутри тоже копируй.
    """
    raise NotImplementedError


def render_main_context(main):
    """Собрать промпт из main context. Вернуть строку.

    Формат ровно такой (две шапки, отступ в два пробела):

        [core]
          persona: helpful
          user: name=ava
        [messages]
          user: hi
          assistant: hey

    Секции core идут в АЛФАВИТНОМ порядке. Словарь сохраняет порядок вставки,
    и без sorted один и тот же набор фактов давал бы разные промпты — а значит
    промахи мимо prompt cache и невоспроизводимые ответы.

    Вытесненные сообщения сюда НЕ попадают: в этом весь смысл вытеснения.
    Чтобы вернуть их в промпт, нужен page_in.
    """
    raise NotImplementedError


def archival_insert(store, text, tags=(), session_id="s0", turn_id=0):
    """Записать факт во внешнее хранилище. Вернуть (новый store, rid).

    archival_insert([], "ava ships agents")
        ->  ([{"rid": "a001", "text": "ava ships agents", "tags": (),
               "session_id": "s0", "turn_id": 0}], "a001")

    rid выдаётся по порядку: a001, a002, ... Он же — цитата: без него агент
    вспомнит «пользователь просил выкатить X», но не сможет показать, где
    именно это было сказано. session_id и turn_id хранятся ровно для этого.

    store менять нельзя: возвращай новый список.
    """
    raise NotImplementedError


def archival_search(store, query, top_k=3):
    """Найти записи, похожие на запрос. Вернуть список записей, лучшие первыми.

    Похожесть — коэффициент Жаккара по множествам слов в нижнем регистре:
    |пересечение| / |объединение|.

    store = [{"rid": "a001", "text": "tool chains drift after 20 steps", ...},
             {"rid": "a002", "text": "sleep-time compute consolidates memory", ...}]
    archival_search(store, "tool chains drift")  ->  [запись a001]
    archival_search(store, "quantum chromodynamics")  ->  []

    Записи с нулевым пересечением не возвращаются вовсе: пустой ответ честнее,
    чем случайный факт, который агент вставит в промпт как «вспомнил».

    Две ловушки:
      * делить надо на объединение, а не на длину запроса. Иначе длинная
        запись выигрывает просто потому, что в ней много слов;
      * при равном счёте порядок обязан быть детерминированным — сортируй
        вторым ключом по rid, иначе одинаковые запросы дадут разные промпты.
    """
    raise NotImplementedError


def conversation_search(main, query):
    """Найти самое свежее сообщение, где встречается query. (role, text) или None.

    Ищем по ВСЕЙ истории: сначала вытесненное, потом то, что ещё в промпте.

    m = {"messages": [("assistant", "let me check")],
         "evicted": [("user", "my retrieval bot has 12 tools")],
         "max_messages": 1}
    conversation_search(m, "retrieval bot")  ->  ("user", "my retrieval bot has 12 tools")
    conversation_search(m, "kubernetes")     ->  None

    Регистр не важен: пользователь пишет как хочет.

    Ловушка порядка: полная история — это evicted + messages именно в таком
    порядке (вытесненное случилось РАНЬШЕ), и идти по ней надо с конца. Если
    вернуть первое совпадение с начала, агент вспомнит устаревшую версию факта.
    """
    raise NotImplementedError


def page_in(main, records):
    """Вернуть найденные записи в main context одним system-сообщением.

    Вернуть НОВЫЙ main context.

    page_in(m, [])  ->  копия m, ничего не добавлено
    page_in(m, [{"rid": "a001", "text": "12 tools", "session_id": "s1",
                 "turn_id": 4, "tags": ()}])
        ->  в messages добавилось ("system", "recall: a001@s1:4 12 tools")

    Это вторая половина page fault: рантайм сходил в archival, и результат
    вклеивается в промпт как обычное наблюдение. Формат одной записи —
    "<rid>@<session_id>:<turn_id> <text>", несколько записей склеиваются
    через "; ". rid в тексте — это и есть цитата, по ней ответ можно проверить.

    Добавление идёт через append_message, а не мимо него: подкачанный факт
    занимает место в промпте наравне с остальными и может вытеснить старое
    сообщение. Память конечна и после подкачки тоже.
    """
    raise NotImplementedError
