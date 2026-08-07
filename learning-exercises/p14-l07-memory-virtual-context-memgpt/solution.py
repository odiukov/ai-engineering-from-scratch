"""
Виртуальный контекст и подкачка памяти (MemGPT) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

MemGPT пересказывает виртуальную память операционной системы: main context —
это RAM (промпт фиксированного размера), archival — это диск (внешнее
хранилище с поиском), а memory tools — это page fault. Здесь мы собираем оба
уровня руками. Соответствие настоящему набору инструментов MemGPT/Letta:

    core_memory_append   <-  core_memory_append(section, text)
    core_memory_replace  <-  core_memory_replace(section, old, new)
    append_message       <-  page out: вытеснение хвоста main context
    render_main_context  <-  сборка промпта из main context
    archival_insert      <-  archival_memory_insert(text)
    archival_search      <-  archival_memory_search(query, top_k)
    conversation_search  <-  conversation_search(query)
    page_in              <-  splice результата обратно в следующий turn

Ни одного вызова LLM и ни одного обращения к сети: main context — обычный
словарь, archival — список словарей. Вся логика управления памятью видна.

Форма main context, она же аргумент почти всех функций:

    {"core": {"persona": "...", "user": "..."},
     "messages": [("user", "hi"), ("assistant", "hey")],
     "evicted": [],
     "max_messages": 3}
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
    updated = dict(core)
    existing = updated.get(section, "")
    # strip на всякий случай: пустая секция не должна давать ведущий пробел
    updated[section] = (existing + " " + text).strip() if existing else text.strip()
    return updated


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
    current = core.get(section, "")
    if old not in current:
        raise ValueError(f"{old!r} not found in core[{section!r}]")
    updated = dict(core)
    updated[section] = current.replace(old, new)
    return updated


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
    messages = list(main["messages"]) + [(role, text)]
    evicted = list(main.get("evicted", ()))
    cap = main["max_messages"]
    # pop(0) — O(n), но n здесь это единицы сообщений, а читаемость дороже
    while len(messages) > cap:
        evicted.append(messages.pop(0))
    updated = dict(main)
    updated["messages"] = messages
    updated["evicted"] = evicted
    return updated


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
    parts = ["[core]"]
    core = main.get("core", {})
    for key in sorted(core):
        parts.append(f"  {key}: {core[key]}")
    parts.append("[messages]")
    for role, text in main["messages"]:
        parts.append(f"  {role}: {text}")
    return "\n".join(parts)


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
    records = list(store)
    rid = f"a{len(records) + 1:03d}"
    records.append({
        "rid": rid,
        "text": text,
        "tags": tuple(tags),
        "session_id": session_id,
        "turn_id": turn_id,
    })
    return records, rid


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
    q_tokens = set(query.lower().split())
    if not q_tokens:
        return []
    scored = []
    for record in store:
        r_tokens = set(record["text"].lower().split())
        overlap = len(q_tokens & r_tokens)
        if overlap == 0:
            continue
        scored.append((overlap / len(q_tokens | r_tokens), record))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["rid"]))
    return [record for _, record in scored[:top_k]]


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
    history = list(main.get("evicted", ())) + list(main["messages"])
    needle = query.lower()
    for role, text in reversed(history):
        if needle in text.lower():
            return (role, text)
    return None


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
    if not records:
        updated = dict(main)
        updated["messages"] = list(main["messages"])
        updated["evicted"] = list(main.get("evicted", ()))
        return updated
    body = "; ".join(
        f"{r['rid']}@{r['session_id']}:{r['turn_id']} {r['text']}" for r in records
    )
    return append_message(main, "system", "recall: " + body)
