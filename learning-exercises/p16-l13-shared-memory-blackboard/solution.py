"""
Общая память и доска объявлений — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def make_entry(writer, topic, value, ts, source=None, cites=(), supersedes=None):
    """Одна запись общей памяти вместе с провенансом.

    make_entry("A", "prices", 4.2, 10, source="page-1")
      ->  {"seq": None, "writer": "A", "topic": "prices", "value": 4.2,
           "ts": 10, "source": "page-1", "cites": (), "supersedes": None}

    Поля:
      writer     — кто написал;
      topic      — тема (по ней подписывается blackboard);
      value      — сам факт (здесь число, чтобы его можно было сверить);
      ts         — когда;
      source     — идентификатор первоисточника, если агент что-то читал;
      cites      — seq записей, на которые эта опирается;
      supersedes — seq записи, которую эта отменяет (append-only: правок на
                   месте не бывает, бывает новая запись поверх старой).

    seq пока None: номер запись получит только при добавлении в лог.
    cites приводи к кортежу — иначе вызывающий сможет дописать в список уже
    после добавления записи, и провенанс перестанет быть неизменяемым.
    """
    return {
        "seq": None,
        "writer": writer,
        "topic": topic,
        "value": value,
        "ts": ts,
        "source": source,
        "cites": tuple(cites),
        "supersedes": supersedes,
    }


def append_entry(pool, entry):
    """Добавить запись в append-only лог. Вернуть присвоенный seq.

    pool = []; append_entry(pool, make_entry("A", "t", 1.0, 0))  ->  0
    следующий вызов на том же пуле                              ->  1

    seq — это просто позиция в логе, поэтому pool[seq] всегда та же запись.

    Три проверки, каждая ловит реальную ошибку:
      * запись с уже проставленным seq добавлять нельзя (ValueError) — иначе
        одна и та же запись окажется в логе дважды под разными номерами;
      * cites и supersedes обязаны указывать на СУЩЕСТВУЮЩИЕ seq (ValueError).
        Ссылка вперёд означала бы цикл в провенансе.
    """
    if entry["seq"] is not None:
        raise ValueError("запись уже в логе")
    for ref in entry["cites"]:
        if not isinstance(ref, int) or not 0 <= ref < len(pool):
            raise ValueError(f"cites ссылается в никуда: {ref}")
    dead = entry["supersedes"]
    if dead is not None and not 0 <= dead < len(pool):
        raise ValueError(f"supersedes ссылается в никуда: {dead}")
    entry["seq"] = len(pool)
    pool.append(entry)
    return entry["seq"]


def subscribe(pool, topics):
    """Проекция доски: только записи по подписанным темам, в порядке seq.

    subscribe(pool, ["prices"])          ->  [запись про prices, ...]
    subscribe(pool, [])                  ->  []

    Полный message pool — это subscribe на все темы сразу. Blackboard
    отличается ровно тем, что агент получает подмножество, и его контекст не
    забивается чужой работой. Экономия видна как len(subscribe) < len(pool).
    """
    wanted = set(topics)
    # pool уже отсортирован по seq: seq присваивается как позиция вставки
    return [e for e in pool if e["topic"] in wanted]


def active_entries(pool):
    """Записи, которые никто не отменил — текущее состояние памяти.

    Если запись 5 пришла с supersedes=2, то запись 2 в активные не попадает,
    но из лога НЕ исчезает: append-only означает, что аудит остаётся полным.

    active_entries([]) -> []
    """
    # один проход собирает все отменённые номера, второй фильтрует: O(n)
    superseded = {e["supersedes"] for e in pool if e["supersedes"] is not None}
    return [e for e in pool if e["seq"] not in superseded]


def provenance_chain(pool, seq):
    """Путь записи к первоисточнику: [seq, ..., seq корня].

    Если запись C цитирует B, а B цитирует A, то provenance_chain даёт
    [C, B, A]. Последний элемент — корень, та самая запись, которая
    единственная в цепочке реально что-то читала (у неё заполнен source).

    provenance_chain(pool, 0)  ->  [0]   (запись без cites — сама себе корень)

    Идём по ПЕРВОЙ ссылке cites: у производной записи главный источник один,
    остальные ссылки — контекст. Зацикливания быть не может: append_entry
    запрещает ссылки вперёд, а значит cites всегда строго убывают.
    """
    chain = [seq]
    while pool[chain[-1]]["cites"]:
        chain.append(pool[chain[-1]]["cites"][0])
    return chain


def spread(pool, readers, topic, ts):
    """Отравление памяти: каждый читатель переписывает активное значение себе.

    spread(pool, ["B", "C"], "research", 20)  ->  [1, 2]  (новые seq)

    Механика ровно та, что в разборе урока: B читает общую память, видит
    число, пишет свой вывод с тем же числом и ссылкой на источник. C читает
    уже двоих и делает то же. Ни один агент не падает, ни один тест не
    краснеет — а галлюцинация к концу цепочки стала «фактом от трёх агентов».

    Читатель берёт последнюю АКТИВНУЮ запись по теме. Если её нет — ValueError.
    """
    new = []
    for i, reader in enumerate(readers):
        visible = [e for e in active_entries(pool) if e["topic"] == topic]
        if not visible:
            raise ValueError(f"нет активной записи по теме {topic}")
        src = visible[-1]
        entry = make_entry(
            reader, topic, src["value"], ts + i + 1, cites=(src["seq"],)
        )
        new.append(append_entry(pool, entry))
    return new


def verify(pool, ground_truth):
    """Верификатор без права записи: seq активных записей, расходящихся с истиной.

    ground_truth — что на самом деле написано в источниках: {source: value}.

    verify(чистый_пул, {"page-1": 4.2})  ->  []
    verify(отравленный, {"page-1": 4.2}) ->  [0, 1, 2]

    Проверка идёт по корню провенанса: у производной записи своего источника
    нет, поэтому сверяем её значение с истиной того источника, из которого
    цепочка выросла. Записи, у корня которых источника нет вообще (агент
    ничего не читал, а рассуждал), пропускаем — сверять не с чем.

    Функция НИЧЕГО не пишет в pool. Это и есть «unwritable verifier»: агент,
    который не может писать в общую память, не может быть ею отравлен.
    """
    flagged = []
    for e in active_entries(pool):
        root = pool[provenance_chain(pool, e["seq"])[-1]]
        if root["source"] is None or root["source"] not in ground_truth:
            continue
        if e["value"] != ground_truth[root["source"]]:
            flagged.append(e["seq"])
    return sorted(flagged)


def correct(pool, flagged, writer, ground_truth, ts):
    """Исправить помеченные записи новыми записями поверх. Вернуть новые seq.

    correct(pool, [0, 1, 2], "verifier-out", {"page-1": 4.2}, 30)  ->  [3, 4, 5]

    Правки на месте нет: на каждую отменяемую запись добавляется новая с
    supersedes=<её seq> и правильным значением из ground_truth. Старая
    остаётся в логе навсегда — по ней потом видно, кто и когда ошибся.

    После вызова verify(pool, ground_truth) обязан вернуть пустой список.
    """
    new = []
    for seq in flagged:
        old = pool[seq]
        root = pool[provenance_chain(pool, seq)[-1]]
        entry = make_entry(
            writer,
            old["topic"],
            ground_truth[root["source"]],
            ts,
            source=root["source"],
            cites=(seq,),
            supersedes=seq,
        )
        new.append(append_entry(pool, entry))
    return new
