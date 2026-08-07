"""
Акторы: почтовые ящики и порядок доставки — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок про AutoGen v0.4, но здесь нет ни autogen, ни сети, ни LLM. Мы собираем
руками то, что фреймворк даёт одной строкой: у актора приватное состояние и
свой почтовый ящик, сообщения — единственный способ взаимодействия, а рантайм
отделяет доставку от обработки и ловит падение одного актора, не роняя
остальных.

Соответствие настоящему API:

    register            <-  AgentRuntime.register() / register_factory()
    send                <-  AgentRuntime.send_message(), кладёт в inbox и
                            возвращается сразу, не дожидаясь обработчика
    publish             <-  AgentRuntime.publish_message(topic_id=...)
    deliver_one         <-  один шаг цикла доставки: Agent.on_message()
    run_round_robin     <-  RoundRobinGroupChat из AgentChat
    run_selector        <-  SelectorGroupChat из AgentChat
    dead_letter_report  <-  разбор dead-letter queue

Рантайм — обычный словарь:

    {"actors": {"reviewer": {"handler": fn, "state": {...}, "inbox": [...]}},
     "dead_letters": [(message, reason)],
     "counter": 3}

Сообщение — тоже словарь: sender, recipient, topic, body, mid.
Обработчик — функция handler(state, message, runtime): состояние своё, чужое
недоступно, ответить можно только через send.
"""

def new_runtime():
    """Пустой рантайм: акторов нет, dead-letter queue пуста, счётчик на нуле.

    new_runtime()  ->  {"actors": {}, "dead_letters": [], "counter": 0}

    Счётчик выдаёт message id: сквозная нумерация с единицы. Она нужна не
    для красоты — по ней потом видно, в каком порядке сообщения РОДИЛИСЬ,
    даже если обработаны они были вперемешку.
    """
    return {"actors": {}, "dead_letters": [], "counter": 0}


def register(runtime, name, handler, state=None):
    """Завести актора: приватное состояние + пустой почтовый ящик.

    register(rt, "reviewer", fn)              ->  rt
    rt["actors"]["reviewer"]["inbox"]         ->  []

    state=None означает пустой словарь, СВОЙ у каждого актора: общий словарь
    по умолчанию превратил бы приватное состояние в разделяемую память, а
    это ровно то, чего актор-модель не допускает.

    Повторная регистрация имени — ValueError: молча затереть живого актора
    вместе с его непрочитанной почтой хуже, чем упасть.
    """
    if name in runtime["actors"]:
        raise ValueError(f"actor already registered: {name!r}")
    runtime["actors"][name] = {
        "handler": handler,
        "state": {} if state is None else state,
        "inbox": [],
    }
    return runtime


def send(runtime, sender, recipient, topic, body):
    """Положить сообщение в ящик получателя и СРАЗУ вернуться. Вернуть mid.

    send(rt, "checklist", "reviewer", "review", "def f(): ...")  ->  1

    Обработчик здесь не вызывается — в этом вся разница с синхронным
    agent_a.chat(agent_b). Отправитель не блокируется, доставку сделает
    рантайм отдельным шагом.

    Неизвестный получатель — не исключение, а dead letter: падать из-за
    чужого опечатанного имени отправитель не должен.
    """
    runtime["counter"] += 1
    message = {"sender": sender, "recipient": recipient, "topic": topic,
               "body": body, "mid": runtime["counter"]}
    actor = runtime["actors"].get(recipient)
    if actor is None:
        runtime["dead_letters"].append((message, f"no actor {recipient!r}"))
    else:
        actor["inbox"].append(message)
    return message["mid"]


def publish(runtime, sender, topic, subscribers, body):
    """Разослать одно событие всем подписчикам топика. Вернуть список mid.

    publish(rt, "__user__", "review", ["a", "b"], {"code": "x"})  ->  [1, 2]

    Каждому подписчику уходит СВОЙ конверт со своим mid, в порядке списка
    подписчиков. Неизвестный подписчик уезжает в dead letters, остальные
    получают сообщение как обычно.

    Соответствует publish_message с TopicId: один вызов, много ящиков.
    """
    return [send(runtime, sender, name, topic, body) for name in subscribers]


def deliver_one(runtime, name):
    """Обработать ОДНО сообщение из головы ящика актора. Вернуть статус.

    "handled"      — обработчик отработал;
    "dead_letter"  — обработчик бросил исключение, сообщение припарковано;
    "idle"         — ящик пуст, делать нечего.

    Порядок FIFO: берём голову, а не хвост. Ящик — очередь, а не стек.

    Исключение из обработчика ловим здесь: падение одного актора не должно
    ронять рантайм и не должно съедать остальную его почту. Причину пишем
    как "TypeError: текст" — по ней потом строится отчёт по DLQ.
    """
    actor = runtime["actors"].get(name)
    if actor is None:
        raise KeyError(name)
    if not actor["inbox"]:
        return "idle"
    message = actor["inbox"].pop(0)
    try:
        actor["handler"](actor["state"], message, runtime)
    except Exception as exc:  # noqa: BLE001 — в этом и смысл fault isolation
        runtime["dead_letters"].append((message, f"{type(exc).__name__}: {exc}"))
        return "dead_letter"
    return "handled"


def run_round_robin(runtime, order, rounds, max_deliveries=100):
    """RoundRobinGroupChat: по кругу даём каждому актору обработать одно письмо.

    Вернуть журнал доставок [(имя актора, mid), ...] в порядке обработки.

    Актор с пустым ящиком круг пропускает, а не блокирует остальных. Письма,
    которые обработчик отправил прямо сейчас, разойдутся на следующем круге —
    доставка отделена от обработки.

    max_deliveries — страховка: два актора, которые перекидываются мячом,
    иначе крутили бы цикл вечно.
    """
    log = []
    for _ in range(rounds):
        for name in order:
            if len(log) >= max_deliveries:
                return log
            inbox = runtime["actors"][name]["inbox"]
            if not inbox:
                continue
            mid = inbox[0]["mid"]
            deliver_one(runtime, name)
            log.append((name, mid))
    return log


def run_selector(runtime, selector, max_deliveries=100):
    """SelectorGroupChat: следующего выбирает selector, а не фиксированный круг.

    selector(runtime) -> имя актора или None. None означает "хватит".

    Вернуть журнал доставок [(имя актора, mid), ...].

    Селектор видит рантайм ПОСЛЕ предыдущей доставки, поэтому может
    маршрутизировать по состоянию разговора — в этом и разница с
    round-robin. Выбор актора с пустым ящиком заканчивает прогон: селектор,
    который зовёт того, кому нечего читать, зациклился бы навсегда.
    """
    log = []
    for _ in range(max_deliveries):
        name = selector(runtime)
        if name is None:
            return log
        actor = runtime["actors"].get(name)
        if actor is None or not actor["inbox"]:
            return log
        mid = actor["inbox"][0]["mid"]
        deliver_one(runtime, name)
        log.append((name, mid))
    return log


def dead_letter_report(runtime):
    """Свести dead-letter queue: {причина: сколько раз}.

    dead_letter_report(rt)  ->  {"RuntimeError: boom": 1, "no actor 'x'": 2}

    Пустая очередь — пустой словарь. Группировка именно по причине: сто
    писем с одной причиной — это одна поломка, а не сто.
    """
    report = {}
    for _message, reason in runtime["dead_letters"]:
        report[reason] = report.get(reason, 0) + 1
    return report
