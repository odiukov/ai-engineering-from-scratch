"""
Хендоффы и рутины

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l11-handoffs-and-routines
Разбор:  /check-code p16-l11-handoffs-and-routines
"""

CONTEXT_POLICIES = ("full", "last_n", "summary")
DEFAULT_MAX_HOPS = 10
DEFAULT_RING = 4


class HandoffLoopError(Exception):
    """Агенты передают диалог по кругу и не могут остановиться.

    Свой класс, а не RuntimeError: NotImplementedError наследуется от
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """
    pass


def make_agent(name, instructions, handoffs=()):
    """Рутина в терминах OpenAI Swarm: имя, системный промпт и список хендоффов.

    make_agent("triage", "Route the user.", ["refund"])
        ->  {'name': 'triage', 'instructions': 'Route the user.',
             'handoffs': ('refund',)}

    handoffs — имена агентов, которым этот агент имеет право отдать диалог.
    В реальном Swarm это функции-инструменты, возвращающие Agent; здесь —
    просто белый список имён, потому что важна не форма, а право.

    Ловушка: агент в собственном списке хендоффов — это готовая петля.
    Лови сразу ValueError, а не через десять переходов.
    """
    raise NotImplementedError


def can_handoff(agent, target):
    """Имеет ли агент право передать диалог этому получателю.

    can_handoff(make_agent("triage", "...", ["refund"]), "refund")  ->  True
    can_handoff(make_agent("triage", "...", ["refund"]), "billing")  ->  False

    Это guardrail из чек-листа урока: без него prompt injection уводит
    диалог к агенту с другими правами на инструменты.
    """
    raise NotImplementedError


def resolve_target(agents, target, fallback=None):
    """Существующий получатель: сам target, иначе fallback.

    resolve_target({"a": ..., "safe": ...}, "a")                 ->  'a'
    resolve_target({"a": ..., "safe": ...}, "ghost", "safe")     ->  'safe'

    Модель выдумала имя агента — это норма жизни, а не аварийная ситуация.
    Но если и запасного нет, дальше идти некуда: ValueError.
    """
    raise NotImplementedError


def context_transfer(history, policy="full", keep=2):
    """Что из истории уезжает вместе с диалогом к новому владельцу.

    "full"    — всё, дорого, но ничего не теряется;
    "last_n"  — хвост из keep сообщений;
    "summary" — одна пара ("summary", "...") вместо всей истории.

    context_transfer([("user", "a"), ("bot", "b")], "last_n", 1)  ->  [('bot', 'b')]
    context_transfer([("user", "a")], "summary")
        ->  [('summary', 'handoff summary of 1 message(s)')]

    Ловушка: возвращай НОВЫЙ список. Отдав тот же объект, ты связал старого
    и нового владельца общей изменяемой историей — и «переданное состояние»
    начнёт меняться задним числом.
    """
    raise NotImplementedError


def is_ping_pong(trace, ring=DEFAULT_RING):
    """Пинг-понг: последние ring переходов ходят между ровно двумя агентами.

    is_ping_pong(["a", "b", "a", "b"])       ->  True
    is_ping_pong(["a", "b", "c", "a"])       ->  False
    is_ping_pong(["a", "b", "a"])            ->  False   (окно ещё не набралось)

    Ring-check из чек-листа урока: смотрим только хвост, потому что
    легальный путь triage -> refund -> triage бывает и он не петля.

    Ловушка: «два разных имени в окне» — недостаточное условие. Нужна ещё
    и строгая чередуемость, иначе a,a,b,b тоже сойдёт за пинг-понг.
    """
    raise NotImplementedError


def route(agent, message, rules):
    """Кому текущий агент отдаёт диалог. None — отвечает сам.

    rules — dict {имя агента: последовательность пар (ключевое слово, получатель)}.
    Правила проверяются по порядку, побеждает первое совпадение.

    route(triage, "I want a refund", {"triage": [("refund", "refunds")]})
        ->  'refunds'
    route(triage, "hello", {"triage": [("refund", "refunds")]})
        ->  None

    Это заглушка вместо LLM tool call: настоящий Swarm получает имя
    получателя из инструмента, который вернул Agent.

    Ловушка: право на передачу проверяется ДО, а не после. Агент, которому
    хендофф не разрешён, обязан получить ValueError, иначе белый список
    ничего не охраняет.
    """
    raise NotImplementedError


def run_conversation(
    agents,
    rules,
    start,
    messages,
    max_hops=DEFAULT_MAX_HOPS,
    policy="full",
    keep=2,
    fallback=None,
):
    """Диалог с передачей владения. Возвращает состояние после всех сообщений.

    Состояние: {"active": имя владельца, "history": [(кто, текст), ...],
                "trace": [имена владельцев по порядку]}

    Владение переходит внутри одного сообщения столько раз, сколько скажут
    правила: triage может отдать диалог refund, тот — billing и так далее.
    История переезжает вместе с владением по политике policy — в этом и
    смысл: состояние не теряется, но и не тащится целиком без нужды.

    Бросает HandoffLoopError, когда переходы зациклились: либо пинг-понг
    между двумя агентами, либо больше max_hops переходов на одно сообщение.
    Именно бросает, а не крутится молча — «висит» это худший из отказов.
    """
    raise NotImplementedError


def handoff_stats(state):
    """Сводка по трассе владения: сколько переходов и кто сколько владел.

    handoff_stats({"trace": ["triage", "refund", "triage"]})
        ->  {'hops': 2, 'distinct': 2, 'turns': {'triage': 2, 'refund': 1}}

    hops на единицу меньше длины трассы: стартовый агент не «перешёл» к
    себе. Это та самая цифра, которую пишут в трейс-событие хендоффа.
    """
    raise NotImplementedError
