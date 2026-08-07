"""
Долгоживущие фоновые агенты: durable execution — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь руками собран минимальный движок durable execution: workflow, activity,
журнал событий и replay. Это та же схема, что у Temporal, LangGraph
checkpointing, Microsoft Agent Framework и Claude Code Routines, только журнал
— обычный список словарей в памяти, а не PostgreSQL.

Главное свойство, которое проверяют тесты: после сбоя повтор запуска
восстанавливает то же состояние и НЕ выполняет побочный эффект второй раз.

Ни файлов, ни сети, ни LLM. Время — параметр now, а не time.time():
недетерминированность внутри workflow ломает replay, и один из тестов это
показывает.
"""


class WorkflowCrash(Exception):
    """Смоделированный сбой хоста посреди workflow.

    Специально НЕ наследник RuntimeError: NotImplementedError — это
    RuntimeError, и тест pytest.raises(RuntimeError) зеленел бы на пустой
    заготовке, ничего не проверяя.
    """


def activity_key(name, args):
    """Стабильный ключ активности: по нему replay узнаёт уже выполненный шаг.

    activity_key("double", (21,))        ->  'double|(21,)'
    activity_key("fetch", ("hi", 2))     ->  "fetch|('hi', 2)"
    activity_key("fetch", (2, "hi"))     ->  "fetch|(2, 'hi')"

    Ключ обязан быть детерминированным: одинаковые имя и аргументы дают
    одинаковую строку в любом процессе. Порядок аргументов значим — это
    разные вызовы.

    Ловушка: не суй в аргументы set или frozenset. Их repr зависит от
    PYTHONHASHSEED, ключ поедет между запусками, и replay начнёт
    перевыполнять уже сделанные активности.
    """
    # repr кортежа, а не hash: ключ читаемый, и его видно в журнале глазами.
    return f"{name}|{tuple(args)!r}"


def find_completed(log, thread_id, key):
    """Ищет в журнале завершённую активность этого треда. Возвращает событие или None.

    find_completed([], "t-1", "double|(21,)")   ->  None

    Считаются только события со status == "done": у активности, которая
    успела записать "started" и упала, результата нет, и replay обязан
    выполнить её заново.

    thread_id обязателен: два одновременных сеанса делят один журнал, и
    результат чужого треда подставлять нельзя.
    """
    for event in log:
        if (
            event["thread_id"] == thread_id
            and event["key"] == key
            and event["status"] == "done"
        ):
            return event
    return None


def run_activity(log, thread_id, name, args, fn, now=0.0):
    """Выполняет активность один раз, дальше отдаёт результат из журнала.

    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2)  ->  42
    run_activity(log, "t-1", "double", (21,), lambda x: 0)      ->  42

    Второй вызов возвращает 42, хотя fn вернул бы 0: это и есть replay —
    функция не вызывается вовсе. Именно поэтому LLM-вызов, потраченные
    деньги и отправленное письмо не дублируются после сбоя.

    Порядок записи важен: "started" пишется ДО вызова fn, "done" — после.
    Если поменять местами, сбой посередине оставит журнал, по которому не
    понять, случился побочный эффект или нет.

    now — момент времени параметром. Внутри активности нельзя брать
    time.time(): при replay значение будет другим.
    """
    key = activity_key(name, args)
    hit = find_completed(log, thread_id, key)
    if hit is not None:
        return hit["result"]

    log.append(
        {
            "thread_id": thread_id,
            "key": key,
            "name": name,
            "args": list(args),
            "status": "started",
            "result": None,
            "at": now,
        }
    )
    result = fn(*args)
    log.append(
        {
            "thread_id": thread_id,
            "key": key,
            "name": name,
            "args": list(args),
            "status": "done",
            "result": result,
            "at": now,
        }
    )
    return result


def deterministic_value(log, thread_id, name, produce, now=0.0):
    """Регистрирует недетерминированное значение как побочный эффект.

    log = []
    deterministic_value(log, "t-1", "clock", lambda: 100.0)  ->  100.0
    deterministic_value(log, "t-1", "clock", lambda: 999.0)  ->  100.0

    produce — функция без аргументов: часы, генератор случайных чисел,
    uuid. Первый раз она вызывается и результат уезжает в журнал, при
    replay возвращается записанное.

    Это ровно то, что в Temporal называется side-effect registration, а в
    их API — Workflow.now(). Без этого workflow при replay пойдёт по другой
    ветке, и весь журнал станет бесполезен.
    """
    # Активность без аргументов: fn(*()) — это produce(). Отдельной логики
    # не нужно, вся durability уже реализована в run_activity.
    return run_activity(log, thread_id, name, (), produce, now)


def run_workflow(log, thread_id, value, activities, now=0.0, crash_after=None):
    """Гоняет цепочку активностей: выход одной становится входом следующей.

    activities — кортеж пар (имя, функция одного аргумента).
    crash_after=2 бросает WorkflowCrash сразу после второй активности.

    run_workflow([], "t-1", 3, (("inc", lambda x: x + 1),))         ->  4
    run_workflow([], "t-1", 3, (("inc", lambda x: x + 1),) * 2)     ->  5

    Сам workflow обязан быть детерминированным: те же вход и журнал дают ту
    же последовательность решений. Всё недетерминированное прячется внутрь
    активностей и deterministic_value.

    Повторный запуск с тем же журналом и тем же thread_id — это и есть
    восстановление после сбоя: завершённые активности проигрываются, руками
    доделывается только то, что не успело.
    """
    for step, (name, fn) in enumerate(activities, start=1):
        value = run_activity(log, thread_id, name, (value,), fn, now)
        if crash_after == step:
            raise WorkflowCrash(f"host died after activity {step}: {name}")
    return value


def execution_count(log, thread_id=None):
    """Сколько активностей реально выполнялось (события "started").

    execution_count([])  ->  0

    thread_id=None считает по всему журналу, иначе только по одному треду.
    Реплей "started" не пишет, поэтому эта функция и есть счётчик
    настоящих побочных эффектов: именно им доказывают, что после сбоя
    ничего не выполнилось дважды.
    """
    return sum(
        1
        for event in log
        if event["status"] == "started"
        and (thread_id is None or event["thread_id"] == thread_id)
    )


def replay_state(log, thread_id):
    """Чекпоинт треда: словарь {имя активности: её результат}.

    replay_state([], "t-1")  ->  {}

    Если активность с одним именем завершалась несколько раз с разными
    аргументами, побеждает последняя запись — latest-wins, как в чекпоинтах
    LangGraph по thread_id.

    Это то, что видит аудитор: состояние, из которого workflow продолжится
    после перезапуска.
    """
    state = {}
    for event in log:
        if event["thread_id"] == thread_id and event["status"] == "done":
            state[event["name"]] = event["result"]
    return state


def needs_fresh_approval(log, thread_id, now, max_idle):
    """Нужно ли заново спрашивать человека при возобновлении треда.

    needs_fresh_approval([], "t-1", now=10.0, max_idle=5.0)  ->  True

    Правило: если с последнего события треда прошло больше max_idle, старое
    одобрение считается протухшим. Пустой журнал — тоже True: нет чекпоинта,
    значит нечего считать свежим.

    Durable execution позволяет прожить дольше, чем держится надёжность
    агента (METR: «35-минутная деградация»). Поэтому в связке с
    долговечностью всегда идёт свежий HITL на входе — иначе получится
    многочасовой прогон, который никто не контролировал.
    """
    events = [event for event in log if event["thread_id"] == thread_id]
    if not events:
        return True
    return now - events[-1]["at"] > max_idle
