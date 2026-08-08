"""
Продакшен-масштабирование: очереди, чекпоинты, durability — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import copy

# Состояния агента из MegaAgent (arXiv:2408.09955) и разрешённые переходы.
# Константа уровня модуля: тесты импортируют её из exercise.
TRANSITIONS = {
    ("idle", "take"): "processing",
    ("processing", "finish"): "response",
    ("response", "flush"): "idle",
}


class WorkerCrash(Exception):
    """Воркер умер посреди супершага.

    СВОЙ класс, а не RuntimeError. NotImplementedError наследуется от
    RuntimeError, и `pytest.raises(RuntimeError)` зеленел бы на пустой
    заготовке, ничего не проверив.
    """


class InvalidTransition(Exception):
    """Переход, которого нет в машине состояний очереди."""


def append_checkpoint(log, thread_id, step, state):
    """Дописать чекпоинт в журнал. Вернуть добавленную запись.

    Запись: {"thread_id": ..., "step": ..., "state": ...}.

    log = []
    append_checkpoint(log, "t-1", 0, {"counter": 1})
        ->  {"thread_id": "t-1", "step": 0, "state": {"counter": 1}}
    len(log)  ->  1

    Журнал append-only: старые записи не переписываются. Именно поэтому по
    нему можно и восстанавливаться, и проводить аудит постфактум.

    Ловушка: нужен copy.deepcopy, не dict(state). Иначе вложенный список или
    словарь останется общим, следующий супершаг задним числом испортит уже
    записанный чекпоинт, и восстановление приведёт не туда.
    """
    record = {"thread_id": thread_id, "step": step,
              "state": copy.deepcopy(state)}
    log.append(record)
    return record


def last_checkpoint(log, thread_id):
    """Последний чекпоинт данного thread_id или None, если их не было.

    last_checkpoint([], "t-1")  ->  None

    log с шагами 0 и 1 для "t-1" и шагом 0 для "t-2"
    last_checkpoint(log, "t-1")  ->  запись с step == 1

    «Последний» — по номеру шага, а не по позиции в списке. Журнал общий на
    все треды, и записи разных тредов перемешаны.
    """
    best = None
    for record in log:
        if record["thread_id"] != thread_id:
            continue
        if best is None or record["step"] > best["step"]:
            best = record
    return best


def run_thread(steps, thread_id, log, start_state, crash_at=None):
    """Прогнать супершаги треда, дописывая чекпоинт после каждого.

    steps — список функций state -> state. Если в журнале уже есть чекпоинт
    этого треда, начинать со СЛЕДУЮЩЕГО шага, а не с нуля.
    crash_at — номер шага, перед выполнением которого бросить WorkerCrash.

    steps = [lambda s: {"n": s["n"] + 1}] * 3
    run_thread(steps, "t", [], {"n": 0})            ->  {"n": 3}
    run_thread(steps, "t", log, {"n": 0}, crash_at=1)  ->  WorkerCrash

    Падение происходит ДО выполнения шага, а не после: иначе побочный
    эффект уже случился, а чекпоинта нет, и при повторе он случится дважды.

    Ловушка: возобновление идёт с last_checkpoint["step"] + 1. Начать с
    самого step — значит выполнить его второй раз.
    """
    checkpoint = last_checkpoint(log, thread_id)
    if checkpoint is None:
        state, next_step = copy.deepcopy(start_state), 0
    else:
        state = copy.deepcopy(checkpoint["state"])
        next_step = checkpoint["step"] + 1

    for i in range(next_step, len(steps)):
        if crash_at is not None and i == crash_at:
            raise WorkerCrash("воркер умер на супершаге %d треда %s" % (i, thread_id))
        state = steps[i](state)
        append_checkpoint(log, thread_id, i, state)
    return state


def resume_until_done(steps, thread_id, log, start_state, crash_plan=None,
                      max_attempts=10):
    """Гонять run_thread, пока тред не досчитает. Вернуть финальное состояние.

    crash_plan — список номеров шагов, на которых падает первая, вторая,
    третья попытка. Пустой или исчерпанный план означает «больше не падаем».

    resume_until_done(steps, "t", [], {"n": 0}, [1, 2])  ->  {"n": 3}

    Ровно то, что делает рантайм LangGraph: воркер умер, аренда снята,
    другой воркер подхватил тот же thread_id и продолжил с чекпоинта.

    Проверять надо не «не упало», а «результат тот же, что без падений»,
    и что журнал не распух от повторных чекпоинтов одного и того же шага.
    """
    plan = list(crash_plan or [])
    for _ in range(max_attempts):
        crash_at = plan.pop(0) if plan else None
        try:
            return run_thread(steps, thread_id, log, start_state, crash_at)
        except WorkerCrash:
            continue        # чекпоинты уже в журнале, следующая попытка их подхватит
    raise WorkerCrash("тред %s не досчитал за %d попыток" % (thread_id, max_attempts))


def queue_transition(state, event):
    """Машина состояний агента: idle -> processing -> response -> idle.

    queue_transition("idle", "take")         ->  "processing"
    queue_transition("processing", "finish") ->  "response"
    queue_transition("response", "flush")    ->  "idle"
    queue_transition("idle", "finish")       ->  InvalidTransition

    Три состояния из MegaAgent. Смысл явной машины в том, что «агент завис»
    становится наблюдаемым состоянием, а не догадкой по логам.

    Недопустимый переход обязан падать громко. Молчаливое «остаться как
    было» — это тот самый state drift из таксономии MAST.
    """
    if (state, event) not in TRANSITIONS:
        raise InvalidTransition("нет перехода %r по событию %r" % (state, event))
    return TRANSITIONS[(state, event)]


def claim_task(tasks, worker, now, ttl):
    """Взять первую свободную задачу в аренду. Вернуть задачу или None.

    Задача: {"thread_id": ..., "state": ..., "worker": ..., "leased_at": ...,
             "done": ...}. Свободна, если не done и либо worker is None,
    либо аренда просрочена: now - leased_at >= ttl.

    claim_task([{"thread_id": "t", "state": {}, "worker": None,
                 "leased_at": None, "done": False}], "w1", 0, 5)
        ->  та же задача, но с worker="w1" и leased_at=0

    Аренда — это то, что делает падение воркера безопасным: он не успел
    ничего вернуть, но через ttl задачу заберёт другой.

    Ловушка: сравнение именно >=, а не >. При ttl=0 аренда должна
    отбираться сразу, иначе умерший воркер держит задачу вечно.
    """
    for task in tasks:
        if task["done"]:
            continue
        leased = task["worker"] is not None
        if leased and now - task["leased_at"] < ttl:
            continue
        task["worker"] = worker
        task["leased_at"] = now
        return task
    return None


class TransactionalEffectSink:
    """Учебный sink, атомарно связывающий idempotency key и эффект.

    В продакшене это одна транзакция с UNIQUE(key) либо внешний API,
    который сам принимает idempotency key. Один словарь здесь изображает
    эту границу: отдельно сохраняемых seen и effects нет.
    """

    def __init__(self):
        self._committed = {}

    def effects(self):
        return [copy.deepcopy(payload) for payload in self._committed.values()]

    def apply(self, key, payload, crash_after_commit=False):
        if key in self._committed:
            return False
        self._committed[key] = copy.deepcopy(payload)
        if crash_after_commit:
            raise WorkerCrash("воркер умер после атомарного commit эффекта")
        return True


def dedup_effect(sink, key, payload, crash_after_commit=False):
    """Выполнить эффект через атомарный идемпотентный sink.

    Вернуть True, если эффект выполнен сейчас, и False, если это повтор.

    sink = TransactionalEffectSink()
    dedup_effect(sink, "pay-1", {"amount": 10})  ->  True
    dedup_effect(sink, "pay-1", {"amount": 10})  ->  False
    len(sink.effects())  ->  1

    At-least-once доставка плюс атомарный/idempotent sink даёт
    exactly-once effective. Два независимых хранилища seen и effects не
    дают этой гарантии: падение между эффектом и записью seen оставляет окно.

    crash_after_commit моделирует смерть уже после успешного commit. Повтор
    обязан увидеть ключ и не добавить второй эффект.
    """
    return sink.apply(key, payload, crash_after_commit)


def process_queue(tasks, steps, log, crash_plan=None, ttl=5, max_rounds=1000):
    """Прогнать всю очередь пулом воркеров. Вернуть {thread_id: финальное состояние}.

    crash_plan — {thread_id: [шаги падений по попыткам]}. Упавший воркер
    снимает аренду сразу, задачу подхватывает следующий и продолжает с
    последнего чекпоинта.

    process_queue(tasks, steps, [], {"t-1": [1]})
        ->  то же самое, что process_queue(tasks, steps, []) без падений

    Собрано из claim_task и run_thread: очередь раздаёт работу, журнал
    хранит прогресс. Это минимальная durable execution — Temporal и
    LangGraph делают то же самое, только с настоящей базой под журналом.

    Ловушка: crash_plan копируется. Иначе повторный вызов на том же плане
    отработает уже без падений, и тест «падения ничего не меняют» пройдёт
    по причине, не имеющей отношения к чекпоинтам.
    """
    plan = {tid: list(v) for tid, v in (crash_plan or {}).items()}
    results = {}
    for now in range(1, max_rounds + 1):
        task = claim_task(tasks, "w%d" % now, now, ttl)
        if task is None:
            return results
        thread_id = task["thread_id"]
        pending = plan.get(thread_id) or []
        crash_at = pending.pop(0) if pending else None
        try:
            results[thread_id] = run_thread(
                steps, thread_id, log, task["state"], crash_at
            )
            task["done"] = True
        except WorkerCrash:
            task["worker"] = None       # аренда снята, задачу возьмёт другой воркер
            task["leased_at"] = None
    raise WorkerCrash("очередь не сошлась за %d раундов" % max_rounds)
