"""
Продакшен-масштабирование: очереди, чекпоинты, durability

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l22-production-scaling-queues-checkpoints
Разбор:  /check-code p16-l22-production-scaling-queues-checkpoints
"""

import copy

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
    pass


class InvalidTransition(Exception):
    """Переход, которого нет в машине состояний очереди."""
    pass


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
    raise NotImplementedError


def last_checkpoint(log, thread_id):
    """Последний чекпоинт данного thread_id или None, если их не было.

    last_checkpoint([], "t-1")  ->  None

    log с шагами 0 и 1 для "t-1" и шагом 0 для "t-2"
    last_checkpoint(log, "t-1")  ->  запись с step == 1

    «Последний» — по номеру шага, а не по позиции в списке. Журнал общий на
    все треды, и записи разных тредов перемешаны.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


class TransactionalEffectSink:
    """Учебный sink, атомарно связывающий idempotency key и эффект.

    В продакшене это одна транзакция с UNIQUE(key) либо внешний API,
    который сам принимает idempotency key. Один словарь здесь изображает
    эту границу: отдельно сохраняемых seen и effects нет.
    """

    def __init__(self):
        raise NotImplementedError

    def effects(self):
        raise NotImplementedError

    def apply(self, key, payload, crash_after_commit=False):
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
