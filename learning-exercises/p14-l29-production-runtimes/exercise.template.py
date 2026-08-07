"""
Production runtimes: очередь, событие, cron

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l29-production-runtimes
Разбор:  /check-code p14-l29-production-runtimes
"""

RUNTIME_SHAPES = (
    "request_response",
    "streaming",
    "queue",
    "event",
    "cron",
    "durable",
)
SHORT_TASK_SECONDS = 30


def pick_runtime_shape(seconds, needs_progress=False, event_triggered=False,
                       periodic=False, resume_cost_high=False):
    """Форма runtime по форме задачи, а не по любимому фреймворку.

    pick_runtime_shape(2)                                 ->  "request_response"
    pick_runtime_shape(2, needs_progress=True)            ->  "streaming"
    pick_runtime_shape(600)                               ->  "queue"
    pick_runtime_shape(600, resume_cost_high=True)        ->  "durable"
    pick_runtime_shape(1, periodic=True)                  ->  "cron"

    Порядок проверок, сверху вниз:
      1. запускается по расписанию — "cron";
      2. запускается по внешнему событию — "event";
      3. дольше SHORT_TASK_SECONDS и перезапуск с нуля дорог — "durable"
         (это про checkpointing LangGraph: состояние сохраняется после
         каждого шага);
      4. просто дольше SHORT_TASK_SECONDS — "queue" (Celery, BullMQ,
         SQS + Lambda);
      5. нужен прогресс по мере готовности — "streaming" (SSE/WebSocket);
      6. иначе — "request_response".

    seconds < 0 -> ValueError.

    Ловушка: resume_cost_high у короткой задачи ничего не меняет. Дорогое
    восстановление имеет смысл только там, где есть что восстанавливать —
    урок называет ошибкой и обратное, выбор request-response под
    пятиминутную задачу.
    """
    raise NotImplementedError


def enqueue(pending, job_id, payload):
    """Постановка задачи в очередь. Вернуть НОВЫЙ список, вход не менять.

    Задача — словарь {"id": ..., "payload": ..., "attempt": 0}.

    enqueue([], "j1", "send report")
      ->  [{"id": "j1", "payload": "send report", "attempt": 0}]
    enqueue([{"id": "j1", "payload": "a", "attempt": 0}], "j1", "b")
      ->  [{"id": "j1", "payload": "a", "attempt": 0}]        (дубликат отброшен)

    Идемпотентность по job_id — не украшение. Продюсер, который не получил
    подтверждения, шлёт задачу ещё раз; без dedup один и тот же отчёт уйдёт
    клиенту дважды. Повторная постановка НЕ перезаписывает payload: первым
    пришёл — первым записан.
    """
    raise NotImplementedError


def run_worker(pending, handler, max_attempts=3):
    """Воркер: разбирает очередь FIFO, падения ретраит, безнадёжные — в DLQ.

    Вернуть {"done": [(id, результат)], "dlq": [id], "attempts": {id: число}}.

    handler(payload) возвращает результат или бросает исключение.

    run_worker([{"id": "j1", "payload": "x", "attempt": 0}], str.upper)
      ->  {"done": [("j1", "X")], "dlq": [], "attempts": {"j1": 1}}

    Упавшая задача уходит в КОНЕЦ очереди, а не повторяется на месте: иначе
    один ядовитый payload заблокирует всех, кто стоит за ним. Исчерпала
    max_attempts — уезжает в dead-letter queue. Очередь без DLQ урок
    называет отдельной ошибкой: упавшие задачи просто исчезают.

    max_attempts < 1 -> ValueError. Вход не мутировать.
    """
    raise NotImplementedError


def apply_once(store, job_id, key, delta):
    """Идемпотентная запись: эффект job_id применяется ровно один раз.

    store — {"counters": {ключ: число}, "applied": [id в порядке применения]}.
    Пустой словарь {} тоже годится как начальное состояние.

    apply_once({}, "j1", "emails", 1)
      ->  {"counters": {"emails": 1}, "applied": ["j1"]}
    apply_once(_то_же_состояние_, "j1", "emails", 1)
      ->  {"counters": {"emails": 1}, "applied": ["j1"]}      (повтор не удваивает)

    Очередь даёт гарантию at-least-once: воркер упал после работы, но до
    подтверждения — задача приедет снова. Без этой функции ретрай списывает
    деньги дважды. Вернуть НОВЫЙ словарь, вход не менять.
    """
    raise NotImplementedError


def percentile(values, pct):
    """Персентиль методом ближайшего ранга. Пустой список -> None.

    percentile([1, 2, 3, 4], 50)   ->  2
    percentile([1, 2, 3, 4], 95)   ->  4
    percentile([], 50)             ->  None

    Ближайший ранг: отсортировать, взять элемент под номером
    ceil(pct/100 * n), нумерация с единицы. Никакой интерполяции между
    соседями — по четырём замерам «p95 = 3.85» это выдумка, а не измерение.

    Хвост распределения задержек и есть то, на что жалуются пользователи:
    среднее по очереди почти всегда выглядит прилично.
    """
    raise NotImplementedError


def queue_metrics(records, now):
    """Наблюдаемость очереди: глубина, размер DLQ, распределение задержек.

    records — список {"id", "enqueued_at", "finished_at" (или None), "status"}.
    Время — целые «минуты» модельных часов, now передаётся параметром: воркер,
    который смотрит на настоящие часы, не воспроизводится в тестах.

    Вернуть {"depth", "dlq", "p50", "p95", "oldest_wait"}.

    queue_metrics([{"id": "a", "enqueued_at": 0, "finished_at": 4,
                    "status": "done"}], now=10)
      ->  {"depth": 0, "dlq": 0, "p50": 4, "p95": 4, "oldest_wait": 0}

    depth — сколько задач ещё не завершено, oldest_wait — сколько ждёт самая
    старая из них (now минус enqueued_at). Персентиль считается методом
    ближайшего ранга по завершённым задачам; завершённых нет -> p50 и p95
    равны None, а не нулю: ноль соврал бы про «мгновенно».

    finished_at < enqueued_at или now раньше постановки -> ValueError.
    Часы, идущие назад, — это баг сбора метрик, а не ноль задержки.
    """
    raise NotImplementedError


def resume_from_checkpoint(steps, checkpoint, runner):
    """Durable execution: продолжить с последнего сохранённого шага.

    checkpoint — {"completed": [имена], "results": {имя: значение}}.
    runner(имя) возвращает результат шага или бросает исключение.

    Вернуть {"completed", "results", "failed"}; failed — имя упавшего шага
    или None, если дошли до конца.

    resume_from_checkpoint(["a", "b"], {}, str.upper)
      ->  {"completed": ["a", "b"], "results": {"a": "A", "b": "B"},
           "failed": None}

    Уже выполненные шаги НЕ перезапускаются — в этом весь смысл. Агент,
    упавший на шаге 37, после починки обязан продолжить с 37-го, а не
    заново дёргать тридцать шесть оплаченных вызовов. Так работает
    checkpointing в LangGraph.

    Упал шаг — останавливаемся на нём: следующие шаги могут зависеть от его
    результата. Повторяющиеся имена в steps -> ValueError, иначе непонятно,
    какой из одноимённых шагов уже сделан.
    """
    raise NotImplementedError


def cron_due(schedule, last_run, now):
    """Какие задания пора запустить на тике часов now.

    schedule — {имя: интервал в минутах}, last_run — {имя: минута последнего
    запуска}; отсутствующее имя означает «ещё ни разу не запускалось».

    cron_due({"evals": 60}, {}, now=0)                ->  ["evals"]
    cron_due({"evals": 60}, {"evals": 0}, now=30)     ->  []
    cron_due({"evals": 60}, {"evals": 0}, now=60)     ->  ["evals"]

    Имена возвращаются отсортированными: порядок словаря не должен влиять на
    порядок запуска, иначе ночной прогон перестанет воспроизводиться.

    Пропущенные тики НЕ копятся: задание, не запускавшееся сутки, запустится
    один раз, а не двадцать четыре. Догонять расписание — работа durable
    execution, а не планировщика.

    Интервал <= 0 -> ValueError: такое задание было бы «пора» всегда.
    """
    raise NotImplementedError
