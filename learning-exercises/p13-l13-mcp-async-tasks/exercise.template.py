"""
Async Tasks в MCP (SEP-1686)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l13-mcp-async-tasks
Разбор:  /check-code p13-l13-mcp-async-tasks
"""

STATES = ("working", "input_required", "completed", "failed", "cancelled")
TERMINAL_STATES = ("completed", "failed", "cancelled")
ALLOWED_TRANSITIONS = {
    "working": ("input_required", "completed", "failed", "cancelled"),
    "input_required": ("working", "failed", "cancelled"),
}


def choose_task_support(estimated_seconds):
    """Какое значение taskSupport поставить инструменту, зная время его работы.

    choose_task_support(0.2)  ->  "forbidden"
    choose_task_support(12)   ->  "optional"
    choose_task_support(180)  ->  "required"

    Правило урока: быстрее 5 секунд — только синхронный вызов ("forbidden"),
    от 5 до 30 секунд включительно — клиент решает сам ("optional"),
    дольше 30 секунд — task-augmentation обязательна ("required").

    Отрицательное время — ValueError. Это не «мгновенный tool», это ошибка
    в замерах, и молча возвращать "forbidden" тут опаснее, чем упасть.

    Аннотация taskSupport едет в tools/list рядом с описанием инструмента,
    и по ней клиент решает, ставить ли params._meta.task.required.
    """
    raise NotImplementedError


def new_task(task_id, ttl_ms, now):
    """Свежая задача в состоянии working. Время везде в МИЛЛИСЕКУНДАХ.

    new_task("tsk_1", 900000, 1000)
      ->  {"id": "tsk_1", "state": "working", "ttl": 900000,
           "createdAt": 1000, "updatedAt": 1000,
           "progress": 0.0, "result": None, "error": None}

    ttl — обещание сервера хранить состояние. По истечении ttl результат
    выбрасывается, и tasks/result отвечает 404.

    now передаётся параметром, а не берётся из time.time(): иначе тест на
    протухание задачи пришлось бы ждать 15 минут.
    """
    raise NotImplementedError


def is_terminal(state):
    """Терминально ли состояние: из него уже никуда не уйти.

    is_terminal("working")    ->  False
    is_terminal("completed")  ->  True
    is_terminal("cancelled")  ->  True

    input_required НЕ терминально: задача ждёт elicitation и вернётся
    в working.
    """
    raise NotImplementedError


def is_expired(task, now):
    """Истёк ли ttl задачи к моменту now (всё в миллисекундах).

    is_expired(new_task("t", 1000, 0), 999)   ->  False
    is_expired(new_task("t", 1000, 0), 1000)  ->  True

    Граница включительная: ровно в createdAt + ttl задача уже протухла.
    ttl отсчитывается от СОЗДАНИЯ, а не от последнего обновления — иначе
    вечно работающая задача жила бы вечно.
    """
    raise NotImplementedError


def advance(task, new_state, now, progress=None, payload=None):
    """Перевести задачу в new_state. Вернуть НОВЫЙ dict, вход не менять.

    advance(new_task("t", 1000, 0), "completed", 5, payload="report")["result"]
      ->  "report"
    advance(new_task("t", 1000, 0), "input_required", 5)["state"]
      ->  "input_required"

    payload уезжает в "result" при переходе в completed и в "error" при
    переходе в failed; в остальных случаях он игнорируется.

    Три отказа, все через ValueError:
      * new_state вне STATES — опечатка в имени состояния;
      * задача уже терминальна — машина append-only, воскрешать нельзя;
      * переход не разрешён (working -> working, например).

    Почему возвращается копия: состояние задачи лежит в durable store, и
    мутация на месте прячет от тебя момент, когда запись надо сохранить.
    """
    raise NotImplementedError


def cancel_task(task, now):
    """tasks/cancel: идемпотентная отмена. Вернуть НОВЫЙ dict.

    cancel_task(new_task("t", 1000, 0), 5)["state"]  ->  "cancelled"

    Терминальную задачу отмена не трогает и НЕ роняет — второй вызов
    tasks/cancel обязан быть no-op, иначе клиент с ретраями получит
    ошибку на пустом месте.

    Именно этим cancel_task отличается от advance(task, "cancelled", now):
    advance на терминальной задаче бросает ValueError.
    """
    raise NotImplementedError


def tasks_result(store, task_id, now):
    """tasks/result: забрать результат. store — dict вида {id: task}.

    Всегда возвращает dict с ключами status, state, result, error.

    tasks_result({}, "нет", 0)
      ->  {"status": 404, "state": None, "result": None, "error": "unknown_task"}
    tasks_result({"t": new_task("t", 1000, 0)}, "t", 5)
      ->  {"status": 404, "state": "working", "result": None, "error": "not_ready"}

    Четыре случая: задачи нет (404 unknown_task), ttl истёк (404 expired,
    состояние уже выброшено), задача ещё не терминальна (404 not_ready),
    задача терминальна (200 с result или error).

    404 на «ещё не готово» — это не баг, а контракт SEP-1686: клиент
    поллит tasks/status и приходит за результатом только после
    терминального состояния.
    """
    raise NotImplementedError


def recover_after_crash(store, now):
    """Перезапуск сервера: починить store, поднятый с диска. Вернуть НОВЫЙ dict.

    Три правила из урока:
      * задачи с истёкшим ttl выбрасываются из store целиком;
      * незавершённые (working / input_required) — их поток умер вместе
        с процессом, помечаем failed с error "CRASH_RECOVERY";
      * терминальные сохраняются как есть до конца ttl.

    recover_after_crash({"t": new_task("t", 1000, 0)}, 5)["t"]["error"]
      ->  "CRASH_RECOVERY"

    Без этого шага клиент вечно поллит задачу в working, которую уже никто
    не считает.
    """
    raise NotImplementedError
