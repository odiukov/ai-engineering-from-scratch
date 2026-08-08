"""
Async Tasks в MCP 2025-11-25 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Упражнение хранит вместе две сущности: публичный Task и результат
обёрнутого запроса. На провод выходит только Task; tasks/result
возвращает ровно сохранённый результат исходного метода.
"""

import copy
from datetime import datetime, timezone


STATES = ("working", "input_required", "completed", "failed", "cancelled")
TERMINAL_STATES = ("completed", "failed", "cancelled")
ALLOWED_TRANSITIONS = {
    "working": ("input_required", "completed", "failed", "cancelled"),
    "input_required": ("working", "completed", "failed", "cancelled"),
}


class InvalidParams(ValueError):
    """JSON-RPC -32602: параметры запроса недопустимы."""

    code = -32602


def choose_task_support(estimated_seconds):
    """Выбрать execution.taskSupport для описания tool в tools/list.

    choose_task_support(0.2)  ->  "forbidden"
    choose_task_support(12)   ->  "optional"
    choose_task_support(180)  ->  "required"

    Быстрее 5 секунд — синхронно, 5..30 — клиент решает, дольше
    30 — только Task. При вызове клиент добавляет params.task,
    а не params._meta.task.
    """
    if estimated_seconds < 0:
        raise ValueError("estimated_seconds не может быть отрицательным")
    if estimated_seconds < 5:
        return "forbidden"
    if estimated_seconds <= 30:
        return "optional"
    return "required"


def _parse_time(value):
    if not isinstance(value, str):
        raise TypeError("timestamps must be ISO 8601 strings")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def new_task(task_id, ttl_ms, now):
    """Создать durable record с каноничным Task внутри.

    now — ISO 8601, потому что createdAt и lastUpdatedAt на проводе
    обязаны быть ISO-строками. ttl и pollInterval — миллисекунды.

    new_task("tsk_1", 900000, "2025-11-25T10:30:00Z")["task"]["status"]
      ->  "working"
    """
    _parse_time(now)
    if ttl_ms is not None and (not isinstance(ttl_ms, int) or ttl_ms < 0):
        raise ValueError("ttl must be a non-negative integer or None")
    return {
        "task": {
            "taskId": task_id,
            "status": "working",
            "createdAt": now,
            "lastUpdatedAt": now,
            "ttl": ttl_ms,
            "pollInterval": 1000,
        },
        "result": None,
    }


def create_task_result(record):
    """Initial response на task-augmented request: {"task": <Task>}."""
    return {"task": copy.deepcopy(record["task"])}


def is_terminal(state):
    """Терминально ли состояние; input_required не терминально."""
    return state in TERMINAL_STATES


def is_expired(record, now):
    """ttl идёт от createdAt; None означает бессрочно."""
    task = record["task"]
    if task["ttl"] is None:
        return False
    elapsed = (_parse_time(now) - _parse_time(task["createdAt"])).total_seconds() * 1000
    return elapsed >= task["ttl"]


def advance(record, new_state, now, status_message=None, result=None):
    """Перевести Task и вернуть новую durable запись, не меняя вход.

    result — точный result или JSON-RPC error исходного запроса.
    tasks/result вернёт его без своей обёртки.
    """
    if new_state not in STATES:
        raise ValueError(f"неизвестное состояние: {new_state}")
    current = record["task"]["status"]
    if is_terminal(current):
        raise ValueError(f"задача уже терминальна: {current}")
    if new_state not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"переход {current} -> {new_state} запрещён")
    _parse_time(now)
    updated = copy.deepcopy(record)
    task = updated["task"]
    task["status"] = new_state
    task["lastUpdatedAt"] = now
    if status_message is None:
        task.pop("statusMessage", None)
    else:
        task["statusMessage"] = status_message
    if is_terminal(new_state):
        updated["result"] = copy.deepcopy(result)
    return updated


def tasks_get(store, task_id, now):
    """tasks/get: вернуть полный Task с taskId/status/временами."""
    record = store.get(task_id)
    if record is None or is_expired(record, now):
        raise KeyError(task_id)
    return copy.deepcopy(record["task"])


def cancel_task(record, now):
    """tasks/cancel: отменить нетерминальную задачу.

    Спецификация не делает этот метод идемпотентным: любое терминальное
    состояние, включая cancelled, даёт JSON-RPC -32602 Invalid params.
    """
    if is_terminal(record["task"]["status"]):
        raise InvalidParams("cannot cancel a task in a terminal status")
    cancelled_result = {
        "isError": True,
        "content": [{"type": "text", "text": "Task cancelled"}],
    }
    return advance(record, "cancelled", now, "Cancelled by requestor", cancelled_result)


def tasks_result(store, task_id, now, wait=None):
    """tasks/result: блокироваться до терминального status и вернуть result.

    В чистой учебной функции wait(store, task_id) имитирует ожидание
    condition/event реального сервера. Без wait незавершённый вызов
    поднимает BlockingIOError, а не притворяется HTTP 404.
    """
    while True:
        record = store.get(task_id)
        if record is None or is_expired(record, now):
            raise KeyError(task_id)
        if is_terminal(record["task"]["status"]):
            return copy.deepcopy(record["result"])
        if wait is None:
            raise BlockingIOError("tasks/result blocks until the task is terminal")
        wait(store, task_id)


def recover_after_crash(store, now):
    """После crash удалить expired, а in-flight закончить JSON-RPC error."""
    recovered = {}
    for task_id, record in store.items():
        if is_expired(record, now):
            continue
        if is_terminal(record["task"]["status"]):
            recovered[task_id] = copy.deepcopy(record)
        else:
            error = {"code": -32000, "message": "CRASH_RECOVERY"}
            recovered[task_id] = advance(
                record, "failed", now, "Worker lost during restart", error
            )
    return recovered
