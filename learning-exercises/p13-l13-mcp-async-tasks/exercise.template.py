"""
Async Tasks в MCP 2025-11-25

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
    pass


def choose_task_support(estimated_seconds):
    """Выбрать execution.taskSupport для описания tool в tools/list.

    choose_task_support(0.2)  ->  "forbidden"
    choose_task_support(12)   ->  "optional"
    choose_task_support(180)  ->  "required"

    Быстрее 5 секунд — синхронно, 5..30 — клиент решает, дольше
    30 — только Task. При вызове клиент добавляет params.task,
    а не params._meta.task.
    """
    raise NotImplementedError


def _parse_time(value):
    raise NotImplementedError


def new_task(task_id, ttl_ms, now):
    """Создать durable record с каноничным Task внутри.

    now — ISO 8601, потому что createdAt и lastUpdatedAt на проводе
    обязаны быть ISO-строками. ttl и pollInterval — миллисекунды.

    new_task("tsk_1", 900000, "2025-11-25T10:30:00Z")["task"]["status"]
      ->  "working"
    """
    raise NotImplementedError


def create_task_result(record):
    """Initial response на task-augmented request: {"task": <Task>}."""
    raise NotImplementedError


def is_terminal(state):
    """Терминально ли состояние; input_required не терминально."""
    raise NotImplementedError


def is_expired(record, now):
    """ttl идёт от createdAt; None означает бессрочно."""
    raise NotImplementedError


def advance(record, new_state, now, status_message=None, result=None):
    """Перевести Task и вернуть новую durable запись, не меняя вход.

    result — точный result или JSON-RPC error исходного запроса.
    tasks/result вернёт его без своей обёртки.
    """
    raise NotImplementedError


def tasks_get(store, task_id, now):
    """tasks/get: вернуть полный Task с taskId/status/временами."""
    raise NotImplementedError


def cancel_task(record, now):
    """tasks/cancel: отменить нетерминальную задачу.

    Спецификация не делает этот метод идемпотентным: любое терминальное
    состояние, включая cancelled, даёт JSON-RPC -32602 Invalid params.
    """
    raise NotImplementedError


def tasks_result(store, task_id, now, wait=None):
    """tasks/result: блокироваться до терминального status и вернуть result.

    В чистой учебной функции wait(store, task_id) имитирует ожидание
    condition/event реального сервера. Без wait незавершённый вызов
    поднимает BlockingIOError, а не притворяется HTTP 404.
    """
    raise NotImplementedError


def recover_after_crash(store, now):
    """После crash удалить expired, а in-flight закончить JSON-RPC error."""
    raise NotImplementedError
