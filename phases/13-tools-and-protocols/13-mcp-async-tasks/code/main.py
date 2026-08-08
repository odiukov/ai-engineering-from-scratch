"""Phase 13 Lesson 13 - MCP Tasks in protocol revision 2025-11-25.

Implements the wire shapes from docs/en.md with only the Python stdlib:
params.task augmentation, CreateTaskResult, tasks/get, blocking tasks/result,
tasks/cancel, ISO timestamps, and a durable filesystem-backed task store.

Spec: https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks
Run: python3 main.py
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


STORE_DIR = Path("/tmp/lesson-13-tasks")
STORE_DIR.mkdir(parents=True, exist_ok=True)
TERMINAL = {"completed", "failed", "cancelled"}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class InvalidParams(ValueError):
    code = -32602


@dataclass
class Task:
    task_id: str
    status: str = "working"
    status_message: str | None = None
    created_at: str = field(default_factory=iso_now)
    last_updated_at: str = field(default_factory=iso_now)
    ttl: int | None = 900_000
    poll_interval: int = 250
    result: dict | None = None
    cancel_requested: bool = False

    def protocol_shape(self) -> dict:
        task = {
            "taskId": self.task_id,
            "status": self.status,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at,
            "ttl": self.ttl,
            "pollInterval": self.poll_interval,
        }
        if self.status_message is not None:
            task["statusMessage"] = self.status_message
        return task

    def persist(self) -> None:
        (STORE_DIR / f"{self.task_id}.json").write_text(
            json.dumps(asdict(self), indent=2)
        )

    @classmethod
    def load(cls, task_id: str) -> "Task | None":
        path = STORE_DIR / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            # A previous lesson revision used a different local-only record
            # shape. Ignore incompatible demo data instead of hanging startup.
            return None


class TaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.condition = threading.Condition()
        self.crash_recover()

    def crash_recover(self) -> None:
        for path in STORE_DIR.glob("*.json"):
            task = Task.load(path.stem)
            if task is None:
                continue
            if task.status in {"working", "input_required"}:
                task.status = "failed"
                task.status_message = "Worker lost during restart"
                task.last_updated_at = iso_now()
                task.result = {
                    "code": -32000,
                    "message": "CRASH_RECOVERY",
                }
                task.persist()
            self.tasks[task.task_id] = task

    def create(self, ttl: int | None) -> Task:
        task = Task(task_id=f"tsk_{uuid.uuid4().hex[:12]}", ttl=ttl)
        with self.condition:
            task.persist()
            self.tasks[task.task_id] = task
            self.condition.notify_all()
        return task

    def update(self, task_id: str, **changes: object) -> Task:
        with self.condition:
            task = self.tasks[task_id]
            for key, value in changes.items():
                setattr(task, key, value)
            task.last_updated_at = iso_now()
            task.persist()
            self.condition.notify_all()
            return task


STORE = TaskStore()


def worker_generate_report(task: Task, size: str) -> None:
    """Simulate a long tool while respecting terminal cancellation."""
    try:
        for _ in range(12):
            time.sleep(0.05)
            if task.cancel_requested or task.status == "cancelled":
                return
        result = {
            "content": [
                {"type": "text", "text": f"Report size={size} with 30 sections"}
            ],
            "isError": False,
        }
        STORE.update(task.task_id, status="completed", result=result)
    except Exception as exc:  # noqa: BLE001 - preserve the underlying RPC error
        STORE.update(
            task.task_id,
            status="failed",
            status_message=str(exc),
            result={"code": -32000, "message": str(exc)},
        )


def tools_call(params: dict) -> dict:
    """Handle the params object of a tools/call request."""
    name = params.get("name")
    if name != "generate_report":
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"unknown tool {name}"}],
        }
    if "task" not in params:
        time.sleep(0.1)
        return {
            "isError": False,
            "content": [{"type": "text", "text": "Report generated synchronously"}],
        }

    requested_ttl = params["task"].get("ttl", 900_000)
    task = STORE.create(ttl=requested_ttl)
    size = params.get("arguments", {}).get("size", "medium")
    threading.Thread(
        target=worker_generate_report, args=(task, size), daemon=True
    ).start()
    return {"task": task.protocol_shape()}


def tasks_get(task_id: str) -> dict:
    task = STORE.tasks.get(task_id)
    if task is None:
        raise KeyError(task_id)
    return task.protocol_shape()


def tasks_result(task_id: str) -> dict:
    """Block until terminal, then return the underlying CallToolResult/error."""
    with STORE.condition:
        while True:
            task = STORE.tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status in TERMINAL:
                return task.result or {
                    "code": -32000,
                    "message": f"task ended as {task.status}",
                }
            STORE.condition.wait(timeout=task.poll_interval / 1000)


def tasks_cancel(task_id: str) -> dict:
    task = STORE.tasks.get(task_id)
    if task is None:
        raise KeyError(task_id)
    if task.status in TERMINAL:
        raise InvalidParams("cannot cancel a task in a terminal status")
    cancelled_result = {
        "isError": True,
        "content": [{"type": "text", "text": "Task cancelled"}],
    }
    task = STORE.update(
        task_id,
        cancel_requested=True,
        status="cancelled",
        status_message="Cancelled by requestor",
        result=cancelled_result,
    )
    return task.protocol_shape()


def demo() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 13 - MCP TASKS (2025-11-25)")
    print("=" * 72)

    created = tools_call(
        {
            "name": "generate_report",
            "arguments": {"size": "large"},
            "task": {"ttl": 900_000},
        }
    )
    task_id = created["task"]["taskId"]
    print(f"\nCreateTaskResult: {json.dumps(created, indent=2)}")

    print("\nPolling tasks/get:")
    while True:
        task = tasks_get(task_id)
        print(f"  {task['taskId']} -> {task['status']}")
        if task["status"] in TERMINAL:
            break
        time.sleep(task["pollInterval"] / 1000)

    print("\ntasks/result (underlying CallToolResult):")
    print(json.dumps(tasks_result(task_id), indent=2))

    created2 = tools_call(
        {
            "name": "generate_report",
            "arguments": {"size": "small"},
            "task": {"ttl": 900_000},
        }
    )
    task_id2 = created2["task"]["taskId"]
    print("\ntasks/cancel:")
    print(json.dumps(tasks_cancel(task_id2), indent=2))
    try:
        tasks_cancel(task_id2)
    except InvalidParams as exc:
        print(f"  repeat cancellation -> {exc.code} Invalid params")


if __name__ == "__main__":
    demo()
