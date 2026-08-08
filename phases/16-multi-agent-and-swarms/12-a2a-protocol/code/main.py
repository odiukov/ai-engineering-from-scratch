"""A2A v1 educational HTTP+JSON server and client.
Lesson: phases/16-multi-agent-and-swarms/12-a2a-protocol/docs/en.md
Spec: https://a2a-protocol.org/latest/specification/
Flow: GET /.well-known/agent-card.json; POST /message:send;
then GET /tasks/{id} for the current server-created Task snapshot.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from uuid import uuid4


AGENT_CARD = {
    "name": "code-review-agent",
    "description": "Reviews small Python snippets",
    "supportedInterfaces": [{
        "url": "http://localhost:8765",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    }],
    "version": "0.1.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["text/plain", "application/json"],
    "skills": [{
        "id": "review-python",
        "name": "Review Python",
        "description": "Finds basic structural issues in Python code",
        "tags": ["review", "python"],
    }],
}


class TaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_from_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Server assigns Task.id; a new client message cannot choose it."""
        if message.get("taskId"):
            raise ValueError("taskId may only reference an existing task")
        tid = str(uuid4())
        content = message["parts"][0]["data"]
        task = {
            "id": tid,
            "contextId": message.get("contextId", f"ctx-{tid}"),
            "status": {"state": "TASK_STATE_SUBMITTED"},
            "artifacts": [],
            "history": [copy.deepcopy(message)],
        }
        with self._lock:
            self.tasks[tid] = task
        threading.Thread(
            target=self._run,
            args=(tid, content.get("skill", ""), content.get("payload", {})),
            daemon=True,
        ).start()
        return copy.deepcopy(task)

    def _run(self, tid: str, skill: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.tasks[tid]["status"] = {"state": "TASK_STATE_WORKING"}
        time.sleep(0.2)
        with self._lock:
            task = self.tasks[tid]
            if skill == "review-python":
                code = payload.get("code", "")
                issues = []
                if "return" not in code:
                    issues.append("no return statement")
                if "def " not in code:
                    issues.append("no function definition")
                task["artifacts"] = [{
                    "artifactId": f"artifact-{tid}",
                    "name": "Review result",
                    "parts": [{
                        "data": {"issues": issues, "lines": code.count("\n") + 1},
                        "mediaType": "application/json",
                    }],
                }]
                task["status"] = {"state": "TASK_STATE_COMPLETED"}
            else:
                task["status"] = {"state": "TASK_STATE_FAILED"}
                task["artifacts"] = [{
                    "artifactId": f"artifact-{tid}-error",
                    "parts": [{
                        "text": f"unknown skill {skill!r}",
                        "mediaType": "text/plain",
                    }],
                }]

    def get(self, tid: str) -> dict[str, Any] | None:
        with self._lock:
            task = self.tasks.get(tid)
            return copy.deepcopy(task) if task is not None else None


STORE = TaskStore()


class A2AHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/a2a+json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/.well-known/agent-card.json":
            self._send_json(200, AGENT_CARD)
            return
        if self.path.startswith("/tasks/"):
            tid = self.path.split("/tasks/", 1)[1]
            task = STORE.get(tid)
            if task is None:
                self._send_json(404, {"error": {"message": "task not found"}})
                return
            self._send_json(200, {"task": task})
            return
        self._send_json(404, {"error": {"message": "route not found"}})

    def do_POST(self) -> None:
        if self.path == "/message:send":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                task = STORE.create_from_message(body["message"])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                self._send_json(400, {"error": {"message": str(exc)}})
                return
            self._send_json(200, {"task": task})
            return
        self._send_json(404, {"error": {"message": "route not found"}})


def run_server() -> HTTPServer:
    server = HTTPServer(("localhost", 8765), A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def http_json(method: str, url: str, body: Any = None) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/a2a+json")
    req.add_header("A2A-Version", "1.0")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_client() -> None:
    print("\n[1] discovery: GET /.well-known/agent-card.json")
    card = http_json("GET", "http://localhost:8765/.well-known/agent-card.json")
    interface = card["supportedInterfaces"][0]
    print(f"    name={card['name']}, skills={[s['id'] for s in card['skills']]}")

    print("\n[2] send message: POST /message:send")
    request = {
        "message": {
            "messageId": str(uuid4()),
            "role": "ROLE_USER",
            "parts": [{
                "data": {
                    "skill": "review-python",
                    "payload": {"code": "x = 1\nprint(x)\n"},
                },
                "mediaType": "application/json",
            }],
        },
        "configuration": {"returnImmediately": True},
    }
    response = http_json("POST", interface["url"] + "/message:send", request)
    task = response["task"]
    tid = task["id"]
    print(f"    server task id={tid}, state={task['status']['state']}")

    print("\n[3] poll until completed")
    for attempt in range(10):
        task = http_json("GET", f"{interface['url']}/tasks/{tid}")["task"]
        state = task["status"]["state"]
        print(f"    attempt {attempt + 1}: state={state}")
        if state in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED"):
            print(f"    artifacts: {task['artifacts']}")
            break
        time.sleep(0.1)


def main() -> None:
    print("A2A v1 educational protocol demo")
    print("-" * 36)
    server = run_server()
    time.sleep(0.1)
    try:
        run_client()
    finally:
        server.shutdown()
        server.server_close()
    print("\nKey insight: clients create Message IDs; servers create new Task IDs.")


if __name__ == "__main__":
    main()
