"""Phase 13 Lesson 19 - A2A v1.0 wire-shape harness.

Builds current AgentCard, Message, Part, TaskStatus, Task, and Artifact shapes
by hand, then simulates the JSON-RPC SendMessage operation in memory.

Spec: https://a2a-protocol.org/latest/specification/
Run: python3 main.py
"""

from __future__ import annotations

import base64
import json
import uuid


AGENT_CARD_PATH = "/.well-known/agent-card.json"
WRITER_AGENT_CARD = {
    "name": "writer-agent",
    "description": "Drafts technical summaries and reports from source material.",
    "supportedInterfaces": [
        {
            "url": "https://writer.example.com/a2a",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ],
    "version": "1.0.0",
    "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text/plain", "application/pdf", "application/json"],
    "defaultOutputModes": ["text/markdown"],
    "skills": [
        {
            "id": "draft_report",
            "name": "Draft report",
            "description": "Produce a report from source material.",
            "tags": ["writing", "summarization"],
            "inputModes": ["text/plain", "application/pdf", "application/json"],
            "outputModes": ["text/markdown"],
        }
    ],
}

TASK_STORE: dict[str, dict] = {}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def text_part(text: str, media_type: str = "text/plain") -> dict:
    return {"text": text, "mediaType": media_type}


def data_part(data: object) -> dict:
    return {"data": data, "mediaType": "application/json"}


def message(role: str, parts: list[dict], **links: str) -> dict:
    value = {"messageId": new_id("msg"), "role": role, "parts": parts}
    value.update(links)
    return value


def artifact(artifact_id: str, name: str, text: str) -> dict:
    return {
        "artifactId": artifact_id,
        "name": name,
        "parts": [text_part(text, "text/markdown")],
    }


def finish(task: dict, length: str) -> None:
    text = (
        f"[writer agent] {length} summary: topic identified, key points "
        "extracted, conclusion drafted."
    )
    task["artifacts"] = [artifact(new_id("artifact"), "summary", text)]
    task["status"] = {"state": "TASK_STATE_COMPLETED"}
    print(f"    WRITER  : completed task {task['id']}")


def send_message(request: dict) -> dict:
    """Simulate the A2A v1 JSON-RPC SendMessage operation."""
    incoming = request["message"]
    task_id = incoming.get("taskId")
    if task_id is None:
        task_id = new_id("task")
        context_id = incoming.get("contextId", new_id("context"))
        task = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "TASK_STATE_WORKING"},
            "history": [incoming],
            "artifacts": [],
        }
        TASK_STORE[task_id] = task
    else:
        task = TASK_STORE[task_id]
        task["history"].append(incoming)
        task["status"] = {"state": "TASK_STATE_WORKING"}

    supplied = [part["data"] for part in incoming["parts"] if "data" in part]
    if not supplied or "targetLength" not in supplied[0]:
        task["status"] = {
            "state": "TASK_STATE_INPUT_REQUIRED",
            "message": message(
                "ROLE_AGENT",
                [text_part("Please provide targetLength as a data Part.")],
                taskId=task["id"],
                contextId=task["contextId"],
            ),
        }
    else:
        finish(task, str(supplied[0]["targetLength"]))
    return task


def research_agent_flow() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 19 - A2A V1 SENDMESSAGE")
    print("=" * 72)
    print(f"\nGET {AGENT_CARD_PATH}")
    print(
        json.dumps(
            {
                "name": WRITER_AGENT_CARD["name"],
                "supportedInterfaces": WRITER_AGENT_CARD["supportedInterfaces"],
                "skills": WRITER_AGENT_CARD["skills"],
            },
            indent=2,
        )
    )

    fake_pdf = base64.b64encode(b"fake-pdf").decode()
    initial = message(
        "ROLE_USER",
        [
            text_part("Summarize the attached paper."),
            {
                "raw": fake_pdf,
                "filename": "paper.pdf",
                "mediaType": "application/pdf",
            },
        ],
    )
    task = send_message({"message": initial})
    print(f"\nSendMessage -> {task['status']['state']}")

    if task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED":
        followup = message(
            "ROLE_USER",
            [data_part({"targetLength": "3 paragraphs"})],
            taskId=task["id"],
            contextId=task["contextId"],
        )
        task = send_message({"message": followup})

    print("\nTask response:")
    print(json.dumps(task, indent=2))
    print(f"\nartifactId: {task['artifacts'][0]['artifactId']}")


if __name__ == "__main__":
    research_agent_flow()
