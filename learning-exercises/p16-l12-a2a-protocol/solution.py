"""
Протокол A2A v1 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import copy
import json
from uuid import uuid4


WELL_KNOWN_PATH = "/.well-known/agent-card.json"
MESSAGE_SEND_PATH = "/message:send"
CARD_REQUIRED = (
    "name", "description", "supportedInterfaces", "version", "capabilities",
    "defaultInputModes", "defaultOutputModes", "skills",
)
SKILL_REQUIRED = ("id", "name", "description", "tags")
MEDIA_TYPES = ("text/plain", "application/json", "image/png", "audio/mpeg", "video/mp4")
TERMINAL_STATES = (
    "TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
)
TRANSITIONS = {
    "TASK_STATE_SUBMITTED": (
        "TASK_STATE_WORKING", "TASK_STATE_CANCELED", "TASK_STATE_FAILED",
        "TASK_STATE_REJECTED",
    ),
    "TASK_STATE_WORKING": (
        "TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_AUTH_REQUIRED",
    ),
    "TASK_STATE_INPUT_REQUIRED": ("TASK_STATE_WORKING", "TASK_STATE_CANCELED"),
    "TASK_STATE_AUTH_REQUIRED": ("TASK_STATE_WORKING", "TASK_STATE_CANCELED"),
    "TASK_STATE_COMPLETED": (),
    "TASK_STATE_FAILED": (),
    "TASK_STATE_CANCELED": (),
    "TASK_STATE_REJECTED": (),
}


class A2AProtocolError(Exception):
    """Нарушение учебного подмножества A2A v1."""


def make_agent_card(name, description, version, skills, base_url,
                    input_modes=("text/plain",), output_modes=("text/plain",),
                    streaming=False):
    """Собрать Agent Card v1 для HTTP+JSON interface.

    skills — список объектов как минимум с id, name, description и tags.
    Версия протокола живёт внутри supportedInterfaces, а не в корне card.
    Поля JSON записаны camelCase, как в ProtoJSON-контракте A2A v1.
    """
    if not skills:
        raise A2AProtocolError("agent card must declare at least one skill")
    for skill in skills:
        missing = [key for key in SKILL_REQUIRED if key not in skill]
        if missing:
            raise A2AProtocolError(f"agent skill is missing required keys: {missing}")
    unknown = [mode for mode in (*input_modes, *output_modes) if mode not in MEDIA_TYPES]
    if unknown:
        raise A2AProtocolError(f"unknown media types: {unknown}")
    root = base_url.rstrip("/")
    return {
        "name": name,
        "description": description,
        "supportedInterfaces": [{
            "url": root,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "version": version,
        "capabilities": {"streaming": bool(streaming), "pushNotifications": False},
        "defaultInputModes": list(input_modes),
        "defaultOutputModes": list(output_modes),
        "skills": copy.deepcopy(list(skills)),
    }


def encode_card(card):
    """Проверить обязательные поля и вернуть стабильный JSON Agent Card."""
    missing = [key for key in CARD_REQUIRED if key not in card]
    if missing:
        raise A2AProtocolError(f"agent card is missing required keys: {missing}")
    return json.dumps(card, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def decode_card(text):
    """Разобрать полученный по сети Agent Card и проверить верхний уровень."""
    try:
        card = json.loads(text)
    except json.JSONDecodeError as exc:
        raise A2AProtocolError(f"agent card is not valid JSON: {exc}") from exc
    if not isinstance(card, dict):
        raise A2AProtocolError("agent card must be a JSON object")
    missing = [key for key in CARD_REQUIRED if key not in card]
    if missing:
        raise A2AProtocolError(f"agent card is missing required keys: {missing}")
    return card


def supports_skill(card, skill_id):
    """Объявляет ли Agent Card навык с данным id."""
    return any(skill["id"] == skill_id for skill in card["skills"])


def make_message(message_id, skill_id, payload):
    """Создать клиентское Message для POST /message:send.

    Клиент создаёт messageId, но НЕ taskId. В учебном data-part лежат id
    желаемого навыка и его вход; сервер интерпретирует их и решает, создавать
    ли Task.
    """
    if not message_id:
        raise A2AProtocolError("messageId must not be empty")
    return {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{
            "data": {"skill": skill_id, "payload": copy.deepcopy(payload)},
            "mediaType": "application/json",
        }],
    }


def make_task(message, id_factory=lambda: str(uuid4())):
    """Создать Task на сервере в ответ на новое Message.

    Task id всегда выдаёт id_factory сервера. taskId в клиентском Message
    означает продолжение существующей задачи, поэтому им нельзя создавать
    новую.
    """
    if message.get("taskId"):
        raise A2AProtocolError("client taskId may only reference an existing task")
    task_id = str(id_factory())
    if not task_id:
        raise A2AProtocolError("server generated an empty task id")
    context_id = message.get("contextId") or f"ctx-{task_id}"
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_SUBMITTED"},
        "artifacts": [],
        "history": [copy.deepcopy(message)],
    }


def make_artifact(artifact_id, media_type, data):
    """Собрать Artifact v1 с member-based Part и mediaType."""
    if not artifact_id:
        raise A2AProtocolError("artifactId must not be empty")
    if media_type not in MEDIA_TYPES:
        raise A2AProtocolError(f"unknown artifact media type {media_type!r}")
    part = {"mediaType": media_type}
    if media_type == "text/plain":
        part["text"] = str(data)
    else:
        part["data"] = copy.deepcopy(data)
    return {"artifactId": artifact_id, "parts": [part]}


def advance_task(task, new_state, artifact=None):
    """Вернуть новый снимок Task после разрешённого перехода."""
    state = task["status"]["state"]
    if state not in TRANSITIONS:
        raise A2AProtocolError(f"unknown task state {state!r}")
    if new_state not in TRANSITIONS[state]:
        raise A2AProtocolError(f"illegal transition {state!r} -> {new_state!r}")
    updated = copy.deepcopy(task)
    updated["status"] = {"state": new_state}
    if artifact is not None:
        updated["artifacts"].append(copy.deepcopy(artifact))
    return updated


def _message_input(message):
    try:
        content = message["parts"][0]["data"]
        return content["skill"], content["payload"]
    except (KeyError, IndexError, TypeError) as exc:
        raise A2AProtocolError("message must contain a skill/payload data part") from exc


def run_task(card, message, worker, id_factory=lambda: str(uuid4())):
    """Обработать POST /message:send и вернуть снимки созданного Task."""
    task = make_task(message, id_factory)
    trace = [task]
    skill_id, payload = _message_input(message)
    if not supports_skill(card, skill_id):
        reason = make_artifact(
            f"artifact-{task['id']}-error", "text/plain",
            f"unknown skill {skill_id!r}",
        )
        trace.append(advance_task(trace[-1], "TASK_STATE_FAILED", reason))
        return trace
    trace.append(advance_task(trace[-1], "TASK_STATE_WORKING"))
    trace.append(advance_task(
        trace[-1], "TASK_STATE_COMPLETED", worker(copy.deepcopy(payload))
    ))
    return trace
