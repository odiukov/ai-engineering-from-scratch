"""
A2A v1.0 — протокол общения агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь мы собираем актуальные AgentCard / Task / Message / Part /
Artifact руками. JSON-RPC binding называет операцию SendMessage;
HTTP+JSON binding кодирует её как POST /message:send.
"""

import base64
import copy
import hashlib
import hmac
import json


AGENT_CARD_PATH = "/.well-known/agent-card.json"
PROTOCOL_VERSION = "1.0"
PART_FIELDS = ("text", "raw", "url", "data")

TASK_STATES = (
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
)
TASK_EVENTS = ("accept", "reject", "need_input", "provide_input", "finish", "fail", "cancel")
TASK_TRANSITIONS = {
    ("TASK_STATE_SUBMITTED", "accept"): "TASK_STATE_WORKING",
    ("TASK_STATE_SUBMITTED", "reject"): "TASK_STATE_REJECTED",
    ("TASK_STATE_SUBMITTED", "cancel"): "TASK_STATE_CANCELED",
    ("TASK_STATE_WORKING", "need_input"): "TASK_STATE_INPUT_REQUIRED",
    ("TASK_STATE_WORKING", "finish"): "TASK_STATE_COMPLETED",
    ("TASK_STATE_WORKING", "fail"): "TASK_STATE_FAILED",
    ("TASK_STATE_WORKING", "cancel"): "TASK_STATE_CANCELED",
    ("TASK_STATE_INPUT_REQUIRED", "provide_input"): "TASK_STATE_WORKING",
    ("TASK_STATE_INPUT_REQUIRED", "finish"): "TASK_STATE_COMPLETED",
    ("TASK_STATE_INPUT_REQUIRED", "fail"): "TASK_STATE_FAILED",
    ("TASK_STATE_INPUT_REQUIRED", "cancel"): "TASK_STATE_CANCELED",
}
TERMINAL_STATES = frozenset(
    {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
)


def build_agent_card(name, description, url, version, skills, capabilities=None):
    """Собрать Agent Card v1.0 для /.well-known/agent-card.json.

    Точка доступа живёт в обязательном supportedInterfaces, а не в
    устаревшем верхнеуровневом url. protocolVersion отделен от
    version самого агента. defaultInputModes/defaultOutputModes — MIME types.
    """
    copied = copy.deepcopy(list(skills))
    ids = [skill.get("id") for skill in copied]
    if any(skill_id is None for skill_id in ids):
        raise ValueError("skill without 'id'")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate skill id")
    input_modes = sorted({mode for skill in copied for mode in skill.get("inputModes", ())})
    output_modes = sorted({mode for skill in copied for mode in skill.get("outputModes", ())})
    caps = {"streaming": False, "pushNotifications": False}
    caps.update(capabilities or {})
    return {
        "name": name,
        "description": description,
        "supportedInterfaces": [
            {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION}
        ],
        "version": version,
        "capabilities": caps,
        "defaultInputModes": input_modes or ["text/plain"],
        "defaultOutputModes": output_modes or ["text/plain"],
        "skills": copied,
    }


def select_skill(card, input_modes, output_mode):
    """Выбрать самый узкий skill по MIME inputModes/outputModes.

    Part больше не имеет kind. Его содержание определяется ровно
    одним из полей text/raw/url/data, а modes в AgentSkill — MIME types.
    """
    if any(not isinstance(mode, str) or not mode for mode in input_modes):
        raise ValueError("input modes must be non-empty strings")
    needed = set(input_modes)
    matches = [
        skill
        for skill in card.get("skills", ())
        if needed <= set(skill.get("inputModes", ()))
        and output_mode in skill.get("outputModes", ())
    ]
    if not matches:
        return None
    matches.sort(key=lambda skill: (len(set(skill.get("inputModes", ()))), skill["id"]))
    return matches[0]["id"]


def canonical_json(obj):
    """Детерминированная компактная JSON-запись для учебной подписи."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_agent_card(card, secret):
    """Добавить учебную JWS-образную подпись в AgentCard.signatures.

    На проводе каждая подпись — {"protected", "signature"}. Реальная v1
    подпись обычно асимметрична; HMAC здесь только чтобы обойтись stdlib.
    """
    signed = copy.deepcopy(card)
    unsigned = copy.deepcopy(signed)
    unsigned.pop("signatures", None)
    protected = _b64url(canonical_json({"alg": "HS256"}).encode())
    signing_input = protected + "." + _b64url(canonical_json(unsigned).encode())
    key = secret.encode() if isinstance(secret, str) else secret
    signature = _b64url(hmac.new(key, signing_input.encode(), hashlib.sha256).digest())
    signed["signatures"] = [{"protected": protected, "signature": signature}]
    return signed


def verify_agent_card(signed, secret):
    """Проверить первую учебную подпись AgentCard за постоянное время."""
    signatures = signed.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        return False
    unsigned = copy.deepcopy(signed)
    unsigned.pop("signatures", None)
    expected = sign_agent_card(unsigned, secret)["signatures"][0]
    actual = signatures[0]
    return (
        actual.get("protected") == expected["protected"]
        and isinstance(actual.get("signature"), str)
        and hmac.compare_digest(actual["signature"], expected["signature"])
    )


def next_task_state(state, event):
    """Перевести TaskState v1; недопустимый переход — ValueError."""
    if state not in TASK_STATES:
        raise ValueError(f"unknown task state: {state}")
    if event not in TASK_EVENTS:
        raise ValueError(f"unknown task event: {event}")
    try:
        return TASK_TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise ValueError(f"illegal transition: {state} --{event}-->") from exc


def make_artifact(artifact_id, name, mime_type, chunks):
    """Собрать Artifact v1 с обязательным artifactId и Part без kind."""
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(f"artifact chunk must be str, got {type(chunk).__name__}")
    return {
        "artifactId": artifact_id,
        "name": name,
        "parts": [{"text": "".join(chunks), "mediaType": mime_type}],
    }


def _part_field(part):
    present = [field for field in PART_FIELDS if field in part]
    if len(present) != 1:
        raise ValueError("Part must contain exactly one of text/raw/url/data")
    return present[0]


def run_task(task_id, card, skill_id, messages):
    """Проиграть SendMessage и вернуть каноничный Task.

    Task содержит status: {state, message?}, history и artifacts[].
    Единственный Artifact этого демо всё равно живёт в массиве и имеет artifactId.
    """
    task = {
        "id": task_id,
        "contextId": f"ctx_{task_id}",
        "status": {"state": "TASK_STATE_SUBMITTED"},
        "history": [],
        "artifacts": [],
    }
    skill = next((skill for skill in card.get("skills", ()) if skill["id"] == skill_id), None)
    if skill is None:
        task["status"] = {"state": next_task_state(task["status"]["state"], "reject")}
        return task

    task["status"] = {"state": next_task_state(task["status"]["state"], "accept")}
    required = tuple(skill.get("requiredData", ()))
    collected = {}
    for message in messages:
        if task["status"]["state"] in TERMINAL_STATES:
            break
        snapshot = copy.deepcopy(message)
        if "messageId" not in snapshot:
            raise ValueError("Message.messageId is required")
        if snapshot.get("role") not in {"ROLE_USER", "ROLE_AGENT"}:
            raise ValueError("Message.role must be ROLE_USER or ROLE_AGENT")
        for part in snapshot.get("parts", ()):
            field = _part_field(part)
            if field == "data" and isinstance(part["data"], dict):
                collected.update(part["data"])
        task["history"].append(snapshot)
        missing = [name for name in required if name not in collected]
        if missing:
            if task["status"]["state"] == "TASK_STATE_WORKING":
                task["status"]["state"] = next_task_state(
                    task["status"]["state"], "need_input"
                )
            task["status"]["message"] = {
                "messageId": f"msg_need_{task_id}",
                "taskId": task_id,
                "contextId": task["contextId"],
                "role": "ROLE_AGENT",
                "parts": [{"text": "input required: " + ", ".join(missing)}],
            }
            continue
        if task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED":
            task["status"] = {
                "state": next_task_state(task["status"]["state"], "provide_input")
            }
        chunks = [
            part["text"]
            for saved in task["history"]
            if saved["role"] == "ROLE_USER"
            for part in saved["parts"]
            if "text" in part
        ]
        task["artifacts"] = [
            make_artifact(f"art_{task_id}", "summary", "text/markdown", chunks)
        ]
        task["status"] = {"state": next_task_state(task["status"]["state"], "finish")}
    return task
