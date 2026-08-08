"""
A2A v1.0 — протокол общения агентов

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l19-a2a-protocol
Разбор:  /check-code p13-l19-a2a-protocol
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
    raise NotImplementedError


def select_skill(card, input_modes, output_mode):
    """Выбрать самый узкий skill по MIME inputModes/outputModes.

    Part больше не имеет kind. Его содержание определяется ровно
    одним из полей text/raw/url/data, а modes в AgentSkill — MIME types.
    """
    raise NotImplementedError


def canonical_json(obj):
    """Детерминированная компактная JSON-запись для учебной подписи."""
    raise NotImplementedError


def _b64url(raw):
    raise NotImplementedError


def sign_agent_card(card, secret):
    """Добавить учебную JWS-образную подпись в AgentCard.signatures.

    На проводе каждая подпись — {"protected", "signature"}. Реальная v1
    подпись обычно асимметрична; HMAC здесь только чтобы обойтись stdlib.
    """
    raise NotImplementedError


def verify_agent_card(signed, secret):
    """Проверить первую учебную подпись AgentCard за постоянное время."""
    raise NotImplementedError


def next_task_state(state, event):
    """Перевести TaskState v1; недопустимый переход — ValueError."""
    raise NotImplementedError


def make_artifact(artifact_id, name, mime_type, chunks):
    """Собрать Artifact v1 с обязательным artifactId и Part без kind."""
    raise NotImplementedError


def _part_field(part):
    raise NotImplementedError


def run_task(task_id, card, skill_id, messages):
    """Проиграть SendMessage и вернуть каноничный Task.

    Task содержит status: {state, message?}, history и artifacts[].
    Единственный Artifact этого демо всё равно живёт в массиве и имеет artifactId.
    """
    raise NotImplementedError
