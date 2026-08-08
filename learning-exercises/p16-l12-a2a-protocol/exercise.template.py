"""
Протокол A2A v1

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l12-a2a-protocol
Разбор:  /check-code p16-l12-a2a-protocol
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
    pass


def make_agent_card(name, description, version, skills, base_url,
                    input_modes=("text/plain",), output_modes=("text/plain",),
                    streaming=False):
    """Собрать Agent Card v1 для HTTP+JSON interface.

    skills — список объектов как минимум с id, name, description и tags.
    Версия протокола живёт внутри supportedInterfaces, а не в корне card.
    Поля JSON записаны camelCase, как в ProtoJSON-контракте A2A v1.
    """
    raise NotImplementedError


def encode_card(card):
    """Проверить обязательные поля и вернуть стабильный JSON Agent Card."""
    raise NotImplementedError


def decode_card(text):
    """Разобрать полученный по сети Agent Card и проверить верхний уровень."""
    raise NotImplementedError


def supports_skill(card, skill_id):
    """Объявляет ли Agent Card навык с данным id."""
    raise NotImplementedError


def make_message(message_id, skill_id, payload):
    """Создать клиентское Message для POST /message:send.

    Клиент создаёт messageId, но НЕ taskId. В учебном data-part лежат id
    желаемого навыка и его вход; сервер интерпретирует их и решает, создавать
    ли Task.
    """
    raise NotImplementedError


def make_task(message, id_factory=lambda: str(uuid4())):
    """Создать Task на сервере в ответ на новое Message.

    Task id всегда выдаёт id_factory сервера. taskId в клиентском Message
    означает продолжение существующей задачи, поэтому им нельзя создавать
    новую.
    """
    raise NotImplementedError


def make_artifact(artifact_id, media_type, data):
    """Собрать Artifact v1 с member-based Part и mediaType."""
    raise NotImplementedError


def advance_task(task, new_state, artifact=None):
    """Вернуть новый снимок Task после разрешённого перехода."""
    raise NotImplementedError


def _message_input(message):
    raise NotImplementedError


def run_task(card, message, worker, id_factory=lambda: str(uuid4())):
    """Обработать POST /message:send и вернуть снимки созданного Task."""
    raise NotImplementedError
