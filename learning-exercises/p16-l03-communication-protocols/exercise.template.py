"""
Протоколы общения агентов

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l03-communication-protocols
Разбор:  /check-code p16-l03-communication-protocols
"""

import hashlib
import hmac
import json

TASK_STATES = (
    "submitted", "working", "input-required", "auth-required",
    "completed", "failed", "canceled", "rejected",
)
TERMINAL_STATES = ("completed", "failed", "canceled", "rejected")


class ProtocolError(Exception):
    """Базовая ошибка протокола.

    Собственный класс, потому что RuntimeError и его потомок
    NotImplementedError неотличимы: тест на RuntimeError зеленел бы на
    пустой заготовке.
    """
    pass


class TaskStateError(ProtocolError):
    """Недопустимый переход в жизненном цикле задачи A2A."""
    pass


def agent_card(name, url, skills, default_input_modes, default_output_modes,
               streaming=False, version="1.0.0", description=None):
    """Agent Card в духе A2A: что агент умеет и как с ним говорить.

    skills — список dict с ключами "id", "name", "description", "tags",
    "inputModes", "outputModes".

    card = agent_card("researcher", "https://r.local/a2a/v1",
                      [{"id": "web-research", "name": "Web research",
                        "description": "Search and summarize", "tags": ["research"],
                        "inputModes": ["text/plain"],
                        "outputModes": ["application/json"]}],
                      ["text/plain"], ["application/json"])
    card["capabilities"]["streaming"]  ->  False
    card["version"]                    ->  "1.0.0"

    В боевом A2A этот документ лежит по адресу
    GET /.well-known/agent-card.json и читается ДО первого запроса —
    клиент заранее знает и умения, и требуемую аутентификацию.
    """
    raise NotImplementedError


def discover(cards, tag=None, media_type=None):
    """Поиск агентов по тегу умения и/или по принимаемому MIME-типу.

    Оба фильтра необязательны и складываются логическим И. Без фильтров
    возвращаются все карточки в порядке регистрации.

    discover(cards)                          ->  все карточки
    discover(cards, tag="research")          ->  только исследователи
    discover(cards, media_type="text/plain") ->  кто принимает такой вход

    MIME засчитывается и из defaultInputModes карточки, и из
    inputModes любого её умения: умение может принимать больше, чем
    агент по умолчанию.
    """
    raise NotImplementedError


def new_task(task_id, context_id):
    """Свежая задача A2A в состоянии submitted.

    new_task("t-1", "ctx-1")["state"]      ->  "submitted"
    new_task("t-1", "ctx-1")["artifacts"]  ->  []

    context_id живёт дольше задачи: продолжение разговора — это новая
    задача с тем же context_id.
    """
    raise NotImplementedError


def apply_event(task, event):
    """Применить событие потока к задаче. Возвращает НОВУЮ задачу.

    Событие бывает двух видов:
      {"kind": "statusUpdate", "state": "working"}
      {"kind": "artifactUpdate", "artifact": {...}, "append": False}

    apply_event(new_task("t", "c"), {"kind": "statusUpdate",
                                     "state": "working"})["state"]
        ->  "working"

    Терминальная задача неизменяема  ->  TaskStateError
    Состояние не из TASK_STATES      ->  TaskStateError

    Ловушка: append=True дописывает parts в артефакт с тем же id, а не
    кладёт рядом второй артефакт. Именно так работает потоковая доставка
    по частям в A2A SSE.
    """
    raise NotImplementedError


def _canonical_payload(payload):
    """Стабильные байты строки или полного JSON-сообщения для подписи."""
    raise NotImplementedError


def sign(secret, payload):
    """Подпись строки или полного JSON-сообщения: HMAC-SHA256 в hex.

    sign("k", "msg-1") == sign("k", "msg-1")   ->  True
    sign("k", "msg-1") == sign("k", "msg-2")   ->  False

    Боевой ANP подписывает асимметрично (ключ did:wba из DID-документа),
    здесь для наглядности симметричный HMAC — проверяемое свойство то же:
    подпись привязана и к секрету, и ко всем полям сообщения. JSON
    канонизируется, поэтому порядок ключей не меняет подпись, а правка даже
    вложенного text/data делает её недействительной.
    """
    raise NotImplementedError


def verify(secrets, did, payload, signature):
    """Проверка подписи по реестру секретов. Неизвестный DID — сразу False.

    verify({"did:wba:a": "k"}, "did:wba:a", "msg-1", sign("k", "msg-1"))
        ->  True
    verify({"did:wba:a": "k"}, "did:wba:ghost", "msg-1", "deadbeef")
        ->  False

    Неизвестный отправитель отвергается, а не пропускается: это принцип
    fail closed из урока. Сравнение — через hmac.compare_digest, обычное
    == течёт по времени.
    """
    raise NotImplementedError


def audit_run(run_id, agent_name, message, handler, session_id=None):
    """Запуск агента с журналом ACP: вход, выход, траектория, статус.

    handler(message) -> (output, trajectory), где trajectory — список
    шагов вида {"reasoning": ..., "tool_name": ...}.

    Упавший handler не роняет запуск: статус становится "failed", а
    причина уходит в траекторию отдельным шагом. Ради этого журнал и
    заводят — регулируемым отраслям нужен след, а не исключение.

    audit_run("r-1", "researcher", msg, lambda m: ("ok", []))["status"]
        ->  "completed"
    """
    raise NotImplementedError


def delegate(cards, secrets, from_did, signature, skill_tag, message,
             task_id, context_id, handler):
    """Шлюз из урока: ANP проверяет личность, A2A ищет агента и ведёт задачу,
    ACP пишет журнал.

    Возвращает {"error": ...} при отказе, иначе
    {"agent": имя, "task": задача, "audit": запись журнала}.

    Порядок проверок важен: сначала личность, потом поиск. Иначе
    неопознанный отправитель узнаёт, какие агенты есть в реестре.

    Успешный запуск доводит задачу до "completed" и кладёт вывод агента
    артефактом. Упавший — до "failed", артефактов нет.
    """
    raise NotImplementedError
