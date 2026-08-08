"""
Протоколы общения агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import hmac
import json

# Жизненный цикл задачи A2A. UNSPECIFIED-сентинел из спецификации сюда не
# берём — работать с ним всё равно нельзя.
TASK_STATES = (
    "submitted", "working", "input-required", "auth-required",
    "completed", "failed", "canceled", "rejected",
)

# Терминальные состояния необратимы: задача в них больше не меняется,
# продолжение — это новая задача в том же context_id.
TERMINAL_STATES = ("completed", "failed", "canceled", "rejected")


class ProtocolError(Exception):
    """Базовая ошибка протокола.

    Собственный класс, потому что RuntimeError и его потомок
    NotImplementedError неотличимы: тест на RuntimeError зеленел бы на
    пустой заготовке.
    """


class TaskStateError(ProtocolError):
    """Недопустимый переход в жизненном цикле задачи A2A."""


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
    return {
        "name": name,
        "description": description or f"{name} agent",
        "supportedInterfaces": [{
            "url": url,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "version": version,
        "capabilities": {"streaming": streaming, "pushNotifications": False},
        "defaultInputModes": list(default_input_modes),
        "defaultOutputModes": list(default_output_modes),
        "skills": [
            dict(skill,
                 tags=list(skill.get("tags", [])),
                 inputModes=list(skill.get("inputModes", [])),
                 outputModes=list(skill.get("outputModes", [])))
            for skill in skills
        ],
    }


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
    found = []
    for card in cards:
        if tag is not None:
            if not any(tag in skill.get("tags", []) for skill in card["skills"]):
                continue
        if media_type is not None:
            accepted = set(card["defaultInputModes"])
            for skill in card["skills"]:
                accepted.update(skill.get("inputModes", []))
            if media_type not in accepted:
                continue
        found.append(card)
    return found


def new_task(task_id, context_id):
    """Свежая задача A2A в состоянии submitted.

    new_task("t-1", "ctx-1")["state"]      ->  "submitted"
    new_task("t-1", "ctx-1")["artifacts"]  ->  []

    context_id живёт дольше задачи: продолжение разговора — это новая
    задача с тем же context_id.
    """
    return {
        "id": task_id,
        "context_id": context_id,
        "state": "submitted",
        "artifacts": [],
    }


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
    if task["state"] in TERMINAL_STATES:
        raise TaskStateError(f"task is terminal: {task['state']}")
    # копия артефактов на два уровня: parts мы дописываем, и делиться
    # списком со старой задачей нельзя
    updated = dict(task)
    updated["artifacts"] = [dict(a, parts=list(a["parts"])) for a in task["artifacts"]]

    if event["kind"] == "statusUpdate":
        if event["state"] not in TASK_STATES:
            raise TaskStateError(f"unknown task state: {event['state']}")
        updated["state"] = event["state"]
        return updated

    if event["kind"] == "artifactUpdate":
        artifact = event["artifact"]
        if event.get("append"):
            for existing in updated["artifacts"]:
                if existing["id"] == artifact["id"]:
                    existing["parts"].extend(artifact["parts"])
                    return updated
        updated["artifacts"].append(dict(artifact, parts=list(artifact["parts"])))
        return updated

    raise ProtocolError(f"unknown event kind: {event['kind']}")


def _canonical_payload(payload):
    """Стабильные байты строки или полного JSON-сообщения для подписи."""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


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
    encoded = _canonical_payload(payload).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), encoded,
                    hashlib.sha256).hexdigest()


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
    secret = secrets.get(did)
    if secret is None:
        return False
    return hmac.compare_digest(sign(secret, payload), signature)


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
    entry = {
        "run_id": run_id,
        "agent_name": agent_name,
        "session_id": session_id,
        "input": message,
        "output": None,
        "trajectory": [],
        "status": "in-progress",
    }
    try:
        output, trajectory = handler(message)
    except Exception as err:  # noqa: BLE001 — журнал обязан пережить падение
        entry["status"] = "failed"
        entry["trajectory"] = [{"reasoning": f"Error: {err}"}]
        return entry
    entry["output"] = output
    entry["trajectory"] = list(trajectory)
    entry["status"] = "completed"
    return entry


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
    if not verify(secrets, from_did, message, signature):
        return {"error": "identity verification failed"}

    candidates = discover(cards, tag=skill_tag)
    if not candidates:
        return {"error": f"no agent with skill tag: {skill_tag}"}
    target = candidates[0]

    audit = audit_run(task_id, target["name"], message, handler,
                      session_id=context_id)

    task = new_task(task_id, context_id)
    task = apply_event(task, {"kind": "statusUpdate", "state": "working"})
    if audit["status"] == "completed":
        task = apply_event(task, {
            "kind": "artifactUpdate",
            "artifact": {"id": f"art-{task_id}", "name": "result",
                         "parts": [audit["output"]]},
            "append": False,
        })
        task = apply_event(task, {"kind": "statusUpdate", "state": "completed"})
    else:
        task = apply_event(task, {"kind": "statusUpdate", "state": "failed"})
    return {"agent": target["name"], "task": task, "audit": audit}
