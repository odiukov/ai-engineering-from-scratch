"""
Протокол A2A — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json

# Кусочки спецификации A2A, которые нужны кодеку. Транспорт (HTTP) тут
# намеренно отсутствует: протокол — это формат сообщений и правила смены
# состояний, а не сокет.
WELL_KNOWN_PATH = "/.well-known/agent.json"
CARD_REQUIRED = ("name", "version", "skills", "endpoints", "auth", "modalities", "protocol_version")
MODALITIES = ("text", "structured", "image", "audio", "video")
TERMINAL_STATES = ("completed", "failed", "canceled")
# Жизненный цикл задачи: submitted -> working -> completed / failed / canceled.
TRANSITIONS = {
    "submitted": ("working", "canceled", "failed"),
    "working": ("completed", "failed", "canceled"),
    "completed": (),
    "failed": (),
    "canceled": (),
}


class A2AProtocolError(Exception):
    """Нарушение протокола: кривая карточка, чужая модальность, запрещённый переход.

    Свой класс, а не RuntimeError: NotImplementedError наследуется от
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """


def make_agent_card(name, version, skills, base_url, auth="none", modalities=("text",)):
    """Agent Card — визитка агента, которую он выкладывает по WELL_KNOWN_PATH.

    make_agent_card("code-review-agent", "0.1.0", ["review-python"],
                    "http://localhost:8765")
        ->  {'name': 'code-review-agent', 'version': '0.1.0',
             'skills': ('review-python',),
             'endpoints': {'card': 'http://localhost:8765/.well-known/agent.json',
                           'tasks': 'http://localhost:8765/tasks'},
             'auth': {'type': 'none'}, 'modalities': ('text',),
             'protocol_version': 'a2a-0.3'}

    Агент без навыков бесполезен для discovery: клиенту не за чем приходить.
    Такую карточку надо отвергать (A2AProtocolError), а не выкладывать.

    Ловушка: лишний слэш в base_url («.../» + «/tasks») даёт битый URL.
    Обрежь хвост перед склейкой.
    """
    if not skills:
        raise A2AProtocolError("agent card must declare at least one skill")
    unknown = [m for m in modalities if m not in MODALITIES]
    if unknown:
        raise A2AProtocolError(f"unknown modalities: {unknown}")
    root = base_url.rstrip("/")
    return {
        "name": name,
        "version": version,
        "skills": tuple(skills),
        "endpoints": {"card": root + WELL_KNOWN_PATH, "tasks": root + "/tasks"},
        "auth": {"type": auth},
        "modalities": tuple(modalities),
        "protocol_version": "a2a-0.3",
    }


def encode_card(card):
    """Карточка в JSON-строку. Ключи отсортированы, поэтому байты стабильны.

    encode_card(card)  ->  '{"auth": {"type": "none"}, "endpoints": ...}'

    Перед кодированием проверь, что все CARD_REQUIRED на месте: выложить
    карточку без "auth" — значит заставить клиента гадать, как к тебе
    стучаться. Это A2AProtocolError, а не «ну и ладно».

    Сортировка ключей нужна не для красоты: одинаковая карточка обязана
    давать одинаковую строку, иначе ETag и кэш discovery не работают.
    """
    missing = [key for key in CARD_REQUIRED if key not in card]
    if missing:
        raise A2AProtocolError(f"agent card is missing required keys: {missing}")
    return json.dumps(card, sort_keys=True, ensure_ascii=False)


def decode_card(text):
    """Разбор карточки, пришедшей по сети. Кортежи приезжают списками.

    decode_card(encode_card(card))["name"]  ->  'code-review-agent'

    Две разные беды с одинаковым исходом: битый JSON и валидный JSON без
    обязательных полей. Обе — A2AProtocolError, чтобы вызывающему не
    приходилось ловить ещё и json.JSONDecodeError отдельно.

    Ловушка: JSON не знает кортежей. Поле skills вернётся списком, и
    сравнивать карточку «до» и «после» надо с учётом этого.
    """
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


def supports_skill(card, skill):
    """Умеет ли агент этот навык. Это весь discovery со стороны клиента.

    supports_skill(card, "review-python")  ->  True
    supports_skill(card, "summarize")      ->  False

    Работает и с карточкой после decode_card, где skills стал списком:
    проверка на вхождение не зависит от типа контейнера.
    """
    return skill in card["skills"]


def make_task(task_id, skill, payload):
    """Новая задача в состоянии submitted.

    make_task("t-1", "review-python", {"code": "x = 1"})
        ->  {'id': 't-1', 'skill': 'review-python', 'payload': {'code': 'x = 1'},
             'state': 'submitted', 'artifact': None}

    Пустой id ломает идемпотентность из чек-листа урока: повторную отправку
    после ретрая нечем узнать. Требуй непустой — A2AProtocolError.
    """
    if not task_id:
        raise A2AProtocolError("task id must not be empty")
    return {
        "id": task_id,
        "skill": skill,
        "payload": payload,
        "state": "submitted",
        "artifact": None,
    }


def make_artifact(kind, data):
    """Типизированный результат задачи.

    make_artifact("text", "looks fine")   ->  {'type': 'text', 'data': 'looks fine'}
    make_artifact("structured", {"issues": []})
        ->  {'type': 'structured', 'data': {'issues': []}}

    Модальность вне MODALITIES — A2AProtocolError: смысл типизированных
    артефактов ровно в том, что принимающая сторона знает заранее, что ей
    приедет. Свободная строка в поле type это поле обесценивает.
    """
    if kind not in MODALITIES:
        raise A2AProtocolError(f"unknown artifact modality {kind!r}")
    return {"type": kind, "data": data}


def advance_task(task, new_state, artifact=None):
    """Переход задачи в новое состояние. Возвращает НОВУЮ задачу.

    advance_task(make_task("t", "s", {}), "working")["state"]  ->  'working'

    Разрешённые переходы лежат в TRANSITIONS. Из терминального состояния
    выхода нет: completed -> working это A2AProtocolError, а не «ну ладно,
    переоткроем». Клиент, который уже забрал артефакт, не должен однажды
    увидеть задачу снова работающей.

    Ловушка: не правь входную задачу на месте. Опрос статуса возвращает
    снимки, и мутация задним числом сделает историю опроса ложью.
    """
    state = task["state"]
    if state not in TRANSITIONS:
        raise A2AProtocolError(f"unknown task state {state!r}")
    if new_state not in TRANSITIONS[state]:
        raise A2AProtocolError(f"illegal transition {state!r} -> {new_state!r}")
    updated = dict(task)
    updated["state"] = new_state
    if artifact is not None:
        updated["artifact"] = artifact
    return updated


def run_task(card, task, worker):
    """Весь жизненный цикл задачи. Возвращает СНИМКИ задачи по состояниям.

    worker — вызываемый объект worker(payload) -> artifact.

    [t["state"] for t in run_task(card, task, worker)]
        ->  ['submitted', 'working', 'completed']

    Навык, которого нет в карточке, обрывает цикл сразу:
        ->  ['submitted', 'failed']
    и в артефакте лежит текстовое объяснение — клиент читает причину, а не
    гадает по коду ответа.

    Это и есть opaque lifecycle из урока: клиент видит только смену
    состояний и артефакт, а чем внутри считал worker — не его дело.
    """
    trace = [dict(task)]
    if not supports_skill(card, task["skill"]):
        reason = make_artifact("text", f"unknown skill {task['skill']!r}")
        trace.append(advance_task(trace[-1], "failed", reason))
        return trace
    trace.append(advance_task(trace[-1], "working"))
    trace.append(advance_task(trace[-1], "completed", worker(task["payload"])))
    return trace
