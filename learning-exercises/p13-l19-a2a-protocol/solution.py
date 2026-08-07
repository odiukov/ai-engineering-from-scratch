"""
A2A — протокол общения агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

A2A (Agent2Agent, v1.0, Linux Foundation) — это JSON-RPC поверх HTTP плюс
несколько соглашений о форме данных. Пакет `a2a-sdk` прячет их за классами
AgentCard / Task / Message / Artifact; здесь мы собираем те же структуры
руками. Соответствие настоящему API:

    build_agent_card   <-  документ, который публикуется по /.well-known/agent.json
    select_skill       <-  клиентский выбор skill из карточки по modes
    canonical_json     <-  канонизация карточки перед подписью (расширение AP2)
    sign_agent_card    <-  подпись карточки издателем (AP2)
    verify_agent_card  <-  проверка подписи потребителем (AP2)
    next_task_state    <-  автомат состояний Task из спецификации
    make_artifact      <-  накопление Artifact из потоковых чанков
    run_task           <-  серверная сторона tasks/send и tasks/sendSubscribe

Сеть не нужна: транспорт (JSON-RPC over HTTP или gRPC) только доставляет
словари. Мы работаем сразу со словарями.
"""

import hashlib
import hmac
import json

# Путь, по которому агент обязан отдавать свою карточку.
AGENT_CARD_PATH = "/.well-known/agent.json"

SCHEMA_VERSION = "1.0"

# Типы Part в сообщении. Больше в спецификации нет.
PART_KINDS = ("text", "file", "data")

TASK_STATES = (
    "submitted",
    "working",
    "input_required",
    "completed",
    "failed",
    "canceled",
    "rejected",
)

TASK_EVENTS = (
    "accept",
    "reject",
    "need_input",
    "provide_input",
    "finish",
    "fail",
    "cancel",
)

# Автомат состояний Task. Ключ — (состояние, событие), значение — новое
# состояние. Пары, которых здесь нет, спецификацией запрещены.
TASK_TRANSITIONS = {
    ("submitted", "accept"): "working",
    ("submitted", "reject"): "rejected",
    ("submitted", "cancel"): "canceled",
    ("working", "need_input"): "input_required",
    ("working", "finish"): "completed",
    ("working", "fail"): "failed",
    ("working", "cancel"): "canceled",
    ("input_required", "provide_input"): "working",
    ("input_required", "fail"): "failed",
    ("input_required", "cancel"): "canceled",
}

# Из этих состояний выхода нет: задача закончена навсегда.
TERMINAL_STATES = frozenset({"completed", "failed", "canceled", "rejected"})


def build_agent_card(name, description, url, version, skills, capabilities=None):
    """Собрать Agent Card — документ, который агент публикует о себе.

    build_agent_card("writer-agent", "Drafts reports.",
                     "https://writer.example.com/a2a", "1.0.0",
                     [{"id": "draft_report", "name": "Draft report",
                       "inputModes": ["text"], "outputModes": ["text"]}])
        ->  {"schemaVersion": "1.0", "name": "writer-agent",
             "description": "Drafts reports.",
             "url": "https://writer.example.com/a2a", "version": "1.0.0",
             "skills": [...],
             "capabilities": {"streaming": False, "pushNotifications": False}}

    capabilities дополняет умолчания, а не заменяет их: передали
    {"streaming": True} — pushNotifications всё равно останется в карточке
    со значением False. Потребитель не должен угадывать отсутствующие ключи.

    Два skill с одинаковым id — это ValueError. Клиент выбирает skill по id,
    и дубликат сделал бы выбор неоднозначным.

    Ловушка: skills нужно скопировать. Если положить в карточку сам список
    вызывающего кода, его позднейшая правка тихо поменяет уже опубликованную
    карточку — а её к этому моменту могли подписать.
    """
    copied = []
    seen = set()
    for skill in skills:
        if "id" not in skill:
            raise ValueError("skill without 'id'")
        if skill["id"] in seen:
            raise ValueError(f"duplicate skill id: {skill['id']}")
        seen.add(skill["id"])
        # dict(skill) — копия верхнего уровня: этого хватает, потому что
        # значения внутри skill (строки и списки строк) мы не правим.
        copied.append(dict(skill))
    caps = {"streaming": False, "pushNotifications": False}
    caps.update(capabilities or {})
    return {
        "schemaVersion": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "url": url,
        "version": version,
        "skills": copied,
        "capabilities": caps,
    }


def select_skill(card, input_kinds, output_mode):
    """Выбрать skill, который примет такие Part и вернёт такой выход. Id или None.

    Пусть в карточке два skill: "quick_note" (inputModes ["text"]) и
    "draft_report" (inputModes ["text", "file", "data"]), оба с
    outputModes ["text"].

    select_skill(card, ["text"], "text")          ->  "quick_note"
    select_skill(card, ["text", "file"], "text")  ->  "draft_report"
    select_skill(card, ["text"], "audio")         ->  None

    Skill подходит, если ВСЕ запрошенные input_kinds есть в его inputModes,
    а output_mode есть в outputModes. Подходящих может оказаться несколько.

    Выбор обязан быть детерминированным и не зависеть от порядка skills в
    карточке: сначала самый узкий (меньше inputModes), при равенстве — по
    возрастанию id. Иначе один и тот же вызов у двух клиентов уедет в разные
    skill только потому, что агент переставил их местами в JSON.

    Неизвестный вид Part — ValueError: молча вернуть None значит спрятать
    опечатку в имени вида.
    """
    needed = set(input_kinds)
    unknown = sorted(needed - set(PART_KINDS))
    if unknown:
        raise ValueError(f"unknown part kinds: {', '.join(unknown)}")
    matches = [
        s
        for s in card.get("skills", ())
        if needed <= set(s.get("inputModes", ()))
        and output_mode in s.get("outputModes", ())
    ]
    if not matches:
        return None
    # sort, а не min по одному полю: ключ составной, и порядок в карточке в
    # него не входит намеренно.
    matches.sort(key=lambda s: (len(set(s.get("inputModes", ()))), s["id"]))
    return matches[0]["id"]


def canonical_json(obj):
    """Каноническая JSON-запись: та же структура — та же строка, байт в байт.

    canonical_json({"b": 1, "a": 2})  ->  '{"a":2,"b":1}'
    canonical_json({"a": 2, "b": 1})  ->  '{"a":2,"b":1}'

    Нужна перед подписью. Обычный json.dumps сохраняет порядок вставки
    ключей и ставит пробелы после запятых — два одинаковых по смыслу словаря
    дадут разные строки, и подпись не сойдётся у того, кто пересобрал
    карточку из своей структуры.

    Три обязательных условия: sort_keys, компактные разделители,
    ensure_ascii=False (иначе кириллица уедет в \\uXXXX и длина строки
    станет зависеть от настроек сериализатора).
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sign_agent_card(card, secret):
    """Подписать Agent Card по-AP2: HMAC-SHA256 над канонической записью.

    sign_agent_card({"name": "writer"}, "s3cret")
        ->  {"alg": "HS256", "card": {"name": "writer"}, "signature": "<64 hex>"}

    Возвращается НЕ исходный словарь, а его снимок: карточку внутри подписи
    правит только тот, кто готов переподписать. Позднейшая правка исходного
    card подписанный снимок не задевает.

    В настоящем AP2 подпись асимметричная (JWT), и потребителю нужен только
    публичный ключ издателя. HMAC здесь ради того, чтобы обойтись стандартной
    библиотекой; форма проверки та же.
    """
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    payload = canonical_json(card)
    digest = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    # json.loads(payload) — заодно и глубокая копия, и доказательство того,
    # что подписали именно то, что сериализуется обратно один в один.
    return {"alg": "HS256", "card": json.loads(payload), "signature": digest}


def verify_agent_card(signed, secret):
    """Проверить подпись карточки. True только если всё сошлось.

    verify_agent_card(sign_agent_card(card, "s3cret"), "s3cret")  ->  True
    verify_agent_card(sign_agent_card(card, "s3cret"), "other")   ->  False

    Испорченная карточка обязана дать False: в этом весь смысл подписи.
    Неизвестный alg — тоже False, иначе злоумышленник пришлёт alg "none" и
    проверка пропустит что угодно (классическая дыра JWT-библиотек).

    Сравнивать подписи через == нельзя: обычное сравнение строк выходит на
    первом же несовпавшем символе, и по времени ответа подпись подбирается
    посимвольно. Нужен hmac.compare_digest.
    """
    if signed.get("alg") != "HS256":
        return False
    signature = signed.get("signature")
    if not isinstance(signature, str):
        return False
    expected = sign_agent_card(signed.get("card", {}), secret)["signature"]
    return hmac.compare_digest(expected, signature)


def next_task_state(state, event):
    """Перевести Task в новое состояние по событию. ValueError, если нельзя.

    next_task_state("submitted", "accept")            ->  "working"
    next_task_state("working", "need_input")          ->  "input_required"
    next_task_state("input_required", "provide_input")->  "working"
    next_task_state("completed", "cancel")            ->  ValueError

    Три разных сорта отказа, и все они ValueError:
      * состояния нет в TASK_STATES — опечатка в коде;
      * события нет в TASK_EVENTS — опечатка в коде;
      * пара есть, но перехода нет — попытка нарушить спецификацию.

    Из терминального состояния (TERMINAL_STATES) выхода нет ни по какому
    событию. Отменить завершённую задачу нельзя: клиент, который получил
    completed, уже забрал артефакт.
    """
    if state not in TASK_STATES:
        raise ValueError(f"unknown task state: {state}")
    if event not in TASK_EVENTS:
        raise ValueError(f"unknown task event: {event}")
    key = (state, event)
    if key not in TASK_TRANSITIONS:
        raise ValueError(f"illegal transition: {state} --{event}-->")
    return TASK_TRANSITIONS[key]


def make_artifact(name, mime_type, chunks):
    """Собрать Artifact из потока текстовых чанков.

    make_artifact("summary", "text/markdown", ["Hel", "lo"])
        ->  {"name": "summary", "mimeType": "text/markdown",
             "parts": [{"kind": "text", "text": "Hello"}]}
    make_artifact("summary", "text/markdown", [])
        ->  {"name": "summary", "mimeType": "text/markdown",
             "parts": [{"kind": "text", "text": ""}]}

    Стриминг в A2A нарезает артефакт произвольно, границы чанков ничего не
    значат. Поэтому склейка обязана быть без разделителей: ["Hel", "lo"] и
    ["Hello"] — один и тот же артефакт. Поставишь " ".join — и получишь
    разный результат в зависимости от того, как сеть порезала поток.

    Нестроковый чанк — TypeError, а не тихий str(): в поток случайно попал
    объект, и молча превращать его в "<object at 0x...>" нельзя.
    """
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError(f"artifact chunk must be str, got {type(chunk).__name__}")
    return {
        "name": name,
        "mimeType": mime_type,
        "parts": [{"kind": "text", "text": "".join(chunks)}],
    }


def run_task(task_id, card, skill_id, messages):
    """Проиграть tasks/send: принять сообщения и довести Task до конца.

    Возвращает словарь ровно с пятью ключами:
        {"id", "skillId", "state", "messages", "artifact"}

    Это и есть граница непрозрачности A2A: наружу видно состояние, переписку
    и артефакт — и ничего о том, как вызванный агент думал.

    Skill может требовать данные: {"requiredData": ["targetLength"]}. Пока
    их нет ни в одном data-part, задача висит в input_required и добавляет
    сообщение от агента с перечнем недостающего.

    run_task("t1", card, "no_such_skill", [])
        ->  state "rejected", artifact None
    run_task("t1", card, "draft_report", [msg_без_data])
        ->  state "input_required", artifact None
    run_task("t1", card, "draft_report", [msg_без_data, msg_с_targetLength])
        ->  state "completed", artifact со склеенным текстом обоих сообщений

    Артефакт собирается из text-part ВСЕХ пользовательских сообщений: то,
    что клиент дослал во втором сообщении, — часть той же задачи.

    Ловушка: входной список messages и его элементы править нельзя. Это
    данные вызывающего кода, а Task обязан хранить свою копию.
    """
    task = {
        "id": task_id,
        "skillId": skill_id,
        "state": "submitted",
        "messages": [],
        "artifact": None,
    }
    skill = next((s for s in card.get("skills", ()) if s["id"] == skill_id), None)
    if skill is None:
        # Незнакомый skill отклоняем сразу: rejected, а не failed. failed —
        # это "взялся и не смог", а мы даже не брались.
        task["state"] = next_task_state(task["state"], "reject")
        return task

    task["state"] = next_task_state(task["state"], "accept")
    required = tuple(skill.get("requiredData", ()))
    collected = {}
    for message in messages:
        if task["state"] in TERMINAL_STATES:
            break
        # своя копия: parts кладём в новый список, чужой мы не держим
        task["messages"].append(
            {"role": message["role"], "parts": list(message["parts"])}
        )
        for part in message["parts"]:
            if part["kind"] == "data":
                collected.update(part["data"])
        missing = [k for k in required if k not in collected]
        if missing:
            if task["state"] == "working":
                task["state"] = next_task_state(task["state"], "need_input")
            task["messages"].append(
                {
                    "role": "agent",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "input_required: " + ", ".join(missing),
                        }
                    ],
                }
            )
            continue
        if task["state"] == "input_required":
            task["state"] = next_task_state(task["state"], "provide_input")
        chunks = [
            part["text"]
            for m in task["messages"]
            if m["role"] == "user"
            for part in m["parts"]
            if part["kind"] == "text"
        ]
        task["artifact"] = make_artifact("summary", "text/markdown", chunks)
        task["state"] = next_task_state(task["state"], "finish")
    return task
