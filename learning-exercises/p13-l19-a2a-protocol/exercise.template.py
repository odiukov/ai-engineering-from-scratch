"""
A2A — протокол общения агентов

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

import hashlib
import hmac
import json

AGENT_CARD_PATH = "/.well-known/agent.json"
SCHEMA_VERSION = "1.0"
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
