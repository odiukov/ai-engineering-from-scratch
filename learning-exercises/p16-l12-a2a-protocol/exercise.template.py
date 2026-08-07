"""
Протокол A2A

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

import json

WELL_KNOWN_PATH = "/.well-known/agent.json"
CARD_REQUIRED = ("name", "version", "skills", "endpoints", "auth", "modalities", "protocol_version")
MODALITIES = ("text", "structured", "image", "audio", "video")
TERMINAL_STATES = ("completed", "failed", "canceled")
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
    pass


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
    raise NotImplementedError


def encode_card(card):
    """Карточка в JSON-строку. Ключи отсортированы, поэтому байты стабильны.

    encode_card(card)  ->  '{"auth": {"type": "none"}, "endpoints": ...}'

    Перед кодированием проверь, что все CARD_REQUIRED на месте: выложить
    карточку без "auth" — значит заставить клиента гадать, как к тебе
    стучаться. Это A2AProtocolError, а не «ну и ладно».

    Сортировка ключей нужна не для красоты: одинаковая карточка обязана
    давать одинаковую строку, иначе ETag и кэш discovery не работают.
    """
    raise NotImplementedError


def decode_card(text):
    """Разбор карточки, пришедшей по сети. Кортежи приезжают списками.

    decode_card(encode_card(card))["name"]  ->  'code-review-agent'

    Две разные беды с одинаковым исходом: битый JSON и валидный JSON без
    обязательных полей. Обе — A2AProtocolError, чтобы вызывающему не
    приходилось ловить ещё и json.JSONDecodeError отдельно.

    Ловушка: JSON не знает кортежей. Поле skills вернётся списком, и
    сравнивать карточку «до» и «после» надо с учётом этого.
    """
    raise NotImplementedError


def supports_skill(card, skill):
    """Умеет ли агент этот навык. Это весь discovery со стороны клиента.

    supports_skill(card, "review-python")  ->  True
    supports_skill(card, "summarize")      ->  False

    Работает и с карточкой после decode_card, где skills стал списком:
    проверка на вхождение не зависит от типа контейнера.
    """
    raise NotImplementedError


def make_task(task_id, skill, payload):
    """Новая задача в состоянии submitted.

    make_task("t-1", "review-python", {"code": "x = 1"})
        ->  {'id': 't-1', 'skill': 'review-python', 'payload': {'code': 'x = 1'},
             'state': 'submitted', 'artifact': None}

    Пустой id ломает идемпотентность из чек-листа урока: повторную отправку
    после ретрая нечем узнать. Требуй непустой — A2AProtocolError.
    """
    raise NotImplementedError


def make_artifact(kind, data):
    """Типизированный результат задачи.

    make_artifact("text", "looks fine")   ->  {'type': 'text', 'data': 'looks fine'}
    make_artifact("structured", {"issues": []})
        ->  {'type': 'structured', 'data': {'issues': []}}

    Модальность вне MODALITIES — A2AProtocolError: смысл типизированных
    артефактов ровно в том, что принимающая сторона знает заранее, что ей
    приедет. Свободная строка в поле type это поле обесценивает.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
