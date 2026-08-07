"""
Наследие FIPA-ACL и речевые акты

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p16-l02-fipa-acl-heritage
Разбор:  /check-code p16-l02-fipa-acl-heritage
"""

PERFORMATIVES = frozenset({
    "inform", "request", "query-if", "query-ref", "propose",
    "accept-proposal", "reject-proposal", "agree", "refuse",
    "confirm", "disconfirm", "not-understood", "cfp",
    "subscribe", "cancel", "failure",
})
ENVELOPE = (
    ("sender", ":sender"),
    ("receiver", ":receiver"),
    ("content", ":content"),
    ("language", ":language"),
    ("ontology", ":ontology"),
    ("protocol", ":protocol"),
    ("conversation_id", ":conversation-id"),
    ("reply_with", ":reply-with"),
    ("in_reply_to", ":in-reply-to"),
)
REQUIRED = ("performative", "sender", "receiver")


class ACLError(Exception):
    """Базовая ошибка конверта FIPA-ACL.

    Собственный класс, а не ValueError и тем более не RuntimeError: тесты
    обязаны отличать «функция отвергла плохое сообщение» от «функция ещё
    не написана».
    """
    pass


class UnknownPerformativeError(ACLError):
    """Performative нет в каталоге FIPA."""
    pass


class MissingFieldError(ACLError):
    """В конверте нет обязательного поля."""
    pass


def make_message(performative, sender, receiver, content, *, language="SL0",
                 ontology="default", protocol=None, conversation_id=None,
                 reply_with=None, in_reply_to=None):
    """Конверт FIPA-ACL как обычный dict. Проверяет performative и обязательные поля.

    make_message("inform", "a1", "a2", "((price IBM 83))")["performative"]
        ->  "inform"
    make_message("inform", "a1", "a2", "x")["language"]   ->  "SL0"

    make_message("yell", "a1", "a2", "x")   ->  UnknownPerformativeError
    make_message("inform", "", "a2", "x")   ->  MissingFieldError

    Ловушка: пустая строка в sender — это отсутствующее поле, а не «поле
    есть, просто короткое». FIPA требует адресуемость обоих концов.

    Соответствует разделу 4.1 FIPA ACL Message Structure: семь полей
    конверта плюс content с полезной нагрузкой.
    """
    raise NotImplementedError


def render(message):
    """Каноническая текстовая форма конверта. Пустые поля не печатаются.

    render(make_message("inform", "a1", "a2", "((price IBM 83))")) даёт
    строку из шести строчек:

        (inform
          :sender a1
          :receiver a2
          :content '((price IBM 83))'
          :language SL0
          :ontology default
        )

    content печатается через repr (он может быть dict), остальные поля —
    как есть. Поля со значением None пропускаются: конверт без protocol не
    должен печатать пустой ":protocol".
    """
    raise NotImplementedError


def reply_to(message, performative, content, **overrides):
    """Ответ на сообщение: адреса меняются местами, нить разговора сохраняется.

    conversation_id уходит в ответ БЕЗ изменений — это и есть нить.
    reply_with исходного сообщения становится in_reply_to ответа.

    m = make_message("cfp", "mgr", "w1", "task", conversation_id="cn-1",
                     reply_with="cfp-w1")
    r = reply_to(m, "propose", {"price": 3})
    r["sender"], r["receiver"]        ->  ("w1", "mgr")
    r["conversation_id"]              ->  "cn-1"
    r["in_reply_to"]                  ->  "cfp-w1"

    Ловушка: новый conversation_id на ответ — самый частый способ порвать
    нить. Тогда менеджер не сможет связать заявку со своим cfp.
    """
    raise NotImplementedError


def mcp_to_acl(request):
    """JSON-RPC запрос MCP -> конверт FIPA-ACL по таблице соответствий урока.

    tools/call     ->  request    (это «сделай X»)
    resources/read ->  query-ref  (это «какое значение у X»)

    mcp_to_acl({"jsonrpc": "2.0", "method": "tools/call", "id": 42,
                "params": {"name": "lookup_stock",
                           "arguments": {"symbol": "IBM"}}})["performative"]
        ->  "request"

    Любой другой метод  ->  ACLError

    conversation_id собирается из id запроса: именно он играет роль
    correlation id, ради которого FIPA держал отдельное поле.
    """
    raise NotImplementedError


def cfp(manager, bidders, task, conversation_id):
    """Объявление задачи: по одному cfp каждому исполнителю. Порядок сохраняется.

    len(cfp("mgr", ["w1", "w2"], "compress logs", "cn-1"))  ->  2
    cfp("mgr", ["w1"], "t", "cn-1")[0]["reply_with"]        ->  "cfp-w1"
    cfp("mgr", [], "t", "cn-1")                             ->  []

    reply_with уникален для каждого адресата, иначе менеджер не поймёт, на
    какое именно приглашение пришла заявка.
    """
    raise NotImplementedError


def collect_proposals(log, conversation_id):
    """Заявки из журнала: только propose и только по нужной нити разговора.

    Журнал общий на все переговоры, поэтому фильтр по conversation_id
    обязателен — иначе задача уйдёт исполнителю из чужого аукциона.

    collect_proposals([], "cn-1")  ->  []
    """
    raise NotImplementedError


def award(proposals, score=None):
    """Присуждение: (accept-proposal победителю, [reject-proposal остальным]).

    score(content) -> число, меньше значит лучше. По умолчанию цена.

    award([])  ->  (None, [])

    Отправителя брать неоткуда не надо: получатель заявки и есть менеджер,
    так что ответы строятся через reply_to.

    Без заявок присуждать нечего: возвращается пара (None, []), а не
    исключение — молчание исполнителей это нормальный исход аукциона.
    При равенстве очков выигрывает тот, чья заявка пришла раньше.
    """
    raise NotImplementedError


def run_contract_net(manager, bidders, task, conversation_id, bids, score=None):
    """Полный протокол Contract Net: cfp -> propose -> accept/reject. Журнал целиком.

    bids — dict {исполнитель: содержимое заявки}. Кто не в bids, тот
    промолчал: приглашение ему ушло, заявки от него нет.

    log = run_contract_net("mgr", ["w1", "w2"], "t", "cn-1",
                           {"w1": {"price": 3}, "w2": {"price": 2}})
    [m["performative"] for m in log]
        ->  ['cfp', 'cfp', 'propose', 'propose', 'accept-proposal', 'reject-proposal']

    run_contract_net("mgr", ["w1"], "t", "cn-1", {}) даёт журнал из одного
    cfp: без заявок присуждения не происходит.

    Это FIPA Contract Net Interaction Protocol (fipa00029) в миниатюре.
    """
    raise NotImplementedError
