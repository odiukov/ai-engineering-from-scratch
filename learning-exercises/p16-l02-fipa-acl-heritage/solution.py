"""
Наследие FIPA-ACL и речевые акты — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Каталог performatives из FIPA ACL Message Structure (fipa00037), тот же
# набор, что перечислен в уроке. Полный список спецификации длиннее.
PERFORMATIVES = frozenset({
    "inform", "request", "query-if", "query-ref", "propose",
    "accept-proposal", "reject-proposal", "agree", "refuse",
    "confirm", "disconfirm", "not-understood", "cfp",
    "subscribe", "cancel", "failure",
})

# Порядок полей конверта при рендере. Слева — ключ в dict, справа — имя
# поля в текстовом синтаксисе FIPA.
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

# Поля, без которых конверт не конверт.
REQUIRED = ("performative", "sender", "receiver")


class ACLError(Exception):
    """Базовая ошибка конверта FIPA-ACL.

    Собственный класс, а не ValueError и тем более не RuntimeError: тесты
    обязаны отличать «функция отвергла плохое сообщение» от «функция ещё
    не написана».
    """


class UnknownPerformativeError(ACLError):
    """Performative нет в каталоге FIPA."""


class MissingFieldError(ACLError):
    """В конверте нет обязательного поля."""


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
    if performative not in PERFORMATIVES:
        raise UnknownPerformativeError(f"unknown performative: {performative}")
    msg = {
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "language": language,
        "ontology": ontology,
        "protocol": protocol,
        "conversation_id": conversation_id,
        "reply_with": reply_with,
        "in_reply_to": in_reply_to,
    }
    # проверяем после сборки: так одинаково ловятся и None, и ""
    for field in REQUIRED:
        if not msg.get(field):
            raise MissingFieldError(f"missing required field: {field}")
    return msg


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
    lines = [f"({message['performative']}"]
    for key, label in ENVELOPE:
        value = message.get(key)
        if value is None:
            continue
        # content — полезная нагрузка, у неё бывает структура; repr делает
        # её однозначной. Остальные поля — плоские идентификаторы.
        shown = repr(value) if key == "content" else value
        lines.append(f"  {label} {shown}")
    lines.append(")")
    return "\n".join(lines)


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
    fields = {
        "language": message.get("language", "SL0"),
        "ontology": message.get("ontology", "default"),
        "protocol": message.get("protocol"),
        "conversation_id": message.get("conversation_id"),
        "in_reply_to": message.get("reply_with"),
    }
    fields.update(overrides)
    return make_message(performative, message["receiver"], message["sender"],
                        content, **fields)


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
    method = request.get("method")
    params = request.get("params", {})
    conv = f"jsonrpc-{request.get('id')}"
    if method == "tools/call":
        return make_message(
            "request", "host", "tool-server", params.get("arguments", {}),
            language="JSON", ontology=params["name"], protocol="fipa-request",
            conversation_id=conv, reply_with=f"msg-{request.get('id')}",
        )
    if method == "resources/read":
        return make_message(
            "query-ref", "host", "resource-server", params["uri"],
            language="URI", ontology="mcp-resource", protocol="fipa-query",
            conversation_id=conv, reply_with=f"msg-{request.get('id')}",
        )
    raise ACLError(f"no FIPA mapping for method: {method}")


def cfp(manager, bidders, task, conversation_id):
    """Объявление задачи: по одному cfp каждому исполнителю. Порядок сохраняется.

    len(cfp("mgr", ["w1", "w2"], "compress logs", "cn-1"))  ->  2
    cfp("mgr", ["w1"], "t", "cn-1")[0]["reply_with"]        ->  "cfp-w1"
    cfp("mgr", [], "t", "cn-1")                             ->  []

    reply_with уникален для каждого адресата, иначе менеджер не поймёт, на
    какое именно приглашение пришла заявка.
    """
    return [
        make_message(
            "cfp", manager, bidder, task,
            ontology="contract-net", protocol="fipa-contract-net",
            conversation_id=conversation_id, reply_with=f"cfp-{bidder}",
        )
        for bidder in bidders
    ]


def collect_proposals(log, conversation_id):
    """Заявки из журнала: только propose и только по нужной нити разговора.

    Журнал общий на все переговоры, поэтому фильтр по conversation_id
    обязателен — иначе задача уйдёт исполнителю из чужого аукциона.

    collect_proposals([], "cn-1")  ->  []
    """
    return [
        m for m in log
        if m["performative"] == "propose" and m["conversation_id"] == conversation_id
    ]


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
    if not proposals:
        return (None, [])
    key = score if score is not None else (lambda content: content["price"])
    # min стабилен: при равных очках берётся первый в списке
    winner = min(proposals, key=lambda m: key(m["content"]))
    accept = reply_to(winner, "accept-proposal", "awarded")
    rejects = [
        reply_to(m, "reject-proposal", "not awarded")
        for m in proposals if m is not winner
    ]
    return (accept, rejects)


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
    log = cfp(manager, bidders, task, conversation_id)
    invitations = {m["receiver"]: m for m in log}
    for bidder, content in bids.items():
        if bidder not in invitations:
            continue  # заявка от того, кого не звали, в аукцион не входит
        # ответ строим через reply_to, чтобы нить разговора не порвалась
        log.append(reply_to(invitations[bidder], "propose", content,
                            reply_with=f"propose-{bidder}"))
    accept, rejects = award(collect_proposals(log, conversation_id), score)
    if accept is not None:
        log.append(accept)
        log.extend(rejects)
    return log
