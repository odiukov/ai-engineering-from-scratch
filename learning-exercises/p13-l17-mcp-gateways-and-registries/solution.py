"""
MCP-гейтвеи и реестры — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib

# Приоритет источников серверов. Меньше — надёжнее.
# Official MCP Registry namespace-verified, метареестры агрегируют кого попало,
# unlisted — то, что кто-то принёс сам.
REGISTRY_RANK = {"official": 0, "metaregistry": 1, "unlisted": 2}


def merge_tool_namespaces(backends):
    """Слить tools/list всех бэкендов в одну таблицу маршрутизации.

    backends — dict {сервер: список имён инструментов}.
    Вернуть dict {внешнее имя: "сервер::инструмент"}.

    merge_tool_namespaces({"notes": ["search"], "github": ["open_pr"]})
      ->  {"search": "notes::search", "open_pr": "github::open_pr"}

    merge_tool_namespaces({"notes": ["search"], "archive": ["search"]})
      ->  {"archive.search": "archive::search", "notes.search": "notes::search"}

    Политика prefix-on-collision: уникальное имя остаётся коротким, а при
    столкновении ОБА имени получают префикс сервера. Именно оба: если
    префикс получит только второй, то «кто первый пришёл, того и имя»,
    и порядок обхода начнёт решать, чей инструмент зовёт разработчик.

    Серверы обходятся отсортированно — таблица маршрутизации обязана быть
    воспроизводимой.
    """
    owners = {}
    for server in sorted(backends):
        for tool in backends[server]:
            owners.setdefault(tool, []).append(server)

    routes = {}
    for tool, servers in owners.items():
        for server in servers:
            exposed = tool if len(servers) == 1 else f"{server}.{tool}"
            routes[exposed] = f"{server}::{tool}"
    return routes


def rbac_allows(policy, user, server, tool):
    """Разрешает ли политика этому пользователю звать сервер::инструмент.

    policy — dict {пользователь: набор шаблонов}. Шаблон — либо
    "сервер::инструмент", либо "сервер::*".

    rbac_allows({"alice": ["notes::*"]}, "alice", "notes", "search")  ->  True
    rbac_allows({"alice": ["notes::*"]}, "alice", "github", "open_pr") -> False
    rbac_allows({}, "bob", "notes", "search")                          -> False

    Запрет по умолчанию: пользователя нет в политике — значит нельзя.
    Гейтвей, который на неизвестного пользователя отвечает «ну ладно»,
    не гейтвей.

    Звёздочка работает только на уровне инструмента. "*::*" — обычная
    строка, которая не совпадёт ни с чем: раздавать доступ ко всем
    бэкендам одной записью слишком легко случайно.
    """
    patterns = policy.get(user, ())
    return f"{server}::{tool}" in patterns or f"{server}::*" in patterns


def pin_filter(tools, manifest):
    """Оставить только инструменты, чьё описание совпало с манифестом хэшей.

    tools — список dict с ключами server, name, description.
    manifest — dict {"сервер::инструмент": sha256 описания}.

    pin_filter([{"server": "n", "name": "s", "description": "Find"}],
               {"n::s": hashlib.sha256(b"Find").hexdigest()})
      ->  тот же список

    Выкидываются двое: инструменты, которых в манифесте нет (их никто
    не одобрял), и инструменты с изменившимся описанием (rug pull).
    Порядок оставшихся сохраняется.

    Это защита из урока 15, применённая централизованно: подменённое
    описание не доедет ни до одного разработчика в компании.
    """
    kept = []
    for tool in tools:
        key = f"{tool['server']}::{tool['name']}"
        digest = hashlib.sha256(tool["description"].encode("utf-8")).hexdigest()
        if manifest.get(key) == digest:
            kept.append(tool)
    return kept


def token_bucket_take(bucket, now, capacity, refill_per_second):
    """Снять один токен из бакета. Вернуть (allowed, новый бакет).

    bucket — dict {"tokens": float, "updated": время в секундах}.
    Вход НЕ менять: гейтвей обязан уметь откатить решение.

    token_bucket_take({"tokens": 1.0, "updated": 0}, 0, 3, 0.5)
      ->  (True, {"tokens": 0.0, "updated": 0})
    token_bucket_take({"tokens": 0.0, "updated": 0}, 0, 3, 0.5)
      ->  (False, {"tokens": 0.0, "updated": 0})

    Токены докапываются со скоростью refill_per_second и упираются
    в capacity — иначе простаивавший пользователь накопит тысячу
    вызовов и выпустит их залпом.

    Часы, ушедшие назад (now меньше bucket["updated"]), не должны
    добавлять токены: отрицательный интервал зажимается в ноль.
    """
    elapsed = max(0.0, now - bucket["updated"])
    tokens = min(float(capacity), bucket["tokens"] + elapsed * refill_per_second)
    if tokens >= 1.0:
        return (True, {"tokens": tokens - 1.0, "updated": now})
    return (False, {"tokens": tokens, "updated": now})


def audit_event(user, tool, verdict, now):
    """Одна запись в append-only журнале: кто, что, когда, чем кончилось.

    audit_event("alice", "notes.search", "ok", 1000)
      ->  {"at": 1000, "user": "alice", "tool": "notes.search", "verdict": "ok"}

    Неаутентифицированный вызов тоже пишется, с user = None. Журнал, в
    котором нет отказов, бесполезен для расследования: интересны как раз
    попытки, которые не прошли.
    """
    return {"at": now, "user": user, "tool": tool, "verdict": verdict}


def registry_rank(source):
    """Приоритет источника сервера: меньше — надёжнее.

    registry_rank("official")      ->  0
    registry_rank("metaregistry")  ->  1

    Неизвестный источник — ValueError. Молча дать неизвестному источнику
    любой ранг означает пустить в прод сервер, про который непонятно,
    откуда он взялся.
    """
    if source not in REGISTRY_RANK:
        raise ValueError(f"неизвестный источник реестра: {source!r}")
    return REGISTRY_RANK[source]


def choose_server(candidates):
    """Выбрать, откуда ставить сервер. Вернуть имя или None.

    candidates — список dict с ключами name, source, verified.

    choose_server([{"name": "io.github.a/notes", "source": "metaregistry",
                    "verified": True},
                   {"name": "io.github.a/notes", "source": "official",
                    "verified": True}])
      ->  "io.github.a/notes"   (official побеждает метареестр)

    Правила: кандидаты без namespace-верификации отбрасываются целиком,
    из оставшихся берётся наименьший registry_rank, при равенстве —
    первый по алфавиту (чтобы выбор был воспроизводим).

    None означает «ставить нечего» — и это нормальный ответ. Именно
    попытка «ну хоть что-нибудь поставить» привела к истории с
    поддельным Postmark MCP.
    """
    allowed = [c for c in candidates if c.get("verified")]
    if not allowed:
        return None
    best = min(allowed, key=lambda c: (registry_rank(c["source"]), c["name"]))
    return best["name"]


def handle_call(gateway, bearer, exposed_name, now):
    """Полный путь вызова через гейтвей. Вернуть dict со status и подробностями.

    gateway — dict с ключами routes, policy, sessions, buckets, limit, audit.

    Порядок проверок и коды:
      1. bearer нет в sessions            -> 401 unauthenticated;
      2. лимит исчерпан                   -> 429 rate_limited;
      3. exposed_name нет в routes        -> 404 unknown_tool;
      4. RBAC запрещает                   -> 403 forbidden;
      5. всё сошлось                      -> 200 ok, в ответе "route".

    Лимит снимается сразу после аутентификации, ДО проверки прав: иначе
    пользователь без доступа бесплатно долбит гейтвей отказами, и защита
    от абьюза защищает только тех, кому и так можно.

    Каждый исход дописывается в gateway["audit"] через audit_event —
    журнал заполняется всегда, включая отказы.
    """
    user = gateway["sessions"].get(bearer)
    if user is None:
        gateway["audit"].append(audit_event(None, exposed_name, "unauthenticated", now))
        return {"status": 401, "verdict": "unauthenticated"}

    bucket = gateway["buckets"].get(user, {"tokens": 0.0, "updated": now})
    allowed, bucket = token_bucket_take(
        bucket, now, gateway["limit"]["capacity"], gateway["limit"]["refill_per_second"]
    )
    gateway["buckets"][user] = bucket
    if not allowed:
        gateway["audit"].append(audit_event(user, exposed_name, "rate_limited", now))
        return {"status": 429, "verdict": "rate_limited"}

    route = gateway["routes"].get(exposed_name)
    if route is None:
        gateway["audit"].append(audit_event(user, exposed_name, "unknown_tool", now))
        return {"status": 404, "verdict": "unknown_tool"}

    server, tool = route.split("::")
    if not rbac_allows(gateway["policy"], user, server, tool):
        gateway["audit"].append(audit_event(user, exposed_name, "forbidden", now))
        return {"status": 403, "verdict": "forbidden"}

    gateway["audit"].append(audit_event(user, exposed_name, "ok", now))
    return {"status": 200, "verdict": "ok", "route": route}
