"""
Безопасность: секреты, PII и неизменяемый аудит — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import json
import re

# Детекторы секретов в порядке приоритета: чем выше, тем «главнее» при
# пересечении. sk-ant-... подходит и под шаблон OpenAI, поэтому Anthropic
# стоит первым — иначе ключ Anthropic будет отмаскирован как OpenAI.
#
# \b перед префиксом обязателен. Без него "task-1234567890abcdefgh" внутри
# обычного текста опознаётся как ключ: подстрока "sk-..." там есть.
SECRET_PATTERNS = (
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("AWS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b")),
)

# RBAC: роль -> что ей разрешено. Неизвестная роль не получает ничего.
POLICY = {
    "admin": ("call_model", "read_audit", "read_raw_prompt", "rotate_key"),
    "engineer": ("call_model", "read_audit"),
    "support": ("call_model",),
    "auditor": ("read_audit",),
}

# Точка отсчёта цепочки хешей. Первая запись ссылается на неё.
GENESIS_HASH = "0" * 64


def find_secrets(text):
    """Найти секреты и PII в тексте. Список словарей, отсортированный по start.

    Ключи словаря: kind, value, start, end.

    find_secrets("ключ sk-ant-api03-AAAABBBBCCCCDDDD в середине")
        ->  [{'kind': 'ANTHROPIC_KEY', 'value': 'sk-ant-api03-AAAABBBBCCCCDDDD',
              'start': 5, 'end': 34}]
    find_secrets("Ask-me about task-2024")   ->  []
    find_secrets("нет секретов")             ->  []

    Две ловушки сразу:

    1. Секрет живёт не только в начале строки. Ищи по всему тексту, а не
       проверяй префикс.
    2. Похожий несекрет резать нельзя. "Ask-me", "task-2024", "skylight"
       содержат "sk-"/"sk" внутри слова — граница слова \\b их отсекает.

    Пересечения снимаются по приоритету SECRET_PATTERNS: если два шаблона
    поймали один и тот же кусок, побеждает тот, что выше в списке.
    """
    hits = []
    for priority, (kind, pattern) in enumerate(SECRET_PATTERNS):
        for m in pattern.finditer(text):
            hits.append((m.start(), priority, m.end(), kind, m.group(0)))
    # сортировка по (start, priority): при одинаковом начале первым идёт
    # шаблон с большим приоритетом, и он занимает диапазон
    hits.sort()
    out = []
    covered_until = -1
    for start, _priority, end, kind, value in hits:
        if start < covered_until:
            continue
        out.append({"kind": kind, "value": value, "start": start, "end": end})
        covered_until = end
    return out


def placeholder_for(kind, value, table):
    """Стабильный placeholder для значения. table пополняется на месте.

    placeholder_for("SSN", "123-45-6789", {})   ->  '[SSN_001]'

    t = {}
    placeholder_for("SSN", "123-45-6789", t)    ->  '[SSN_001]'
    placeholder_for("SSN", "999-88-7777", t)    ->  '[SSN_002]'
    placeholder_for("SSN", "123-45-6789", t)    ->  '[SSN_001]'  (то же значение)

    Consistent tokenization: одно и то же значение обязано получать один и тот
    же placeholder всегда. Иначе модель перестаёт видеть, что «отправитель» и
    «получатель» в двух предложениях — один человек, и ответ разваливается.

    Нумерация идёт внутри kind: счётчик SSN не зависит от счётчика EMAIL.
    """
    if value in table:
        return table[value]
    prefix = "[" + kind + "_"
    # номер выводится из уже выданных placeholder'ов, а не хранится отдельным
    # счётчиком: одна структура — один источник правды
    number = 1 + sum(1 for p in table.values() if p.startswith(prefix))
    token = "%s%03d]" % (prefix, number)
    table[value] = token
    return token


def redact(text, table):
    """Заменить все секреты в тексте на стабильные placeholder'ы.

    t = {}
    redact("SSN 123-45-6789, mail a@b.com", t)  ->  'SSN [SSN_001], mail [EMAIL_001]'
    redact("снова 123-45-6789", t)              ->  'снова [SSN_001]'
    redact("чистый текст", t)                   ->  'чистый текст'

    Ловушка со смещениями: placeholder короче или длиннее оригинала, поэтому
    после первой замены все последующие start/end уже врут. Режь с конца строки
    к началу — тогда несделанные замены остаются левее и не съезжают.

    Вторая ловушка: номера placeholder'ов должны раздаваться в порядке чтения,
    слева направо, иначе второй секрет в строке получит номер 001.
    """
    findings = find_secrets(text)
    # номера раздаём в порядке чтения...
    tokens = [placeholder_for(f["kind"], f["value"], table) for f in findings]
    out = text
    # ...а склеиваем с конца, чтобы смещения слева оставались верными
    for finding, token in zip(reversed(findings), reversed(tokens)):
        out = out[: finding["start"]] + token + out[finding["end"] :]
    return out


def is_allowed(role, action, policy=POLICY):
    """Разрешено ли роли выполнить действие. Deny by default.

    is_allowed("engineer", "call_model")     ->  True
    is_allowed("auditor", "call_model")      ->  False
    is_allowed("intern", "call_model")       ->  False   (роли нет в политике)

    Незнакомая роль обязана получить False, а не KeyError: опечатка в имени
    роли не должна ронять gateway, но и не должна открывать доступ.
    """
    return action in policy.get(role, ())


def chain_hash(prev_hash, record):
    """Хеш записи, привязанный к хешу предыдущей. Шестнадцатеричная строка.

    chain_hash("0"*64, {"a": 1}) == chain_hash("0"*64, {"a": 1})   ->  True
    chain_hash("0"*64, {"a": 1}) == chain_hash("1"*64, {"a": 1})   ->  False

    Именно привязка к prev_hash делает журнал цепочкой: подменить запись в
    середине нельзя, не пересчитав все последующие.

    Ловушка сериализации: json.dumps без sort_keys выдаёт разный текст для
    одинаковых по смыслу словарей, и хеш начинает зависеть от порядка ключей.
    Проверка целостности после этого краснеет на ровном месте.
    """
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return hashlib.sha256((prev_hash + "|" + payload).encode("utf-8")).hexdigest()


def append_audit(chain, record):
    """Добавить запись в журнал. Вернуть НОВЫЙ список, старый не трогать.

    append_audit([], {"a": 1})  ->  [{'record': {'a': 1}, 'prev': '000...0',
                                      'hash': '<sha256>'}]

    Журнал append-only: старые записи не редактируются никогда. Возврат нового
    списка вместо мутации — самая дешёвая страховка от того, что кто-то
    случайно перепишет историю через ссылку на неё.

    Первая запись ссылается на GENESIS_HASH.
    """
    prev = chain[-1]["hash"] if chain else GENESIS_HASH
    entry = {"record": record, "prev": prev, "hash": chain_hash(prev, record)}
    return chain + [entry]


def verify_chain(chain):
    """Проверить цепочку. Вернуть индекс первой битой записи или -1.

    verify_chain([])  ->  -1   (пустой журнал цел по определению)

    Битой считается запись, у которой либо prev не совпал с хешем предыдущей,
    либо собственный hash не сходится с пересчитанным.

    Честная граница: цепочка ловит правку и перестановку записей, но НЕ ловит
    обрезание хвоста — первые k записей сами по себе валидны. Поэтому хеш
    последней записи (head) хранят отдельно, вне журнала.
    """
    prev = GENESIS_HASH
    for i, entry in enumerate(chain):
        if entry["prev"] != prev:
            return i
        if entry["hash"] != chain_hash(prev, entry["record"]):
            return i
        prev = entry["hash"]
    return -1


def audit_llm_call(chain, table, now, user, role, tenant, model, prompt, response,
                   policy=POLICY):
    """Полный путь вызова: проверка прав, редактирование, запись в журнал.

    Вернуть кортеж (новая цепочка, отредактированный промпт).
    Если роли нельзя call_model — вернуть (цепочка с записью об отказе, None).

    chain, safe = audit_llm_call([], {}, "2026-08-07T10:00:00Z", "u1",
                                 "engineer", "t1", "claude",
                                 "SSN 123-45-6789", "ok")
    safe                      ->  'SSN [SSN_001]'
    chain[0]["record"]["allowed"]        ->  True
    chain[0]["record"]["secrets_found"]  ->  1

    Время приходит параметром now, а не берётся из time.time(): аудит должен
    воспроизводиться в тесте и в постмортеме, а не зависеть от момента запуска.

    В запись НИКОГДА не попадает сырой промпт — только хеш отредактированного.
    Аудит-журнал живёт дольше, чем данные в нём: SOC 2 требует год, HIPAA — шесть.
    """
    if not is_allowed(role, "call_model", policy):
        denied = {
            "ts": now, "user": user, "role": role, "tenant": tenant,
            "model": model, "action": "call_model", "allowed": False,
            "prompt_hash": None, "response_hash": None, "secrets_found": 0,
        }
        return (append_audit(chain, denied), None)

    secrets_found = len(find_secrets(prompt))
    safe_prompt = redact(prompt, table)
    record = {
        "ts": now, "user": user, "role": role, "tenant": tenant,
        "model": model, "action": "call_model", "allowed": True,
        # хеш считается от ОТРЕДАКТИРОВАННОГО текста: хеш сырого промпта — это
        # всё ещё канал утечки, по нему подбирается короткий секрет вроде SSN
        "prompt_hash": hashlib.sha256(safe_prompt.encode("utf-8")).hexdigest()[:16],
        "response_hash": hashlib.sha256(response.encode("utf-8")).hexdigest()[:16],
        "secrets_found": secrets_found,
    }
    return (append_audit(chain, record), safe_prompt)
