"""
Безопасность: секреты, PII и неизменяемый аудит

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l25-security-secrets-audit
Разбор:  /check-code p17-l25-security-secrets-audit
"""

import hashlib
import json
import re

SECRET_PATTERNS = (
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("AWS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b")),
)
POLICY = {
    "admin": ("call_model", "read_audit", "read_raw_prompt", "rotate_key"),
    "engineer": ("call_model", "read_audit"),
    "support": ("call_model",),
    "auditor": ("read_audit",),
}
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def is_allowed(role, action, policy=POLICY):
    """Разрешено ли роли выполнить действие. Deny by default.

    is_allowed("engineer", "call_model")     ->  True
    is_allowed("auditor", "call_model")      ->  False
    is_allowed("intern", "call_model")       ->  False   (роли нет в политике)

    Незнакомая роль обязана получить False, а не KeyError: опечатка в имени
    роли не должна ронять gateway, но и не должна открывать доступ.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def append_audit(chain, record):
    """Добавить запись в журнал. Вернуть НОВЫЙ список, старый не трогать.

    append_audit([], {"a": 1})  ->  [{'record': {'a': 1}, 'prev': '000...0',
                                      'hash': '<sha256>'}]

    Журнал append-only: старые записи не редактируются никогда. Возврат нового
    списка вместо мутации — самая дешёвая страховка от того, что кто-то
    случайно перепишет историю через ссылку на неё.

    Первая запись ссылается на GENESIS_HASH.
    """
    raise NotImplementedError


def verify_chain(chain):
    """Проверить цепочку. Вернуть индекс первой битой записи или -1.

    verify_chain([])  ->  -1   (пустой журнал цел по определению)

    Битой считается запись, у которой либо prev не совпал с хешем предыдущей,
    либо собственный hash не сходится с пересчитанным.

    Честная граница: цепочка ловит правку и перестановку записей, но НЕ ловит
    обрезание хвоста — первые k записей сами по себе валидны. Поэтому хеш
    последней записи (head) хранят отдельно, вне журнала.
    """
    raise NotImplementedError


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
    raise NotImplementedError
