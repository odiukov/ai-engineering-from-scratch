"""
Compliance: матрица контролей, полей и сроков хранения

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l26-compliance-frameworks
Разбор:  /check-code p17-l26-compliance-frameworks
"""

from datetime import date

CONTROL_MAP = {
    "access logging": {
        "ISO 27001": "A.5.15-5.18",
        "GDPR": "Art. 32",
        "HIPAA": "§164.312(a)",
        "SOC 2": "CC6",
    },
    "change management": {
        "ISO 27001": "A.8.32",
        "PCI DSS": "Req. 6",
        "SOC 2": "CC8",
    },
    "encryption in transit": {
        "ISO 27001": "A.8.24",
        "GDPR": "Art. 32",
        "HIPAA": "§164.312(e)",
        "PCI DSS": "Req. 4",
    },
    "secrets management": {
        "ISO 27001": "A.8.19",
        "PCI DSS": "Req. 8",
        "SOC 2": "CC6.1",
    },
    "PII redaction (inference-time)": {
        "GDPR": "Art. 25",
        "EU AI Act": "Art. 10",
        "HIPAA": "§164.514",
    },
    "audit log retention": {
        "SOC 2": "CC7",
        "HIPAA": "§164.312(b)",
        "ISO 27001": "A.8.15",
    },
    "conformity assessment": {"EU AI Act": "Art. 43"},
    "impact assessment": {
        "EU AI Act": "Art. 27",
        "Colorado AI Act": "SB24-205",
        "ISO 42001": "6.1.4",
    },
    "data subject rights": {"GDPR": "Ch. III"},
    "BAA signed": {"HIPAA": "§164.504(e)"},
    "human oversight": {"EU AI Act": "Art. 14", "ISO 42001": "8.3"},
}
PROFILE_MAP = {
    ("US", "B2B SaaS"): ("SOC 2", "ISO 27001"),
    ("US", "healthcare"): ("SOC 2", "HIPAA", "ISO 27001"),
    ("US", "fintech"): ("SOC 2", "PCI DSS", "ISO 27001"),
    ("US-CO", "B2B SaaS"): ("SOC 2", "Colorado AI Act", "ISO 27001"),
    ("EU", "B2B SaaS"): ("GDPR", "SOC 2", "ISO 27001", "EU AI Act"),
    ("EU", "healthcare"): ("GDPR", "SOC 2", "EU AI Act", "ISO 27001"),
    ("Global", "enterprise"): ("SOC 2", "ISO 27001", "ISO 42001", "GDPR",
                               "HIPAA", "EU AI Act"),
}
RETENTION_DAYS = {
    "SOC 2": 365,
    "GDPR": 365,
    "PCI DSS": 365,
    "ISO 27001": 1095,
    "ISO 42001": 1095,
    "Colorado AI Act": 1095,
    "HIPAA": 2190,
    "EU AI Act": 2190,
}
FIELD_REQUIREMENTS = {
    "SOC 2": ("ts", "user", "action", "model", "prompt_hash", "response_hash"),
    "HIPAA": ("ts", "user", "tenant", "action", "prompt_hash", "phi_redacted"),
    "GDPR": ("ts", "user", "prompt_hash", "pii_redacted", "legal_basis"),
    "PCI DSS": ("ts", "user", "action", "prompt_hash"),
    "EU AI Act": ("ts", "user", "model", "model_version", "risk_tier",
                  "prompt_hash", "response_hash"),
    "Colorado AI Act": ("ts", "user", "decision_outcome", "appeal_channel"),
    "ISO 27001": ("ts", "user", "action"),
    "ISO 42001": ("ts", "model", "model_version", "human_review"),
}


def frameworks_for(control, control_map=CONTROL_MAP):
    """Какие фреймворки закрывает один контроль. Отсортированный список.

    frameworks_for("data subject rights")  ->  ['GDPR']
    frameworks_for("access logging")
        ->  ['GDPR', 'HIPAA', 'ISO 27001', 'SOC 2']

    Ловушка: соблазн вернуть [] для неизвестного контроля. Тогда опечатка в
    названии («acess logging») превращается в тихий ноль покрытия, и матрица
    врёт аудитору. Незнакомый контроль — это ValueError.
    """
    raise NotImplementedError


def required_frameworks(geo, segment, profile_map=PROFILE_MAP):
    """Какие фреймворки требует профиль клиента. Отсортированный список.

    required_frameworks("US", "healthcare")  ->  ['HIPAA', 'ISO 27001', 'SOC 2']
    required_frameworks("EU", "B2B SaaS")
        ->  ['EU AI Act', 'GDPR', 'ISO 27001', 'SOC 2']

    Неизвестная пара — ValueError по той же причине: «клиент не подошёл ни под
    один профиль» обязано быть видно, а не выглядеть как «требований нет».
    """
    raise NotImplementedError


def required_controls(frameworks, control_map=CONTROL_MAP):
    """Какие контроли надо внедрить ради заданного набора фреймворков.

    required_controls(["GDPR"])
        ->  ['PII redaction (inference-time)', 'access logging',
             'data subject rights', 'encryption in transit']
    required_controls([])  ->  []

    Контроль попадает в список, если закрывает хотя бы один из фреймворков.
    Сортировка — обычная строковая, поэтому заглавные буквы идут раньше строчных.
    """
    raise NotImplementedError


def coverage_gaps(implemented, frameworks, control_map=CONTROL_MAP):
    """Чего не хватает. Словарь: фреймворк -> отсортированный список дыр.

    Фреймворки без дыр в словарь не попадают — пустой словарь значит «покрыто».

    coverage_gaps(required_controls(["GDPR"]), ["GDPR"])  ->  {}
    coverage_gaps([], ["HIPAA"])
        ->  {'HIPAA': ['BAA signed', 'PII redaction (inference-time)',
                       'access logging', 'audit log retention',
                       'encryption in transit']}

    Фреймворк, для которого в матрице нет НИ ОДНОГО контроля, дырой не считается
    — это дыра в самой матрице, и её видно по пустому required_controls.
    """
    raise NotImplementedError


def required_log_fields(frameworks, field_requirements=FIELD_REQUIREMENTS):
    """Объединение полей журнала по всем фреймворкам. Отсортированный список.

    required_log_fields(["ISO 27001"])  ->  ['action', 'ts', 'user']
    required_log_fields(["ISO 27001", "GDPR"])
        ->  ['action', 'legal_basis', 'pii_redacted', 'prompt_hash', 'ts', 'user']

    Объединение, а не пересечение: журнал один на всех, и он обязан удовлетворить
    самый требовательный фреймворк в списке.
    """
    raise NotImplementedError


def record_is_complete(record, frameworks, field_requirements=FIELD_REQUIREMENTS):
    """Хватает ли записи полей. Кортеж (полна ли, отсортированный список дыр).

    record_is_complete({"ts": 1, "user": "u", "action": "call"}, ["ISO 27001"])
        ->  (True, [])
    record_is_complete({"ts": 1, "user": "u"}, ["ISO 27001"])
        ->  (False, ['action'])

    Ловушка: поле, лежащее в записи со значением None, аудит не засчитает —
    «legal_basis: null» это отсутствие правового основания, а не его наличие.
    Пустая строка и ноль тоже не годятся, а вот False — вполне значение.
    """
    raise NotImplementedError


def retention_days(frameworks, table=RETENTION_DAYS):
    """Сколько дней хранить журнал: максимум по всем фреймворкам.

    retention_days(["SOC 2"])            ->  365
    retention_days(["SOC 2", "HIPAA"])   ->  2190
    retention_days([])                   ->  0

    Максимум, а не минимум и не среднее: один HIPAA в списке тянет хранение
    с года до шести лет для всего журнала.
    """
    raise NotImplementedError


def records_to_purge(records, frameworks, now, table=RETENTION_DAYS):
    """Какие записи journal'а пора удалить на дату now.

    Записи с полем ts вида "YYYY-MM-DD..." (годится и полный ISO-таймстамп).
    Возвращается список самих записей в исходном порядке.

    recs = [{"ts": "2024-01-01"}, {"ts": "2026-08-01"}]
    records_to_purge(recs, ["SOC 2"], "2026-08-07")  ->  [{'ts': '2024-01-01'}]

    Граница включающая: запись возрастом ровно retention_days ещё хранится,
    удаляется та, что старше. Ошибка на единицу тут стоит дорого — удалённая
    на день раньше запись это провал аудита.
    """
    raise NotImplementedError
