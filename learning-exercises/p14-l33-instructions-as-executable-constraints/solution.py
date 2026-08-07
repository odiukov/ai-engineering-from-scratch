"""
Инструкции агента как исполняемые ограничения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import fnmatch
import re

# Пять категорий, которые покрывают почти любое правило. Порядок канонический:
# по нему сортируется lock-файл.
CATEGORIES = ("startup", "forbidden", "definition_of_done", "uncertainty", "approval")

# Severity ставится при написании правила, а не потом под давлением дедлайна.
SEVERITIES = ("block", "warn", "info")
DEFAULT_SEVERITY = "warn"

# Имена проверок, которые умеет compile_rule. Правило со ссылкой на что-то
# другое неисполнимо, и это ошибка автора правила, а не агента.
KNOWN_CHECKS = (
    "must_read_state",
    "no_edits_to",
    "tests_exit_zero",
    "ask_when_unsure",
    "approve_new_dependency",
)

# Ниже этой уверенности агент обязан задать вопрос, а не гадать.
CONFIDENCE_THRESHOLD = 0.7


def parse_rules(markdown):
    """Разобрать agent-rules.md в список словарей-правил.

    Формат блока: заголовок '## <slug>', затем строки '- ключ: значение',
    затем одна строка описания.

    md = ("## forbidden/no-release\\n"
          "- category: forbidden\\n"
          "- check: no_edits_to\\n"
          "- arg: scripts/release.sh\\n"
          "- severity: block\\n"
          "Не трогай релизный скрипт.\\n")
    parse_rules(md)[0]["arg"]       ->  'scripts/release.sh'
    parse_rules(md)[0]["severity"]  ->  'block'

    Ключи результата: slug, category, check, arg, severity, expires_at,
    description. Отсутствующие check/arg/expires_at — None, severity по
    умолчанию DEFAULT_SEVERITY.

    Ловушка: категория вне CATEGORIES и severity вне SEVERITIES — ValueError.
    Правило, которое не влезло в пять категорий, почти всегда хочет быть
    двумя правилами; молча пропустить его значит потерять его навсегда.
    """
    rules = []
    for block in re.split(r"(?m)^##\s+", markdown)[1:]:
        lines = block.splitlines()
        slug = lines[0].strip()
        fields = {}
        description = []
        for line in lines[1:]:
            match = re.match(r"\s*-\s*(\w+):\s*(.*)$", line)
            if match:
                fields[match.group(1)] = match.group(2).strip()
            elif line.strip():
                description.append(line.strip())
        category = fields.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"{slug}: категория {category!r} не из {CATEGORIES}")
        severity = fields.get("severity", DEFAULT_SEVERITY)
        if severity not in SEVERITIES:
            raise ValueError(f"{slug}: severity {severity!r} не из {SEVERITIES}")
        rules.append(
            {
                "slug": slug,
                "category": category,
                "check": fields.get("check"),
                "arg": fields.get("arg"),
                "severity": severity,
                "expires_at": fields.get("expires_at"),
                "description": " ".join(description),
            }
        )
    return rules


def is_operational(rule):
    """Правило исполнимо, если его check есть в KNOWN_CHECKS.

    is_operational({"check": "tests_exit_zero"})  ->  True
    is_operational({"check": None})               ->  False
    is_operational({"check": "be_careful"})       ->  False

    Правило без исполнимой проверки — пожелание. Его либо удаляют, либо
    дописывают до проверки; оставлять «будь аккуратен» в файле правил значит
    делать вид, что воркбенч что-то контролирует.
    """
    return rule.get("check") in KNOWN_CHECKS


def compile_rule(rule):
    """Скомпилировать правило в предикат trace -> bool. True = правило соблюдено.

    Трейс хода — словарь с ключами read_state, edited_files, tests_exit_code,
    confidence, asked_for_help, added_dependencies, approvals.

    predicate = compile_rule({"check": "no_edits_to", "arg": "scripts/*.sh"})
    predicate({"edited_files": ["scripts/release.sh"]})  ->  False
    predicate({"edited_files": ["app.py"]})              ->  True

    Проверки:
      must_read_state        — агент прочитал файл состояния до первого действия;
      no_edits_to            — ни один изменённый файл не подходит под glob в arg;
      tests_exit_zero        — tests_exit_code == 0 (None значит «не запускали»);
      ask_when_unsure        — confidence >= CONFIDENCE_THRESHOLD или задан вопрос;
      approve_new_dependency — каждая новая зависимость есть в approvals.

    Ловушка: неисполнимое правило (нет check или он неизвестен) и no_edits_to
    без arg — это ValueError на этапе компиляции, а не молчаливое True в
    рантайме. Правило, которое всегда проходит, хуже отсутствующего.
    """
    check = rule.get("check")
    if check not in KNOWN_CHECKS:
        raise ValueError(f"неисполнимая проверка: {check!r}")

    if check == "must_read_state":
        return lambda trace: bool(trace.get("read_state", False))

    if check == "no_edits_to":
        pattern = rule.get("arg")
        if not pattern:
            raise ValueError("no_edits_to требует arg с glob запрещённых путей")
        # fnmatch, а не равенство: репозитории двигают файлы, а контракт
        # должен пережить переезд scripts/release.sh -> scripts/ci/release.sh
        return lambda trace: not any(
            fnmatch.fnmatch(path, pattern) for path in trace.get("edited_files", [])
        )

    if check == "tests_exit_zero":
        return lambda trace: trace.get("tests_exit_code") == 0

    if check == "ask_when_unsure":
        return lambda trace: (
            trace.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD
            or bool(trace.get("asked_for_help", False))
        )

    return lambda trace: all(
        dep in trace.get("approvals", []) for dep in trace.get("added_dependencies", [])
    )


def check_rules(rules, trace):
    """Прогнать все правила по трейсу. Список словарей в исходном порядке.

    Каждый результат: {"slug", "severity", "status"}, где status —
    'pass' | 'fail' | 'unchecked'.

    rules = [{"slug": "s", "check": "tests_exit_zero", "severity": "block"}]
    check_rules(rules, {"tests_exit_code": 1})[0]["status"]  ->  'fail'
    check_rules(rules, {"tests_exit_code": 0})[0]["status"]  ->  'pass'

    Неисполнимые правила получают 'unchecked', а не 'pass': отчёт обязан
    отличать «проверили и всё хорошо» от «проверить нечем».
    """
    results = []
    for rule in rules:
        if not is_operational(rule):
            status = "unchecked"
        else:
            status = "pass" if compile_rule(rule)(trace) else "fail"
        results.append(
            {
                "slug": rule.get("slug"),
                "severity": rule.get("severity", DEFAULT_SEVERITY),
                "status": status,
            }
        )
    return results


def severity_verdict(results):
    """Итог прогона по severity упавших правил: 'block' | 'warn' | 'pass'.

    severity_verdict([{"severity": "warn", "status": "fail"}])   ->  'warn'
    severity_verdict([{"severity": "info", "status": "fail"}])   ->  'pass'
    severity_verdict([])                                         ->  'pass'

    info не останавливает прогон по замыслу: severity ставят при написании
    правила, и если всё подряд объявить block, гейт отключит первая же
    команда, которой он помешал.
    """
    failed = {r["severity"] for r in results if r.get("status") == "fail"}
    if "block" in failed:
        return "block"
    if "warn" in failed:
        return "warn"
    return "pass"


def expired_rules(rules, now):
    """Слаги правил, у которых expires_at строго раньше now. Отсортированы.

    now и expires_at — строки ISO 'YYYY-MM-DD', сравниваются как строки.

    rules = [{"slug": "a", "expires_at": "2026-01-01"},
             {"slug": "b", "expires_at": "2027-01-01"},
             {"slug": "c"}]
    expired_rules(rules, "2026-06-01")  ->  ['a']

    Правило без expires_at не устаревает никогда — и именно так набор правил
    дорастает до восьмидесяти штук, из которых срабатывают три.
    Время приходит параметром: функция, зовущая внутри себя сегодняшнюю дату,
    непроверяема.
    """
    return sorted(
        rule["slug"]
        for rule in rules
        if rule.get("expires_at") and rule["expires_at"] < now
    )


def rules_lock(rules):
    """Кэш правил для горячего пути: только исполнимые, в устойчивом порядке.

    Сортировка — по позиции категории в CATEGORIES, затем по slug.
    Каждая запись: {"slug", "category", "check", "arg", "severity"}.

    rules = [{"slug": "z", "category": "approval", "check": "tests_exit_zero",
              "arg": None, "severity": "warn"}]
    rules_lock(rules)[0]["slug"]  ->  'z'

    Markdown — источник, lock — кэш: переставили правила местами в файле,
    lock обязан остаться прежним, иначе каждый коммит шумит в diff.
    Неисполнимые правила в lock не попадают: в горячем пути их нечем звать.
    """
    picked = [rule for rule in rules if is_operational(rule)]
    picked.sort(key=lambda r: (CATEGORIES.index(r["category"]), r["slug"]))
    return [
        {
            "slug": rule["slug"],
            "category": rule["category"],
            "check": rule["check"],
            "arg": rule.get("arg"),
            "severity": rule.get("severity", DEFAULT_SEVERITY),
        }
        for rule in picked
    ]
