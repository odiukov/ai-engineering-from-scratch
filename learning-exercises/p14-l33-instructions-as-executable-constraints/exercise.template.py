"""
Инструкции агента как исполняемые ограничения

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l33-instructions-as-executable-constraints
Разбор:  /check-code p14-l33-instructions-as-executable-constraints
"""

import fnmatch
import re

CATEGORIES = ("startup", "forbidden", "definition_of_done", "uncertainty", "approval")
SEVERITIES = ("block", "warn", "info")
DEFAULT_SEVERITY = "warn"
KNOWN_CHECKS = (
    "must_read_state",
    "no_edits_to",
    "tests_exit_zero",
    "ask_when_unsure",
    "approve_new_dependency",
)
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
    raise NotImplementedError


def is_operational(rule):
    """Правило исполнимо, если его check есть в KNOWN_CHECKS.

    is_operational({"check": "tests_exit_zero"})  ->  True
    is_operational({"check": None})               ->  False
    is_operational({"check": "be_careful"})       ->  False

    Правило без исполнимой проверки — пожелание. Его либо удаляют, либо
    дописывают до проверки; оставлять «будь аккуратен» в файле правил значит
    делать вид, что воркбенч что-то контролирует.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def severity_verdict(results):
    """Итог прогона по severity упавших правил: 'block' | 'warn' | 'pass'.

    severity_verdict([{"severity": "warn", "status": "fail"}])   ->  'warn'
    severity_verdict([{"severity": "info", "status": "fail"}])   ->  'pass'
    severity_verdict([])                                         ->  'pass'

    info не останавливает прогон по замыслу: severity ставят при написании
    правила, и если всё подряд объявить block, гейт отключит первая же
    команда, которой он помешал.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
