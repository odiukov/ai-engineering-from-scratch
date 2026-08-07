"""
Гейты верификации

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l38-verification-gates
Разбор:  /check-code p14-l38-verification-gates
"""

import fnmatch

SEVERITIES = ("warn", "block")
REPORT_DIR = "outputs/verification"
COVERAGE_FLOOR = 80.0
COVERAGE_DROP_LIMIT = 1.0
GATE_ORDER = ("feedback", "scope", "rules", "coverage")
AGENT_PREFIX = "agent:"


def finding(code, severity, message, source):
    """Собрать одну находку гейта.

    finding("NULL_EXIT", "block", "нет exit_code", "feedback")
        ->  {"code": "NULL_EXIT", "severity": "block",
             "message": "нет exit_code", "source": "feedback",
             "overridden": False}

    severity вне SEVERITIES — ValueError. Опечатка в градации страшнее
    отсутствия находки: "blok" молча не заблокирует ничего.
    """
    raise NotImplementedError


def gate_feedback(artifacts):
    """Гейт обратной связи: acceptance-команды прогнаны и вышли нулём.

    gate_feedback({"scope": {"acceptance": ["pytest -q"]},
                   "feedback": [{"command": "pytest -q", "exit_code": 0}]})
        ->  []

    Три находки, все block:
      ACCEPTANCE_NOT_RUN  — команды из контракта нет в журнале;
      ACCEPTANCE_FAILED   — команда была, но код не нулевой;
      NULL_EXIT           — в журнале есть запись без exit_code.

    NULL_EXIT ищется по ВСЕМ записям, а не только по acceptance: сорванный
    прогон где угодно означает, что журналу нельзя верить целиком.
    """
    raise NotImplementedError


def gate_scope(artifacts):
    """Гейт границ: запретные зоны — block, выход за разрешённые — warn.

    gate_scope({"scope": {"allowed_files": ["app/*.py"],
                          "forbidden_files": ["scripts/*"]},
                "diff": {"touched_files": ["app/main.py"]}})
        ->  []

    Запрет сильнее разрешения: файл, попавший под оба шаблона, считается
    запретным. Иначе достаточно расширить allowed_files, чтобы обойти запрет.

    Правка вне разрешённых, но и вне запретных — warn: расширение границ
    бывает осознанным, и хоронить задачу из-за него не за что. Ровно этот
    warn превращается в block в строгом режиме.
    """
    raise NotImplementedError


def gate_rules(artifacts):
    """Гейт правил: каждое непройденное правило даёт находку своей градации.

    gate_rules({"rules": [{"id": "no-todo", "severity": "block",
                           "passed": False}]})
        ->  [{"code": "RULE_FAILED", "severity": "block", ...}]

    Градацию задаёт само правило, а не гейт: часть правил стилевые, они
    только аннотируют вердикт. Пройденные правила находок не порождают.
    """
    raise NotImplementedError


def gate_coverage(artifacts, floor=COVERAGE_FLOOR, drop_limit=COVERAGE_DROP_LIMIT):
    """Гейт покрытия: пол по абсолюту и запрет на просадку относительно прошлого.

    gate_coverage({"coverage": {"measured": 84.0, "previous": 84.5}})  ->  []
    gate_coverage({"coverage": {"measured": 70.0, "previous": 70.0}})
        ->  одна находка COVERAGE_BELOW_FLOOR

    Второй порог обязателен: без него агент удаляет падающий тест, покрытие
    падает с 95 до 81, пол формально соблюдён, отчёт зелёный.

    Отсутствие coverage-артефакта — warn, а не block: не в каждом репозитории
    он вообще собирается, и заваливать задачу за это нельзя.
    """
    raise NotImplementedError


def run_gates(artifacts, gates=None, strict=False):
    """Прогнать гейты по порядку с коротким замыканием на первом block.

    Вернуть {"findings": [...], "ran": [имена], "skipped": [имена],
             "passed": bool}.

    run_gates({"scope": {"acceptance": []}, "diff": {"touched_files": []},
               "rules": [], "coverage": {"measured": 99.0}})
        ->  passed True, ran все четыре, skipped пусто

    gates — последовательность пар (имя, функция). None означает стандартный
    GATE_ORDER.

    Короткое замыкание принципиально: упавший гейт не пропускает дальше, и
    следующие ГЕЙТЫ НЕ ЗАПУСКАЮТСЯ. Смысл не в экономии — прогонять проверку
    покрытия по диффу, который вообще не должен был появиться, значит
    показать человеку список претензий, половина которых исчезнет сама.

    strict=True поднимает каждый warn до block. Тогда замыкание может
    случиться раньше — это и есть режим для релизной ветки.
    """
    raise NotImplementedError


def verification_report(task_id, artifacts, strict=False, now=0, gates=None):
    """Полный отчёт закрытия задачи. Один task_id — один путь.

    verification_report("T-1", clean_artifacts)["path"]
        ->  "outputs/verification/T-1.json"

    passed=True только когда ни одной неперекрытой находки уровня block.
    Пустой task_id — ValueError: отчёт без адреса не найдёт ни CI, ни человек.
    """
    raise NotImplementedError


def apply_override(report, code, reason, user_id, commit, now):
    """Перекрыть находку уровня block подписью человека.

    Вернуть (новый отчёт, строка для overrides.jsonl).

    apply_override(report, "OFF_SCOPE_WRITE", "согласовано", "u-42", "abc123", 7)
        ->  KeyError, потому что это warn: перекрывать нечего

    Пустая причина, пустой подписант или пустой коммит — ValueError.
    Подписант с префиксом "agent:" — ValueError: перекрытие это подписанное
    решение человека, а не решение агента. Без этой проверки гейт становится
    театром: агент сам себе выписывает пропуск.

    Кода нет среди block-находок — KeyError.
    """
    raise NotImplementedError
