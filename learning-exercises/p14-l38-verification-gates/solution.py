"""
Гейты верификации — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Гейт — это детерминированная функция над артефактами верстака, которая
отвечает на один вопрос: задача действительно закрыта? Агент не имеет права
поставить себе зачёт сам.

Соответствие настоящей системе:

    finding             <-  одна строка в verification_report.json
    gate_feedback       <-  сверка acceptance-команд с feedback_record.jsonl
    gate_scope          <-  чтение scope_report.json (Phase 14 · 36)
    gate_rules          <-  чтение rule_report.json (Phase 14 · 33)
    gate_coverage       <-  coverage_report.json и порог покрытия
    run_gates           <-  сам verify_agent.py: порядок и короткое замыкание
    apply_override      <-  подписанная строка в overrides.jsonl
    verification_report <-  outputs/verification/<task_id>.json

Никаких LLM-судей: гейт обязан выдавать один и тот же вердикт на одном и том
же наборе артефактов. Качественная оценка живёт в ревьюере (Phase 14 · 39).

Время приходит параметром now, файловая система смоделирована словарями.
"""

import fnmatch

# Две градации, и только две. "info" сюда не добавляют: находка, на которую
# никто не обязан реагировать, засоряет отчёт и обесценивает block.
SEVERITIES = ("warn", "block")

# Один путь на одну закрытую задачу. Второй путь = вторая версия правды.
REPORT_DIR = "outputs/verification"

# Порог покрытия по умолчанию и допустимое падение относительно прошлого
# мержа. Без второй проверки агент тихо удаляет падающие тесты, и покрытие
# формально остаётся выше пола.
COVERAGE_FLOOR = 80.0
COVERAGE_DROP_LIMIT = 1.0

# Порядок гейтов. Сначала самое дешёвое и самое частое — отсутствие прогона;
# покрытие последним, потому что оно бессмысленно, если тесты не запускались.
GATE_ORDER = ("feedback", "scope", "rules", "coverage")

# Префикс идентификатора, который гейт не примет как подписанта override.
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
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity: {severity!r}")
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "source": source,
        "overridden": False,
    }


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
    feedback = artifacts.get("feedback", [])
    acceptance = artifacts.get("scope", {}).get("acceptance", [])
    # индекс по команде: последний прогон команды и есть тот, по которому
    # закрывают задачу
    last = {}
    for record in feedback:
        last[record["command"]] = record
    out = []
    for command in acceptance:
        record = last.get(command)
        if record is None:
            out.append(
                finding("ACCEPTANCE_NOT_RUN", "block", f"не запускалось: {command}", "feedback")
            )
        elif record.get("exit_code") not in (0, None):
            out.append(
                finding("ACCEPTANCE_FAILED", "block", f"ненулевой выход: {command}", "feedback")
            )
    for record in feedback:
        if record.get("exit_code") is None:
            out.append(
                finding(
                    "NULL_EXIT", "block", f"нет exit_code: {record['command']}", "feedback"
                )
            )
    return out


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
    scope = artifacts.get("scope", {})
    allowed = scope.get("allowed_files", [])
    forbidden = scope.get("forbidden_files", [])
    out = []
    for path in artifacts.get("diff", {}).get("touched_files", []):
        if any(fnmatch.fnmatch(path, pattern) for pattern in forbidden):
            out.append(finding("FORBIDDEN_WRITE", "block", f"запретная зона: {path}", "scope"))
        elif not any(fnmatch.fnmatch(path, pattern) for pattern in allowed):
            out.append(finding("OFF_SCOPE_WRITE", "warn", f"вне контракта: {path}", "scope"))
    return out


def gate_rules(artifacts):
    """Гейт правил: каждое непройденное правило даёт находку своей градации.

    gate_rules({"rules": [{"id": "no-todo", "severity": "block",
                           "passed": False}]})
        ->  [{"code": "RULE_FAILED", "severity": "block", ...}]

    Градацию задаёт само правило, а не гейт: часть правил стилевые, они
    только аннотируют вердикт. Пройденные правила находок не порождают.
    """
    out = []
    for rule in artifacts.get("rules", []):
        if not rule.get("passed", False):
            out.append(
                finding(
                    "RULE_FAILED",
                    rule["severity"],
                    f"правило не выполнено: {rule['id']}",
                    "rules",
                )
            )
    return out


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
    coverage = artifacts.get("coverage")
    if not coverage:
        return [finding("COVERAGE_MISSING", "warn", "нет coverage_report", "coverage")]
    out = []
    measured = coverage["measured"]
    if measured < coverage.get("floor", floor):
        out.append(
            finding("COVERAGE_BELOW_FLOOR", "block", f"покрытие {measured}", "coverage")
        )
    previous = coverage.get("previous")
    if previous is not None and previous - measured > drop_limit:
        out.append(
            finding(
                "COVERAGE_REGRESSION",
                "block",
                f"просадка {previous} -> {measured}",
                "coverage",
            )
        )
    return out


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
    if gates is None:
        table = {
            "feedback": gate_feedback,
            "scope": gate_scope,
            "rules": gate_rules,
            "coverage": gate_coverage,
        }
        gates = [(name, table[name]) for name in GATE_ORDER]
    gates = list(gates)

    findings = []
    ran = []
    for index, (name, gate) in enumerate(gates):
        ran.append(name)
        produced = gate(artifacts)
        if strict:
            for item in produced:
                if item["severity"] == "warn":
                    # градацию поднимаем, происхождение сохраняем: иначе в
                    # отчёте не отличить строгий режим от настоящего блока
                    item["severity"] = "block"
                    item["promoted_from"] = "warn"
        findings.extend(produced)
        if any(item["severity"] == "block" for item in produced):
            skipped = [n for n, _ in gates[index + 1 :]]
            return {"findings": findings, "ran": ran, "skipped": skipped, "passed": False}
    return {"findings": findings, "ran": ran, "skipped": [], "passed": True}


def verification_report(task_id, artifacts, strict=False, now=0, gates=None):
    """Полный отчёт закрытия задачи. Один task_id — один путь.

    verification_report("T-1", clean_artifacts)["path"]
        ->  "outputs/verification/T-1.json"

    passed=True только когда ни одной неперекрытой находки уровня block.
    Пустой task_id — ValueError: отчёт без адреса не найдёт ни CI, ни человек.
    """
    if not task_id:
        raise ValueError("task_id is required: a report needs one path")
    result = run_gates(artifacts, gates=gates, strict=strict)
    return {
        "task_id": task_id,
        "path": f"{REPORT_DIR}/{task_id}.json",
        "passed": result["passed"],
        "findings": result["findings"],
        "gates_ran": result["ran"],
        "gates_skipped": result["skipped"],
        "strict": strict,
        "generated_at": now,
        "overrides": [],
    }


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
    if not reason or not user_id or not commit:
        raise ValueError("override must be signed: reason, user_id, commit")
    if user_id.startswith(AGENT_PREFIX):
        raise ValueError(f"agent cannot sign its own override: {user_id}")

    matched = [
        i
        for i, item in enumerate(report["findings"])
        if item["code"] == code and item["severity"] == "block" and not item["overridden"]
    ]
    if not matched:
        raise KeyError(code)

    # копия отчёта: исходный вердикт остаётся как был, перекрытие — новая
    # версия, а не правка истории. Одна подпись закрывает все претензии с
    # этим кодом: причина у них одна, разводить их по строкам незачем.
    findings = [dict(item) for item in report["findings"]]
    for i in matched:
        findings[i]["overridden"] = True

    row = {
        "task_id": report["task_id"],
        "code": code,
        "reason": reason,
        "overridden_by": user_id,
        "commit": commit,
        "at": now,
    }
    new_report = dict(report)
    new_report["findings"] = findings
    new_report["overrides"] = list(report["overrides"]) + [row]
    new_report["passed"] = not any(
        item["severity"] == "block" and not item["overridden"] for item in findings
    )
    return new_report, row
