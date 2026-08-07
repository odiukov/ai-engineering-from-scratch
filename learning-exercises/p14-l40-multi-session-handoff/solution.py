"""
Хендофф между сессиями — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Семь полей пакета из урока: summary, changed_files, commands_run,
# failed_attempts, open_risks, next_action, verdict_pointer.
HANDOFF_FIELDS = (
    "summary",
    "changed_files",
    "commands_run",
    "failed_attempts",
    "open_risks",
    "next_action",
    "verdict_pointer",
)

# Поля, которые обязаны быть непустыми: без них следующая сессия не стартует.
REQUIRED_NONEMPTY = ("summary", "next_action", "verdict_pointer")

# Порядок серьёзности: block строго важнее warn, info в риски не попадает.
SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}

# Порядок проверок чистоты рабочего места из таблицы урока.
CLEAN_CHECKS = ("working_tree", "temp_artifacts", "tests", "feature_board", "branch")

# Сколько последних записей фидбека кладём в пакет по умолчанию.
TAIL_K = 5


def trim_feedback(records, tail_k=TAIL_K):
    """Обрезать журнал фидбека: последние tail_k записей плюс ВСЕ с ненулевым exit_code.

    Порядок исходного журнала сохраняется, дублей нет.

    recs = [{"cmd": "a", "exit_code": 1}, {"cmd": "b", "exit_code": 0},
            {"cmd": "c", "exit_code": 0}]
    trim_feedback(recs, 1)  ->  [{"cmd": "a", ...}, {"cmd": "c", ...}]
    trim_feedback(recs, 0)  ->  [{"cmd": "a", ...}]
    trim_feedback([], 5)    ->  []

    Ловушка: если провальная запись попала и в хвост, и в выборку по exit_code,
    она обязана появиться в результате ОДИН раз. Дедупликация по id() объекта
    ломается на одинаковых словарях — считай по индексам.

    Зачем: полный feedback_record.jsonl бывает на сотни строк, а пакет должен
    оставаться маленьким. Провалы не выбрасываем никогда — это то, ради чего
    следующая сессия вообще читает журнал.
    """
    if tail_k < 0:
        raise ValueError("tail_k не может быть отрицательным")
    n = len(records)
    # множество индексов, а не объектов: одинаковые словари не склеиваются
    keep = set(range(max(0, n - tail_k), n))
    for i, rec in enumerate(records):
        if rec.get("exit_code", 0) not in (0, None):
            keep.add(i)
    return [records[i] for i in sorted(keep)]


def derive_open_risks(verdict, review):
    """Собрать open_risks из отчёта верификации и отчёта ревьюера.

    Берём только severity "block" и "warn"; "info" — это шум, не риск.
    У каждого риска появляется поле source: "verification" или "review".
    Сортировка: сначала block, потом warn; внутри — по source, затем по detail.

    v = {"findings": [{"severity": "warn", "detail": "slow test"}]}
    r = {"findings": [{"severity": "block", "detail": "no rollback"}]}
    derive_open_risks(v, r)
      ->  [{"severity": "block", "detail": "no rollback", "source": "review"},
           {"severity": "warn", "detail": "slow test", "source": "verification"}]
    derive_open_risks({}, {})  ->  []

    Ловушка: порядок результата не должен зависеть от порядка findings во
    входных отчётах — иначе два запуска генератора дадут разные пакеты, и
    идемпотентность из упражнения 4 урока не выполнится.
    """
    risks = []
    for source, report in (("verification", verdict), ("review", review)):
        for finding in report.get("findings", ()) or ():
            severity = finding.get("severity")
            if severity not in SEVERITY_ORDER:
                continue
            if SEVERITY_ORDER[severity] > SEVERITY_ORDER["warn"]:
                continue
            risks.append(
                {
                    "severity": severity,
                    "detail": str(finding.get("detail", "")),
                    "source": source,
                }
            )
    # полный ключ сортировки: при равной severity порядок всё равно фиксирован
    risks.sort(key=lambda r: (SEVERITY_ORDER[r["severity"]], r["source"], r["detail"]))
    return risks


def choose_next_action(verdict, open_risks, feature_board):
    """Выбрать next_action — единственный конкретный первый шаг следующей сессии.

    Приоритет строгий и детерминированный:
      1. есть риск severity "block"  ->  устранить его;
      2. verdict["status"] != "pass" ->  перепрогнать верификацию;
      3. первая фича доски со статусом "in_progress"  ->  продолжить её;
      4. первая фича доски со статусом "todo"         ->  начать её;
      5. есть риск severity "warn"   ->  разобрать предупреждение;
      6. иначе                       ->  закрыть задачу.

    choose_next_action({"status": "pass"}, [], [{"id": "F1", "status": "todo"}])
      ->  "начать фичу F1"
    choose_next_action({"status": "pass"}, [], [])
      ->  "закрыть задачу: открытых пунктов на доске нет"

    Функция НИКОГДА не возвращает пустую строку: пакет без next_action —
    это статус-репорт, а не хендофф.
    """
    for risk in open_risks:
        if risk["severity"] == "block":
            return "устранить блокер: " + risk["detail"]
    if verdict.get("status") != "pass":
        report = verdict.get("report_path", "verification_report.json")
        return "перепрогнать верификацию: " + str(report)
    # доску читаем в её собственном порядке: он и есть приоритет команды
    for status, verb in (("in_progress", "продолжить"), ("todo", "начать")):
        for feature in feature_board:
            if feature.get("status") == status:
                title = str(feature.get("title", "")).strip()
                head = "%s фичу %s" % (verb, feature.get("id"))
                return head + (": " + title if title else "")
    for risk in open_risks:
        if risk["severity"] == "warn":
            return "разобрать предупреждение: " + risk["detail"]
    return "закрыть задачу: открытых пунктов на доске нет"


def clean_state_issues(workbench, open_risks=()):
    """Проверка чистоты рабочего места: список блокирующих проблем.

    workbench — словарь наблюдений:
      uncommitted_files, stash_note, temp_artifacts,
      tests = {"status": "green"|"red", "failure": "..."},
      feature_board = [{"id", "status", "actual_done"}],
      branch, expected_branch, orphan_branches.

    Каждая проблема — {"check": ..., "detail": ...}; результат отсортирован в
    порядке CLEAN_CHECKS. Пустой список — предусловие, которое проверяет
    build_handoff.

    clean_state_issues({"branch": "x", "expected_branch": "x"})  ->  []
    clean_state_issues({"temp_artifacts": ["a.tmp"], "branch": "x",
                        "expected_branch": "x"})
      ->  [{"check": "temp_artifacts", "detail": "мусор в дереве: a.tmp"}]

    Тонкость про тесты: красный тест НЕ является блокером, если его падение
    уже названо в open_risks — урок разрешает уходить с красным, но только
    честно записанным. Именно поэтому open_risks — аргумент этой функции.
    """
    issues = []

    uncommitted = sorted(workbench.get("uncommitted_files", ()) or ())
    if uncommitted and not workbench.get("stash_note"):
        issues.append(
            {"check": "working_tree", "detail": "не закоммичено: " + ", ".join(uncommitted)}
        )

    temp = sorted(workbench.get("temp_artifacts", ()) or ())
    if temp:
        issues.append({"check": "temp_artifacts", "detail": "мусор в дереве: " + ", ".join(temp)})

    tests = workbench.get("tests", {}) or {}
    if tests.get("status") == "red":
        failure = str(tests.get("failure", ""))
        named = {risk["detail"] for risk in open_risks}
        if failure not in named:
            issues.append(
                {"check": "tests", "detail": "красный тест не назван в open_risks: " + failure}
            )

    # доска врёт, если объявленный статус расходится с наблюдаемой реальностью
    stale = sorted(
        str(f.get("id"))
        for f in workbench.get("feature_board", ()) or ()
        if (f.get("status") == "done") != bool(f.get("actual_done"))
    )
    if stale:
        issues.append(
            {"check": "feature_board", "detail": "доска расходится с репозиторием: " + ", ".join(stale)}
        )

    branch = workbench.get("branch")
    expected = workbench.get("expected_branch")
    orphans = sorted(workbench.get("orphan_branches", ()) or ())
    if branch != expected:
        issues.append(
            {"check": "branch", "detail": "ветка %r, ожидалась %r" % (branch, expected)}
        )
    elif orphans:
        issues.append({"check": "branch", "detail": "осиротевшие ветки: " + ", ".join(orphans)})

    issues.sort(key=lambda i: CLEAN_CHECKS.index(i["check"]))
    return issues


def build_handoff(snapshot, workbench, now, tail_k=TAIL_K):
    """Собрать пакет хендоффа из артефактов воркбенча.

    snapshot: task_id, topic, last_known_good_commit,
              state = {"summary", "commands_run", "failed_attempts"},
              verdict, review, feedback, diff_summary = {"changed": [...]}.
    now: строка-момент генерации; берётся ПАРАМЕТРОМ, а не из time.time(),
         иначе два запуска дадут разные пакеты.

    Возвращает словарь с семью полями HANDOFF_FIELDS плюс служебные
    task_id, topic, branch, last_known_good_commit, status, generated_at,
    feedback_tail.

    Если clean_state_issues() не пуст — ValueError. Хендофф, собранный на
    грязном дереве, это не хендофф, а переадресованный беспорядок.

    Идемпотентность: при одинаковых snapshot, workbench и now два вызова
    дают равные словари.
    """
    verdict = snapshot.get("verdict", {}) or {}
    review = snapshot.get("review", {}) or {}
    risks = derive_open_risks(verdict, review)

    blockers = clean_state_issues(workbench, risks)
    if blockers:
        raise ValueError(
            "рабочее место грязное: " + ", ".join(b["check"] for b in blockers)
        )

    state = snapshot.get("state", {}) or {}
    diff = snapshot.get("diff_summary", {}) or {}
    return {
        "task_id": snapshot.get("task_id", ""),
        "topic": snapshot.get("topic", ""),
        "branch": workbench.get("branch"),
        "last_known_good_commit": snapshot.get("last_known_good_commit", ""),
        "status": "active",
        "generated_at": now,
        "summary": state.get("summary", ""),
        # сортируем, чтобы порядок файлов не зависел от порядка обхода диффа
        "changed_files": sorted(diff.get("changed", ()) or ()),
        "commands_run": list(state.get("commands_run", ()) or ()),
        "failed_attempts": list(state.get("failed_attempts", ()) or ()),
        "open_risks": risks,
        "next_action": choose_next_action(
            verdict, risks, workbench.get("feature_board", ()) or ()
        ),
        "verdict_pointer": {
            "verification": str(verdict.get("report_path", "")),
            "review": str(review.get("report_path", "")),
        },
        "feedback_tail": trim_feedback(list(snapshot.get("feedback", ()) or ()), tail_k),
    }


def render_markdown(payload):
    """Отрендерить handoff.md из пакета: заголовок, шапка и семь секций.

    Каждое поле HANDOFF_FIELDS даёт секцию "## <имя поля>" — ровно в том
    порядке, в каком они перечислены в HANDOFF_FIELDS. Пустое поле
    отображается строкой "_none_", а не исчезает: читателю важно видеть, что
    поле пусто, а не гадать, забыли его или нет.

    render_markdown(pkt).startswith("# Handoff ")  ->  True
    "## next_action" in render_markdown(pkt)       ->  True

    JSON — источник истины, markdown — производная. Разойтись им негде,
    потому что оба делаются из одного payload.
    """
    out = [
        "# Handoff " + str(payload.get("task_id", "")),
        "",
        "- branch: %s" % payload.get("branch"),
        "- status: %s" % payload.get("status"),
        "- generated_at: %s" % payload.get("generated_at"),
        "- last_known_good_commit: %s" % payload.get("last_known_good_commit"),
        "",
    ]
    for field in HANDOFF_FIELDS:
        out.append("## " + field)
        value = payload.get(field)
        if isinstance(value, str):
            out.append(value if value else "_none_")
        elif isinstance(value, dict):
            rows = ["- %s: %s" % (k, value[k]) for k in sorted(value)]
            out.extend(rows or ["_none_"])
        else:
            rows = []
            for item in value or ():
                if isinstance(item, dict):
                    rows.append("- " + ", ".join("%s=%s" % (k, item[k]) for k in sorted(item)))
                else:
                    rows.append("- %s" % (item,))
            out.extend(rows or ["_none_"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def resume_blockers(payload):
    """Чего не хватает пакету, чтобы следующая сессия стартовала БЕЗ исходной.

    Возвращает список причин; пустой список — пакет самодостаточен.

    Проверяем: есть task_id и branch; присутствуют все семь HANDOFF_FIELDS;
    поля из REQUIRED_NONEMPTY непусты; verdict_pointer несёт обе ссылки.

    resume_blockers(build_handoff(...))  ->  []
    resume_blockers({**pkt, "next_action": ""})
      ->  ["поле next_action пустое"]

    Смысл проверки: документ, где есть всё кроме next_action, — это
    статус-репорт. Он полезен, но следующая сессия по нему не стартует.
    """
    blockers = []
    for key in ("task_id", "branch"):
        if not payload.get(key):
            blockers.append("нет %s" % key)
    for field in HANDOFF_FIELDS:
        if field not in payload:
            blockers.append("нет поля %s" % field)
        elif field in REQUIRED_NONEMPTY and not payload[field]:
            blockers.append("поле %s пустое" % field)
    pointer = payload.get("verdict_pointer") or {}
    if isinstance(pointer, dict):
        for key in ("verification", "review"):
            if not pointer.get(key):
                blockers.append("verdict_pointer без ссылки %s" % key)
    return blockers


def select_active_handoff(packets, branch, topic):
    """На пару (branch, topic) активным остаётся ровно один пакет.

    Самый свежий по generated_at становится "active", остальные пакеты той же
    пары — "superseded". Пакеты со статусом "archived" не участвуют вовсе и
    активными не становятся. Пакеты других веток или тем не трогаются.

    Ничья по generated_at разрешается по task_id: побеждает лексикографически
    больший. Так порядок входного списка не влияет на исход.

    select_active_handoff([], "main", "auth")  ->  []

    Возвращается НОВЫЙ список новых словарей: входные пакеты не мутируются —
    иначе повторный вызов на тех же данных дал бы другой результат.
    """
    result = [dict(p) for p in packets]

    def matches(p):
        return (
            p.get("branch") == branch
            and p.get("topic") == topic
            and p.get("status") != "archived"
        )

    candidates = [p for p in result if matches(p)]
    if not candidates:
        return result
    # ключ включает task_id: при равном времени исход всё равно однозначен
    best = max(candidates, key=lambda p: (str(p.get("generated_at", "")), str(p.get("task_id", ""))))
    for p in candidates:
        p["status"] = "active" if p is best else "superseded"
    return result
