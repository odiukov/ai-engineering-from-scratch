"""
Воркбенч на реальном репозитории — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import fnmatch

# Пять исходов из таблицы урока — порядок фиксирован, отчёт читают по нему.
OUTCOME_KEYS = (
    "tests_actually_run",
    "acceptance_met",
    "files_outside_scope",
    "handoff_quality",
    "reviewer_total",
)

# Куда для каждого исхода «лучше»: files_outside_scope — единственный,
# у которого меньше значит лучше.
OUTCOME_DIRECTION = {
    "tests_actually_run": "higher",
    "acceptance_met": "higher",
    "files_outside_scope": "lower",
    "handoff_quality": "higher",
    "reviewer_total": "higher",
}

# Приманки запретной зоны: то, что в реальном репозитории агент трогать не
# должен, даже если технически может.
FORBIDDEN_HINTS = ("scripts/*", ".github/*", "README.md", "*.lock")

# Признак -> команда приёмки. Порядок важен: питон проверяем раньше npm.
ACCEPTANCE_EVIDENCE = (
    (("pytest.ini", "conftest.py", "test_*.py"), "python3 -m pytest -q"),
    (("package.json",), "npm test"),
)

# Виды задач, на которых prompt-only честно быстрее воркбенча.
FAST_PATH_KINDS = ("formatter", "one_line_lint", "single_fact")


def adapt_scope_contract(repo_tree, protected=()):
    """Вывести scope-контракт из ДЕРЕВА реального репозитория, а не из шаблона.

    repo_tree — список путей. Возвращает словарь:
      allowed_globs      — "<каталог>/*.py" для каждого каталога, где реально
                           лежат .py файлы ("*.py" для корня);
      forbidden_globs    — те FORBIDDEN_HINTS, под которые в дереве есть хотя
                           бы одно совпадение, плюс всё из protected;
      acceptance_command — по признакам из ACCEPTANCE_EVIDENCE, иначе None.

    adapt_scope_contract(["app.py", "test_app.py", "README.md",
                          "scripts/release.sh"])
      ->  {"allowed_globs": ["*.py"],
           "forbidden_globs": ["README.md", "scripts/*"],
           "acceptance_command": "python3 -m pytest -q"}

    Ловушки:
      * запрет на то, чего в дереве нет, ничего не защищает и только шумит —
        такие маски выбрасываем (protected остаются всегда: это явная воля
        человека, а не догадка по дереву);
      * "*.py" по правилам fnmatch совпадает и с "scripts/release.py":
        звёздочка проходит сквозь слэш. Поэтому запретная зона обязана
        побеждать разрешённую — см. classify_touched_files.
    """
    dirs = set()
    for path in repo_tree:
        if not path.endswith(".py"):
            continue
        head, _, tail = path.rpartition("/")
        dirs.add(head)
    allowed = sorted(("%s/*.py" % d) if d else "*.py" for d in dirs)

    forbidden = {p for p in protected}
    for hint in FORBIDDEN_HINTS:
        # маска остаётся в контракте только если в дереве есть что запрещать
        if any(fnmatch.fnmatchcase(path, hint) for path in repo_tree):
            forbidden.add(hint)

    command = None
    for markers, cmd in ACCEPTANCE_EVIDENCE:
        if any(fnmatch.fnmatchcase(p, m) for p in repo_tree for m in markers):
            command = cmd
            break

    return {
        "allowed_globs": allowed,
        "forbidden_globs": sorted(forbidden),
        "acceptance_command": command,
    }


def classify_touched_files(touched, contract):
    """Разложить тронутые файлы на in_scope и outside_scope.

    Файл вне контракта, если он попал под ЛЮБУЮ запретную маску либо не попал
    ни под одну разрешённую. Запрет сильнее разрешения.

    c = {"allowed_globs": ["*.py"], "forbidden_globs": ["scripts/*"]}
    classify_touched_files(["app.py", "scripts/release.py", "docs/x.md"], c)
      ->  {"in_scope": ["app.py"],
           "outside_scope": ["docs/x.md", "scripts/release.py"]}

    Оба списка отсортированы: отчёт о выходе за скоуп не должен зависеть от
    порядка обхода диффа.
    """
    allowed = contract.get("allowed_globs", ()) or ()
    forbidden = contract.get("forbidden_globs", ()) or ()
    inside, outside = [], []
    for path in touched:
        banned = any(fnmatch.fnmatchcase(path, g) for g in forbidden)
        permitted = any(fnmatch.fnmatchcase(path, g) for g in allowed)
        # порядок условий и есть правило "запрет сильнее разрешения"
        (outside if banned or not permitted else inside).append(path)
    return {"in_scope": sorted(inside), "outside_scope": sorted(outside)}


def simulate_test_run(repo, checks):
    """Смоделировать прогон тестов данными: никаких настоящих процессов.

    repo — {путь: содержимое}. checks — список проверок вида
    {"name": ..., "file": ..., "requires": [подстрока, ...]}.
    Проверка проходит, если файл есть в repo и содержит все подстроки.

    repo = {"app.py": "if len(pw) < 8: raise Invalid"}
    checks = [{"name": "t_short_pw", "file": "app.py", "requires": ["len(pw) < 8"]}]
    simulate_test_run(repo, checks)
      ->  {"ran": True, "passed": ["t_short_pw"], "failed": [], "exit_code": 0}
    simulate_test_run(repo, [])
      ->  {"ran": False, "passed": [], "failed": [], "exit_code": 1}

    Ключевая тонкость: пустой список проверок — это НЕ успех. Ноль тестов
    даёт exit_code 1 и ran=False, потому что «тесты прошли» без прогона —
    ровно то непроверяемое утверждение, из-за которого урок и написан.
    """
    passed, failed = [], []
    for check in checks:
        content = repo.get(check.get("file"))
        ok = content is not None and all(
            marker in content for marker in check.get("requires", ()) or ()
        )
        (passed if ok else failed).append(check.get("name"))
    ran = bool(checks)
    return {
        "ran": ran,
        "passed": passed,
        "failed": failed,
        "exit_code": 0 if ran and not failed else 1,
    }


def handoff_quality(packet):
    """Оценить пакет хендоффа в 0..3 балла по трём признакам.

    Балл за непустой next_action, балл за verdict_pointer с обеими ссылками,
    балл за непустой changed_files.

    handoff_quality(None)  ->  0
    handoff_quality({"next_action": "починить", "changed_files": ["app.py"],
                     "verdict_pointer": {"verification": "v", "review": "r"}})  ->  3

    Зачем именно эти три: next_action делает первый шаг следующей сессии
    определённым, verdict_pointer даёт трассируемость, changed_files —
    дифф одним взглядом.
    """
    if not packet:
        return 0
    score = 0
    if str(packet.get("next_action", "")).strip():
        score += 1
    pointer = packet.get("verdict_pointer") or {}
    if pointer.get("verification") and pointer.get("review"):
        score += 1
    if packet.get("changed_files"):
        score += 1
    return score


def measure_run(run, contract):
    """Измерить один прогон по пяти исходам урока.

    run: touched, repo_after, checks, acceptance_test, commands,
         handoff, reviewer_scores.

    tests_actually_run — команда приёмки из контракта реально попала в
                         commands И прогон состоялся (ran=True);
    acceptance_met     — тест, доказывающий цель, оказался среди passed;
    files_outside_scope— сколько тронутых файлов вне контракта;
    handoff_quality    — handoff_quality(run["handoff"]);
    reviewer_total     — сумма оценок ревьюера.

    Ловушка: acceptance_met считается по РЕЗУЛЬТАТУ прогона, а не по
    содержимому репозитория. Если нужный тест не запускался, цель не
    доказана, сколько бы правильного кода ни лежало в файлах.
    """
    result = simulate_test_run(run.get("repo_after", {}) or {}, run.get("checks", ()) or ())
    command = contract.get("acceptance_command")
    commands = list(run.get("commands", ()) or ())
    scoped = classify_touched_files(run.get("touched", ()) or (), contract)
    scores = run.get("reviewer_scores", {}) or {}
    return {
        "tests_actually_run": bool(command) and command in commands and result["ran"],
        "acceptance_met": run.get("acceptance_test") in result["passed"],
        "files_outside_scope": len(scoped["outside_scope"]),
        "handoff_quality": handoff_quality(run.get("handoff")),
        "reviewer_total": sum(scores.values()),
    }


def compare_pipelines(baseline, candidate):
    """Сравнить два набора исходов: prompt-only против воркбенча.

    Возвращает список строк в порядке OUTCOME_KEYS, каждая —
    {"outcome", "baseline", "candidate", "winner"}. Победитель считается с
    учётом OUTCOME_DIRECTION; равенство даёт "tie", а не случайный выбор.
    Отсутствующий исход читается как 0: строка про него всё равно появится,
    потому что «не измерили» и «измерили и получили ноль» отчёт различать
    не обязан, а вот молча пропасть исход не должен.

    compare_pipelines({"files_outside_scope": 3, ...},
                      {"files_outside_scope": 0, ...})[2]["winner"]  ->  "candidate"

    Булевы значения сравниваются как числа: True > False.
    """
    rows = []
    for key in OUTCOME_KEYS:
        left, right = baseline.get(key, 0), candidate.get(key, 0)
        # int() выравнивает bool и число: сравнение одно для всех пяти исходов
        lo, hi = int(left), int(right)
        if lo == hi:
            winner = "tie"
        elif OUTCOME_DIRECTION[key] == "higher":
            winner = "candidate" if hi > lo else "baseline"
        else:
            winner = "candidate" if hi < lo else "baseline"
        rows.append(
            {"outcome": key, "baseline": left, "candidate": right, "winner": winner}
        )
    return rows


def render_before_after(comparison):
    """Собрать before-after-report.md: таблица исходов плюс итоговая строка.

    Строк ровно столько, сколько исходов, плюс заголовок, разделитель и
    строка "выиграл воркбенч в N из M исходов".

    render_before_after(rows).startswith("| outcome |")  ->  True

    Это тот артефакт, который отдают скептику: числа спорят лучше объяснений.
    """
    lines = [
        "| outcome | prompt-only | workbench | winner |",
        "| --- | --- | --- | --- |",
    ]
    for row in comparison:
        lines.append(
            "| %s | %s | %s | %s |"
            % (row["outcome"], row["baseline"], row["candidate"], row["winner"])
        )
    wins = sum(1 for row in comparison if row["winner"] == "candidate")
    lines.append("")
    lines.append("выиграл воркбенч в %d из %d исходов" % (wins, len(comparison)))
    return "\n".join(lines) + "\n"


def false_negative_reason(task):
    """Честно назвать задачи, где prompt-only быстрее и воркбенч — накладные.

    Возвращает строку-причину, если задача попадает в быстрый путь, и пустую
    строку, если воркбенч свою цену отрабатывает.

    Быстрый путь: task["kind"] в FAST_PATH_KINDS, шагов не больше одного и
    задача не залезает в запретную зону.

    false_negative_reason({"kind": "formatter", "steps": 1})
      ->  "formatter: один шаг, prompt-only быстрее"
    false_negative_reason({"kind": "formatter", "steps": 4})  ->  ""

    Зачем это в наборе: урок требует перечислять ложноотрицательные случаи
    открыто, иначе воркбенч выглядит как оверинжиниринг, и его выкинут
    целиком вместе с полезной частью.
    """
    if task.get("kind") not in FAST_PATH_KINDS:
        return ""
    if int(task.get("steps", 1)) > 1:
        return ""
    if task.get("touches_forbidden"):
        # запретная зона отменяет быстрый путь: цена ошибки тут выше экономии
        return ""
    return "%s: один шаг, prompt-only быстрее" % task["kind"]
