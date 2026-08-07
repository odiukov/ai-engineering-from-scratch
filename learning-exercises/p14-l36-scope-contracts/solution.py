"""
Контракты области изменений — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import fnmatch

# Поля, без которых контракт неполон. forbidden_files здесь не случайно:
# урок прямо говорит, что негативное пространство — половина контракта.
CONTRACT_REQUIRED_FIELDS = (
    "task_id",
    "goal",
    "allowed_files",
    "forbidden_files",
    "acceptance_criteria",
    "rollback_plan",
)

# Три уровня серьёзности находки. block останавливает merge, warn тратит
# бюджет нарушений, info только пишется в отчёт.
SEVERITIES = ("block", "warn", "info")


def path_matches(path, pattern):
    """Совпадает ли путь с glob-шаблоном контракта.

    Сегмент "**" покрывает ноль или больше сегментов пути, "*" — любые
    символы ВНУТРИ одного сегмента и через "/" не перелезает.

    path_matches("app.py", "app.py")            ->  True
    path_matches("app/x.py", "app/**/*.py")     ->  True   ("**" покрывает ноль папок)
    path_matches("app/sub/x.py", "app/*.py")    ->  False  ("*" не пересекает "/")
    path_matches("docs/guide.md", "*.md")       ->  False
    path_matches("docs/guide.md", "**/*.md")    ->  True

    Ловушка, на которой контракты и протекают: fnmatch про "/" ничего не знает,
    и голый fnmatch("app/sub/x.py", "app/*.py") вернёт True. Поэтому путь и
    шаблон надо разбить по "/" и сопоставлять сегменты.

    Контракты пишут на globs, а не на списках путей, чтобы рефакторинг между
    сессиями не обнулял контракт.
    """
    parts = [p for p in path.split("/") if p]
    pats = [p for p in pattern.split("/") if p]
    # reach[i] == "первые i сегментов пути уже покрыты просмотренными шаблонами".
    # Динамика нужна именно из-за "**": он покрывает произвольное число
    # сегментов, и жадный проход тут ошибается.
    reach = [True] + [False] * len(parts)
    for pat in pats:
        if pat == "**":
            first = next((i for i, ok in enumerate(reach) if ok), None)
            if first is None:
                return False
            reach = [i >= first for i in range(len(parts) + 1)]
        else:
            nxt = [False] * (len(parts) + 1)
            for i, segment in enumerate(parts):
                if reach[i] and fnmatch.fnmatchcase(segment, pat):
                    nxt[i + 1] = True
            reach = nxt
    return reach[len(parts)]


def classify_write(path, contract):
    """К какой категории контракта относится правка файла.

    Вернуть "forbidden", "allowed", "soft" или "off_scope".

    contract — словарь с ключами "allowed_files", "forbidden_files" и
    необязательным "soft_files" (обычно документация).

    classify_write("app.py", C)            ->  "allowed"
    classify_write("scripts/deploy.sh", C) ->  "forbidden"
    classify_write("docs/api.md", C)       ->  "soft"
    classify_write("driver.c", C)          ->  "off_scope"

    Порядок проверок важен: forbidden сильнее allowed. Файл, попавший в оба
    списка, запрещён — иначе широкий allowed ("**/*.py") тихо разрешил бы
    правку миграций, которую контракт явно запретил.
    """
    if any(path_matches(path, p) for p in contract["forbidden_files"]):
        return "forbidden"
    if any(path_matches(path, p) for p in contract["allowed_files"]):
        return "allowed"
    if any(path_matches(path, p) for p in contract.get("soft_files", ())):
        return "soft"
    return "off_scope"


def contract_gaps(contract):
    """Чего не хватает контракту. Вернуть отсортированный список имён полей.

    Поле считается пропущенным, если его нет или оно пустое.

    contract_gaps(полный_контракт)  ->  []
    contract_gaps({"task_id": "T-1", "goal": "g", "allowed_files": ["app.py"],
                   "forbidden_files": [], "acceptance_criteria": ["pytest"],
                   "rollback_plan": "revert"})
      ->  ["forbidden_files"]

    Пустой forbidden_files — не «нечего запрещать», а незаполненный контракт:
    негативное пространство половина смысла. Пустой rollback_plan означает,
    что контракт нельзя откатить, а такой контракт не должен проходить
    approval. Пустой acceptance_criteria — что «сделано» никто не докажет.
    """
    return sorted(
        field for field in CONTRACT_REQUIRED_FIELDS if not contract.get(field)
    )


def merge_egress(parent, child):
    """Слияние сетевых allowlist по правилу наименьших привилегий.

    None означает «не проверяем», [] — «запрещено всё», список — allowlist.

    merge_egress(None, None)                     ->  None
    merge_egress(None, ["api.anthropic.com"])    ->  ["api.anthropic.com"]
    merge_egress([], ["api.anthropic.com"])      ->  []
    merge_egress(["a", "b"], ["b", "c"])         ->  ["b"]

    None уступает второй стороне: «я не проверяю» не должно ослаблять того,
    кто проверяет. Deny-all остаётся deny-all — он и получается пересечением
    пустого списка с любым. Результат отсортирован, чтобы отчёты диффались.

    Сеть — такое же измерение области, как файлы: агент, тихо сходивший на
    внешний API, вышел за рамки задачи ровно так же, как правкой лишнего файла.
    """
    if parent is None and child is None:
        return None
    if parent is None:
        return sorted(child)
    if child is None:
        return sorted(parent)
    return sorted(set(parent) & set(child))


def merge_contracts(parent, child):
    """Слияние двух контрактов (проектного и задачного) по наименьшим привилегиям.

    allowed_files ПЕРЕСЕКАЮТСЯ (разрешают оба), forbidden_files
    ОБЪЕДИНЯЮТСЯ (запретить достаточно одному), time_budget_minutes —
    минимальный из заданных, approvals_required накапливаются без дублей,
    violation_budget — минимальный, network_egress — через merge_egress.

    task_id, goal и rollback_plan берутся у child, если непустые, иначе
    у parent: конкретная задача уточняет проектные значения по умолчанию.

    merge_contracts({"allowed_files": ["app.py", "lib/**"], ...},
                    {"allowed_files": ["app.py", "docs/**"], ...})
      ->  контракт с allowed_files == ["app.py"]

    Обратное направление (объединять allowed) — самая дорогая ошибка в этом
    уроке: два безобидных контракта дали бы права, которых не давал ни один.
    """
    return {
        "task_id": child.get("task_id") or parent.get("task_id"),
        "goal": child.get("goal") or parent.get("goal"),
        "allowed_files": sorted(set(parent["allowed_files"]) & set(child["allowed_files"])),
        "forbidden_files": sorted(set(parent["forbidden_files"]) | set(child["forbidden_files"])),
        "soft_files": sorted(set(parent.get("soft_files", ())) | set(child.get("soft_files", ()))),
        "acceptance_criteria": list(
            dict.fromkeys(
                list(parent.get("acceptance_criteria", ()))
                + list(child.get("acceptance_criteria", ()))
            )
        ),
        "rollback_plan": child.get("rollback_plan") or parent.get("rollback_plan"),
        "approvals_required": list(
            dict.fromkeys(
                list(parent.get("approvals_required", ()))
                + list(child.get("approvals_required", ()))
            )
        ),
        "time_budget_minutes": min(
            (
                v
                for v in (parent.get("time_budget_minutes"), child.get("time_budget_minutes"))
                if v is not None
            ),
            default=None,
        ),
        "violation_budget": min(
            parent.get("violation_budget", 0), child.get("violation_budget", 0)
        ),
        "network_egress": merge_egress(
            parent.get("network_egress"), child.get("network_egress")
        ),
    }


def scope_check(contract, run):
    """Сверка прогона с контрактом. Вернуть отчёт с находками и вердиктом.

    run — {"touched_files", "commands_run", "elapsed_minutes", "network_hosts"}.

    Вернуть {"in_scope", "off_scope", "soft", "forbidden", "missing_acceptance",
             "findings", "warnings", "over_budget", "passed"}.

    Находка — {"code", "severity", "detail"}. Серьёзность:
      * "scope.forbidden"        block — тронут явно запрещённый путь;
      * "acceptance.missing"     block — не запущено то, что доказывает «готово»;
      * "time.over_budget"       block — превышен бюджет времени;
      * "network.unallowed_host" block — егресс на хост вне allowlist;
      * "scope.off_scope"        warn  — правка вне разрешённых путей;
      * "scope.soft_off_scope"   info  — документация вне разрешённых путей.

    over_budget = число warn БОЛЬШЕ contract["violation_budget"].
    passed = нет block и не over_budget.

    Асимметрия здесь не косметическая: гейт, который блокирует за правку
    README, отключит первая же команда, которой он помешал. Бюджет нарушений
    и есть разница между гейтом, который живёт в проекте, и гейтом, который
    выключили.
    """
    buckets = {"allowed": [], "off_scope": [], "soft": [], "forbidden": []}
    for path in run["touched_files"]:
        buckets[classify_write(path, contract)].append(path)

    missing = [
        cmd for cmd in contract.get("acceptance_criteria", ()) if cmd not in run["commands_run"]
    ]
    findings = []
    if buckets["forbidden"]:
        findings.append({"code": "scope.forbidden", "severity": "block",
                         "detail": f"тронуты запрещённые пути: {buckets['forbidden']}"})
    if missing:
        findings.append({"code": "acceptance.missing", "severity": "block",
                         "detail": f"не запущено: {missing}"})
    budget = contract.get("time_budget_minutes")
    if budget is not None and run.get("elapsed_minutes", 0) > budget:
        findings.append({"code": "time.over_budget", "severity": "block",
                         "detail": f"{run['elapsed_minutes']} мин > бюджета {budget} мин"})
    allowlist = contract.get("network_egress")
    if allowlist is not None:
        bad = [h for h in run.get("network_hosts", ()) if h not in allowlist]
        if bad:
            findings.append({"code": "network.unallowed_host", "severity": "block",
                             "detail": f"егресс на хосты вне allowlist: {bad}"})
    if buckets["off_scope"]:
        findings.append({"code": "scope.off_scope", "severity": "warn",
                         "detail": f"правки вне области: {buckets['off_scope']}"})
    if buckets["soft"]:
        findings.append({"code": "scope.soft_off_scope", "severity": "info",
                         "detail": f"документация вне области: {buckets['soft']}"})

    warnings = sum(1 for f in findings if f["severity"] == "warn")
    over_budget = warnings > contract.get("violation_budget", 0)
    blocked = any(f["severity"] == "block" for f in findings)
    return {
        "in_scope": buckets["allowed"],
        "off_scope": buckets["off_scope"],
        "soft": buckets["soft"],
        "forbidden": buckets["forbidden"],
        "missing_acceptance": missing,
        "findings": findings,
        "warnings": warnings,
        "over_budget": over_budget,
        "passed": not blocked and not over_budget,
    }


def pick_feature(feature_list):
    """Какую единственную фичу можно трогать в этой сессии.

    feature_list — {"project", "active", "features": [{"id", "status", ...}]}.
    Статусы: "todo", "in_progress", "done", "blocked".

    pick_feature({"active": "import-pdf", "features": [...]})   ->  "import-pdf"
    pick_feature({"active": "", "features": [todo_a, todo_b]})  ->  id первой todo
    pick_feature({"active": "", "features": [done, blocked]})   ->  None

    Пустой active — «выбери и запиши»: берём ПЕРВУЮ фичу со статусом "todo" в
    порядке файла. Порядок здесь и есть приоритет, поэтому сортировать по id
    нельзя.

    Два элемента в статусе "in_progress" -> ValueError, и это стартовая
    проверка, а не мелочь: список с двумя начатыми фичами означает, что
    прошлая сессия закончилась не там, где думает человек. Разбираться должен
    человек. active, которого нет среди features, — тоже ValueError.

    «Одна фича за раз» перестаёт быть строкой в промпте, которую агент умеет
    себе объяснить, и становится значением с диска.
    """
    in_progress = [f["id"] for f in feature_list["features"] if f["status"] == "in_progress"]
    if len(in_progress) > 1:
        raise ValueError(f"начато больше одной фичи: {in_progress}")
    active = feature_list.get("active")
    if active:
        known = {f["id"] for f in feature_list["features"]}
        if active not in known:
            raise ValueError(f"active={active!r} нет среди фич {sorted(known)}")
        return active
    return next((f["id"] for f in feature_list["features"] if f["status"] == "todo"), None)
