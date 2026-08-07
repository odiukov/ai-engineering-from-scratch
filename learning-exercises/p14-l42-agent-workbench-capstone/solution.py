"""
Капстоун: переносимый пакет воркбенча — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Корень пакета внутри целевого репозитория.
PACK_ROOT = "agent-workbench-pack"

# Файл-замок, который установщик кладёт в целевой репозиторий.
LOCK_FILE = ".workbench-version"

# Минимальный состав пакета: схемы — контракт, скрипты — рантайм,
# документы — правила и рубрика. Без любого из них пакет не пакет.
REQUIRED_PACK_FILES = (
    "AGENTS.md",
    "README.md",
    "VERSION",
    "docs/agent-rules.md",
    "docs/reliability-policy.md",
    "docs/handoff-protocol.md",
    "docs/reviewer-rubric.md",
    "schemas/agent_state.schema.json",
    "schemas/task_board.schema.json",
    "schemas/scope_contract.schema.json",
    "scripts/init_agent.py",
    "scripts/run_with_feedback.py",
    "scripts/verify_agent.py",
    "scripts/generate_handoff.py",
    "bin/install.sh",
)

# Что в пакет НЕ входит, с формулировкой причины из урока.
EXCLUDED_KINDS = {
    "project_task": "задачи живут на доске целевого репозитория, не в пакете",
    "vendor_sdk": "пакет не привязан к фреймворку",
    "onboarding_prose": "онбординг лежит рядом с пакетом, а не внутри",
}

# Состояние принадлежит пользователю: пакет его не создаёт и не удаляет.
STATE_FILES = ("agent_state.json", "task_board.json")

# Единый источник правды разъезжается по всем агентским инструментам.
FANOUT_TARGETS = ("CLAUDE.md", ".cursor/rules/agents.md", ".github/copilot-instructions.md")

# Что требует бамп версии: major — миграцию состояния, minor — прогон чекера.
BUMP_ACTION = {
    "major": "migrate state",
    "minor": "re-run checker",
    "patch": "nothing",
    "same": "nothing",
}

# Путь, который установщик дописывает, если в репозитории есть CI.
CI_WORKFLOW = ".github/workflows/agent-workbench.yml"

# Порядок этапов сквозного конвейера и обязательность каждого.
SHIP_STAGES = (
    ("assemble", True),
    ("lint", True),
    ("install", True),
    ("fanout", False),
    ("ci_wiring", False),
)


def classify_pack_candidates(candidates):
    """Решить, что остаётся в пакете, а что нет.

    candidates — [{"path": ..., "kind": ...}]. Виды из EXCLUDED_KINDS
    отбраковываются с причиной, остальное попадает в пакет.

    classify_pack_candidates([{"path": "docs/agent-rules.md", "kind": "doc"},
                              {"path": "tasks/T-1.md", "kind": "project_task"}])
      ->  {"included": ["docs/agent-rules.md"],
           "excluded": [{"path": "tasks/T-1.md",
                         "reason": "задачи живут на доске целевого репозитория, не в пакете"}]}

    Оба списка отсортированы по path: состав пакета не должен зависеть от
    порядка обхода каталога, иначе один и тот же пакет собирается по-разному
    на разных машинах.
    """
    included, excluded = [], []
    for item in candidates:
        kind = item.get("kind")
        if kind in EXCLUDED_KINDS:
            excluded.append({"path": item["path"], "reason": EXCLUDED_KINDS[kind]})
        else:
            included.append(item["path"])
    return {
        "included": sorted(included),
        "excluded": sorted(excluded, key=lambda e: e["path"]),
    }


def assemble_pack(parts, version):
    """Собрать дерево пакета: {PACK_ROOT/<путь>: содержимое} плюс VERSION.

    parts — {относительный путь: содержимое}. VERSION генерируется из version,
    а не берётся из parts: версия — свойство сборки, не файла на диске.

    Если не хватает хотя бы одного файла из REQUIRED_PACK_FILES — ValueError,
    в тексте перечислены недостающие. Пакет без схемы или без скрипта
    выглядит установленным и ломается на первом же прогоне.

    assemble_pack(all_parts, "1.2.3")["agent-workbench-pack/VERSION"]  ->  "1.2.3\\n"

    Сборка детерминирована: одинаковые parts и version дают равные словари.
    """
    tree = {PACK_ROOT + "/" + rel: content for rel, content in parts.items()}
    tree[PACK_ROOT + "/VERSION"] = version + "\n"
    missing = sorted(
        rel for rel in REQUIRED_PACK_FILES if (PACK_ROOT + "/" + rel) not in tree
    )
    if missing:
        raise ValueError("в пакете не хватает: " + ", ".join(missing))
    return tree


def classify_bump(old, new):
    """Какого рода переход между версиями и что он требует от установки.

    Версия — "MAJOR.MINOR.PATCH". Возвращает {"kind": ..., "action": ...},
    где kind — "major" | "minor" | "patch" | "same", а action берётся из
    BUMP_ACTION.

    classify_bump("1.2.3", "2.0.0")  ->  {"kind": "major", "action": "migrate state"}
    classify_bump("1.2.3", "1.2.3")  ->  {"kind": "same", "action": "nothing"}

    Две ошибки, которые ловятся здесь:
      * версия не из трёх целых чисел — ValueError, а не молчаливое сравнение
        строк ("1.10.0" < "1.9.0" как строка, но не как версия);
      * new старше old — тоже ValueError: установщик не имеет права тихо
        откатить целевой репозиторий на предыдущий пакет.
    """

    def parse(text):
        chunks = str(text).split(".")
        if len(chunks) != 3:
            raise ValueError("версия должна быть MAJOR.MINOR.PATCH, получено %r" % (text,))
        out = []
        for chunk in chunks:
            if not chunk.isdigit():
                raise ValueError("версия должна быть MAJOR.MINOR.PATCH, получено %r" % (text,))
            out.append(int(chunk))
        return tuple(out)

    a, b = parse(old), parse(new)
    if b < a:
        raise ValueError("откат версии запрещён: %s -> %s" % (old, new))
    for kind, index in (("major", 0), ("minor", 1), ("patch", 2)):
        if a[index] != b[index]:
            return {"kind": kind, "action": BUMP_ACTION[kind]}
    return {"kind": "same", "action": BUMP_ACTION["same"]}


def fanout_targets(pack):
    """Куда разъезжается единый AGENTS.md пакета.

    Возвращает список {"link": ..., "source": ...} по FANOUT_TARGETS; source у
    всех один и тот же — AGENTS.md пакета. Если AGENTS.md в пакете нет,
    список пуст: ссылаться не на что.

    fanout_targets({})  ->  []
    len(fanout_targets(pack))  ->  3

    Смысл: форк пакета ради одного инструмента — это провал. Один источник
    правды, симлинки наружу.
    """
    source = PACK_ROOT + "/AGENTS.md"
    if source not in pack:
        return []
    return [{"link": target, "source": source} for target in FANOUT_TARGETS]


def install_pack(repo, pack, version, force=False):
    """Положить пакет в целевой репозиторий. Возвращает НОВЫЙ словарь файлов.

    repo и pack — модели файловой системы {путь: содержимое}.

    * если хотя бы один путь пакета уже есть в repo и force=False — ValueError;
    * пишется LOCK_FILE с версией;
    * раскладываются симлинки из fanout_targets (моделируем строкой
      "-> <источник>");
    * если в репозитории есть что-нибудь под .github/workflows/, дописывается
      CI_WORKFLOW.

    install_pack({}, pack, "1.0.0")[".workbench-version"]  ->  "1.0.0\\n"

    Идемпотентность: install_pack(installed, pack, version, force=True) даёт
    дерево, равное installed. Входной repo не мутируется — иначе повторный
    вызов на тех же данных дал бы другой результат.
    """
    collisions = sorted(path for path in pack if path in repo)
    if collisions and not force:
        raise ValueError(
            "пакет уже установлен, нужен force: " + ", ".join(collisions[:3])
        )
    out = dict(repo)
    out.update(pack)
    out[LOCK_FILE] = version + "\n"
    for link in fanout_targets(pack):
        out[link["link"]] = "-> " + link["source"] + "\n"
    # CI трогаем только там, где он уже есть: пакет не навязывает пайплайн
    if any(path.startswith(".github/workflows/") for path in repo):
        out[CI_WORKFLOW] = "# wired by " + PACK_ROOT + "\n"
    return out


def lint_pack(repo, pack_version):
    """Проверить установленный пакет: список проблем, пустой — значит всё сошлось.

    Проверяем три вещи:
      * все REQUIRED_PACK_FILES на месте;
      * PACK_ROOT/VERSION совпадает с pack_version;
      * LOCK_FILE есть и совпадает с pack_version.

    lint_pack(install_pack({}, pack, "1.0.0"), "1.0.0")  ->  []
    lint_pack({}, "1.0.0")  ->  список из отсутствующих файлов и двух версий

    Замок — главное здесь: если в целевом репозитории лежит версия от старого
    пакета, ставить новый нельзя, пока миграция не проведена. Расхождение
    замка и VERSION — это и есть дрейф, который CI обязан ловить.
    """
    problems = []
    for rel in REQUIRED_PACK_FILES:
        if (PACK_ROOT + "/" + rel) not in repo:
            problems.append("нет файла %s" % rel)
    declared = repo.get(PACK_ROOT + "/VERSION")
    if declared is None or declared.strip() != pack_version:
        problems.append("VERSION пакета не равен %s" % pack_version)
    lock = repo.get(LOCK_FILE)
    if lock is None:
        problems.append("нет замка %s" % LOCK_FILE)
    elif lock.strip() != pack_version:
        problems.append("замок %s расходится с версией %s" % (lock.strip(), pack_version))
    return problems


def uninstall_pack(repo, dirty_state_files=(), keep_agents_md=False):
    """Снять пакет, не тронув состояние пользователя.

    Удаляются файлы пакета, LOCK_FILE и симлинки FANOUT_TARGETS. Остаются
    STATE_FILES, всё под outputs/ и любые прочие файлы репозитория.

    Если хоть один файл из STATE_FILES перечислен в dirty_state_files —
    ValueError: снимать пакет поверх незакоммиченного состояния нельзя, оно
    принадлежит пользователю, а не пакету.

    keep_agents_md=True оставляет PACK_ROOT/AGENTS.md на месте.

    uninstall_pack({"agent_state.json": "{}"})  ->  {"agent_state.json": "{}"}

    Входной repo не мутируется.
    """
    dirty = sorted(set(dirty_state_files) & set(STATE_FILES))
    if dirty:
        raise ValueError("состояние не закоммичено: " + ", ".join(dirty))
    keep_path = PACK_ROOT + "/AGENTS.md"
    out = {}
    for path, content in repo.items():
        if path == keep_path and keep_agents_md:
            out[path] = content
            continue
        if path.startswith(PACK_ROOT + "/") or path == LOCK_FILE or path in FANOUT_TARGETS:
            continue
        out[path] = content
    return out


def ship_pack(parts, repo, version, force=False):
    """Сквозной конвейер капстоуна: собрать, проверить, установить, развести.

    Этапы и их обязательность заданы в SHIP_STAGES. Обязательные —
    assemble, lint, install; необязательные — fanout и ci_wiring.

    Возвращает {"ok": bool, "stages": [{"name", "required", "ok", "detail"}],
                "repo": ...}.

    Два свойства, ради которых конвейер и написан:
      * отказ ЛЮБОГО обязательного этапа валит конвейер целиком: ok=False,
        оставшиеся этапы помечаются detail="skipped", а "repo" возвращается
        ИСХОДНЫМ. Половина установленного пакета хуже, чем ни одного;
      * отказ необязательного этапа конвейер не валит: ok остаётся True,
        а repo — установленным.

    lint запускается ДО установки, по паре «дерево пакета + замок целевого
    репозитория». Поэтому force=True не протаскивает пакет в репозиторий с
    чужим замком: устаревший замок ловится до того, как что-то записано.
    """
    stages = []
    failed = False
    pack = None
    installed = None

    for name, required in SHIP_STAGES:
        if failed:
            stages.append({"name": name, "required": required, "ok": False, "detail": "skipped"})
            continue
        try:
            if name == "assemble":
                pack = assemble_pack(parts, version)
                ok, detail = True, "файлов: %d" % len(pack)
            elif name == "lint":
                # замок берём из целевого репозитория, а не из пакета:
                # именно его расхождение и обязано остановить поставку
                view = dict(pack)
                view[LOCK_FILE] = repo.get(LOCK_FILE, version + "\n")
                problems = lint_pack(view, version)
                ok = not problems
                detail = "ok" if ok else "; ".join(problems)
            elif name == "install":
                installed = install_pack(repo, pack, version, force)
                ok, detail = True, "файлов: %d" % len(installed)
            elif name == "fanout":
                links = fanout_targets(pack)
                ok = bool(links)
                detail = "ссылок: %d" % len(links)
            else:
                ok = CI_WORKFLOW in (installed or {})
                detail = "CI подключён" if ok else "в репозитории нет .github/workflows"
        except ValueError as exc:
            ok, detail = False, str(exc)
        stages.append({"name": name, "required": required, "ok": ok, "detail": detail})
        if not ok and required:
            failed = True

    return {
        "ok": not failed,
        "stages": stages,
        # всё или ничего: при провале обязательного этапа репозиторий как был
        "repo": repo if failed else installed,
    }
