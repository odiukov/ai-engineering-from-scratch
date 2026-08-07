"""
Капстоун: переносимый пакет воркбенча

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l42-agent-workbench-capstone
Разбор:  /check-code p14-l42-agent-workbench-capstone
"""

PACK_ROOT = "agent-workbench-pack"
LOCK_FILE = ".workbench-version"
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
EXCLUDED_KINDS = {
    "project_task": "задачи живут на доске целевого репозитория, не в пакете",
    "vendor_sdk": "пакет не привязан к фреймворку",
    "onboarding_prose": "онбординг лежит рядом с пакетом, а не внутри",
}
STATE_FILES = ("agent_state.json", "task_board.json")
FANOUT_TARGETS = ("CLAUDE.md", ".cursor/rules/agents.md", ".github/copilot-instructions.md")
BUMP_ACTION = {
    "major": "migrate state",
    "minor": "re-run checker",
    "patch": "nothing",
    "same": "nothing",
}
CI_WORKFLOW = ".github/workflows/agent-workbench.yml"
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
