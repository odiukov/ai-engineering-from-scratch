"""
Контракты области изменений

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l36-scope-contracts
Разбор:  /check-code p14-l36-scope-contracts
"""

import fnmatch

CONTRACT_REQUIRED_FIELDS = (
    "task_id",
    "goal",
    "allowed_files",
    "forbidden_files",
    "acceptance_criteria",
    "rollback_plan",
)
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
