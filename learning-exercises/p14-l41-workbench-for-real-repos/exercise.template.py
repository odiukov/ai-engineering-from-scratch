"""
Воркбенч на реальном репозитории

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l41-workbench-for-real-repos
Разбор:  /check-code p14-l41-workbench-for-real-repos
"""

import fnmatch

OUTCOME_KEYS = (
    "tests_actually_run",
    "acceptance_met",
    "files_outside_scope",
    "handoff_quality",
    "reviewer_total",
)
OUTCOME_DIRECTION = {
    "tests_actually_run": "higher",
    "acceptance_met": "higher",
    "files_outside_scope": "lower",
    "handoff_quality": "higher",
    "reviewer_total": "higher",
}
FORBIDDEN_HINTS = ("scripts/*", ".github/*", "README.md", "*.lock")
ACCEPTANCE_EVIDENCE = (
    (("pytest.ini", "conftest.py", "test_*.py"), "python3 -m pytest -q"),
    (("package.json",), "npm test"),
)
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def render_before_after(comparison):
    """Собрать before-after-report.md: таблица исходов плюс итоговая строка.

    Строк ровно столько, сколько исходов, плюс заголовок, разделитель и
    строка "выиграл воркбенч в N из M исходов".

    render_before_after(rows).startswith("| outcome |")  ->  True

    Это тот артефакт, который отдают скептику: числа спорят лучше объяснений.
    """
    raise NotImplementedError


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
    raise NotImplementedError
