"""
Воркбенч агента: почему сильные модели всё равно ошибаются

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l31-agent-workbench-why-models-fail
Разбор:  /check-code p14-l31-agent-workbench-why-models-fail
"""

SURFACES = (
    "instructions",
    "state",
    "scope",
    "feedback",
    "verification",
    "review",
    "handoff",
)
FAILURE_MODES = ("loop", "premature_stop", "scope_creep", "unverified_success")
MODE_SURFACE = {
    "loop": "state",
    "premature_stop": "instructions",
    "scope_creep": "scope",
    "unverified_success": "verification",
}
LOOP_THRESHOLD = 3


def missing_surfaces(present):
    """Какие поверхности воркбенча отсутствуют. Порядок — канонический SURFACES.

    missing_surfaces(["scope", "state"])  ->  ['instructions', 'feedback',
                                               'verification', 'review', 'handoff']
    missing_surfaces(SURFACES)            ->  []

    Ловушка: не возвращай set — порядок несёт смысл, отчёты сравниваются
    дословно. И не сортируй по алфавиту: канонический порядок — это порядок
    таблицы из урока, а не буквенный.
    """
    raise NotImplementedError


def weakest_surface(scores):
    """Самая слабая поверхность: имя с минимальной оценкой 0..2.

    weakest_surface({"scope": 2, "state": 0, "review": 1})  ->  'state'
    weakest_surface({"handoff": 1, "state": 1})             ->  'state'

    Ничья разрешается каноническим порядком SURFACES, а не алфавитом: в
    примере выше state идёт раньше handoff, поэтому чинят его.
    Пустой словарь и незнакомое имя поверхности — ValueError, потому что
    аудит по опечатке в имени хуже, чем отсутствие аудита.
    """
    raise NotImplementedError


def repeated_steps(trace, threshold=LOOP_THRESHOLD):
    """Пары (action, target), встретившиеся в трейсе не меньше threshold раз.

    Шаг трейса — словарь {"action": ..., "target": ..., "ok": ...}.

    repeated_steps([{"action": "read", "target": "app.py"}] * 3)
        ->  [('read', 'app.py')]
    repeated_steps([{"action": "read", "target": "app.py"}] * 2)
        ->  []

    Результат отсортирован по (action, target), чтобы отчёт был стабильным.
    Это детектор зацикливания: агент без файла состояния перечитывает одно и
    то же, потому что не помнит, что уже прочитал.
    """
    raise NotImplementedError


def off_scope_writes(trace, allowed_files):
    """Записи в файлы вне разрешённого списка, в порядке первого появления.

    trace = [{"action": "write", "target": "app.py"},
             {"action": "write", "target": "README.md"},
             {"action": "write", "target": "README.md"}]
    off_scope_writes(trace, ["app.py"])  ->  ['README.md']

    Учитываются только шаги с action == "write": чтение чужого файла — не
    нарушение области, запись — нарушение. Дубли схлопываются, но порядок
    первого появления сохраняется: он подсказывает, где агент свернул не туда.
    """
    raise NotImplementedError


def acceptance_status(trace, acceptance):
    """Состояние каждого критерия приёмки: 'passed' | 'failed' | 'not_run'.

    trace = [{"action": "run", "target": "pytest", "ok": False}]
    acceptance_status(trace, ["pytest"])        ->  {'pytest': 'failed'}
    acceptance_status(trace, ["lint"])          ->  {'lint': 'not_run'}
    acceptance_status([], [])                   ->  {}

    'passed' — если хотя бы один запуск критерия завершился ok. Один зелёный
    прогон после красного считается за успех: агент чинил и починил.
    Разница между 'failed' и 'not_run' принципиальна: первое — сломанный код,
    второе — сломанный воркбенч.
    """
    raise NotImplementedError


def classify_failures(trace, allowed_files, acceptance):
    """Режимы отказа, видимые в трейсе. Порядок — канонический FAILURE_MODES.

    Правила:
      loop                — какая-то пара (action, target) повторилась >= LOOP_THRESHOLD раз;
      scope_creep         — была запись вне allowed_files;
      premature_stop      — есть шаг "stop", и хотя бы один критерий не запускался вообще;
      unverified_success  — есть шаг "stop" с ok=True, все критерии запускались,
                            но хотя бы один провалился.

    trace = [{"action": "stop", "target": "done", "ok": True}]
    classify_failures(trace, ["app.py"], ["pytest"])  ->  ['premature_stop']

    Ловушка: premature_stop и unverified_success взаимоисключающи. Первое —
    агент не проверял вовсе, второе — проверял, увидел красное и всё равно
    объявил победу. Это разные поломки и чинят их разные поверхности.
    Трейс без шага "stop" — незавершённый прогон, ни один из этих двух
    режимов по нему не ставится.
    """
    raise NotImplementedError


def surfaces_to_fix(modes):
    """Поверхности, которые надо починить, по списку режимов отказа.

    surfaces_to_fix(["scope_creep", "loop"])  ->  ['state', 'scope']
    surfaces_to_fix([])                       ->  []

    Дубли схлопываются, порядок — канонический SURFACES (state идёт раньше
    scope, поэтому в примере он первый). Незнакомый режим — ValueError:
    молча проглоченная опечатка превращает отчёт в ложное «всё хорошо».
    """
    raise NotImplementedError


def failure_report(trace, allowed_files, acceptance):
    """Полный отчёт по прогону: что сломалось и какую поверхность чинить.

    Ключи результата: modes, surfaces_to_fix, off_scope_writes,
    repeated_steps, acceptance, clean.

    trace = [{"action": "run", "target": "pytest", "ok": True},
             {"action": "stop", "target": "done", "ok": True}]
    failure_report(trace, ["app.py"], ["pytest"])["clean"]  ->  True

    clean == True означает «ни один режим отказа не сработал», а не «код
    хороший»: воркбенч проверяет процесс, а качество кода проверяет ревьюер.
    """
    raise NotImplementedError
