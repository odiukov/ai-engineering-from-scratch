"""
Хендофф между сессиями

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l40-multi-session-handoff
Разбор:  /check-code p14-l40-multi-session-handoff
"""

HANDOFF_FIELDS = (
    "summary",
    "changed_files",
    "commands_run",
    "failed_attempts",
    "open_risks",
    "next_action",
    "verdict_pointer",
)
REQUIRED_NONEMPTY = ("summary", "next_action", "verdict_pointer")
SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}
CLEAN_CHECKS = ("working_tree", "temp_artifacts", "tests", "feature_board", "branch")
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
