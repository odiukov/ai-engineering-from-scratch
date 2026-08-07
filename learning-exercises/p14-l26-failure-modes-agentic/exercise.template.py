"""
Failure modes: почему агенты ломаются

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l26-failure-modes-agentic
Разбор:  /check-code p14-l26-failure-modes-agentic
"""

KNOWN_TOOLS = ("search", "read_file", "write_file", "send_email", "list_dir")
TOOL_SCHEMA = {
    "search": ("query",),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "send_email": ("to", "body"),
    "list_dir": ("path",),
}
EFFECT_TOOLS = ("write_file", "send_email")
TARGET_KEYS = {"write_file": "path", "send_email": "to"}
ADDRESS_KEYS = ("path", "to")
FAILURE_MODES = (
    "cascading_error",
    "context_loss",
    "hallucinated_action",
    "repeat_loop",
    "scope_creep",
    "success_hallucination",
    "tool_misuse",
)


def tool_problems(steps):
    """Две беды уровня вызова: несуществующий инструмент и кривые аргументы.

    Вернуть {"unknown": отсортированные имена, "bad_args": отсортированные индексы}.

    tool_problems([{"tool": "search", "args": {"query": "x"}}])
      ->  {"unknown": [], "bad_args": []}
    tool_problems([{"tool": "magic_scan", "args": {}}])
      ->  {"unknown": ["magic_scan"], "bad_args": []}
    tool_problems([{"tool": "read_file", "args": {"file": "a"}}])
      ->  {"unknown": [], "bad_args": [0]}

    Аргументы считаются кривыми, если не хватает обязательного ключа ИЛИ
    есть лишний, которого нет в схеме. Второе не придирка: лишний ключ почти
    всегда означает, что модель перепутала схемы двух инструментов.

    Выдуманный инструмент в bad_args не попадает: схемы у него нет, ругаться
    на аргументы бессмысленно — это уже другой failure mode.
    """
    raise NotImplementedError


def first_repeat_index(steps, limit=3):
    """Индекс шага, на котором агент limit раз ПОДРЯД повторил один и тот же вызов.

    first_repeat_index([{"tool": "search", "args": {"query": "x"}}] * 3)  ->  2
    first_repeat_index([{"tool": "search", "args": {"query": "x"}}] * 2)  ->  None

    Ключевое слово — «подряд». Три обращения к search в разные моменты
    длинной сессии это нормальная работа; три обращения без единого другого
    действия между ними — зацикливание, агент не понял, что уже получил
    ответ, и крутит один и тот же шаг.

    Повтор — это совпадение и инструмента, и аргументов. limit < 2 ->
    ValueError: «повтор из одного шага» смысла не имеет.
    """
    raise NotImplementedError


def cascade_radius(steps):
    """Сколько шагов агент успел сделать ПОСЛЕ первой ошибки.

    cascade_radius([{"tool": "search", "args": {}, "status": "error"},
                    {"tool": "read_file", "args": {}},
                    {"tool": "write_file", "args": {}}])   ->  2
    cascade_radius([{"tool": "search", "args": {}}])       ->  0

    Считаем от ПЕРВОЙ ошибки, а не от последней: урок про то, что одна
    ранняя ошибка тянет за собой хвост. Если взять последнюю, радиус всегда
    выйдет маленьким, и самый дорогой каскад окажется невидимым.

    Шаг без ключа "status" считается успешным.
    """
    raise NotImplementedError


def context_violations(steps, constraints):
    """Индексы шагов, нарушивших ограничение, объявленное в начале сессии.

    constraints — dict с необязательными ключами "forbidden_tools" и
    "forbidden_paths" (проверяются как префиксы пути или адреса).

    context_violations([{"tool": "read_file", "args": {"path": "README.md"}},
                        {"tool": "write_file", "args": {"path": "src/a.py",
                                                        "content": ""}}],
                       {"forbidden_paths": ("src/",)})   ->  [1]
    context_violations([], {"forbidden_tools": ("send_email",)})  ->  []

    Пустой constraints -> пустой список: нечего терять.

    Почему это называется context loss, а не «непослушание»: интересен не
    сам факт нарушения, а его позиция. Нарушение на шаге 1 — агент не понял
    задачу. Нарушение на шаге 30, когда первые 29 ограничение соблюдали, —
    ограничение вытеснилось из контекста.
    """
    raise NotImplementedError


def scope_creep_targets(steps, allowed_targets):
    """Цели, которые агент изменил, хотя его об этом не просили.

    scope_creep_targets([{"tool": "write_file",
                          "args": {"path": "README.md", "content": "x"}},
                         {"tool": "write_file",
                          "args": {"path": "src/a.py", "content": "x"}}],
                        ("README.md",))            ->  ["src/a.py"]
    scope_creep_targets([{"tool": "read_file", "args": {"path": "secret"}}], ())
      ->  []

    Чтение целью не считается: агент имеет право осмотреться. Считаются
    только EFFECT_TOOLS — запись и отправка. Именно так выглядит «создал
    лишний PR» и «отправил лишнее письмо» из урока.
    """
    raise NotImplementedError


def success_hallucination(trace):
    """Агент отчитался об успехе, но мир не изменился.

    success_hallucination({"steps": [{"tool": "write_file",
                                      "args": {"path": "a", "content": "b"},
                                      "status": "error"}],
                           "claims_success": True, "state_changed": False})
      ->  True
    success_hallucination({"steps": [{"tool": "search", "args": {"query": "x"}}],
                           "claims_success": True, "state_changed": False})
      ->  False

    Второй случай — чистое чтение: успех там и не должен ничего менять.
    Проверка срабатывает, только если агент ПЫТАЛСЯ что-то изменить.

    Урок формулирует это жёстко: агент не отличает «у меня не получилось» от
    «задача невыполнима» и на 400 нередко закрывает цикл фразой об успехе.
    Поймать это можно только повторной пробой состояния, а не текстом ответа.
    """
    raise NotImplementedError


def tag_trace(trace):
    """Все failure modes одного трейса, отсортированный список меток.

    trace — dict с ключами "steps" и необязательными "constraints",
    "allowed_targets", "claims_success", "state_changed", "repeat_limit".

    tag_trace({"steps": [{"tool": "magic_scan", "args": {}}]})
      ->  ["hallucinated_action"]
    tag_trace({"steps": [{"tool": "search", "args": {"query": "x"}}]})
      ->  []

    Каскадом считается ошибка, после которой было минимум два шага: один
    шаг после ошибки — это обычно корректный retry, а не каскад.

    Меток может быть несколько: один трейс спокойно ломается сразу тремя
    способами, и разделять их важно — чинятся они по-разному.
    """
    raise NotImplementedError


def mode_distribution(traces):
    """Сколько трейсов поймано каждым модом: {метка: число трейсов}.

    Считаются трейсы, а не срабатывания: трейс с двумя repeat_loop подряд
    всё равно даёт единицу.

    mode_distribution([{"steps": [{"tool": "magic_scan", "args": {}}]},
                       {"steps": [{"tool": "search", "args": {"query": "x"}}]}])
      ->  {"hallucinated_action": 1}

    Моды, не встретившиеся ни разу, в результат не попадают, ключи
    отсортированы. Это дешёвая замена кластеризации трейсов в Phoenix:
    сначала распределение, потом решение, какой мод чинить первым.
    """
    raise NotImplementedError
