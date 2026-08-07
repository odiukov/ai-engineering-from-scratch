"""
Петли обратной связи во время выполнения

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l37-runtime-feedback-loops
Разбор:  /check-code p14-l37-runtime-feedback-loops
"""

import re

HEAD_LINES = 3
TAIL_LINES = 5
TRUNCATION_MARKER = "...truncated {n} lines..."
REDACTED = "[REDACTED]"
SECRET_PATTERNS = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer " + REDACTED),
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?key|token)"
            r"\s*[:=]\s*\S+"
        ),
        r"\1=" + REDACTED,
    ),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA" + REDACTED),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]+"), "xox-" + REDACTED),
)
ROTATE_BYTES = 1000
MAX_ROTATIONS = 5
MAX_TURNS = 5


def deterministic_tail(text, head=HEAD_LINES, tail=TAIL_LINES):
    """Усечь вывод до головы и хвоста. Вернуть (текст, сколько строк выброшено).

    deterministic_tail("a\\nb\\nc", head=1, tail=1)
        ->  ("a\\nb\\nc", 0)          коротко, резать нечего
    deterministic_tail("1\\n2\\n3\\n4\\n5", head=1, tail=1)
        ->  ("1\\n...truncated 3 lines...\\n5", 3)

    Детерминированно: один и тот же вывод обязан давать одну и ту же запись,
    иначе две одинаковые попытки выглядят как разные. Никакого сэмплирования.

    Ловушка: резать надо только когда строк СТРОГО больше head + tail. Если
    резать при равенстве, маркер вытеснит ноль строк и запись станет длиннее
    оригинала.
    """
    raise NotImplementedError


def redact(text):
    """Вычистить секреты. Вернуть (текст, сколько подстановок сделано).

    redact("Authorization: Bearer abc123")
        ->  ("Authorization: Bearer [REDACTED]", 1)
    redact("all green")  ->  ("all green", 0)

    Чистим при ЗАПИСИ, а не при чтении: на диске лежит именно то, до чего
    доберётся атакующий. Чистка при чтении защищает только вежливых.

    Подставляй шаблоны из SECRET_PATTERNS по порядку.
    """
    raise NotImplementedError


def make_record(
    command,
    exit_code,
    stdout,
    stderr,
    agent_note,
    started_at,
    duration_ms,
    command_id,
    parent_command_id=None,
    error=None,
):
    """Собрать одну запись обратной связи.

    make_record(["pytest"], 0, "1 passed", "", "жду зелёный", 100, 42, "a-0")
        ->  {"command": ("pytest",), "exit_code": 0, "stdout_tail": "1 passed",
             "stderr_tail": "", "redacted": 0, "duration_ms": 42, ...}

    Команда хранится КОРТЕЖЕМ argv, а не строкой: строку кто-нибудь снова
    прогонит через shell, и запись перестанет описывать то, что выполнялось.

    exit_code=None означает "прогон сорвался, кода нет" — вместе с error это
    единственный случай, когда петля обязана остановиться.

    Порядок обработки: сначала redact, потом deterministic_tail. Наоборот
    нельзя — секрет мог бы уцелеть в оставленной голове или хвосте.

    Пустая команда и отрицательная длительность — ValueError: такая запись
    ничего не документирует, а гейт верификации потом на неё сошлётся.
    """
    raise NotImplementedError


def loop_can_advance(record):
    """Можно ли двигать петлю дальше по этой записи.

    loop_can_advance(make_record(["pytest"], 1, "", "fail", "", 0, 1, "a"))
        ->  True    ненулевой код — это сигнал, по нему есть что чинить
    loop_can_advance(make_record(["pytest"], None, "", "", "", 0, 1, "a",
                                 error="timeout"))
        ->  False   кода нет, чинить нечего, заявлять успех нельзя

    Главное недоразумение: кажется, что блокировать надо провал. Нет.
    Провал — самая полезная обратная связь. Блокирует ОТСУТСТВИЕ результата:
    нет exit_code — нет прогресса.
    """
    raise NotImplementedError


def retry_chain(records, command_id):
    """Цепочка попыток от самой первой до указанной, в порядке выполнения.

    retry_chain(recs, "a-2")  ->  [запись a-0, запись a-1, запись a-2]

    Идём по parent_command_id вверх и разворачиваем. Без этой связи повторные
    попытки выглядят как независимые успехи, и аудит прячет историю провалов.

    Неизвестный id — KeyError. Цикл в ссылках — ValueError: молча зациклиться
    здесь хуже, чем упасть.
    """
    raise NotImplementedError


def rotate(store, incoming, limit=ROTATE_BYTES, max_rotations=MAX_ROTATIONS):
    """Дописать incoming байт в лог, при переполнении сдвинуть поколения.

    store — словарь {поколение: размер в байтах}, 0 это текущий файл.

    rotate({0: 100}, 50, limit=1000)   ->  {0: 150}
    rotate({0: 990}, 50, limit=1000)   ->  {0: 50, 1: 990}

    Поколение старше max_rotations выбрасывается: архив ограничен, иначе
    ротация лечила бы симптом и не лечила болезнь.

    Отдельный случай: одна запись сама больше лимита. Её всё равно пишем —
    потерять обратную связь хуже, чем превысить порог на один файл.

    Отрицательный incoming — ValueError.
    """
    raise NotImplementedError


def run_feedback_loop(runner, fixer, agent_note, clock, max_turns=MAX_TURNS):
    """Петля: прогон -> сигнал -> правка -> повторный прогон.

    runner(turn, state) -> {"command": [...], "exit_code": int|None,
                            "stdout": str, "stderr": str,
                            "duration_ms": int, "error": str|None}
    fixer(record, state) -> новое state, либо None если чинить нечем.
    clock() -> метка времени начала попытки.

    Вернуть {"status": ..., "records": [...], "turns": int, "state": ...}.

    Четыре исхода, и все они конечны:
      "passed"    — очередной прогон дал нулевой код;
      "blocked"   — прогон сорвался, exit_code=None, двигаться нельзя;
      "stuck"     — fixer вернул None, сигнал неисправим, петлю крутить незачем;
      "exhausted" — израсходованы max_turns попыток.

    Ловушка: без ветки "stuck" петля будет гонять один и тот же провальный
    прогон, пока не кончатся попытки. Формально она завершится, фактически
    сожжёт бюджет впустую.

    Каждая попытка ссылается на предыдущую через parent_command_id, поэтому
    retry_chain по последней записи возвращает всю историю.
    """
    raise NotImplementedError
