"""
Петли обратной связи во время выполнения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь собирается руками то, что в реальном харнессе делает обёртка над
subprocess: запуск -> захват вывода -> структурная запись -> следующий ход
агента читает факты, а не свою фантазию о фактах.

Соответствие настоящей системе:

    deterministic_tail  <-  усечение головы и хвоста в run_with_feedback.py
    redact              <-  вычистка секретов ПЕРЕД записью в JSONL
    make_record         <-  одна строка feedback_record.jsonl
    loop_can_advance    <-  правило "нет exit_code — нет прогресса"
    retry_chain         <-  parent_command_id, по которому ревьюер читает
                            историю попыток (Phase 14 · 40)
    rotate              <-  ротация feedback_record.jsonl на 1 MB
    run_feedback_loop   <-  сама петля: прогон, сигнал, правка, повторный прогон

Ни сети, ни subprocess, ни time.time(): прогон приходит функцией runner,
время — функцией clock. Иначе тест был бы невоспроизводим.
"""

import re

# Сколько строк оставляем с начала и с конца вывода. Хвост важнее головы:
# итоговая ошибка и summary живут в конце, поэтому его берём длиннее.
HEAD_LINES = 3
TAIL_LINES = 5

# Маркер вставляется вместо выброшенных строк. Он же — доказательство, что
# усечение было: без маркера запись врёт, будто вывод был короткий.
TRUNCATION_MARKER = "...truncated {n} lines..."

REDACTED = "[REDACTED]"

# Шаблоны секретов. Пересматривать раз в квартал против того, что реально
# течёт в проде: список ниже покрывает Bearer-заголовки, пары вида
# key=value, ключи AWS и токены Slack.
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

# Порог ротации и глубина архива. В уроке байты условные, чтобы тест был
# быстрым; в проде это 1 MB и пять поколений.
ROTATE_BYTES = 1000
MAX_ROTATIONS = 5

# Потолок числа попыток в петле. Без него неисправимый сигнал крутил бы
# петлю до конца бюджета токенов.
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
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return text, 0
    dropped = len(lines) - head - tail
    # хвост берём срезом с конца, а не по индексу head+dropped: так формула
    # не ломается, если кто-то передаст head=0
    kept = lines[:head] + [TRUNCATION_MARKER.format(n=dropped)] + lines[len(lines) - tail :]
    return "\n".join(kept), dropped


def redact(text):
    """Вычистить секреты. Вернуть (текст, сколько подстановок сделано).

    redact("Authorization: Bearer abc123")
        ->  ("Authorization: Bearer [REDACTED]", 1)
    redact("all green")  ->  ("all green", 0)

    Чистим при ЗАПИСИ, а не при чтении: на диске лежит именно то, до чего
    доберётся атакующий. Чистка при чтении защищает только вежливых.

    Подставляй шаблоны из SECRET_PATTERNS по порядку.
    """
    total = 0
    for pattern, replacement in SECRET_PATTERNS:
        # subn сразу возвращает счётчик — второй проход по тексту не нужен
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


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
    if not command:
        raise ValueError("empty command has nothing to document")
    if duration_ms < 0:
        raise ValueError(f"negative duration_ms: {duration_ms}")

    clean_out, n_out = redact(stdout)
    clean_err, n_err = redact(stderr)
    out_tail, dropped_out = deterministic_tail(clean_out)
    err_tail, dropped_err = deterministic_tail(clean_err)
    return {
        "command": tuple(command),
        "command_id": command_id,
        "parent_command_id": parent_command_id,
        "exit_code": exit_code,
        "stdout_tail": out_tail,
        "stderr_tail": err_tail,
        "dropped_stdout_lines": dropped_out,
        "dropped_stderr_lines": dropped_err,
        "redacted": n_out + n_err,
        "duration_ms": duration_ms,
        "started_at": started_at,
        "agent_note": agent_note,
        "error": error,
    }


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
    return record["exit_code"] is not None and record["error"] is None


def retry_chain(records, command_id):
    """Цепочка попыток от самой первой до указанной, в порядке выполнения.

    retry_chain(recs, "a-2")  ->  [запись a-0, запись a-1, запись a-2]

    Идём по parent_command_id вверх и разворачиваем. Без этой связи повторные
    попытки выглядят как независимые успехи, и аудит прячет историю провалов.

    Неизвестный id — KeyError. Цикл в ссылках — ValueError: молча зациклиться
    здесь хуже, чем упасть.
    """
    by_id = {r["command_id"]: r for r in records}
    if command_id not in by_id:
        raise KeyError(command_id)
    chain = []
    seen = set()
    current = command_id
    while current is not None:
        if current in seen:
            raise ValueError(f"cycle in retry chain at {current}")
        seen.add(current)
        record = by_id[current]
        chain.append(record)
        current = record["parent_command_id"]
    chain.reverse()
    return chain


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
    if incoming < 0:
        raise ValueError(f"negative incoming bytes: {incoming}")
    current = store.get(0, 0)
    if current + incoming <= limit:
        return {**store, 0: current + incoming}
    # сдвигаем все поколения на одно вниз, всё что уехало за max_rotations —
    # выкидываем прямо здесь, а не отдельным проходом
    rotated = {k + 1: v for k, v in store.items() if k + 1 <= max_rotations}
    rotated[0] = incoming
    return rotated


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
    records = []
    state = None
    parent = None
    for turn in range(max_turns):
        outcome = runner(turn, state)
        command_id = f"attempt-{turn}"
        record = make_record(
            outcome["command"],
            outcome.get("exit_code"),
            outcome.get("stdout", ""),
            outcome.get("stderr", ""),
            agent_note,
            clock(),
            outcome.get("duration_ms", 0),
            command_id,
            parent_command_id=parent,
            error=outcome.get("error"),
        )
        records.append(record)
        parent = command_id
        if not loop_can_advance(record):
            return {"status": "blocked", "records": records, "turns": len(records), "state": state}
        if record["exit_code"] == 0:
            return {"status": "passed", "records": records, "turns": len(records), "state": state}
        fix = fixer(record, state)
        if fix is None:
            return {"status": "stuck", "records": records, "turns": len(records), "state": state}
        state = fix
    return {"status": "exhausted", "records": records, "turns": len(records), "state": state}
