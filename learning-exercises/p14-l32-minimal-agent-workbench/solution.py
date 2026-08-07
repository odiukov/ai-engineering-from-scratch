"""
Минимальный воркбенч агента — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import copy
import re

# Три файла минимального воркбенча. Файловую систему моделируем словарём
# путь -> содержимое, на диск ничего не пишем.
ROUTER_FILE = "AGENTS.md"
STATE_FILE = "agent_state.json"
BOARD_FILE = "task_board.json"

# Роутер должен помещаться на экран: длинный AGENTS.md перестают читать.
ROUTER_MAX_LINES = 50

# Строка, по которой ищем команду верификации в роутере.
VERIFICATION_MARKER = "Verification command:"

# Что считаем путём внутри обратных кавычек. Без этого списка в ссылки
# попадает всё подряд, включая `state.active_task_id`.
PATH_SUFFIXES = (".md", ".json", ".py", ".sh", ".toml", ".yaml", ".ini")

# Допустимые статусы задачи на доске.
STATUSES = ("todo", "in_progress", "done", "blocked")


def router_links(text):
    """Пути, на которые ссылается роутер: всё в обратных кавычках, похожее на файл.

    router_links("читай `agent_state.json` и `docs/rules.md`")
        ->  ['agent_state.json', 'docs/rules.md']
    router_links("проверь `status == \\"done\\"`")  ->  []

    Путём считается токен без пробелов, у которого есть один из суффиксов
    PATH_SUFFIXES или встречается «/». Иначе в ссылки попадут выражения вроде
    `state.active_task_id`, и линтер начнёт ругаться на несуществующие файлы.
    Дубли схлопываются, порядок появления сохраняется.
    """
    links = []
    for token in re.findall(r"`([^`\n]+)`", text):
        if " " in token:
            continue
        looks_like_path = "/" in token or token.endswith(PATH_SUFFIXES)
        if looks_like_path and token not in links:
            links.append(token)
    return links


def lint_router(text, fs, max_lines=ROUTER_MAX_LINES):
    """Проверить роутер: длина, битые ссылки, ссылки на состояние и доску, команда верификации.

    Возвращает отсортированный список кодов проблем. Пустой список = роутер здоров.

    Коды: 'too_long', 'missing_state_link', 'missing_board_link',
          'no_verification', 'broken_link:<путь>'.

    lint_router("", {})
        ->  ['missing_board_link', 'missing_state_link', 'no_verification']
    lint_router("см. `docs/rules.md`", {})
        ->  ['broken_link:docs/rules.md', 'missing_board_link',
             'missing_state_link', 'no_verification']

    fs — файловая система как словарь путь -> содержимое. Битая ссылка хуже
    отсутствующего правила: агент уходит читать файл, которого нет, и
    возвращается ни с чем.
    """
    links = router_links(text)
    problems = []
    if len(text.splitlines()) > max_lines:
        problems.append("too_long")
    if STATE_FILE not in links:
        problems.append("missing_state_link")
    if BOARD_FILE not in links:
        problems.append("missing_board_link")
    if VERIFICATION_MARKER not in text:
        problems.append("no_verification")
    for link in links:
        if link not in fs:
            problems.append(f"broken_link:{link}")
    # сортировка нужна, чтобы линтер давал одинаковый вывод при одинаковом
    # входе: иначе diff отчёта шумит на каждом прогоне
    return sorted(problems)


def next_task(board):
    """Задача, которую агент возьмёт следующей: самый высокий priority среди 'todo'.

    board = [{"id": "T-001", "status": "todo", "priority": 1},
             {"id": "T-002", "status": "todo", "priority": 5}]
    next_task(board)["id"]  ->  'T-002'
    next_task([])           ->  None

    priority по умолчанию 0. Ничья разрешается порядком на доске: первая
    записанная задача выигрывает, потому что доска — это очередь, а не куча.
    Доску не менять: выбор задачи и её захват — разные операции.
    """
    best = None
    best_key = None
    for index, task in enumerate(board):
        if task.get("status") != "todo":
            continue
        # минус на индексе не нужен: сравниваем (priority, -index) через max,
        # меньший индекс должен побеждать при равном приоритете
        key = (task.get("priority", 0), -index)
        if best_key is None or key > best_key:
            best, best_key = task, key
    return best


def pull_task(state, board):
    """Взять задачу с доски в работу. Вернуть НОВЫЕ (state, board).

    Если state["active_task_id"] уже занят — ничего не меняется: два
    незакрытых задания одновременно и есть та самая потеря фокуса.

    state = {"active_task_id": None, "touched_files": [], "next_action": ""}
    board = [{"id": "T-001", "goal": "валидация", "status": "todo"}]
    new_state, new_board = pull_task(state, board)
    new_state["active_task_id"]  ->  'T-001'
    new_board[0]["status"]       ->  'in_progress'

    Ловушка: входные state и board изменять нельзя. Агент читает файл, пишет
    новый — так же, как это делает атомарная запись на диск.
    """
    state = copy.deepcopy(state)
    board = copy.deepcopy(board)
    if state.get("active_task_id") is not None:
        return state, board
    task = next_task(board)
    if task is None:
        state["next_action"] = "idle: на доске нет todo"
        return state, board
    task["status"] = "in_progress"
    state["active_task_id"] = task["id"]
    state["touched_files"] = []
    state["next_action"] = f"start {task['id']}: {task.get('goal', '')}"
    return state, board


def run_turn(state, board, allowed_files):
    """Один ход агента. Вернуть НОВЫЕ (state, board).

    Порядок: нет активной задачи — взять с доски; есть — тронуть очередной
    файл из allowed_files; все файлы тронуты — закрыть задачу.

    state = {"active_task_id": "T-001", "touched_files": [], "next_action": ""}
    board = [{"id": "T-001", "goal": "g", "status": "in_progress"}]
    run_turn(state, board, ["app.py"])[0]["touched_files"]  ->  ['app.py']

    Отдельный случай: активная задача исчезла с доски (её удалили руками).
    Тогда сбрасываем active_task_id в None и не притворяемся, что работаем.
    """
    state = copy.deepcopy(state)
    board = copy.deepcopy(board)

    if state.get("active_task_id") is None:
        return pull_task(state, board)

    active = next((t for t in board if t.get("id") == state["active_task_id"]), None)
    if active is None:
        state["active_task_id"] = None
        state["next_action"] = "активная задача пропала с доски, беру новую"
        return state, board

    touched = state.get("touched_files", [])
    remaining = [f for f in allowed_files if f not in touched]
    if remaining:
        state["touched_files"] = touched + [remaining[0]]
        state["next_action"] = f"править {remaining[0]} по задаче {active['id']}"
        return state, board

    active["status"] = "done"
    state["active_task_id"] = None
    state["touched_files"] = []
    state["next_action"] = "pick next task from board"
    return state, board


def run_session(state, board, allowed_files, turns):
    """Сессия из turns ходов подряд. Вернуть НОВЫЕ (state, board).

    run_session(state, board, ["app.py"], 0)  ->  копия входа без изменений

    Смысл проверки: сессия из 6 ходов и две сессии по 3 хода должны дать
    одинаковый результат. Если нет — состояние живёт не в файле, а в голове
    у процесса, и следующая сессия начнёт всё заново.
    """
    for _ in range(turns):
        state, board = run_turn(state, board, allowed_files)
    return copy.deepcopy(state), copy.deepcopy(board)


def board_summary(board):
    """Сводка по доске: сколько задач в каждом статусе. Все четыре ключа всегда есть.

    board_summary([])  ->  {'todo': 0, 'in_progress': 0, 'done': 0, 'blocked': 0}
    board_summary([{"id": "T-001", "status": "done"}])["done"]  ->  1

    Нулевые ключи не выкидываем: отчёт с постоянной формой не ломает того,
    кто его читает. Незнакомый статус — ValueError, потому что «почти done»
    на доске означает, что кто-то придумал статус в обход схемы.
    """
    counts = {status: 0 for status in STATUSES}
    for task in board:
        status = task.get("status")
        if status not in counts:
            raise ValueError(f"неизвестный статус задачи: {status!r}")
        counts[status] += 1
    return counts
