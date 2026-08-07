"""
Минимальный воркбенч агента

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l32-minimal-agent-workbench
Разбор:  /check-code p14-l32-minimal-agent-workbench
"""

import copy
import re

ROUTER_FILE = "AGENTS.md"
STATE_FILE = "agent_state.json"
BOARD_FILE = "task_board.json"
ROUTER_MAX_LINES = 50
VERIFICATION_MARKER = "Verification command:"
PATH_SUFFIXES = (".md", ".json", ".py", ".sh", ".toml", ".yaml", ".ini")
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def run_session(state, board, allowed_files, turns):
    """Сессия из turns ходов подряд. Вернуть НОВЫЕ (state, board).

    run_session(state, board, ["app.py"], 0)  ->  копия входа без изменений

    Смысл проверки: сессия из 6 ходов и две сессии по 3 хода должны дать
    одинаковый результат. Если нет — состояние живёт не в файле, а в голове
    у процесса, и следующая сессия начнёт всё заново.
    """
    raise NotImplementedError


def board_summary(board):
    """Сводка по доске: сколько задач в каждом статусе. Все четыре ключа всегда есть.

    board_summary([])  ->  {'todo': 0, 'in_progress': 0, 'done': 0, 'blocked': 0}
    board_summary([{"id": "T-001", "status": "done"}])["done"]  ->  1

    Нулевые ключи не выкидываем: отчёт с постоянной формой не ломает того,
    кто его читает. Незнакомый статус — ValueError, потому что «почти done»
    на доске означает, что кто-то придумал статус в обход схемы.
    """
    raise NotImplementedError
