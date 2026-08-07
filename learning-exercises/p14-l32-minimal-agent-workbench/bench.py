"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_links = "\n".join(f"- `docs/topic_{i}.md`" for i in range(300))
_router = (
    "# AGENTS.md\n\n1. `agent_state.json`\n2. `task_board.json`\n"
    + _links
    + "\n\nVerification command: python3 -m pytest -x\n"
)
_fs = {f"docs/topic_{i}.md": "x" for i in range(0, 300, 2)}
_fs["agent_state.json"] = "{}"
_fs["task_board.json"] = "[]"

_board = [
    {
        "id": f"T-{i:03d}",
        "goal": f"задача {i}",
        "owner": "builder",
        "status": random.choice(["todo", "todo", "in_progress", "done", "blocked"]),
        "priority": random.randint(0, 9),
    }
    for i in range(1, 801)
]
_state = {"active_task_id": None, "touched_files": [], "next_action": ""}
_files = [f"pkg/mod_{i}.py" for i in range(12)]

BENCH = {
    "router_links": (_router,),
    "lint_router": (_router, _fs),
    "next_task": (_board,),
    "pull_task": (_state, _board),
    "run_turn": (_state, _board, _files),
    "run_session": (_state, _board, _files, 20),
    "board_summary": (_board,),
}
