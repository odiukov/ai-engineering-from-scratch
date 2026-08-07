"""Входные данные для замера скорости."""

_read = lambda args: f"content-of:{args.get('path', '')}"
_write = lambda args: f"written:{args.get('path', '')}"

_registry = {"read_file": _read, "write_file": _write, "list_dir": _read}
_hooks = {
    "PreToolUse": [lambda p: None, lambda p: None],
    "PostToolUse": [lambda p: None],
}

_plan = [("read_file", {"path": f"f{i}.txt"}) for i in range(200)]
_tasks = [
    {"name": f"w{i}", "prompt": f"task {i}", "plan": _plan[:20], "allowed": ["read_file"]}
    for i in range(20)
]

_store = {"root": []}
for _i in range(50):
    _store[f"root/w{_i}"] = []
    _store[f"other{_i}"] = []

BENCH = {
    "stub_model": ("a long prompt " * 200,),
    "select_tools": (_registry, ["read_file", "list_dir"]),
    "run_hooks": (_hooks, "PreToolUse", {"tool": "read_file", "args": {}}),
    "call_tool": (_registry, _hooks, "read_file", {"path": "a.txt"}, []),
    "session_subkeys": (dict(_store), "root"),
    "session_delete": (dict(_store), "root"),
    "run_agent": ({}, "s1", "bench", _plan, _registry, _hooks),
    "spawn_subagents": ({}, "root", _tasks, _registry, _hooks),
}
