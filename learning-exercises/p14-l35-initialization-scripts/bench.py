"""Входные данные для замера скорости."""

import json
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_deps = [f"pkg_{i:04d}" for i in range(500)]
_env = [f"AGENT_VAR_{i:03d}" for i in range(200)]
_installed = list(_deps) + [f"extra_{i}" for i in range(500)]
random.shuffle(_installed)

_config = {
    "python": (3, 10),
    "deps": _deps,
    "env": _env,
    "test_command": "pytest -q",
}
_snapshot = {
    "version": (3, 12),
    "installed": _installed,
    "environ": {name: f"value_{name}" for name in _env},
    "changed_files": [f"src/mod_{i}.py" for i in range(40)],
}
_lock = {"fingerprint": "0" * 16, "written_at": 0}
_fs = {"workdir/agent_state.json": json.dumps({"written_at": 9_000})}

BENCH = {
    "probe_runtime": ((3, 12), (3, 10)),
    "probe_dependencies": (_installed, _deps),
    "probe_env": (_snapshot["environ"], _env),
    "probe_state_freshness": ({"written_at": 0}, 10_000),
    "probe_lkg_diff": (_snapshot["changed_files"], 50),
    "deps_fingerprint": (_config,),
    "lock_is_fresh": (_lock, _config, 10_000),
    "run_init": (_fs, _snapshot, _config, 10_000, False),
}
