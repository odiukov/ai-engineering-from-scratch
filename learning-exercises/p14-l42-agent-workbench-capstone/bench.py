"""Входные данные для замера скорости."""

import random

random.seed(0)

_REQUIRED = (
    "AGENTS.md",
    "README.md",
    "VERSION",
    "docs/agent-rules.md",
    "docs/reliability-policy.md",
    "docs/handoff-protocol.md",
    "docs/reviewer-rubric.md",
    "schemas/agent_state.schema.json",
    "schemas/task_board.schema.json",
    "schemas/scope_contract.schema.json",
    "scripts/init_agent.py",
    "scripts/run_with_feedback.py",
    "scripts/verify_agent.py",
    "scripts/generate_handoff.py",
    "bin/install.sh",
)

_parts = {rel: "содержимое " + rel for rel in _REQUIRED if rel != "VERSION"}
_parts.update({"docs/extra%d.md" % i: "x" * 200 for i in range(600)})

_kinds = ("schema", "script", "doc", "project_task", "vendor_sdk", "onboarding_prose")
_candidates = [
    {"path": "file%04d" % i, "kind": random.choice(_kinds)} for i in range(4000)
]

_pack = {"agent-workbench-pack/" + rel: content for rel, content in _parts.items()}
_pack["agent-workbench-pack/VERSION"] = "1.0.0\n"

_repo = {"src/mod%d.py" % i: "pass\n" for i in range(2000)}
_repo[".github/workflows/ci.yml"] = "on: push"

_installed = dict(_repo)
_installed.update(_pack)
_installed[".workbench-version"] = "1.0.0\n"

BENCH = {
    "classify_pack_candidates": (_candidates,),
    "assemble_pack": (_parts, "1.0.0"),
    "classify_bump": ("1.2.3", "2.0.0"),
    "fanout_targets": (_pack,),
    "install_pack": (_repo, _pack, "1.0.0"),
    "lint_pack": (_installed, "1.0.0"),
    "uninstall_pack": (_installed,),
    "ship_pack": (_parts, _repo, "1.0.0"),
}
