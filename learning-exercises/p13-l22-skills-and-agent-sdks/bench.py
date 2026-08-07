"""Входные данные для замера скорости."""

_BODY = "\n".join(
    f"{i}. Step {i}. See ref-{i % 40}.md for details." for i in range(600)
)

_SKILL_MD = (
    "---\n"
    "name: bench-skill\n"
    "description: Benchmark skill: does nothing, quickly.\n"
    "triggers:\n"
    "  - bench\n"
    "  - benchmark this skill\n"
    "---\n\n"
    + _BODY
    + "\n"
)

# много папок skill: наивный обход "для каждого корня пройти все файлы
# повторно" заметно медленнее одного прохода со срезом префикса
_files = {}
for _i in range(400):
    _files[f"skills/skill-{_i:03d}/SKILL.md"] = _SKILL_MD.replace(
        "bench-skill", f"skill-{_i:03d}"
    )
    _files[f"skills/skill-{_i:03d}/ref-0.md"] = "# ref\n\n- rule one\n"

_skill = {
    "name": "skill-000",
    "description": "Benchmark skill",
    "triggers": ("bench", "benchmark this skill"),
    "body": _BODY,
    "root": "skills/skill-000",
    "folder": "skill-000",
    "nameMatchesFolder": True,
}

_registry = {
    f"skill-{i:03d}": dict(_skill, name=f"skill-{i:03d}", root=f"skills/skill-{i:03d}")
    for i in range(400)
}

_AGENTS_MD = "# Project\n\n" + "\n".join(
    f"## Section {i}\n" + "\n".join(f"- rule {j}" for j in range(20))
    for i in range(100)
)

BENCH = {
    "parse_frontmatter": (_SKILL_MD,),
    "parse_skill": ("skills/bench-skill", _SKILL_MD),
    "discover_skills": (_files, None),
    "match_skill": (_registry, "please benchmark this skill for me"),
    "subresource_links": (_BODY,),
    "read_subresource": (_files, _skill, "ref-0.md"),
    "build_system_prompt": (_skill, "do it", _files),
    "parse_agents_md": (_AGENTS_MD,),
}
