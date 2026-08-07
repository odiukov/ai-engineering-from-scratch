"""Тесты к уроку «Skills и Agent SDK». Правь exercise.py."""

import pytest

from exercise import (
    SKILL_ROOTS,
    build_system_prompt,
    discover_skills,
    match_skill,
    parse_agents_md,
    parse_frontmatter,
    parse_skill,
    read_subresource,
    subresource_links,
)

RELEASE_NOTES = """\
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs.
triggers:
  - release notes
  - changelog
---

# Release notes writer

1. List PRs merged since the last tag.
2. Group by label.

See style-guide.md for the house style rules.
"""

PR_REVIEWER = """\
---
name: pr-reviewer
description: Review a PR diff against the project style guide.
triggers: [review, review pr]
---

# PR reviewer

1. Fetch the PR diff.
"""

STYLE_GUIDE = "# Style\n\n- One line per PR. No prose.\n"

AGENTS_MD = """\
# Project: research-system

Some prose that belongs to nobody.

- orphan bullet before any section

## Conventions

- Python, stdlib only.
- Tests run with `pytest -q`.

## Build and run

- `python code/main.py`
"""

FILES = {
    "skills/release-notes-writer/SKILL.md": RELEASE_NOTES,
    "skills/release-notes-writer/style-guide.md": STYLE_GUIDE,
    "skills/pr-reviewer/SKILL.md": PR_REVIEWER,
    "skills/release-notes-writer/deep/nested/SKILL.md": "---\nname: nested\n---\nno",
    "AGENTS.md": AGENTS_MD,
}


def registry():
    return discover_skills(FILES)


def skill(name):
    return registry()[name]


# ------------------------------------------------------- parse_frontmatter
def test_scalar_fields_are_parsed():
    meta, _ = parse_frontmatter(RELEASE_NOTES)
    assert meta["name"] == "release-notes-writer"


def test_value_containing_a_colon_keeps_its_tail():
    """split(':') без maxsplit отрезал бы половину описания."""
    meta, _ = parse_frontmatter("---\ndescription: Use when: the user ships\n---\nB")
    assert meta["description"] == "Use when: the user ships"


def test_block_list_becomes_a_list():
    meta, _ = parse_frontmatter(RELEASE_NOTES)
    assert meta["triggers"] == ["release notes", "changelog"]


def test_inline_list_becomes_the_same_shape_as_a_block_list():
    meta, _ = parse_frontmatter(PR_REVIEWER)
    assert meta["triggers"] == ["review", "review pr"]


def test_file_without_frontmatter_is_all_body():
    assert parse_frontmatter("# Body only\n") == ({}, "# Body only\n")


def test_unclosed_frontmatter_is_rejected():
    """Иначе YAML уедет модели в качестве инструкций."""
    with pytest.raises(ValueError):
        parse_frontmatter("---\nname: x\n\n# Body\n")


# ------------------------------------------------------------- parse_skill
def test_skill_body_excludes_the_frontmatter():
    parsed = parse_skill("skills/pr-reviewer", PR_REVIEWER)
    assert parsed["body"].startswith("# PR reviewer")
    assert "description:" not in parsed["body"]


def test_skill_without_a_name_is_rejected():
    with pytest.raises(ValueError):
        parse_skill("skills/anon", "---\ndescription: no name here\n---\nBody")


def test_triggers_are_always_a_tuple_even_from_a_single_string():
    parsed = parse_skill("skills/x", "---\nname: x\ntriggers: solo\n---\nB")
    assert parsed["triggers"] == ("solo",)


def test_folder_name_mismatch_is_flagged_not_fatal():
    """Рантайм ищет папку по имени skill — расхождение однажды сломает ссылки."""
    parsed = parse_skill("skills/notes", "---\nname: release-notes-writer\n---\nB")
    assert parsed["nameMatchesFolder"] is False
    assert parse_skill("skills/x", "---\nname: x\n---\nB")["nameMatchesFolder"] is True


# --------------------------------------------------------- discover_skills
def test_every_skill_folder_is_found_and_keyed_by_name():
    assert set(registry()) == {"release-notes-writer", "pr-reviewer"}


def test_nested_skill_md_is_not_a_separate_skill():
    """SKILL.md глубже одного уровня — чужой субресурс, а не skill."""
    assert "nested" not in registry()


def test_project_skill_overrides_the_user_one():
    """Одинаковое имя в двух корнях: побеждает корень, стоящий позже."""
    files = {
        "~/.claude/skills/pr-reviewer/SKILL.md": "---\nname: pr-reviewer\n---\nUSER",
        "skills/pr-reviewer/SKILL.md": "---\nname: pr-reviewer\n---\nPROJECT",
    }
    assert discover_skills(files, SKILL_ROOTS)["pr-reviewer"]["body"] == "PROJECT"


def test_discovery_does_not_depend_on_file_insertion_order():
    forward = discover_skills(FILES)
    backward = discover_skills({k: FILES[k] for k in reversed(list(FILES))})
    assert forward == backward


# ------------------------------------------------------------- match_skill
def test_trigger_substring_selects_the_skill():
    assert match_skill(registry(), "draft the release notes") == "release-notes-writer"


def test_matching_ignores_case():
    assert match_skill(registry(), "DRAFT THE CHANGELOG") == "release-notes-writer"


def test_no_trigger_means_no_skill():
    assert match_skill(registry(), "what is MCP?") is None


def test_the_longer_trigger_wins_over_the_shorter_one():
    """Триггер «review pr» конкретнее, чем «review», и обязан победить."""
    assert match_skill(registry(), "please review pr 42") == "pr-reviewer"


def test_equal_triggers_resolve_the_same_way_in_any_order():
    """Два skill с одним триггером не имеют права зависеть от порядка установки."""
    a = parse_skill("skills/aaa", "---\nname: aaa\ntriggers: [review]\n---\nA")
    b = parse_skill("skills/zzz", "---\nname: zzz\ntriggers: [review]\n---\nB")
    forward = match_skill({"aaa": a, "zzz": b}, "review this")
    backward = match_skill({"zzz": b, "aaa": a}, "review this")
    assert forward == backward == "aaa"


# ------------------------------------------------------- subresource_links
def test_referenced_file_is_listed():
    assert subresource_links(skill("release-notes-writer")["body"]) == ("style-guide.md",)


def test_external_urls_are_not_subresources():
    """Иначе кусок чужого адреса поехал бы искаться на локальном диске."""
    assert subresource_links("Read https://example.com/style-guide.md now") == ()


def test_the_skill_file_is_not_its_own_subresource():
    assert subresource_links("mentions SKILL.md and style-guide.md") == (
        "style-guide.md",
    )


def test_links_come_back_sorted_and_deduplicated():
    body = "see z.md, then a.md, then z.md again"
    assert subresource_links(body) == ("a.md", "z.md")


# -------------------------------------------------------- read_subresource
def test_subresource_is_read_from_the_skill_folder():
    text = read_subresource(FILES, skill("release-notes-writer"), "style-guide.md")
    assert text == STYLE_GUIDE


def test_path_escaping_the_skill_root_is_refused():
    """Тело SKILL.md — недоверенный ввод; ".." отсекается до всякого чтения."""
    with pytest.raises(ValueError):
        read_subresource(FILES, skill("pr-reviewer"), "../release-notes-writer/style-guide.md")


def test_absolute_path_is_refused():
    with pytest.raises(ValueError):
        read_subresource(FILES, skill("pr-reviewer"), "/etc/passwd")


def test_missing_subresource_is_a_different_failure_than_an_escape():
    """Опечатка автора skill — это не атака, и путать их нельзя."""
    with pytest.raises(FileNotFoundError):
        read_subresource(FILES, skill("pr-reviewer"), "style-guide.md")


# ----------------------------------------------------- build_system_prompt
def test_prompt_contains_the_skill_body():
    prompt = build_system_prompt(skill("pr-reviewer"), "review 42", FILES)
    assert "Fetch the PR diff" in prompt


def test_disclosure_off_keeps_the_subresource_out_of_the_prompt():
    """Skill загружен, а детали ещё не стоили ни одного токена."""
    s = skill("release-notes-writer")
    lazy = build_system_prompt(s, "draft 1.4.0", FILES, disclose=False)
    eager = build_system_prompt(s, "draft 1.4.0", FILES, disclose=True)
    assert "One line per PR" not in lazy
    assert "One line per PR" in eager
    assert len(lazy) < len(eager)


def test_agents_md_comes_before_the_skill_body():
    """Порядок блоков повторяет жизненный цикл: проект, потом skill, потом задача."""
    prompt = build_system_prompt(
        skill("pr-reviewer"), "review 42", FILES, agents_md=parse_agents_md(AGENTS_MD)
    )
    assert prompt.index("stdlib only") < prompt.index("Fetch the PR diff")
    assert prompt.index("Fetch the PR diff") < prompt.index("review 42")


def test_missing_subresource_is_marked_instead_of_crashing_the_session():
    s = parse_skill("skills/ghost", "---\nname: ghost\n---\nSee absent.md for rules.")
    prompt = build_system_prompt(s, "go", {})
    assert "(missing)" in prompt


# --------------------------------------------------------- parse_agents_md
def test_bullets_are_grouped_under_their_section():
    parsed = parse_agents_md(AGENTS_MD)
    assert parsed["Conventions"] == ("Python, stdlib only.", "Tests run with `pytest -q`.")


def test_every_section_is_kept():
    assert set(parse_agents_md(AGENTS_MD)) == {"Conventions", "Build and run"}


def test_bullets_before_the_first_section_are_dropped():
    """У них нет темы, и подставить их в промпт некуда."""
    assert all(
        "orphan bullet" not in b
        for bullets in parse_agents_md(AGENTS_MD).values()
        for b in bullets
    )


def test_a_level_one_heading_closes_the_current_section():
    text = "## A\n- one\n# New document\n- stray\n"
    assert parse_agents_md(text) == {"A": ("one",)}
