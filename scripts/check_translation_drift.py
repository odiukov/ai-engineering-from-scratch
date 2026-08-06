#!/usr/bin/env python3
"""Report lessons whose docs/en.md moved after their translation last did.

Translations live on the `translations` branch (see AGENTS.md). Because
`.gitignore` hides `i18n/*/phases/` on the main line, a correction to a lesson
can be committed and shipped while its Russian copy still teaches the old
wrong number, and nothing in `git status` says so. This compares commit dates
across the two branches and names the pairs that have gone out of step.

Advisory, not a gate: an `en.md` edit that only reflows prose needs no mirror.
Exit status is 0 unless --strict is passed.

    python3 scripts/check_translation_drift.py
    python3 scripts/check_translation_drift.py --phase 07 --lang ru
    python3 scripts/check_translation_drift.py --strict     # 1 if drift found
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TRANSLATIONS_REF = "translations"
MANUAL_MARKER = "<!-- i18n:manual -->"


def git(*args: str) -> str:
    """Run git and return stdout, or "" if the command failed."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def commit_epoch(ref: str, path: str) -> int | None:
    """Committer date of the last commit touching path on ref, as a Unix epoch."""
    out = git("log", "-1", "--format=%ct", ref, "--", path)
    return int(out) if out else None


def translated_docs(ref: str) -> list[str]:
    """Every i18n lesson doc tracked on ref."""
    listing = git("ls-tree", "-r", "--name-only", ref, "--", "i18n/")
    return [
        line
        for line in listing.splitlines()
        # i18n/<lang>/phases/<phase>/<lesson>/docs/<lang>.md
        if re.fullmatch(r"i18n/[^/]+/phases/[^/]+/[^/]+/docs/[^/]+\.md", line)
    ]


def en_counterpart(translated: str) -> str:
    """i18n/ru/phases/A/B/docs/ru.md -> phases/A/B/docs/en.md"""
    parts = translated.split("/")
    return "/".join(["phases", parts[3], parts[4], "docs", "en.md"])


def is_manual(ref: str, path: str) -> bool:
    """True if the translation is hand-authored rather than pipeline-generated."""
    return git("show", f"{ref}:{path}").lstrip().startswith(MANUAL_MARKER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", help="only this phase number, e.g. 07")
    parser.add_argument("--lang", help="only this language code, e.g. ru")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when drift is found"
    )
    args = parser.parse_args()

    if not git("rev-parse", "--verify", TRANSLATIONS_REF):
        print(
            f"no `{TRANSLATIONS_REF}` ref in this clone — nothing to compare",
            file=sys.stderr,
        )
        return 0

    candidates = translated_docs(TRANSLATIONS_REF)
    if args.lang:
        candidates = [p for p in candidates if p.split("/")[1] == args.lang]
    if args.phase:
        candidates = [
            p for p in candidates if p.split("/")[3].startswith(f"{args.phase}-")
        ]

    drifted: list[tuple[str, str, int, int]] = []
    unmirrored: list[str] = []
    for translated in sorted(candidates):
        english = en_counterpart(translated)
        if not Path(english).exists():
            unmirrored.append(translated)
            continue
        en_at = commit_epoch("HEAD", english)
        ru_at = commit_epoch(TRANSLATIONS_REF, translated)
        if en_at is None or ru_at is None:
            continue
        if en_at > ru_at and is_manual(TRANSLATIONS_REF, translated):
            drifted.append((english, translated, en_at, ru_at))

    if drifted:
        print(f"{len(drifted)} lesson(s) whose en.md is newer than the translation:\n")
        for english, translated, en_at, ru_at in drifted:
            days = (en_at - ru_at) // 86400
            print(f"  {english}")
            print(f"    {translated}  ({days}d behind)")
        print(
            "\nMirror the correction on the `translations` branch, or confirm the "
            "en.md edit changed no number, formula, snippet, or claim."
        )
    else:
        print(f"no drift across {len(candidates)} translated lesson doc(s)")

    if unmirrored:
        print(
            f"\n{len(unmirrored)} translation(s) point at a lesson missing from this "
            "branch (renamed or not yet merged):"
        )
        for path in unmirrored:
            print(f"  {path}")

    return 1 if drifted and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
