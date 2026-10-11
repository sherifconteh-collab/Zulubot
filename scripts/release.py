#!/usr/bin/env python3
"""Cut a release from CHANGELOG.md: version, release notes, README badge.

Same model as the maintainer's other repositories: every PR adds its entry under
``## [Unreleased]`` in CHANGELOG.md; a weekly workflow (``.github/workflows/release.yml``) turns that
into a dated version, prepends it to RELEASE_NOTES.md and updates the README release badge.

Versions are calendar versions, ``YYYY.M.N``: the year, the month with no leading zero, and a
same-month counter starting at 0 (the first release in October 2026 is ``2026.10.0``, the next
``2026.10.1``). A month with no release simply has none. An empty ``[Unreleased]`` cuts nothing.

Usage::

    python scripts/release.py cut [--dry-run] [--version X] [--date YYYY-MM-DD]
    python scripts/release.py check

Standard library only; no network access.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = "CHANGELOG.md"
RELEASE_NOTES = "RELEASE_NOTES.md"
README = "README.md"

_VERSION_RE = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d+)$")
_UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\s*$", re.IGNORECASE)
_VERSION_HEADING_RE = re.compile(r"^## \[(\d{4}\.\d{1,2}\.\d+)\]")
_RELEASE_NOTES_HEADING_RE = re.compile(r"^## v(\d{4}\.\d{1,2}\.\d+)\b", re.MULTILINE)
_BADGE_RE = re.compile(r"img\.shields\.io/badge/Release-[^)\s\"']*")

CHANGELOG_PREAMBLE = (
    "# Changelog\n\n"
    "Every pull request adds its entry under **[Unreleased]**. A weekly workflow "
    "(`.github/workflows/release.yml`) turns it into a dated release `YYYY.M.N` and copies it to "
    "[RELEASE_NOTES.md](RELEASE_NOTES.md). See [CONTRIBUTING.md](CONTRIBUTING.md).\n\n"
)


class ReleaseError(RuntimeError):
    """Something in the release inputs is malformed."""


def next_version(current: str | None, today: date) -> str:
    """The calendar version for a release cut on ``today`` given the latest released version."""
    if not current:
        return f"{today.year}.{today.month}.0"
    match = _VERSION_RE.match(current)
    if not match:
        raise ReleaseError(f"Not a calendar version: {current!r}")
    year, month, counter = (int(part) for part in match.groups())
    if (year, month) == (today.year, today.month):
        return f"{year}.{month}.{counter + 1}"
    return f"{today.year}.{today.month}.0"


def latest_released_version(release_notes: str, changelog: str = "") -> str | None:
    """Top version in RELEASE_NOTES.md (falling back to CHANGELOG.md), or None before the first release."""
    found = _RELEASE_NOTES_HEADING_RE.search(release_notes)
    if found:
        return found.group(1)
    for line in changelog.splitlines():
        match = _VERSION_HEADING_RE.match(line)
        if match:
            return match.group(1)
    return None


@dataclass
class Sections:
    before: list[str]       # everything above the Unreleased heading (title, preamble)
    unreleased: list[str]   # the lines of the Unreleased body
    after: list[str]        # everything from the first version heading on


def split_changelog(text: str) -> Sections:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if _UNRELEASED_RE.match(line)), None)
    if start is None:
        raise ReleaseError("CHANGELOG.md has no '## [Unreleased]' section.")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return Sections(lines[:start], lines[start + 1:end], lines[end:])


def has_content(body: list[str]) -> bool:
    """Real entries, ignoring blank lines, HTML comments and blockquote placeholders."""
    text = "\n".join(body)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(">") and stripped != "---":
            return True
    return False


def _tidy(body: list[str]) -> str:
    return "\n".join(body).strip("\n")


def cut_changelog(text: str, version: str, today: date) -> tuple[str, str]:
    """Rotate ``[Unreleased]`` into ``[version]``; returns (new changelog text, the release body)."""
    sections = split_changelog(text)
    if not has_content(sections.unreleased):
        raise ReleaseError("[Unreleased] has no entries.")
    if any(line.startswith(f"## [{version}]") for line in sections.after):
        raise ReleaseError(f"CHANGELOG.md already has version {version}.")
    body = _tidy(sections.unreleased)
    out = list(sections.before) + ["## [Unreleased]", "", f"## [{version}] — {today.isoformat()}", ""]
    out += body.splitlines() + [""]
    out += sections.after
    return "\n".join(out).rstrip("\n") + "\n", body


def prepend_release_notes(existing: str, version: str, today: date, body: str) -> str:
    """Add a release entry to the top of RELEASE_NOTES.md (creating the file's heading if needed)."""
    heading = "# Release notes"
    previous = existing.strip()
    if previous.startswith("#") and previous.splitlines()[0].lower().startswith("# release notes"):
        previous = "\n".join(previous.splitlines()[1:]).strip("\n")
    entry = (
        f"## v{version} — {today.isoformat()}\n\n"
        f"> **Version**: {version} · **Released**: {today.isoformat()}\n\n{body}\n"
    )
    pieces = [heading, entry]
    if previous:
        pieces.append(previous)
    return "\n\n".join(pieces).rstrip("\n") + "\n"


def update_badge(readme: str, version: str) -> str:
    """Point the README's Release badge at ``version`` (no change if the badge isn't there)."""
    return _BADGE_RE.sub(f"img.shields.io/badge/Release-v{version}-green", readme)


def cut(root: Path, *, today: date, version: str | None = None, dry_run: bool = False) -> dict[str, str]:
    """Run the whole cut against files under ``root``. Returns ``{"version", "changed", "notes_file"}``."""
    changelog_path, notes_path, readme_path = root / CHANGELOG, root / RELEASE_NOTES, root / README
    changelog = changelog_path.read_text(encoding="utf-8")
    sections = split_changelog(changelog)
    if not has_content(sections.unreleased):
        return {"version": "", "changed": "false", "notes_file": ""}
    existing_notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    chosen = version or next_version(latest_released_version(existing_notes, changelog), today)
    if not _VERSION_RE.match(chosen):
        raise ReleaseError(f"Not a calendar version: {chosen!r}")
    new_changelog, body = cut_changelog(changelog, chosen, today)
    new_notes = prepend_release_notes(existing_notes, chosen, today, body)
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    new_readme = update_badge(readme, chosen)
    if not dry_run:
        changelog_path.write_text(new_changelog, encoding="utf-8")
        notes_path.write_text(new_notes, encoding="utf-8")
        if readme_path.exists() and new_readme != readme:
            readme_path.write_text(new_readme, encoding="utf-8")
    notes_file = root / f".release-notes-{chosen}.md"
    if not dry_run:
        notes_file.write_text(body + "\n", encoding="utf-8")
    return {"version": chosen, "changed": "false" if dry_run else "true", "notes_file": str(notes_file)}


def check(root: Path) -> list[str]:
    """Problems that would break the weekly cut (used by the tests and runnable by hand)."""
    problems: list[str] = []
    try:
        split_changelog((root / CHANGELOG).read_text(encoding="utf-8"))
    except (ReleaseError, OSError) as exc:
        problems.append(str(exc))
    readme = root / README
    if readme.exists() and not _BADGE_RE.search(readme.read_text(encoding="utf-8")):
        problems.append("README.md has no Release badge (img.shields.io/badge/Release-...) to keep current.")
    return problems


def _write_outputs(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cut_cmd = sub.add_parser("cut", help="Turn [Unreleased] into a dated release")
    cut_cmd.add_argument("--dry-run", action="store_true", help="Report the version without writing files")
    cut_cmd.add_argument("--version", help="Force a version (YYYY.M.N) instead of computing it")
    cut_cmd.add_argument("--date", help="Release date (YYYY-MM-DD); default today (UTC)")
    sub.add_parser("check", help="Validate the files the weekly cut depends on")
    args = parser.parse_args(argv)

    if args.command == "check":
        problems = check(ROOT)
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1 if problems else 0

    today = date.fromisoformat(args.date) if args.date else datetime.now(timezone.utc).date()
    try:
        result = cut(ROOT, today=today, version=args.version, dry_run=args.dry_run)
    except ReleaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not result["version"]:
        print("[Unreleased] has no entries: nothing to release.")
        _write_outputs({"changed": "false", "version": ""})
        return 0
    print(f"Release {result['version']}{' (dry run)' if args.dry_run else ''}")
    _write_outputs({**result, "dry_run": "true" if args.dry_run else "false"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
