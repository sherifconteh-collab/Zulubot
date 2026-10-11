from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parents[1] / "scripts" / "release.py")
release = importlib.util.module_from_spec(_SPEC)
sys.modules["release"] = release   # dataclasses look the module up by name
_SPEC.loader.exec_module(release)

_CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

### Parlays
- Added a thing.

#### Details
- More detail.

## [2026.9.1] — 2026-09-20

### Older
- Old entry.
"""


class VersionTests(unittest.TestCase):
    def test_first_release_a_new_month_and_the_same_month(self):
        self.assertEqual(release.next_version(None, date(2026, 10, 12)), "2026.10.0")
        self.assertEqual(release.next_version("2026.9.1", date(2026, 10, 12)), "2026.10.0")   # new month: back to 0
        self.assertEqual(release.next_version("2026.10.0", date(2026, 10, 19)), "2026.10.1")  # same month: +1
        self.assertEqual(release.next_version("2025.12.4", date(2026, 1, 5)), "2026.1.0")     # new year, no leading zero
        with self.assertRaises(release.ReleaseError):
            release.next_version("v1.2", date(2026, 10, 12))

    def test_latest_version_comes_from_release_notes_then_the_changelog(self):
        notes = "# Release notes\n\n## v2026.9.1 — 2026-09-20\n\nx\n\n## v2026.9.0 — 2026-09-01\n"
        self.assertEqual(release.latest_released_version(notes), "2026.9.1")
        self.assertEqual(release.latest_released_version("", _CHANGELOG), "2026.9.1")
        self.assertIsNone(release.latest_released_version("", "# Changelog\n\n## [Unreleased]\n"))


class ChangelogTests(unittest.TestCase):
    def test_unreleased_with_only_placeholders_is_empty(self):
        for body in ("", "\n\n", "> Nothing yet\n", "<!-- add entries -->\n", "---\n"):
            sections = release.split_changelog(f"# C\n\n## [Unreleased]\n\n{body}\n## [2026.9.0] — 2026-09-01\n")
            self.assertFalse(release.has_content(sections.unreleased), repr(body))
        self.assertTrue(release.has_content(["- real entry"]))

    def test_missing_unreleased_section_is_an_error(self):
        with self.assertRaises(release.ReleaseError):
            release.split_changelog("# Changelog\n\n## [2026.9.0] — 2026-09-01\n")

    def test_rotation_moves_the_entries_under_the_new_version(self):
        new, body = release.cut_changelog(_CHANGELOG, "2026.10.0", date(2026, 10, 12))
        self.assertIn("## [Unreleased]\n\n## [2026.10.0] — 2026-10-12\n\n### Parlays\n- Added a thing.", new)
        self.assertIn("#### Details", body)
        self.assertLess(new.index("## [2026.10.0]"), new.index("## [2026.9.1]"))   # newest first
        self.assertIn("Intro text.", new)
        self.assertEqual(new.count("## [Unreleased]"), 1)
        self.assertTrue(new.endswith("\n") and not new.endswith("\n\n"))

    def test_cutting_a_version_that_exists_or_with_nothing_to_cut_fails(self):
        with self.assertRaises(release.ReleaseError):
            release.cut_changelog(_CHANGELOG, "2026.9.1", date(2026, 10, 12))
        with self.assertRaises(release.ReleaseError):
            release.cut_changelog("# C\n\n## [Unreleased]\n\n", "2026.10.0", date(2026, 10, 12))


class ReleaseNotesAndBadgeTests(unittest.TestCase):
    def test_new_entries_go_on_top_and_old_ones_are_kept(self):
        first = release.prepend_release_notes("", "2026.10.0", date(2026, 10, 12), "### A\n- one")
        self.assertTrue(first.startswith("# Release notes\n\n## v2026.10.0 — 2026-10-12"))
        second = release.prepend_release_notes(first, "2026.10.1", date(2026, 10, 19), "### B\n- two")
        self.assertLess(second.index("## v2026.10.1"), second.index("## v2026.10.0"))
        self.assertEqual(second.count("# Release notes"), 1)
        self.assertEqual(release.latest_released_version(second), "2026.10.1")

    def test_badge_is_pointed_at_the_new_version(self):
        for old in ("Release-pending-lightgrey.svg", "Release-v2026.9.0-green", "Release-v2026.9.0-green.svg"):
            readme = f"[![Release](https://img.shields.io/badge/{old})](RELEASE_NOTES.md)\ntext"
            fixed = release.update_badge(readme, "2026.10.1")
            self.assertIn("img.shields.io/badge/Release-v2026.10.1-green)", fixed)
            self.assertIn("](RELEASE_NOTES.md)", fixed)
        self.assertEqual(release.update_badge("no badge here", "2026.10.1"), "no badge here")


class CutTests(unittest.TestCase):
    def _repo(self, changelog=_CHANGELOG, notes="", readme="[![R](https://img.shields.io/badge/Release-pending-lightgrey.svg)](x)\n"):
        root = Path(tempfile.mkdtemp())
        (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
        if notes:
            (root / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")
        (root / "README.md").write_text(readme, encoding="utf-8")
        return root

    def test_a_full_cut_writes_all_three_files_and_the_release_body(self):
        root = self._repo()
        out = release.cut(root, today=date(2026, 10, 12))
        self.assertEqual((out["version"], out["changed"]), ("2026.10.0", "true"))
        self.assertIn("## [2026.10.0]", (root / "CHANGELOG.md").read_text())
        self.assertIn("## v2026.10.0", (root / "RELEASE_NOTES.md").read_text())
        self.assertIn("Release-v2026.10.0-green", (root / "README.md").read_text())
        self.assertIn("- Added a thing.", Path(out["notes_file"]).read_text())
        # the next cut with nothing new does nothing, and a second one numbers on from the first
        self.assertEqual(release.cut(root, today=date(2026, 10, 19))["version"], "")
        (root / "CHANGELOG.md").write_text(
            (root / "CHANGELOG.md").read_text().replace("## [Unreleased]\n", "## [Unreleased]\n\n- New.\n", 1))
        self.assertEqual(release.cut(root, today=date(2026, 10, 19))["version"], "2026.10.1")

    def test_dry_run_writes_nothing(self):
        root = self._repo()
        before = (root / "CHANGELOG.md").read_text()
        out = release.cut(root, today=date(2026, 10, 12), dry_run=True)
        self.assertEqual((out["version"], out["changed"]), ("2026.10.0", "false"))
        self.assertEqual((root / "CHANGELOG.md").read_text(), before)
        self.assertFalse((root / "RELEASE_NOTES.md").exists())

    def test_empty_unreleased_makes_no_release(self):
        root = self._repo(changelog="# C\n\n## [Unreleased]\n\n## [2026.9.1] — 2026-09-20\n\n- Old.\n")
        self.assertEqual(release.cut(root, today=date(2026, 10, 12)), {"version": "", "changed": "false", "notes_file": ""})
        self.assertFalse((root / "RELEASE_NOTES.md").exists())

    def test_forced_version_must_be_a_calendar_version(self):
        with self.assertRaises(release.ReleaseError):
            release.cut(self._repo(), today=date(2026, 10, 12), version="1.0")

    def test_check_reports_a_missing_section_or_badge(self):
        self.assertEqual(release.check(self._repo()), [])
        self.assertTrue(release.check(self._repo(readme="no badge\n")))
        self.assertTrue(release.check(self._repo(changelog="# nothing\n")))

    def test_this_repository_is_ready_for_the_weekly_cut(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(release.check(root), [])
        sections = release.split_changelog((root / "CHANGELOG.md").read_text(encoding="utf-8"))
        self.assertIsNotNone(sections)


if __name__ == "__main__":
    unittest.main()
