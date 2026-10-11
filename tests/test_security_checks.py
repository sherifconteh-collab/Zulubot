"""The repo's static security controls: each rule flags a bad sample, and the repo itself passes."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "security_checks", Path(__file__).resolve().parent.parent / "scripts" / "security_checks.py")
sc = importlib.util.module_from_spec(_SPEC)
sys.modules["security_checks"] = sc
_SPEC.loader.exec_module(sc)  # type: ignore[union-attr]


def _tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="seccheck-"))
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class RepoPassesTests(unittest.TestCase):
    def test_this_repository_has_no_blocking_findings(self):
        for name, check in sc.CHECKS.items():
            with self.subTest(check=name):
                self.assertEqual(check(sc.ROOT).blocking, [], f"{name} found blocking issues")


class PatternTests(unittest.TestCase):
    def test_dangerous_calls_block_and_comments_are_ignored(self):
        root = _tree({"trading_bot/a.py": (
            "x = eval(s)\n"
            "import subprocess\nsubprocess.run(c, shell=True)\n"
            "import pickle\npickle.loads(b)\n"
            "yaml.load(f)\n"
            "# eval(commented)\n"
            "ok = literal_eval(s)\n"
            "model.eval()\n"
        )})
        report = sc.check_patterns(root)
        kinds = " ".join(report.blocking)
        self.assertEqual(len(report.blocking), 4, report.blocking)
        for word in ("eval()/exec()", "shell=True", "unpickling", "yaml.load"):
            self.assertIn(word, kinds)

    def test_advisory_findings_do_not_block(self):
        root = _tree({"trading_bot/a.py": (
            "m = __import__('x')\nimport base64\nd = base64.b64decode(s)\nu = 'http://evil.example.org/x'\n"
            "ok = 'http://localhost:8080'\n")})
        report = sc.check_patterns(root)
        self.assertEqual(report.blocking, [])
        self.assertEqual(len(report.advisory), 3)


class CryptoTests(unittest.TestCase):
    def test_weak_crypto_and_disabled_tls_block(self):
        root = _tree({"trading_bot/c.py": (
            "import hashlib\nh = hashlib.md5(x)\ng = hashlib.sha1(x)\n"
            "okay = hashlib.md5(x, usedforsecurity=False)\n"
            "fine = hashlib.sha256(x)\n"
            "requests.get(u, verify=False)\n"
            "mode = modes.ECB()\n"
            "token = random.choice(chars)\n"
            "jitter = random.random()\n"
        )})
        report = sc.check_crypto(root)
        self.assertEqual(len(report.blocking), 5, report.blocking)  # md5, sha1, verify=False, ECB, random token

    def test_secure_hashing_passes(self):
        root = _tree({"trading_bot/c.py": "import hashlib, secrets\nh = hashlib.sha256(b'x')\nt = secrets.token_hex(8)\n"})
        self.assertEqual(sc.check_crypto(root).blocking, [])


class TelegramAuthTests(unittest.TestCase):
    TEMPLATE = (
        "class Bot:\n"
        "    async def _cmd_ok(self, update, context):\n"
        "        if not self._check_auth(update):\n            return\n"
        "    async def _cmd_bad(self, update, context):\n"
        "        await update.message.reply_text('hi')\n"
        "    async def _on_text(self, update, context):\n"
        "        if not self._check_auth(update):\n            return\n"
        "    def helper(self):\n        pass\n"
    )

    def test_handler_without_the_chat_check_is_flagged(self):
        root = _tree({"trading_bot/telegram_bot.py": self.TEMPLATE})
        report = sc.check_telegram_auth(root)
        self.assertEqual(len(report.blocking), 1)
        self.assertIn("_cmd_bad", report.blocking[0])

    def test_missing_handlers_are_reported_not_silently_passed(self):
        root = _tree({"trading_bot/telegram_bot.py": "class Bot:\n    pass\n"})
        self.assertEqual(len(sc.check_telegram_auth(root).blocking), 1)


class SupplyChainTests(unittest.TestCase):
    def test_unpinned_actions_and_loose_requirements(self):
        sha = "a" * 40
        root = _tree({
            ".gitignore": ".env\n",
            "requirements.txt": "requests>=2.0,<3.0\nnumpy\nrich>=13\n",
            "requirements-extra.txt": "-r requirements.txt\nfoo>=1.0\npkg @ git+https://github.com/x/y.git\n",
            ".github/workflows/ci.yml": (
                f"jobs:\n  a:\n    steps:\n      - uses: actions/checkout@v4\n"
                f"      - uses: actions/setup-python@{sha} # v5\n      - uses: ./local\n"),
        })
        report = sc.check_supply_chain(root)
        blocking = " ".join(report.blocking)
        self.assertIn("actions/checkout@v4", blocking)
        self.assertNotIn("setup-python", blocking)           # pinned
        self.assertNotIn("local", blocking)                   # local action allowed
        self.assertIn("'numpy' has no version constraint", blocking)
        self.assertIn("'rich' has no upper bound", blocking)
        advisory = " ".join(report.advisory)
        self.assertIn("'foo' has no upper bound", advisory)   # extras are advisory
        self.assertIn("not pinned to a commit", advisory)

    def test_env_files_must_be_ignored(self):
        root = _tree({"requirements.txt": "requests>=2,<3\n", ".gitignore": "*.pyc\n"})
        self.assertIn(".env", " ".join(sc.check_supply_chain(root).blocking))

    def test_alternate_package_index_blocks(self):
        root = _tree({".gitignore": ".env\n", "requirements.txt": "--extra-index-url http://evil/simple\nrequests>=2,<3\n"})
        self.assertIn("alternate package index", " ".join(sc.check_supply_chain(root).blocking))


class EndpointTests(unittest.TestCase):
    def test_tunnel_urls_block(self):
        # Built at runtime so this file itself does not contain a tunnel URL (the check scans tests too).
        url = "https://abc-def.try" + "cloudflare.com"
        root = _tree({"railway.toml": f'URL = "{url}"\n', "ok.md": "no tunnels here\n"})
        report = sc.check_endpoints(root)
        self.assertEqual(len(report.blocking), 1)
        self.assertIn("railway.toml:1", report.blocking[0])


class CliTests(unittest.TestCase):
    def test_unknown_check_is_an_error(self):
        self.assertEqual(sc.main(["x", "nope"]), 2)


if __name__ == "__main__":
    unittest.main()
