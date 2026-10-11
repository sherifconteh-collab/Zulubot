#!/usr/bin/env python3
"""Static security controls for this repository (run in CI and locally).

    python scripts/security_checks.py            # run every check
    python scripts/security_checks.py crypto     # run one check

Checks: patterns, crypto, telegram-auth, supply-chain, endpoints.
Blocking findings exit 1; advisory findings are only reported (and written to the
GitHub Actions job summary when available). These are cheap guard rails, not a
substitute for review, CodeQL or a dependency audit.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("trading_bot", "scripts")
SELF = "scripts/security_checks.py"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Report:
    blocking: list[str] = field(default_factory=list)
    advisory: list[str] = field(default_factory=list)

    def merge(self, other: "Report") -> None:
        self.blocking += other.blocking
        self.advisory += other.advisory


def tracked_files(root: Path, suffixes: tuple[str, ...], dirs: tuple[str, ...] | None = None) -> list[Path]:
    """Git-tracked files (falls back to a directory walk outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        paths = [root / p for p in out]
    except (OSError, subprocess.CalledProcessError):
        paths = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    result = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if rel == SELF or not path.is_file() or not rel.endswith(suffixes):
            continue
        if dirs is not None and rel.split("/")[0] not in dirs:
            continue
        result.append(path)
    return result


def _lines(path: Path) -> list[tuple[int, str]]:
    try:
        return list(enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1))
    except OSError:
        return []


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

BLOCKING_PATTERNS = [
    (re.compile(r"\b(?<![.\w])(eval|exec)\("), "eval()/exec() runs arbitrary code"),
    (re.compile(r"\bos\.system\("), "os.system() runs a shell command"),
    (re.compile(r"shell\s*=\s*True"), "shell=True invites command injection"),
    (re.compile(r"\b(pickle|marshal|shelve)\.loads?\("), "unpickling untrusted data runs code"),
    (re.compile(r"\byaml\.load\((?![^)]*SafeLoader)"), "yaml.load without SafeLoader runs code"),
]
ADVISORY_PATTERNS = [
    (re.compile(r"__import__\("), "dynamic import"),
    (re.compile(r"\bimportlib\.import_module\("), "dynamic import"),
    (re.compile(r"\bb64decode\("), "base64 decoding (common obfuscation)"),
    (re.compile(r"\bcompile\((?=.*['\"]exec['\"])"), "compile() of dynamic code"),
]
HTTP_URL = re.compile(r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|example\.com|www\.w3\.org)[\w.-]+")


def check_patterns(root: Path = ROOT) -> Report:
    report = Report()
    for path in tracked_files(root, (".py",), CODE_DIRS):
        rel = path.relative_to(root).as_posix()
        for number, line in _lines(path):
            if _is_comment(line):
                continue
            for pattern, why in BLOCKING_PATTERNS:
                if pattern.search(line):
                    report.blocking.append(f"{rel}:{number}: {why}")
            for pattern, why in ADVISORY_PATTERNS:
                if pattern.search(line):
                    report.advisory.append(f"{rel}:{number}: {why}")
            if HTTP_URL.search(line):
                report.advisory.append(f"{rel}:{number}: non-HTTPS URL")
    return report


CRYPTO_RULES = [
    (re.compile(r"hashlib\.(md5|sha1)\((?![^)]*usedforsecurity\s*=\s*False)"),
     "MD5/SHA-1 are broken for security use; use SHA-256, or pass usedforsecurity=False for non-security hashing"),
    (re.compile(r"\b(TripleDES|ARC4|Blowfish|CAST5|IDEA)\b"), "weak or legacy cipher"),
    (re.compile(r"modes\.ECB\b|MODE_ECB\b"), "ECB mode leaks plaintext patterns"),
    (re.compile(r"verify\s*=\s*False"), "TLS certificate verification disabled"),
    (re.compile(r"_create_unverified_context|CERT_NONE"), "TLS certificate verification disabled"),
]
RANDOM_NEAR_SECRET = re.compile(r"\brandom\.(random|randint|choice|choices|randrange|getrandbits)\(")
SECRETISH = re.compile(r"token|secret|password|passwd|nonce|salt|otp|session|api_?key|private", re.IGNORECASE)


def check_crypto(root: Path = ROOT) -> Report:
    report = Report()
    for path in tracked_files(root, (".py",), CODE_DIRS):
        rel = path.relative_to(root).as_posix()
        for number, line in _lines(path):
            if _is_comment(line):
                continue
            for pattern, why in CRYPTO_RULES:
                if pattern.search(line):
                    report.blocking.append(f"{rel}:{number}: {why}")
            if RANDOM_NEAR_SECRET.search(line) and SECRETISH.search(line):
                report.blocking.append(
                    f"{rel}:{number}: random is not cryptographically secure; use the secrets module")
    return report


def check_telegram_auth(root: Path = ROOT) -> Report:
    """Every Telegram handler must check the chat ID itself (single-user access control)."""
    report = Report()
    path = root / "trading_bot" / "telegram_bot.py"
    if not path.exists():
        return report
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^    async def (?P<name>_cmd_\w+|_on_\w+|_confirm_pending)\(.*?(?=^    (?:async )?def |^class |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    handlers = list(pattern.finditer(text))
    if not handlers:
        report.blocking.append("trading_bot/telegram_bot.py: no handlers found (check pattern is out of date)")
    for match in handlers:
        if "_check_auth" not in match.group(0):
            report.blocking.append(
                f"trading_bot/telegram_bot.py: handler {match.group('name')} does not call _check_auth")
    return report


_REQ_LINE = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(?P<spec>.*)$")


def check_supply_chain(root: Path = ROOT) -> Report:
    report = Report()
    gitignore = root / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if not re.search(r"^\.env\*?$", ignored, re.MULTILINE) and ".env" not in ignored.split():
        report.blocking.append(".gitignore: .env files are not ignored")

    for req in sorted(root.glob("requirements*.txt")):
        runtime = req.name == "requirements.txt"
        bucket = report.blocking if runtime else report.advisory
        for number, raw in _lines(req):
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-r"):
                continue
            if re.search(r"--(trusted-host|extra-index-url|index-url)|http://", line):
                report.blocking.append(f"{req.name}:{number}: insecure or alternate package index")
                continue
            if "@" in line and "git+" in line:
                if "#" not in raw and not re.search(r"@[0-9a-f]{7,40}\b", line.split("git+", 1)[1]):
                    bucket.append(f"{req.name}:{number}: VCS dependency not pinned to a commit")
                continue
            match = _REQ_LINE.match(line)
            if not match:
                continue
            spec = match.group("spec")
            if not spec or "*" in spec:
                bucket.append(f"{req.name}:{number}: '{match.group('name')}' has no version constraint")
            elif "==" not in spec and "<" not in spec and "~=" not in spec:
                bucket.append(f"{req.name}:{number}: '{match.group('name')}' has no upper bound")

    for wf in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        for number, line in _lines(wf):
            m = re.match(r"\s*-?\s*uses:\s*(?P<ref>\S+)", line)
            if not m or m.group("ref").startswith("./"):
                continue
            ref = m.group("ref").split("#", 1)[0]
            if "@" not in ref or not SHA_RE.match(ref.split("@", 1)[1]):
                report.blocking.append(
                    f".github/workflows/{wf.name}:{number}: action '{ref}' is not pinned to a commit SHA")
    return report


TUNNELS = re.compile(r"trycloudflare\.com|\.ngrok(-free)?\.(io|app|dev)|loca\.lt|serveo\.net")


def check_endpoints(root: Path = ROOT) -> Report:
    """Personal tunnel URLs must never be committed (they expose a private machine)."""
    report = Report()
    for path in tracked_files(root, (".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".example", ".json", ".sh")):
        for number, line in _lines(path):
            if TUNNELS.search(line):
                report.blocking.append(f"{path.relative_to(root).as_posix()}:{number}: tunnel URL committed")
    return report


CHECKS = {
    "patterns": check_patterns,
    "crypto": check_crypto,
    "telegram-auth": check_telegram_auth,
    "supply-chain": check_supply_chain,
    "endpoints": check_endpoints,
}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"Unknown check(s): {', '.join(unknown)}. Choose from: {', '.join(CHECKS)}", file=sys.stderr)
        return 2
    total = Report()
    for name in names:
        report = CHECKS[name]()
        status = "FAIL" if report.blocking else "ok"
        print(f"[{status}] {name}: {len(report.blocking)} blocking, {len(report.advisory)} advisory")
        for item in report.blocking:
            print(f"  ERROR  {item}")
        for item in report.advisory:
            print(f"  warn   {item}")
        total.merge(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and (total.blocking or total.advisory):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("### Security checks\n\n")
            for item in total.blocking:
                fh.write(f"- :x: {item}\n")
            for item in total.advisory:
                fh.write(f"- :warning: {item} (advisory)\n")
    return 1 if total.blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
