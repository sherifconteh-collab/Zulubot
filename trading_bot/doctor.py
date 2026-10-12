"""`python -m trading_bot doctor`: say which settings are present and what each one unlocks.

Reads only environment variables (after `.env` is loaded). It never starts the service, never
touches the network and never prints a secret value, only "set" or "missing".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Finding:
    level: str
    text: str


def _value(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _is_placeholder(value: str) -> bool:
    """`.env.example` values such as `your_bot_token_from_botfather` count as missing."""
    lowered = value.lower()
    return lowered.startswith("your_") or lowered.endswith("_here") or lowered in {"changeme", "xxx"}


def _has(env: Mapping[str, str], *names: str) -> bool:
    return all(_value(env, n) and not _is_placeholder(_value(env, n)) for n in names)


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _kalshi_key_looks_valid(env: Mapping[str, str]) -> bool:
    key = _value(env, "KALSHI_PRIVATE_KEY").replace("\\n", "\n")
    return "BEGIN" in key and "PRIVATE KEY" in key


def diagnose(env: Mapping[str, str]) -> list[Finding]:
    out: list[Finding] = []
    paper = _flag(env, "PAPER_MODE", True)
    kalshi = _has(env, "KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY")

    # Mode: the single most important line.
    if paper:
        out.append(Finding(OK, "Mode: PAPER (PAPER_MODE=true). Nothing is sent to any broker. Safe to try."))
    elif kalshi:
        out.append(Finding(WARN, "Mode: LIVE. Sports, parlays and the other Kalshi strategies place REAL orders."))
    else:
        out.append(Finding(WARN, "PAPER_MODE=false but the Kalshi key is missing: orders stay on paper "
                                 "until KALSHI_KEY_ID and KALSHI_PRIVATE_KEY are set."))

    # Kalshi.
    if kalshi and not _kalshi_key_looks_valid(env):
        out.append(Finding(FAIL, "KALSHI_PRIVATE_KEY is set but does not look like a PEM key "
                                 "(use the full text, with \\n for newlines)."))
    elif kalshi:
        out.append(Finding(OK, "Kalshi: key set. Real bets are possible when PAPER_MODE=false."))
    else:
        out.append(Finding(INFO, "Kalshi: no key. Markets are read-only and bets are simulated. "
                                 "Needs KALSHI_KEY_ID and KALSHI_PRIVATE_KEY to bet for real."))

    # Telegram.
    token, chat = _has(env, "TELEGRAM_BOT_TOKEN"), _has(env, "TELEGRAM_CHAT_ID")
    if token and chat:
        out.append(Finding(OK, "Telegram: configured. The bot answers only that chat."))
    elif token or chat:
        missing = "TELEGRAM_CHAT_ID" if token else "TELEGRAM_BOT_TOKEN"
        out.append(Finding(FAIL, f"Telegram: {missing} is missing, so Telegram stays off."))
    else:
        out.append(Finding(INFO, "Telegram: not set. The bot runs but you get no commands or alerts. "
                                 "Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."))

    # Anthropic.
    nl = _flag(env, "TELEGRAM_NATURAL_LANGUAGE", True)
    if _has(env, "ANTHROPIC_API_KEY"):
        out.append(Finding(OK, "Anthropic: key set. Plain-English chat and bet-slip screenshots work."))
    else:
        note = " TELEGRAM_NATURAL_LANGUAGE is on but has no effect without the key." if nl else ""
        out.append(Finding(INFO, "Anthropic: no key. Slash commands and pasted leg lists still work; "
                                 "plain-English chat and screenshots are off." + note))

    # Strategies that need Kalshi to do anything real.
    wanting_kalshi = [
        label for label, name, default in (
            ("sports", "KALSHI_SPORTS_ENABLED", True),
            ("predictions", "PREDICTION_ENABLED", False),
            ("latency arbitrage", "LATENCY_ARB_ENABLED", False),
            ("perpetuals", "KALSHI_PERP_ENABLED", False),
        ) if _flag(env, name, default)
    ]
    if wanting_kalshi and not kalshi and not paper:
        out.append(Finding(WARN, "Enabled but cannot trade without a Kalshi key: " + ", ".join(wanting_kalshi) + "."))

    # Coinbase: keys matter only for trading.
    coinbase_on = _flag(env, "COINBASE_TRADING_ENABLED", False)
    if coinbase_on and not _has(env, "COINBASE_API_KEY", "COINBASE_API_SECRET"):
        out.append(Finding(FAIL, "COINBASE_TRADING_ENABLED=true but COINBASE_API_KEY / COINBASE_API_SECRET are missing."))
    elif coinbase_on:
        out.append(Finding(WARN, "Coinbase trading is ON: crypto and rotation place real Coinbase orders when live."))
    else:
        out.append(Finding(INFO, "Coinbase: data only (no key needed). Set COINBASE_TRADING_ENABLED=true to trade."))

    # Robinhood options.
    if _flag(env, "OPTIONS_ENABLED", False) and not _has(env, "ROBINHOOD_USERNAME", "ROBINHOOD_PASSWORD"):
        out.append(Finding(INFO, "Options are enabled without Robinhood credentials, so they stay simulated."))

    # State survives redeploys only on a persistent volume.
    storage = _value(env, "BOT_STORAGE_DIR")
    if env.get("RAILWAY_ENVIRONMENT") or env.get("RAILWAY_PROJECT_ID"):
        if storage:
            out.append(Finding(OK, f"Railway: BOT_STORAGE_DIR={storage}. Make sure a Volume is mounted there "
                                   "(send /sports and look at the Storage line)."))
        else:
            out.append(Finding(FAIL, "Railway: BOT_STORAGE_DIR is not set, so state is lost on every redeploy. "
                                     "Attach a Volume at /data and set BOT_STORAGE_DIR=/data."))

    # Leftover keys nothing reads.
    if _has(env, "FMP_API_KEY"):
        out.append(Finding(INFO, "FMP_API_KEY is set but no longer used (stock data comes from yfinance). Safe to remove."))

    # Placeholder values copied from .env.example.
    placeholders = sorted(
        name for name in (
            "KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
            "ANTHROPIC_API_KEY", "COINBASE_API_KEY", "COINBASE_API_SECRET",
        ) if _value(env, name) and _is_placeholder(_value(env, name))
    )
    if placeholders:
        out.append(Finding(WARN, "Still the example text from .env.example (treated as missing): " + ", ".join(placeholders) + "."))
    return out


_MARK = {OK: "[ok]  ", INFO: "[info]", WARN: "[warn]", FAIL: "[FAIL]"}


def render(findings: list[Finding]) -> str:
    lines = ["Zulubot setup check (no secret values are printed)", ""]
    lines += [f"{_MARK[f.level]} {f.text}" for f in findings]
    problems = sum(1 for f in findings if f.level == FAIL)
    lines += ["", "Result: " + (f"{problems} problem(s) to fix." if problems else "nothing blocking.")]
    return "\n".join(lines)


def run(env: Mapping[str, str]) -> int:
    findings = diagnose(env)
    print(render(findings))
    return 1 if any(f.level == FAIL for f in findings) else 0
