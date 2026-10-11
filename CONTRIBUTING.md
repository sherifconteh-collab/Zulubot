# Contributing

Thanks for helping. This bot can place real orders, so changes are reviewed carefully.

## Setup

```bash
git clone https://github.com/sherifconteh-collab/Zulubot.git
cd Zulubot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # keep PAPER_MODE=true while developing
bash scripts/install-hooks.sh   # optional: secret-scanning pre-commit hook
```

Python 3.12. Never put real credentials in a file that git tracks.

## Before you open a pull request

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/security_checks.py
```

- Add tests for new behavior and **mock all external APIs** (Kalshi, ESPN, Polymarket, Telegram, Anthropic).
- Configuration goes through environment variables in frozen dataclasses in `trading_bot/config.py`, with the variable documented in `.env.example`.
- Keep `PAPER_MODE=true` as the default. Anything that can spend real money must be explicit, bounded and previewed.
- New Telegram handlers must call `self._check_auth(update)` first (CI enforces this).
- Update `README.md` and `CHANGELOG.md` for user-facing changes.
- Keep pull requests focused: one change per PR.

## Contributor License Agreement

Before a pull request can be merged you must sign the [CLA](CLA.md). The CLA bot prompts you on your first PR: comment `I have read the CLA Document and I hereby sign the CLA`. You keep the copyright in your contribution; the CLA lets the maintainers license it under both the AGPL-3.0 and the commercial license (see [LICENSE](LICENSE)).

## Security requirements

By submitting a pull request you certify that your contribution:

1. does not contain malicious code, backdoors or undisclosed telemetry,
2. does not contain hardcoded secrets, API keys or personal URLs (tunnels, IPs),
3. is your own work or properly attributed,
4. does not weaken access control, risk limits, the `PAPER_MODE` default or the confirm-before-buying flow,
5. is submitted under the project's dual license (AGPL-3.0 and the commercial license) as set out in the [CLA](CLA.md).

Not accepted: miners, obfuscated code, code that phones home to undisclosed servers, `eval`/`exec`/`shell=True`/pickle on untrusted data, disabling TLS verification, and unpinned GitHub Actions. See [SECURITY.md](SECURITY.md) for the full policy and how to report a vulnerability privately.

## Dependencies

Add runtime packages to `requirements.txt` with a version range that has an upper bound, and check them with `pip-audit`. Optional integrations live in `requirements-*.txt`.
