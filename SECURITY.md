# Security Policy

Zulubot can place real orders with real money, so security problems here can cost money. Thank you for helping keep it safe.

## Reporting a vulnerability

**Please do not open a public issue.** Report privately through GitHub:

1. Go to the repository's **Security** tab and choose **Report a vulnerability** (or open <https://github.com/sherifconteh-collab/Zulubot/security/advisories/new>).
2. Include what you found, how to reproduce it, and the impact. Never include real API keys or account details.

You can expect an acknowledgement within a few days and a fix or mitigation plan after triage. Please allow a reasonable time to fix a problem before disclosing it publicly. We credit reporters who want credit.

### Supported versions

Only the latest commit on `main` receives security fixes.

### In scope

- Anything that could place an order the owner did not authorise (Telegram access control, the confirm flow, plain-text and screenshot parsing)
- Leaking or logging credentials (Kalshi, Coinbase, Telegram, Anthropic, Polymarket keys)
- Bypassing `PAPER_MODE`, risk limits, stake caps or the kill switch
- Vulnerabilities in dependencies that the bot actually uses
- CI/CD weaknesses (unpinned actions, over-broad tokens)

### Out of scope

- Losing money to market moves or model error (see the disclaimer in the README)
- Problems that need an attacker to already control your machine, Railway account or Telegram account
- Findings in third-party services (Kalshi, ESPN, Polymarket, Telegram, Anthropic)

## What the project protects, and how

| Risk | Control |
| --- | --- |
| Someone else controls the bot | Telegram handlers answer **only** `TELEGRAM_CHAT_ID`; every handler calls `_check_auth` (enforced in CI) |
| Accidental real-money trading | `PAPER_MODE=true` by default; parlays refuse to run in paper mode |
| Bad or injected input moving money | Plain text is routed to a fixed list of commands by a model that cannot buy; bets are previewed and need a yes; pause, resume and kill are slash-only; screenshots are read into a validated schema and treated as data |
| Runaway stakes | Per-bet and per-parlay caps, exposure budget, daily halt and kill switch, price ceilings on parlays |
| Phantom positions | Orders are only booked when the exchange confirms fills; positions are read back and reconciled |
| Credential leaks | Secrets only via environment variables; `.env`/key files git-ignored; pre-commit hook and CI scan for keys; Gitleaks scans full history |
| Vulnerable dependencies | Version-bounded requirements, weekly Dependabot, `pip-audit` in CI, SBOM on every push to `main` |
| Supply chain via CI | All GitHub Actions pinned to commit SHAs; read-only workflow tokens |

## Automated checks

| Workflow | What it does |
| --- | --- |
| `test.yml` | Unit tests (external APIs mocked), coverage floor |
| `secrets-check.yml` | Pattern-based secret scan on every push and PR |
| `security.yml` | `pip-audit`, Gitleaks over full history (config in `.gitleaks.toml`), `scripts/security_checks.py` (below), license check; also runs weekly |
| `codeql.yml` | CodeQL `security-extended` for Python (runs once the repo is public) |
| `sbom.yml` | CycloneDX software bill of materials |
| `cla.yml` | CLA bot for pull requests. It runs on `pull_request_target` with write access, so it never checks out or runs PR code, and the action is pinned to a commit SHA. If branch protection stops it recording signatures on `main`, add a fine-grained token as the `PERSONAL_ACCESS_TOKEN` secret. |
| Dependabot | Weekly updates for Python packages and GitHub Actions |

`scripts/security_checks.py` fails the build on: `eval`/`exec`, `os.system`, `shell=True`, unpickling, unsafe `yaml.load`, MD5/SHA-1 without `usedforsecurity=False`, weak ciphers and ECB, disabled TLS verification, `random` used for secrets, Telegram handlers that skip the chat check, unpinned GitHub Actions, and committed tunnel URLs. It warns on dynamic imports, base64 decoding, non-HTTPS URLs and loosely pinned optional requirements. Run it locally with `python scripts/security_checks.py`.

## Repository settings (maintainers)

These live in GitHub settings, not in files, and they match the settings used for [ai-grc-platform](https://github.com/sherifconteh-collab/ai-grc-platform/blob/main/SECURITY.md). Once the repository is public, the maintainers' OpenClaw agent owns them and keeps `main` at this target state.

**Branch protection or ruleset for `main`** (Settings → Branches)
- Require a pull request before merging, with approving reviews. ai-grc-platform asks for 2; GitHub does not let an author approve their own pull request, so use 1 while there is a single maintainer and raise it to 2 when a second one joins.
- Dismiss stale approvals when new commits are pushed.
- Require review from Code Owners (see `.github/CODEOWNERS`).
- Require status checks to pass, and require the branch to be up to date before merging: `test`, `secrets`, and the `security.yml` jobs (Static Security Controls, Dependency Security Audit, Check Dependency Licenses, Scan Full History for Secrets).
- Require signed commits (squash merges made in GitHub's web UI are signed by GitHub).
- Do not allow bypassing these settings, and include administrators.
- Restrict who can push to `main` to the maintainers, and block force pushes and deletion.
- Require linear history (this repository merges by squash).
- The weekly release workflow commits to `main`. Give the Actions bot a bypass for that one workflow, or have it open a pull request, otherwise the Monday release fails against these rules.

**Security features** (Settings → Code security)
- Dependency graph and dependency review.
- Dependabot alerts and security updates.
- Code scanning (CodeQL, already configured in `.github/workflows/codeql.yml`; it runs once the repository is public).
- **Secret scanning with push protection.**
- **Private vulnerability reporting**, and a security contact (see the top of this file).

**Actions** (Settings → Actions)
- Default workflow token set to read-only.
- Require approval for workflows from first-time contributors, including forks.

**Accounts**
- Two-factor authentication for every maintainer.

## Rules for contributors

By opening a pull request you confirm that your contribution:

1. contains no malicious code, backdoors, telemetry or code that phones home without disclosure,
2. contains no hardcoded credentials or personal endpoints,
3. is your own work or properly attributed, and is submitted under the project's dual license as set out in the [CLA](CLA.md),
4. leaves `PAPER_MODE=true` as the default and does not weaken risk limits, access control or the confirm flow.

Not accepted: cryptocurrency miners, obfuscated code, code that sends data to servers not documented in the README, anything that bypasses the safeguards above.

### Review checklist

Reviewers should look for: hardcoded secrets; `eval`/`exec`/`shell=True`/pickle; new network destinations; Telegram handlers without `_check_auth`; orders booked before confirmation; amounts or prices that aren't bounded; new dependencies (check maintainer, downloads, last release, typosquatting; run `pip-audit`); and GitHub Actions that aren't pinned to a commit SHA.

## Dependencies and GitHub Actions

- Runtime requirements carry upper bounds; optional extras too, and the VCS dependency is pinned to a commit.
- Every GitHub Action is pinned to a full commit SHA with the version in a comment, for example `actions/checkout@<sha> # v7.0.1`. Dependabot proposes SHA updates. CI fails on an unpinned action.
- Do not add `--trusted-host`, `--extra-index-url` or plain-HTTP indexes.

## If something goes wrong

1. **Stop the bot trading:** send `/kill` in Telegram (liquidates and halts), or set `PAPER_MODE=true` and redeploy, or stop the Railway service.
2. **Revoke exposed secrets:** Kalshi API key (Kalshi account settings), Telegram token (`/revoke` with @BotFather), Anthropic key, Coinbase and Polymarket keys. Create new ones and update the host's variables.
3. **Check what happened:** review Kalshi order history and the bot's trade audit log (`/audit`).
4. **Remove bad code:** `git revert` the commit; if a secret was committed, rotate it first, then clean history.
5. **Report it** privately as above so others can be warned.

Third-party tooling to review before enabling: GitHub Apps, webhooks, OAuth applications. Abuse of GitHub can be reported to abuse@github.com.
