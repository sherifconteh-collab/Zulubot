# Release notes

## 2026-10-11: Paste a list of legs, no AI needed
- A typed or pasted list of legs (`Lawrence 150+ passing yards`, `Name anytime TD`, `the Eagles +16.5 spread`, ...) is read by a local parser and handled exactly like a screenshot: exact Kalshi matching, preview or one-shot placement with `bet $10`, and nothing placed unless every leg matched. Works with no Anthropic key or credit.
- **Matching fixes** (also benefit screenshots): a surname shared by two players is reported instead of guessed (the game most legs point to settles it when the slip names none), and a spread for a team is only matched in that team's own game.

## 2026-10-11: Clearer Anthropic errors in Telegram
- When the Anthropic key is out of credit, invalid, rate limited or the API is overloaded, picture and plain-text messages now say so (and what to do) instead of "I couldn't work out what you meant" or a raw API error dump. The `/` commands are unaffected.

## 2026-10-11: Open-source readiness
- **Security standards** (matching the practices in the maintainer's other public repository): `SECURITY.md` (private vulnerability reporting, threat model, controls, repository settings, incident response), `CONTRIBUTING.md`, `CODEOWNERS`, PR and issue templates, Dependabot for pip and GitHub Actions, and read-only workflow tokens with every action pinned to a commit SHA.
- **New CI:** `security.yml` (`pip-audit`, Gitleaks over full history, license check, weekly schedule), `codeql.yml`, `sbom.yml` (CycloneDX), and `scripts/security_checks.py` with tests (blocks `eval`/`exec`, `shell=True`, pickle, weak crypto, disabled TLS verification, Telegram handlers without the chat check, unpinned actions, committed tunnel URLs).
- **Fixes found by the new checks:** the explicit `/confirm` command now checks the chat ID itself (defence in depth); a non-security MD5 cache key became SHA-256; optional requirements got version bounds and the TradingAgents dependency is pinned to a commit.
- **License:** dual-licensed, matching the maintainer's other public repository: AGPL-3.0 (full text in `LICENSES/AGPL-3.0-only.txt`) or a commercial license from Conteh Consulting LLC (`LICENSE`, `NOTICE`), with a Contributor License Agreement (`CLA.md`) enforced by the CLA bot (`cla.yml`, signatures in `signatures/`).
- **README rewritten** for people: what it does, quick start, Railway deploy, Telegram commands, safety controls, how the sports model works. The full variable list lives in `.env.example`. Added a risk disclaimer.
- **Security:** `cryptography` raised to `>=50.0.2` (fixes the known advisories on 46.x/47.x; the Kalshi RSA signing path was re-tested). Removed a personal Cloudflare tunnel URL from `railway.toml` (`OLLAMA_BASE_URL` is now blank; set your own in the Railway Variables panel).

## 2026-10-11: Telegram betting, parlays and screenshots

### New: place your own bets from Telegram (#33)
- `/games`, `/bet`, `/confirm`, `/cancel`: list upcoming games with Kalshi prices and the model's view, preview a bet on the team you name, and place it as a live Kalshi order.
- `/props`: search Kalshi's NFL player-stat markets (yards, receptions, passing TDs, anytime TD, alternate spreads).
- `/parlay`: build a Kalshi combo from teams, player stats and spreads. All legs must be in the same sport; same-game legs are allowed; up to 10 legs. The bot asks Kalshi market makers for a price and buys only at or under a price ceiling (the model's fair value, Kalshi's own prices multiplied, or your `max`).
- `/stack`: auto-build a same-game player-stat parlay, like a sportsbook same-game parlay.
- **Bet-slip screenshots:** send a picture of a slip and Claude reads the legs; the bot matches each exactly to Kalshi (or reports the nearest alternatives) and previews the parlay. A caption such as `bet $10 with this parlay` places it directly when every leg matched.
- **Plain-text chat:** type what you want ("bet $5 on the browns", "how am I doing?") instead of remembering commands. Claude only picks a command; bets are still previewed and need a yes. Pause, resume and kill stay slash-only.

### New: small "action picks" when nothing is mispriced (#32)
- When no game has an edge after Kalshi's fee, the bot places small action bets (default up to 2 per scan and 5 per day, about 3% of bankroll each, never a side the model gives under 35%, at most half underdogs). Real mispricings still take priority and are Kelly-sized. Turn off with `KALSHI_SPORTS_ACTION_BETS=false`. Expected cost is about 2¢ per 50¢ contract; this buys action, not edge.

### Earlier in this release line
- **Sports betting on Kalshi** with ESPN, sportsbook lines, Polymarket and Kalshi's own order book as inputs; adaptive Kelly sizing that compounds with the wallet, drawdown throttle, joint exposure fitting, and a source-accuracy model that learns from settled games. Enabled by default (`KALSHI_SPORTS_ENABLED`). All sports, including college football, NFL, college and pro basketball, baseball, hockey and soccer.
- **Real Kalshi fee schedule** (per-series multiplier, per-order rounding) and fee-aware edge (#30).
- **Sports is exempt from the global kill switch** and keeps its own limits (per-bet cap, exposure budget, max open bets).
- **Coinbase is data-only** unless `COINBASE_TRADING_ENABLED=true`; all trading is on Kalshi (#28).
- **Live safety fixes** (#28): failed orders are never booked as filled, Kalshi positions and balance are synced, settled positions are pruned, the live drawdown peak uses the same equity as the risk check, service cycles are serialised.
- **Telegram digest:** routine alerts arrive as one message every 6 hours (`TELEGRAM_DIGEST_HOURS`) (#31). The sports output says "no bet" when there is no edge (#29).

### Configuration added
`KALSHI_SPORTS_USER_BET_DEFAULT_PCT`, `KALSHI_SPORTS_USER_BET_MAX_PCT`, `KALSHI_SPORTS_USER_PARLAY_DEFAULT_PCT`, `KALSHI_SPORTS_USER_PARLAY_MAX_PCT`, `KALSHI_SPORTS_USER_PARLAY_MAX_LEGS`, `KALSHI_SPORTS_PARLAY_QUOTE_WAIT_SECONDS`, `KALSHI_SPORTS_SLIP_MODEL`, `TELEGRAM_NATURAL_LANGUAGE`, `TELEGRAM_NL_MODEL`, and the `KALSHI_SPORTS_ACTION_*` settings. See `.env.example`.

### Upgrade notes
- Plain-text chat and screenshot reading need `ANTHROPIC_API_KEY` on the host (the same variable the AI analysis uses). Without it the bot says so and the slash commands keep working.
- Parlays and single bets place real orders only when `PAPER_MODE=false`; parlays refuse to run in paper mode. `PAPER_MODE=true` remains the default.
- The parlay request-for-quote flow follows Kalshi's published API and has not yet been exercised against a live account. Assumption to check with the first small parlay: accepting the maker's NO bid leaves you long YES at `1 − no_bid`. The bot reads the position back from Kalshi and warns on a mismatch. Start with a small stake (about $1).
- Market makers often don't quote combos at the default ceiling; in that case nothing is bought.
