# Changelog

Every pull request adds its entry under **[Unreleased]**. A weekly workflow (`.github/workflows/release.yml`) turns it into a dated release `YYYY.M.N` and copies it to [RELEASE_NOTES.md](RELEASE_NOTES.md). See [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### SECURITY.md repository settings match ai-grc-platform
- The settings list now mirrors ai-grc-platform's: branch protection (reviews, stale-approval dismissal, Code Owners, required and up-to-date checks, signed commits, no bypass, restricted pushes, linear history, administrators included), the code-security features, the read-only Actions token and approval for first-time contributors. It notes two things to settle before enforcing it: one approving review while there is one maintainer, and a bypass for the weekly release bot.

### Kalshi is the source of truth, not the bot's own files
- **Exposure comes from the account.** The sports exposure budget (and `open $` in `/sports`) now uses the larger of the bot's own records and what Kalshi says is at risk in sports markets (cost of open positions). Before, a redeploy without a persistent volume reset it to $0 and the budget could be spent again. Combo (parlay) positions are only known from the bot's records. If Kalshi can't be read, the bot's records apply and the held-market guard still stops all buying.
- **No more alarm about volumes.** `/sports` and `doctor` now say plainly that without a persistent volume only the bot's own notes (today's counters, learned weights) reset on redeploy, and that your balance and holdings are read from Kalshi. The README marks the Railway Volume as optional.
- Still local: today's action-pick and parlay counters, parlay quote tries, and the learned source weights.

### `/sports` says why the "best value" action picks did or did not fire
- New line `Action picks: on, N of 5 used today (at most 2 per scan)`, or `OFF (KALSHI_SPORTS_ACTION_BETS=false)`. "No bets right now" used to be the only answer whether the picks were switched off, used up for the day, or had nothing to bet.
- The action path now records why it skipped: `action_picks_used_up_today` and `already_bet_on_game` (a game the account already holds) show up in the skip reasons instead of vanishing.

### `python -m trading_bot doctor`
- New command that checks the setup without starting the bot or touching the network: mode (paper or live), Kalshi, Telegram, Anthropic and Coinbase keys, Railway storage, keys left over from older versions, and `.env.example` placeholder text still in place. It prints "set" or "missing", never a value, and exits non-zero only for real problems (half-configured Telegram, a malformed Kalshi key, Coinbase trading without keys, Railway without `BOT_STORAGE_DIR`).

### README audit for the public release
- Strategy table now says, per strategy, whether it is on by default and what happens with `PAPER_MODE=false`: sports, predictions, latency arbitrage, perpetuals and the BTC 5-minute Kalshi leg place real Kalshi orders; crypto and rotation run only with `COINBASE_TRADING_ENABLED=true`; options go through Robinhood only with credentials; stocks, Polymarket and Markov are simulated only.
- Corrected "all live trading happens on Kalshi" and the Coinbase data-only wording, and noted that Railway's Variables panel, not `railway.toml`, decides what is actually enabled.
- Full Telegram command reference with a description for every command (added `/start`, `/ai`, `/config`, `/perps`, `/markov`, `/options`, `/backtest [range]`, `/resume force`, `/kill`).
- Replaced the long secrets table with "Which keys do you need?": almost none, and each key is listed with the feature it unlocks. `FMP_API_KEY` is gone from the docs, `.env.example` and `railway.toml` (nothing reads it any more).
- Documented the `/data` volume requirement and the Robinhood variables. No secrets or personal tokens in the README or `.env.example`.

### Storage check in `/sports`; Kalshi market probe is now a proper script
- `/sports` prints a `Storage:` line saying where the state file lives and whether that is a mounted volume. On Railway with no Volume attached the bot forgets its bets and daily counters at every redeploy (which is how a bet can be placed twice), and now says so plainly with the fix.
- The root `test_predictions.py` (a manual script, not a test) is now `scripts/probe_kalshi_markets.py`: read-only, public data only, with `--series`, `--buy-yes`, `--buy-no` and `--min-volume` options.
- SECURITY.md: repository settings are owned by the maintainers' OpenClaw agent once the repository is public.

### Never buy a market the Kalshi account already holds
- **Duplicate bets fixed at the source.** `/sports` always showed `open $0.00` and the same market was bought again (the Golden Knights bet doubled): the bot was relying on its own records of what it holds, which can be missing or late (for example after a redeploy without a persistent volume). Before any live order the sports engine and the predictions strategy now ask Kalshi what the account actually holds (`GET /portfolio/positions`) and skip any game or market already held. Hand-picked bets and parlays respect it too.
- **Fail safe.** If the account's positions can't be read in live mode, the bot places no new bets that cycle (`positions_unavailable` in `/sports`) rather than risking a duplicate.
- `KalshiClient.get_held_tickers()` is the new source of truth for what is held.

### README describes live trading accurately
- The README no longer calls Zulubot a "paper-trading bot that trades simulated money". It starts in paper mode, but with `PAPER_MODE=false` and Kalshi credentials it places real orders (sports bets, automatic parlays, predictions). Added how to confirm live mode with `/sports` and how Kalshi's per-shard funding affects which markets can be traded.

### Orders on markets Kalshi routes to an unfunded shard
- **Kalshi holds funds per exchange shard and routes each order to the shard its market lives on.** Some sports markets answered `404 insufficient_shard_balance (Exchange user not found)` or `400 insufficient_balance` even though the account total was enough. The bot now treats those as a property of the market series: it skips that series for 6 hours (instead of retrying every game) and keeps placing orders on markets that do work. Hand-picked bets are never skipped silently.
- `/sports` shows `Kalshi balance by shard: #0 $X, #1 $Y` (from Kalshi's balance breakdown) so you can see where the money is, and order errors are no longer cut off (full Kalshi message, with the market ticker).
- The earlier "stop after 3 rejected orders" now counts only non-shard failures; otherwise unfunded-shard markets, which sort first, would have blocked every cycle.

### Parlay stake floor and order-failure details
- **Automatic parlays never stake under $5** (default; `KALSHI_SPORTS_AUTO_PARLAY_MIN_STAKE_USD`): the first live attempt at about $1.50 got no market-maker quote. Open-parlay cap raised to 25% of bankroll so three $5 parlays fit, kickoff window 12 hours, and quote requests for a game are spaced 45 minutes apart (`KALSHI_SPORTS_AUTO_PARLAY_RETRY_MINUTES`) instead of burning all three tries in 90 minutes. A parlay the account can't afford is skipped with a clear reason.
- **Order failures show the order itself** in `/sports` (side, contracts, price, then Kalshi's message), and the bot stops trying more orders in a cycle after three rejections.

### Show Kalshi's real order errors; show cash in `/sports`
- Removed the fallback to Kalshi's old order endpoint. It answers `410 deprecated_v1_order_endpoint`, which hid the real V2 error behind a misleading one. V2 errors now surface as they are. `cancel_order` uses the V2 path too.
- `/sports` shows `cash $X`, what the bot believes Kalshi has available, so it can be compared with the Kalshi app when orders fail with `insufficient_balance`.

### `/sports` shows the running build
- `/sports` now prints `Build: <commit> — <message>` from Railway's deploy variables, so you can tell whether the newest code is live.

### Orders use Kalshi's current API (fixes live bets not being placed)
- **Live orders now go to Kalshi's V2 endpoint** (`POST /portfolio/events/orders`, string count and dollar price, YES-side `bid`/`ask`). Kalshi's current API spec no longer defines the legacy `POST /portfolio/orders` the bot used, so live bets could not be placed. Buying NO at p is an `ask` at 1 - p; selling is the mirror image. The legacy endpoint is only tried if V2 answers 404/405/410 (nothing was placed). All strategies that place orders benefit.
- **`/sports` now shows why an order failed** (Kalshi's HTTP status and message) and examples for unfilled parlays, instead of only counting skips.

### Automatic parlays, and `/sports` shows the mode
- **Automatic parlays (live mode, on by default):** unless `KALSHI_SPORTS_AUTO_PARLAYS=false`, each cycle the bot builds same-game stacks of player-stat legs for upcoming football games and buys them through Kalshi's market makers at or under a price ceiling, then sends a Telegram alert. Defaults: 4 legs, 2% of bankroll per parlay, 3 a day, 10% of bankroll open at once, ceiling = Kalshi's own fair price + 10%. Never runs in paper mode. Reuses the verified hand-parlay path (ceiling enforced before accepting, position read-back, an uncertain outcome stops the cycle).
- **`/sports` now says whether orders really go to Kalshi** (including `PAPER_MODE=false` with missing credentials, which runs on paper) and why nothing was bought (skip reasons).

### Paste a list of legs, no AI needed
- A typed or pasted list of legs (`Lawrence 150+ passing yards`, `Name anytime TD`, `the Eagles +16.5 spread`, ...) is read by a local parser and handled exactly like a screenshot: exact Kalshi matching, preview or one-shot placement with `bet $10`, and nothing placed unless every leg matched. Works with no Anthropic key or credit.
- **Matching fixes** (also benefit screenshots): a surname shared by two players is reported instead of guessed (the game most legs point to settles it when the slip names none), and a spread for a team is only matched in that team's own game.

### Clearer Anthropic errors in Telegram
- When the Anthropic key is out of credit, invalid, rate limited or the API is overloaded, picture and plain-text messages now say so (and what to do) instead of "I couldn't work out what you meant" or a raw API error dump. The `/` commands are unaffected.

### Open-source readiness
- **Security standards** (matching the practices in the maintainer's other public repository): `SECURITY.md` (private vulnerability reporting, threat model, controls, repository settings, incident response), `CONTRIBUTING.md`, `CODEOWNERS`, PR and issue templates, Dependabot for pip and GitHub Actions, and read-only workflow tokens with every action pinned to a commit SHA.
- **New CI:** `security.yml` (`pip-audit`, Gitleaks over full history, license check, weekly schedule), `codeql.yml`, `sbom.yml` (CycloneDX), and `scripts/security_checks.py` with tests (blocks `eval`/`exec`, `shell=True`, pickle, weak crypto, disabled TLS verification, Telegram handlers without the chat check, unpinned actions, committed tunnel URLs).
- **Fixes found by the new checks:** the explicit `/confirm` command now checks the chat ID itself (defence in depth); a non-security MD5 cache key became SHA-256; optional requirements got version bounds and the TradingAgents dependency is pinned to a commit.
- **License:** dual-licensed, matching the maintainer's other public repository: AGPL-3.0 (full text in `LICENSES/AGPL-3.0-only.txt`) or a commercial license from Conteh Consulting LLC (`LICENSE`, `NOTICE`), with a Contributor License Agreement (`CLA.md`) enforced by the CLA bot (`cla.yml`, signatures in `signatures/`).
- **README rewritten** for people: what it does, quick start, Railway deploy, Telegram commands, safety controls, how the sports model works. The full variable list lives in `.env.example`. Added a risk disclaimer.
- **Security:** `cryptography` raised to `>=50.0.2` (fixes the known advisories on 46.x/47.x; the Kalshi RSA signing path was re-tested). Removed a personal Cloudflare tunnel URL from `railway.toml` (`OLLAMA_BASE_URL` is now blank; set your own in the Railway Variables panel).

### Telegram betting, parlays and screenshots

#### New: place your own bets from Telegram (#33)
- `/games`, `/bet`, `/confirm`, `/cancel`: list upcoming games with Kalshi prices and the model's view, preview a bet on the team you name, and place it as a live Kalshi order.
- `/props`: search Kalshi's NFL player-stat markets (yards, receptions, passing TDs, anytime TD, alternate spreads).
- `/parlay`: build a Kalshi combo from teams, player stats and spreads. All legs must be in the same sport; same-game legs are allowed; up to 10 legs. The bot asks Kalshi market makers for a price and buys only at or under a price ceiling (the model's fair value, Kalshi's own prices multiplied, or your `max`).
- `/stack`: auto-build a same-game player-stat parlay, like a sportsbook same-game parlay.
- **Bet-slip screenshots:** send a picture of a slip and Claude reads the legs; the bot matches each exactly to Kalshi (or reports the nearest alternatives) and previews the parlay. A caption such as `bet $10 with this parlay` places it directly when every leg matched.
- **Plain-text chat:** type what you want ("bet $5 on the browns", "how am I doing?") instead of remembering commands. Claude only picks a command; bets are still previewed and need a yes. Pause, resume and kill stay slash-only.

#### New: small "action picks" when nothing is mispriced (#32)
- When no game has an edge after Kalshi's fee, the bot places small action bets (default up to 2 per scan and 5 per day, about 3% of bankroll each, never a side the model gives under 35%, at most half underdogs). Real mispricings still take priority and are Kelly-sized. Turn off with `KALSHI_SPORTS_ACTION_BETS=false`. Expected cost is about 2¢ per 50¢ contract; this buys action, not edge.

#### Earlier in this release line
- **Sports betting on Kalshi** with ESPN, sportsbook lines, Polymarket and Kalshi's own order book as inputs; adaptive Kelly sizing that compounds with the wallet, drawdown throttle, joint exposure fitting, and a source-accuracy model that learns from settled games. Enabled by default (`KALSHI_SPORTS_ENABLED`). All sports, including college football, NFL, college and pro basketball, baseball, hockey and soccer.
- **Real Kalshi fee schedule** (per-series multiplier, per-order rounding) and fee-aware edge (#30).
- **Sports is exempt from the global kill switch** and keeps its own limits (per-bet cap, exposure budget, max open bets).
- **Coinbase is data-only** unless `COINBASE_TRADING_ENABLED=true`; all trading is on Kalshi (#28).
- **Live safety fixes** (#28): failed orders are never booked as filled, Kalshi positions and balance are synced, settled positions are pruned, the live drawdown peak uses the same equity as the risk check, service cycles are serialised.
- **Telegram digest:** routine alerts arrive as one message every 6 hours (`TELEGRAM_DIGEST_HOURS`) (#31). The sports output says "no bet" when there is no edge (#29).

#### Configuration added
`KALSHI_SPORTS_USER_BET_DEFAULT_PCT`, `KALSHI_SPORTS_USER_BET_MAX_PCT`, `KALSHI_SPORTS_USER_PARLAY_DEFAULT_PCT`, `KALSHI_SPORTS_USER_PARLAY_MAX_PCT`, `KALSHI_SPORTS_USER_PARLAY_MAX_LEGS`, `KALSHI_SPORTS_PARLAY_QUOTE_WAIT_SECONDS`, `KALSHI_SPORTS_SLIP_MODEL`, `TELEGRAM_NATURAL_LANGUAGE`, `TELEGRAM_NL_MODEL`, and the `KALSHI_SPORTS_ACTION_*` settings. See `.env.example`.

#### Upgrade notes
- Plain-text chat and screenshot reading need `ANTHROPIC_API_KEY` on the host (the same variable the AI analysis uses). Without it the bot says so and the slash commands keep working.
- Parlays and single bets place real orders only when `PAPER_MODE=false`; parlays refuse to run in paper mode. `PAPER_MODE=true` remains the default.
- The parlay request-for-quote flow follows Kalshi's published API and has not yet been exercised against a live account. Assumption to check with the first small parlay: accepting the maker's NO bid leaves you long YES at `1 − no_bid`. The bot reads the position back from Kalshi and warns on a mismatch. Start with a small stake (about $1).
- Market makers often don't quote combos at the default ceiling; in that case nothing is bought.
