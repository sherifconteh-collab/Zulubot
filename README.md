# Zulubot

A multi-strategy paper-trading bot with a **Kalshi sports-betting engine**, controlled from Telegram. It runs as a long-lived async daemon (built for [Railway](https://railway.app)) and trades simulated money by default.

[![Tests](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/test.yml/badge.svg)](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

> ### ⚠️ Read this first
> **This is experimental software, not financial advice.** Trading and betting can lose money, including all of it. When `PAPER_MODE=false` the bot places **real orders with real money** on Kalshi. It is provided as is, with no warranty (see [LICENSE](LICENSE)). The sports model finds small edges at best; prediction markets are efficient and fees are real. Run it in paper mode first, read the code, start with small stakes, and only risk money you can afford to lose. You are responsible for following the laws and the terms of service that apply to you. Never commit your API keys.

## Contents

- [What it does](#what-it-does)
- [Sports betting from Telegram](#sports-betting-from-telegram)
- [Quick start](#quick-start)
- [Deploy on Railway](#deploy-on-railway)
- [Configuration](#configuration)
- [Telegram commands](#telegram-commands)
- [Betting from Telegram in detail](#betting-from-telegram-in-detail)
- [Safety and risk controls](#safety-and-risk-controls)
- [How the sports model works](#how-the-sports-model-works)
- [Project layout](#project-layout)
- [Development](#development)
- [License](#license)

## What it does

Zulubot runs several independent strategies on a schedule, sizes every trade through one shared risk manager, and reports to you over Telegram.

| Strategy | What it trades | Notes |
| --- | --- | --- |
| **Kalshi sports** | Game-winner markets across NFL, college football, NBA, college basketball, WNBA, MLB, NHL and major soccer leagues | Fuses ESPN, sportsbook lines, Polymarket and Kalshi's own book. On by default. |
| **Kalshi predictions** | Crypto, economics, politics, weather and other Kalshi markets | Threshold-based, spread across categories |
| **Latency arbitrage** | Short-dated Kalshi crypto contracts | Coinbase price feed moves first, Kalshi lags |
| **BTC 5-minute spread** | Kalshi and Polymarket implied price vs Coinbase spot | |
| **Polymarket** | Crypto and finance markets | Paper only |
| **Stocks** | Value screen (P/E, P/B, debt, ROE, free cash flow) gated by VIX | Runs every few days, paper only |
| **Crypto** | BTC and ETH on Fear & Greed with trend and funding filters | Needs `COINBASE_TRADING_ENABLED=true` to trade |
| **Altcoin rotation** | Rotates into coins that lead the market | Needs `COINBASE_TRADING_ENABLED=true` to trade |
| **Options** | Deep in-the-money calls and puts on extreme fear or greed | Opt-in |

Around the strategies:

- **Risk manager:** position caps, daily halt, drawdown kill switch, trailing stops, cooldown after losing streaks, volatility targeting.
- **AI analysis (optional):** Claude or a local Ollama model can confirm or veto trades (`boost`, `require` or `log_only` mode) under a hard monthly budget with a circuit breaker.
- **Backtesting:** Freqtrade integration for backtests and hyperopt.
- **Telegram:** two-way control, a 6-hourly digest, and the betting tools below.

All live trading happens on **Kalshi**. Coinbase is used as a data source only, unless you explicitly enable it.

## Sports betting from Telegram

Message the bot in plain English or send a screenshot, and it does the work on Kalshi:

- *"what games are on today"*, *"put five bucks on the browns"*
- *"parlay hurts 200 passing and barkley 80 rushing for 3 bucks"*
- `/stack eagles jaguars legs 9` builds a same-game player-stat parlay
- **Send a screenshot of a sportsbook parlay** with the caption `bet $10 with this parlay`. The bot reads the legs, matches each one exactly to a Kalshi market, asks Kalshi's market makers for a price and buys it if the price is within your ceiling.

Single bets are previewed and need a "yes". Parlays are only bought at or under a price ceiling you can see. Details in [Betting from Telegram in detail](#betting-from-telegram-in-detail).

## Quick start

You need Python 3.12.

```bash
git clone https://github.com/sherifconteh-collab/Zulubot.git
cd Zulubot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env (it is git-ignored)
python -m trading_bot status        # print portfolio and risk status
python -m trading_bot kalshi_sports # run the sports strategy once (paper)
python -m trading_bot run-daemon    # start the scheduler
```

`PAPER_MODE=true` is the default: nothing is sent to a broker. Other one-shot commands: `stocks`, `crypto`, `rotation`, `predictions`, `polymarket`, `latency_arb`, `btc5min`, `markov` and `all`. `run-daemon --once` runs a single scheduler cycle and exits.

### Connecting Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token into `TELEGRAM_BOT_TOKEN`.
2. Send your bot a message, find your chat ID, and set `TELEGRAM_CHAT_ID`. **The bot only answers this one chat.**
3. For plain-text chat and screenshot reading, also set `ANTHROPIC_API_KEY`.

### Going live on Kalshi

Create an API key in Kalshi, set `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY`, then set `PAPER_MODE=false`. Start with a very small balance. Sports bets, parlays and the other Kalshi strategies will then place real orders.

## Deploy on Railway

The repo ships a `Procfile` and `railway.toml` (nixpacks build, `python -m trading_bot run-daemon`).

1. Create a Railway project from your fork of this repo.
2. Attach a **Volume** mounted at `/data` so state and logs survive deploys (`Service → Volumes → Add`).
3. Add your secrets in the **Variables** panel (see below). They are intentionally not in `railway.toml`.
4. Deploy. The bot starts in paper mode.

`railway.toml` carries non-secret defaults only. If you update the repo and a new variable was added, add it to an existing service by hand in the Variables panel.

## Configuration

Everything is configured through **environment variables**. [`.env.example`](.env.example) lists every setting with a short comment; copy it to `.env` locally, or set the same names in Railway.

### Secrets (set these yourself, never commit them)

| Variable | Used for |
| --- | --- |
| `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY` | Kalshi trading and balance (RSA key, `\n` for newlines) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram control and alerts |
| `ANTHROPIC_API_KEY` | Plain-text chat, slip screenshots, optional AI analysis |
| `FMP_API_KEY` | Stock fundamentals |
| `COINBASE_API_KEY`, `COINBASE_API_SECRET` | Coinbase prices (and trading, if enabled) |
| `POLYMARKET_*` | Polymarket trading (optional) |

### The settings you will actually touch

| Variable | Default | What it does |
| --- | --- | --- |
| `PAPER_MODE` | `true` | `false` places real orders. Leave `true` until you trust it. |
| `INITIAL_CASH` | `100000` | Starting balance in paper mode |
| `KALSHI_SPORTS_ENABLED` | `true` | Turn the sports strategy on or off |
| `KALSHI_SPORTS_LEAGUES` | all major leagues | Which leagues to scan |
| `KALSHI_SPORTS_KELLY_FRACTION` | `0.25` | Fraction of Kelly to stake |
| `KALSHI_SPORTS_MAX_POSITION_PCT` | `0.10` | Largest single automatic bet, as a share of bankroll |
| `KALSHI_SPORTS_CAPITAL_PCT` | `0.50` | Total open sports exposure cap |
| `KALSHI_SPORTS_MIN_TRADE_USD` | `1.0` | Smallest bet worth placing |
| `KALSHI_SPORTS_ACTION_BETS` | `true` | Small "action" bets when nothing is mispriced |
| `KALSHI_SPORTS_USER_BET_MAX_PCT` | `0.25` | Cap for a single bet you place from Telegram |
| `KALSHI_SPORTS_USER_PARLAY_MAX_PCT` | `0.25` | Cap for a parlay you place from Telegram |
| `KALSHI_SPORTS_USER_PARLAY_MAX_LEGS` | `10` | Most legs in a parlay |
| `TELEGRAM_DIGEST_HOURS` | `6` | Batch routine alerts into one message (`0` = immediate) |
| `TELEGRAM_NATURAL_LANGUAGE` | `true` | Understand plain-text messages |
| `COINBASE_TRADING_ENABLED` | `false` | Coinbase stays data-only unless this is `true` |
| `DAILY_HALT_DRAWDOWN` / `KILL_SWITCH_DRAWDOWN` | `0.05` / `0.15` | Risk halts for the non-sports strategies |
| `BOT_STORAGE_DIR` | blank | Set to `/data` on Railway |

Persistent state lives in `BOT_STORAGE_DIR` (`state.json`, the trade audit log, sports calibration and daily session files).

## Telegram commands

You can also type plain English instead of any of these.

| Group | Commands |
| --- | --- |
| **Monitor** | `/status` `/positions` `/risk` `/history` `/audit [N]` `/decisions [N]` `/latency` `/leads` |
| **Run a strategy now** | `/sports` `/crypto` `/stocks` `/rotation` `/predict` `/perps` `/polymarket` `/btc5min` `/markov` `/options` `/backtest [range]` |
| **Your own sports bets** | `/games` `/bet` `/props` `/parlay` `/stack` `/confirm` `/cancel` (screenshots work too) |
| **Control** | `/pause` `/resume` `/kill` and `/resume force` |
| **Settings** | `/config` `/ai` `/help` |

Routine alerts (cycle summaries, risk banners, leads, heartbeats) arrive as **one digest every 6 hours**. Replies to your commands and a kill-switch trigger are always immediate. Pause, resume and kill are slash-only: plain text can never trigger them.

## Betting from Telegram in detail

Everything here places **real Kalshi orders** when `PAPER_MODE=false`, and only the configured `TELEGRAM_CHAT_ID` can use it.

### Just type it

Type what you want ("what games are on today", "bet $5 on the browns", "how am I doing?") and Claude picks the matching command. This needs `ANTHROPIC_API_KEY`, uses `TELEGRAM_NL_MODEL` (default Haiku, one small call per message) and can be switched off with `TELEGRAM_NATURAL_LANGUAGE=false`. The model only *chooses* a command from a fixed list, so the same limits apply and bets are still **previewed** first. Reply "yes", "do it" or "confirm" to place the waiting bet, or "no" or "cancel" to drop it. Those words are matched by simple rules, only while a preview is waiting (10 minutes).

### Commands

| Command | What it does |
| --- | --- |
| `/games [league or team]` | Lists numbered upcoming games with each side's Kalshi price and the model's win probability. |
| `/bet <team> [$amount]` | Previews a single bet, e.g. `/bet browns 5` or `/bet 3 browns`. Ambiguous names (Browns, Guardians) ask you to be specific. Add `now` to skip the confirm step. |
| `/props <player, team or stat>` | Football player-stat markets (NFL passing, rushing and receiving yards, receptions, passing TDs, anytime TD, alternate spreads), grouped per player: `passing yards: p1 125+ 85¢ · p2 150+ 73¢`. |
| `/parlay <leg> + <leg> [$amount] [max <cents>]` | Previews a Kalshi combo (see below). |
| `/stack <game> [legs N] [p %] [$stake] [max c]` | Auto-builds a same-game player-stat parlay. |
| `/confirm` / `/cancel` | Place or drop the previewed bet. Previews expire after 10 minutes. |

A single-bet preview shows contracts, price, Kalshi's fee, what you win or lose, and the model's honest view. If it thinks the price is too high it says so, and the bet is still yours. Orders are immediate-or-cancel and only contracts that actually fill are booked.

### Parlays (Kalshi combos)

Example: `/parlay hurts 200 passing + barkley 80 rushing + browns 3 max 12`.

- Up to `KALSHI_SPORTS_USER_PARLAY_MAX_LEGS` legs (default 10), **all in the same sport** (football with football, never football with baseball).
- A leg is a team to win (from `/games`, one per game), or a player stat or spread: by number from `/props` (`p3`) or in plain words (a number matches the exact `N+` threshold; ambiguous names list the choices).
- Same-game legs are allowed. The same player can't appear twice, and a game gets one winner or spread leg.

Combos have no order book. After you confirm, the bot creates the combo market, sends a **request for quote** to Kalshi's market makers, waits up to `KALSHI_SPORTS_PARLAY_QUOTE_WAIT_SECONDS` and accepts the best quote **only if its all-in price (fee included) is at or under a ceiling**:

- the model's fair probability for all legs when it has a view on every leg, otherwise Kalshi's own mid prices multiplied (the model has no view on player stats);
- or your `max <cents>`; for a slip screenshot with every leg matched, the slip's own price if that is higher.

Contracts are sized so the worst-case cost stays within your stake (default `KALSHI_SPORTS_USER_PARLAY_DEFAULT_PCT` = 2% of bankroll, capped at 25%). If no maker quotes at or under the ceiling, **nothing is bought** and you are told the best price offered. Makers price combos above the product of the legs, and same-game legs are correlated, so expect "no quote" at the default ceiling. Raising `max` means paying their margin, an expected cost rather than an edge. After execution the bot reads the position back from Kalshi and warns if it holds the opposite side. Parlays need live trading. The bot never builds or buys a parlay on its own.

### Same-game stacks (`/stack`)

`/stack eagles jaguars legs 9 p 85 2` builds a same-game parlay like a sportsbook SGP: nine near-certain legs such as low receptions, rushing and receiving yards, a passing floor and an alternate spread. For each player it picks the one liquid Kalshi market whose mid price is closest to `p` (percent chance per leg, default 85), skips one-sided or wide markets, ranks players by closeness then volume, and adds one alternate-spread leg when it fits. Spreads are offered on both sides ("PHI don't win by over 3.5" is Eagles +3.5). Nine legs near 85% multiply to roughly +330. The trailing number is your stake. You then get the usual parlay preview.

### Copy a bet slip from a screenshot

Send a screenshot (photo or image file) of a bet slip, such as a sportsbook same-game parlay. Claude reads the legs (`ANTHROPIC_API_KEY`, model `KALSHI_SPORTS_SLIP_MODEL`, one image call per picture) into a fixed schema; text inside the picture is only data and can't trigger anything. Each leg is mapped onto Kalshi **exactly**: same player, stat and number, or same team and spread line (Team +16.5 means the opponent does *not* win by over 16.5). The bot replies with what matched and what didn't, offering Kalshi's nearest alternatives for near misses ("Kalshi doesn't list 15+ rushing yards for them, it has 20+, 25+..."). Nothing is silently swapped for a different bet. Unders, totals and legs Kalshi doesn't offer are reported as not matched.

- **Caption `bet $10 with this parlay`** (any bet word plus your stake) is the instruction: when every leg matched, the bot places it straight away, no `/confirm`. You can also send the picture first and then text `bet $10 on this`.
- **Caption `10`** (or `what would $10 pay`) shows the preview and waits for `/confirm`.
- If some legs aren't on Kalshi the bot **never** auto-places the shorter parlay: it shows the preview and asks you to reply yes.

### Limits that always apply

Market open, game not started, price within 2¢ of the preview (single bets), one bet per game for game-winner legs, at least `KALSHI_SPORTS_MIN_TRADE_USD`, at most `KALSHI_SPORTS_USER_BET_MAX_PCT` of your bankroll for a single bet and `KALSHI_SPORTS_USER_PARLAY_MAX_PCT` for a parlay, and no more than the cash available. Sports is exempt from the global kill switch (`KALSHI_SPORTS_IGNORE_GLOBAL_HALT`). Hand-picked bets settle and are counted like any other sports bet.

## Safety and risk controls

- **Paper by default.** `PAPER_MODE=true` unless you change it. Parlays refuse to run in paper mode.
- **Coinbase is data-only** unless `COINBASE_TRADING_ENABLED=true`. All live trading is on Kalshi.
- **Orders are checked, not assumed.** A failed or unfilled order is never booked. Live Kalshi balance and positions are synced, settled positions are pruned, and the drawdown peak uses the same equity as the risk check.
- **One lock for state.** Service cycles are serialised so two runs can't place the same bet.
- **Risk manager** (non-sports strategies): position caps, a daily halt at `DAILY_HALT_DRAWDOWN`, a kill switch at `KILL_SWITCH_DRAWDOWN` that liquidates, trailing stops, cooldown after consecutive losses, optional volatility targeting.
- **Sports has its own limits**: per-bet cap, exposure budget, max open bets, and a stake throttle that shrinks bets as the day's drawdown grows.
- **Telegram is single-user.** Messages from any other chat are ignored, and plain text can never pause, resume or kill the bot.
- **Secrets stay out of git.** `.env` is ignored, a CI job and an optional pre-commit hook (`scripts/install-hooks.sh`) scan for keys.

## How the sports model works

1. **Collect.** ESPN supplies the schedule, sportsbook odds (with line movement) and its own win predictor. Polymarket supplies a second real-money price. Kalshi supplies its order book and the fee schedule for each series.
2. **Fuse.** Each source is treated as a noisy estimate of the true win probability. They are combined in log-odds space, weighted by how accurate each has actually been on settled games (learned, not hand-tuned), with outlier rejection and a conflict test that skips a game when sources disagree more than chance allows.
3. **Value both sides net of fees.** Kalshi's real fee, `ceil(0.07 × C × P × (1 − P))` with a per-series multiplier, is subtracted before judging an edge.
4. **Size with Kelly, shrunk by uncertainty.** Stakes scale with edge relative to uncertainty, so there is no arbitrary minimum edge. Bets grow and shrink with your wallet, are capped per bet and in total, and are trimmed together when opportunities pile up.
5. **Action picks.** When nothing is mispriced, small capped "action" bets back the best-value side of the best games. They have no edge after fees and cost a little in expectation; turn them off with `KALSHI_SPORTS_ACTION_BETS=false`.
6. **Settle.** Positions are held to the final whistle and closed at Kalshi's settlement value.

Be realistic: sportsbooks, Polymarket and Kalshi usually agree within a point or two, so real mispricings are rare.

## Project layout

```
trading_bot/
  strategies/     sports, predictions, latency arb, BTC 5-min, Polymarket, stocks, crypto, rotation, options
  integrations/   AI analysis, analyst personas, Freqtrade, slip reader, plain-text router
  service.py      core orchestrator
  scheduler.py    daemon loop
  risk.py         limits, drawdown halts, trailing stops, cooldowns
  broker.py       paper broker
  coinbase_broker.py, polymarket_broker.py
  clients*.py     Kalshi, ESPN, Polymarket, Coinbase and other API clients
  sports_model.py fee-aware probability fusion and Kelly sizing
  telegram_bot.py commands, betting flows, digest
  config.py       all settings (frozen dataclasses read from environment variables)
tests/            unit tests (all network calls mocked)
```

## Development

```bash
python -m unittest discover -s tests -p "test_*.py"
```

CI runs the tests on Python 3.12 with a coverage floor of 50%, plus a secret scan. External APIs are mocked in tests. Configuration is always through environment variables, never hard-coded. Contributions are welcome: please keep `PAPER_MODE=true` as the default and never commit credentials.

## License

[Apache License 2.0](LICENSE). Copyright 2026 sherifconteh-collab. See [NOTICE](NOTICE).
