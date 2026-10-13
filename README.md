# Zulubot

A multi-strategy trading bot with a **Kalshi sports-betting engine** and **crypto trading on Coinbase**, controlled from Telegram. It runs as a long-lived async daemon (built for [Railway](https://railway.app)). It starts in **paper mode** (simulated money) so nothing is risked until you choose; with `PAPER_MODE=false` and Kalshi credentials its Kalshi strategies (sports bets, automatic parlays, predictions) place **real orders with real money**. Its crypto strategies trade on Coinbase only when you also turn that on (see [Crypto trading](#crypto-trading)). Stocks and Polymarket are always paper-only.

[![Tests](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/test.yml/badge.svg)](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/test.yml)
[![Security checks](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/security.yml/badge.svg)](https://github.com/sherifconteh-collab/Zulubot/actions/workflows/security.yml)
[![License: AGPL-3.0 or commercial](https://img.shields.io/badge/License-AGPL--3.0_or_commercial-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![Release](https://img.shields.io/badge/Release-v2026.10.0-green)](RELEASE_NOTES.md)

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
- [Crypto trading](#crypto-trading)
- [Safety and risk controls](#safety-and-risk-controls)
- [How the sports model works](#how-the-sports-model-works)
- [Project layout](#project-layout)
- [Development](#development)
- [Security](#security)
- [License](#license)

## What it does

Zulubot runs several independent strategies on a schedule, sizes every trade through one shared risk manager, and reports to you over Telegram.

| Strategy | What it trades | On by default? | With `PAPER_MODE=false` |
| --- | --- | --- | --- |
| **Kalshi sports** | Game-winner markets across NFL, college football, NBA, college basketball, WNBA, MLB, NHL and major soccer leagues, plus automatic same-game parlays. Fuses ESPN, sportsbook lines, Polymarket and Kalshi's own book. | Yes | Real Kalshi orders |
| **Kalshi predictions** | Crypto, economics, finance, politics, weather and other Kalshi markets, on price thresholds spread across categories | No (`PREDICTION_ENABLED`) | Real Kalshi orders |
| **Latency arbitrage** | Short-dated Kalshi crypto contracts when the Coinbase price feed moves before Kalshi does | No (`LATENCY_ARB_ENABLED`) | Real Kalshi orders |
| **Kalshi perpetuals** | Kalshi crypto perpetuals (BTC, ETH, SOL, XRP) on momentum, leverage capped | No (`KALSHI_PERP_ENABLED`) | Real Kalshi orders |
| **BTC 5-minute spread** | Kalshi and Polymarket implied BTC price vs Coinbase spot | Yes (`BTC5MIN_ENABLED`) | Kalshi leg is real; the Polymarket leg is skipped; spot leg needs Coinbase trading |
| **Polymarket** | Crypto and finance markets | No (`POLYMARKET_ENABLED`) | Never live: live execution is not implemented, so it stays simulated |
| **Stocks** | Value screen (P/E, P/B, debt, ROE, free cash flow) gated by VIX | Yes | Simulated only: no stock broker is connected, so it never sends an order |
| **Crypto** | BTC and ETH on Fear & Greed with trend and funding filters | Yes | Runs only if `COINBASE_TRADING_ENABLED=true`, then real Coinbase orders |
| **Altcoin rotation** | Rotates into coins that lead the market | Yes (`ROTATION_ENABLED`) | Runs only if `COINBASE_TRADING_ENABLED=true`, then real Coinbase orders |
| **Markov regime** | Bull / bear / sideways regime model on daily prices; opens and closes long positions on the signal | Yes (`MARKOV_ENABLED`) | Simulated only: it trades the paper book, never a venue |
| **Options** | Deep in-the-money calls and puts on extreme fear or greed | No (`OPTIONS_ENABLED`) | Real Robinhood orders if `ROBINHOOD_*` credentials are set |

How the crypto rows work, and how to turn them on, is in [Crypto trading](#crypto-trading).

In paper mode (the default) every row is simulated. The "on by default" column is the code default; Railway's Variables panel is what the running bot actually uses, so check it (the `railway.toml` values are only defaults). Send `/status` or `/help` to see what is running.

Around the strategies:

- **Risk manager:** position caps, daily halt, drawdown kill switch, trailing stops, cooldown after losing streaks, volatility targeting.
- **AI analysis (optional):** Claude or a local Ollama model can confirm or veto trades (`boost`, `require` or `log_only` mode) under a hard monthly budget with a circuit breaker.
- **Backtesting:** Freqtrade integration for backtests and hyperopt.
- **Telegram:** two-way control, a 6-hourly digest, and the betting tools below.

Live trading happens on **Kalshi** (sports, predictions, latency arbitrage, perpetuals, BTC 5-minute). Coinbase is a data source only, unless you set `COINBASE_TRADING_ENABLED=true`; options go through Robinhood only if you add credentials; stocks and Polymarket never send real orders.

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
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env (it is git-ignored)
python -m trading_bot doctor        # which settings are present and what each unlocks
python -m trading_bot status        # print portfolio and risk status
python -m trading_bot kalshi_sports # run the sports strategy once (paper)
python -m trading_bot run-daemon    # start the scheduler
```

`doctor` prints which keys are set and what each one unlocks (it never prints a secret value). Run it whenever something does not seem to work, for example if `PAPER_MODE=false` but orders stay on paper.

You do not need any API key to try it: with an empty `.env` the bot runs in paper mode on public data (`status` prints the portfolio; some price lookups may log warnings if a data source is unreachable). Add keys only for the features you want (Telegram, Kalshi, Anthropic: see [Configuration](#configuration)). The optional integrations have their own requirements files (`requirements-freqtrade.txt`, `requirements-polymarket.txt`, `requirements-tradingagents.txt`); install one only if you use that feature.

`PAPER_MODE=true` is the default: nothing is sent to a broker. Other one-shot commands: `stocks`, `crypto`, `rotation`, `predictions`, `polymarket`, `latency_arb`, `btc5min`, `markov` and `all`. `run-daemon --once` runs a single scheduler cycle and exits.

### Connecting Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token into `TELEGRAM_BOT_TOKEN`.
2. Send your new bot any message, then open `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser (keep that URL private, it contains your token) and copy the number at `"chat":{"id":...}`. Set it as `TELEGRAM_CHAT_ID`. **The bot only answers this one chat.**
3. For plain-text chat and screenshot reading, also set `ANTHROPIC_API_KEY`. Pasted leg lists and all slash commands work without it.
4. Start with `python -m trading_bot run-daemon` and send `/help` to the bot.

### Going live on Kalshi

Create an API key in Kalshi, set `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY`, then set `PAPER_MODE=false`. Start with a very small balance. Sports bets, parlays and the other Kalshi strategies will then place real orders. Send `/sports` to confirm: it prints `Mode: LIVE` (or `PAPER`, with the reason) and the running build.

**Funds live on exchange shards.** Kalshi holds your balance per exchange shard and routes each order to the shard its market lives on, so your total can be more than what is spendable on a given market. `/sports` prints `Kalshi balance by shard: #0 $X, #1 $Y, ...`. An order on a market whose shard has no funds is refused (`insufficient_shard_balance`, `Exchange user not found` or `insufficient_balance`); the bot then skips that market series for 6 hours and keeps trading the markets that work. It never moves money between shards for you: to trade the other markets, move funds in Kalshi (see Kalshi's transfer documentation).

## Deploy on Railway

The repo ships a `Procfile` and `railway.toml` (nixpacks build, `python -m trading_bot run-daemon`).

1. Create a Railway project from your fork of this repo.
2. Optional but recommended: attach a **Volume** mounted at `/data` and set `BOT_STORAGE_DIR=/data` so the bot's own notes survive deploys. Without it, today's counters (action picks, parlays tried) and the learned model weights reset on every redeploy. **Kalshi is the source of truth**: your balance, what you hold and how much is at risk are read from your account every cycle, so a redeploy never makes the bot forget a position or buy a game twice.
3. Add your secrets in the **Variables** panel (see below). They are intentionally not in `railway.toml`.
4. Deploy. The bot starts in paper mode.

`railway.toml` carries non-secret defaults only. If you update the repo and a new variable was added, add it to an existing service by hand in the Variables panel.

## Configuration

Everything is configured through **environment variables**. [`.env.example`](.env.example) lists every setting with a short comment; copy it to `.env` locally, or set the same names in Railway.

### Which keys do you need?

Almost none. Start with nothing and add a key only when you want the feature it unlocks.

| You want | Set |
| --- | --- |
| Try it in paper mode, nothing sent anywhere | Nothing |
| Control it and get alerts on your phone | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Real Kalshi bets and parlays | `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY`, then `PAPER_MODE=false` |
| Plain-English chat and bet-slip screenshots | `ANTHROPIC_API_KEY` (slash commands and pasted leg lists work without it) |
| Crypto trading on Coinbase (rare) | `COINBASE_TRADING_ENABLED=true`, `COINBASE_API_KEY`, `COINBASE_API_SECRET`. Prices need no key. |
| Options on Robinhood (rare) | `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`, `ROBINHOOD_MFA_SECRET` |
| Polymarket | `POLYMARKET_*`, optional; the strategy is simulated only |

Set these yourself and never commit them. `FMP_API_KEY` from older versions is no longer used (stock data comes from yfinance); you can delete it. A value copied unchanged from `.env.example` (`your_..._here`) counts as not set.

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

Only the chat in `TELEGRAM_CHAT_ID` is answered. You can also type plain English instead of most of these. `/start` is an alias for `/help`.

**Monitor**

| Command | What it does |
| --- | --- |
| `/status` | Portfolio, equity, cash and open positions |
| `/positions` | Open positions in detail |
| `/risk` | Risk metrics: drawdown, daily halt, kill switch and cooldown state |
| `/history` | Recent trades |
| `/audit [N]` | Tail of the trade audit log |
| `/decisions [N]` | Recent strategy decisions, including why something was skipped |
| `/latency` | Timings of each module |
| `/leads` | Watchlist: skipped buys and top targets per strategy |

**Run a strategy now** (the same code the scheduler runs; it trades for real if `PAPER_MODE=false`)

| Command | What it does |
| --- | --- |
| `/sports` | Run the Kalshi sports strategy and show the mode (LIVE or PAPER), build, balance by shard, cash, what was bought and why nothing was |
| `/predict` | Kalshi prediction markets (crypto, economics, politics, weather) |
| `/perps` | Kalshi crypto perpetuals |
| `/crypto` | BTC and ETH on Fear & Greed |
| `/rotation` | Altcoin rotation |
| `/stocks` | Value screen on stocks (simulated) |
| `/polymarket` | Polymarket strategy (simulated) |
| `/btc5min` | BTC 5-minute spread strategy |
| `/markov` | Markov regime strategy (simulated) |
| `/options` | Options strategy |
| `/backtest [range]` | Freqtrade backtest, e.g. `/backtest 20250101-20250601` |

**Your own sports bets** (details in the next section)

| Command | What it does |
| --- | --- |
| `/games [league or team]` | Upcoming games with Kalshi prices and the model's win probability |
| `/bet <team> [$amount] [now]` | Preview a single bet |
| `/props <player, team or stat>` | Football player-stat markets |
| `/parlay <leg> + <leg> [$amount] [max <cents>]` | Preview a Kalshi combo |
| `/stack <game> [legs N] [p %] [$stake] [max c]` | Auto-build a same-game player-stat parlay |
| `/confirm` / `/cancel` | Place or drop the previewed bet |

Sending a screenshot of a bet slip, or typing a list of legs, works too.

**Control and settings**

| Command | What it does |
| --- | --- |
| `/pause` / `/resume` | Stop or restart all trading. `/resume` also clears the kill switch (new entries then stay blocked for the post-kill cooldown); it may refuse if equity is still below the drawdown limit |
| `/resume force` | Clear the kill switch and reset the peak equity to current equity |
| `/kill` | Engage the kill switch: closes the positions it can and pauses the bot |
| `/config` | Show the active settings |
| `/ai` | AI analysis status, budget and circuit breaker |
| `/help` | The command list |

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

### Automatic parlays

In live mode (`PAPER_MODE=false` with Kalshi credentials) the bot also buys parlays on its own, with no message from you. It **spends real money without asking**; turn it off with `KALSHI_SPORTS_AUTO_PARLAYS=false`. Every cycle (30 minutes) it looks at upcoming football games (kickoff 15 minutes to 12 hours away), builds a same-game stack of high-probability player-stat legs like `/stack` does, asks Kalshi's market makers for a price and buys it only at or under a ceiling. You get a Telegram alert for each one.

The limits (`KALSHI_SPORTS_AUTO_PARLAY*` in `.env.example`):

| Limit | Default |
|---|---|
| Legs per parlay / target chance per leg | 4 / 85% |
| Stake per parlay | 2% of bankroll, never under $5 (market makers don't quote tiny requests) |
| Parlays per day | 3 |
| Open parlay cost at once | 25% of bankroll |
| Price ceiling | Kalshi's own fair price for the legs plus 10% |
| Quote requests per game per day | 3, at least 45 minutes apart |

It never buys the same game twice, never pays above the ceiling, and an outcome it can't confirm stops it for the cycle. **Expect it to lose a little on average:** the model has no view on player props, so these are priced at Kalshi's own odds, and you pay the fee and up to the 10% premium. The point is consistent action within a small, fixed budget, not an edge. Singles on game winners (when the model finds an edge or picks the best value) are separate and unchanged.

Send `/sports` to see the mode (**LIVE** or **PAPER**, including the case where `PAPER_MODE=false` but the Kalshi credentials are missing, which silently runs on paper), what was bought and why nothing was.

### Paste a list of legs instead (no AI, no API cost)

Type or paste the legs one per line (bullets and numbers are fine), with your stake and a bet word above them:

```
Make this bet at $10
• Brian Thomas Jr. 2+ receptions
• Lawrence 150+ passing yards
• Rodriguez 10+ rushing yards
• the Eagles +16.5 spread
```

This is read by a plain parser on the bot itself, so it works without `ANTHROPIC_API_KEY` and costs nothing. It is a fixed set of shapes, not free-form understanding:

| Reads | Examples |
|---|---|
| Player stat, "N or more" | `Lawrence 150+ passing yards`, `Lawrence 150+ pass yds`, `Tuten - Alt Rushing Yds 30+`, `Brian Thomas Jr. to record 2+ receptions` |
| Over a half-point line (same as N+) | `Lawrence over 149.5 passing yards` |
| Anytime touchdown | `Josh Allen anytime TD` |
| Spread and moneyline | `the Eagles +16.5 spread`, `Philadelphia Eagles (+16.5)`, `Ravens ML` |
| Layout | one per line (bullets or numbers are fine), or on one line separated by commas, `+` or `and`; a stake clause such as `bet $10 on` in front |

It does **not** guess. These are left alone or reported rather than interpreted: a whole-number line without a plus (`Lawrence 150 passing yards`: exactly 150 or 150 or more?), `30+ Yards` with no stat (which yards?), game totals, and unders (Kalshi's player markets are all "N or more"). At least two lines must look like legs, and ordinary chat is never mistaken for a slip. It then follows the same rules as a screenshot: exact matching, the same placing and preview behaviour, and nothing is placed unless every leg matched.

A bare surname is fine when it is unambiguous. If the legs point at one game, that game settles a shared surname; otherwise the bot lists the players that match and asks for the full name rather than guessing. It cannot read screenshots (that needs the API).

### Limits that always apply

Market open, game not started, price within 2¢ of the preview (single bets), one bet per game for game-winner legs, at least `KALSHI_SPORTS_MIN_TRADE_USD`, at most `KALSHI_SPORTS_USER_BET_MAX_PCT` of your bankroll for a single bet and `KALSHI_SPORTS_USER_PARLAY_MAX_PCT` for a parlay, and no more than the cash available. Sports is exempt from the global kill switch (`KALSHI_SPORTS_IGNORE_GLOBAL_HALT`). Hand-picked bets settle and are counted like any other sports bet.

## Crypto trading

Zulubot also trades crypto on **Coinbase**. It is **off for real money by default**: in paper mode (the default) the crypto strategies trade simulated cash, and the live Coinbase path needs three things at once: `PAPER_MODE=false`, `COINBASE_TRADING_ENABLED=true` and Coinbase API keys. With `PAPER_MODE=false` and Coinbase trading left off, the crypto and rotation strategies do not run at all (Coinbase then stays a free price source only).

### What trades

| Strategy | Idea | Runs |
| --- | --- | --- |
| **Crypto sentiment** (`/crypto`) | Buys the coins you list when the market is fearful, sells when it is greedy. Signal: the [Fear & Greed index](https://alternative.me/crypto/fear-and-greed-index/). | every 10 minutes |
| **Altcoin rotation** (`/rotation`) | Holds one coin at a time and swaps into another only when the price ratio has moved enough to pay for the fees. | every 30 minutes |

**Crypto sentiment** in detail:
- **Buy** a coin when Fear & Greed is below `CRYPTO_BUY_FEAR_THRESHOLD` (35), **sell** when it is above `CRYPTO_SELL_FEAR_THRESHOLD` (75).
- **Trend filter:** it will not buy a coin whose own price is more than 5% below its 20-day average, so it does not buy a falling knife just because the whole market is scared (`CRYPTO_TREND_FILTER_ENABLED`, `CRYPTO_TREND_SMA_DAYS`, `CRYPTO_TREND_TOLERANCE_PCT`).
- **Funding rate** (from Hyperliquid, free and US-accessible) is recorded as a signal. Set `CRYPTO_REQUIRE_NEGATIVE_FUNDING=true` to buy only when it is negative; by default it does not block a buy.
- **Exits that do not wait for greed:** take profit at +15% and stop loss at -10% from entry, checked every cycle (`CRYPTO_TAKE_PROFIT_PCT`, `CRYPTO_STOP_LOSS_PCT`), with per-coin overrides (`CRYPTO_TAKE_PROFIT_OVERRIDES=XRPUSDT:0.20`).
- **Size:** each buy is about 2% of your equity (`MAX_POSITION_FRACTION`), never more than the cash available, scaled per coin by `CRYPTO_SYMBOL_WEIGHTS`. Optional volatility targeting (`VOL_TARGETING_ENABLED`) scales the size with how wild a coin has been.
- **Coins:** `CRYPTO_SYMBOLS`, default `BTCUSDT,ETHUSDT`. Coins the bot can price and trade on Coinbase: BTC, ETH, SOL, XRP, ADA, DOT, LINK, AVAX, MATIC.
- **Optional AI check:** with `TRADING_AGENTS_ENABLED` an AI model can confirm or veto a buy (`boost`, `require` or `log_only`), under the monthly budget in [Configuration](#configuration).

**Altcoin rotation** in detail:
- It looks at the top 50 coins by volume (`ROTATION_MAX_COINS_TO_TRACK`) and uses up to 60% of your capital (`ROTATION_CAPITAL_PCT`).
- Every swap goes coin -> USD -> new coin, and costs about 0.6% a leg (`ROTATION_FEE_PCT`). It only swaps when the price ratio beats the fees by `ROTATION_SCOUT_MULTIPLIER` (3x), so most cycles do nothing.
- A "never go backwards" rule skips a swap that would leave you with less of the new coin than you held of it before, and that memory expires after about a week (`ROTATION_QTY_MEMORY_CYCLES`) so a crash cannot lock it out forever. Coins you never want: `ROTATION_EXCLUDED_COINS`.

**Related crypto strategies that live on Kalshi, not Coinbase** (all off by default except the BTC 5-minute one): the predictions strategy (it covers crypto markets), latency arbitrage (Kalshi crypto contracts that lag the Coinbase price), Kalshi perpetuals and the BTC 5-minute spread. They place real orders on Kalshi when `PAPER_MODE=false`, whatever the Coinbase setting is. The **Markov regime** model (bull, bear or sideways from daily prices) is simulated only, but its signal can hold back new prediction and Polymarket entries (`PREDICTION_MARKOV_GATE`, `POLYMARKET_MARKOV_GATE`).

### Turning on real Coinbase trading

1. Create an API key for **Coinbase Advanced Trade** (the "Cloud" key format whose name starts with `organizations/`). It needs permission to view and to trade, and it should not be able to withdraw.
2. Fund the account with **USD or USDC**. In live mode the bot treats your Coinbase USD plus USDC balance as the cash it can spend, and it reads that balance from Coinbase on every cycle.
3. Set these in Railway's Variables (or `.env`), never in a file you commit:
   - `COINBASE_API_KEY` (the key name) and `COINBASE_API_SECRET` (the EC private key, with `\n` for line breaks)
   - `COINBASE_TRADING_ENABLED=true`
   - `PAPER_MODE=false` (this one switch also makes the Kalshi strategies live, so check [Going live on Kalshi](#going-live-on-kalshi) too)
4. Start with a small balance. Orders are **market orders**, so you pay the spread and Coinbase's fee on each one.
5. Run `python -m trading_bot doctor` to confirm the keys are present, then watch `/status`, `/positions` and `/crypto`.

### Controls and honest expectations

- **Risk limits apply:** the position cap, the daily halt (`DAILY_HALT_DRAWDOWN`, 5%), the kill switch (`KILL_SWITCH_DRAWDOWN`, 15%), trailing stops and the loss cooldown all cover crypto. `/kill` (or the kill switch firing) sells your crypto positions at market and pauses the bot.
- **Commands:** `/crypto` and `/rotation` run a strategy now, `/status` and `/positions` show holdings, `/pause` stops all trading, and `/config` shows the active crypto settings.
- **It is not an edge.** Fear & Greed is a slow sentiment reading, not a forecast; buying fear can mean buying a long decline. Rotation pays fees on every swap. Crypto can fall a lot in a day, and the stop loss may fill below its level in a fast market. Treat it as an experiment, size it small, and only use money you can afford to lose.

## Safety and risk controls

- **Paper by default, real money when you switch it.** `PAPER_MODE=true` unless you change it; nothing reaches a broker until you set `PAPER_MODE=false` and Kalshi credentials. In live mode the bot spends real money without asking, within the caps below. Parlays refuse to run in paper mode.
- **Coinbase is data-only** unless `COINBASE_TRADING_ENABLED=true` (with it off and `PAPER_MODE=false`, the crypto and rotation strategies do not run at all). Live orders go to Kalshi, plus Coinbase and Robinhood only when you enable them. Stocks and Polymarket are simulated only.
- **No doubling up.** Before it buys, the bot reads your Kalshi positions and skips any market, or any other market in the same event, that your account already holds, even if the bot's own record was lost. If it cannot read your positions in live mode, it places no new bets that cycle.
- **Orders are checked, not assumed.** A failed or unfilled order is never booked. Live Kalshi balance and positions are synced, settled positions are pruned, and the drawdown peak uses the same equity as the risk check.
- **One lock for state.** Service cycles are serialised so two runs can't place the same bet.
- **Risk manager** (non-sports strategies): position caps, a daily halt at `DAILY_HALT_DRAWDOWN`, a kill switch at `KILL_SWITCH_DRAWDOWN` that liquidates, trailing stops, cooldown after consecutive losses, optional volatility targeting.
- **Sports has its own limits**: per-bet cap, exposure budget, max open bets, and a stake throttle that shrinks bets as the day's drawdown grows.
- **Telegram is single-user.** Messages from any other chat are ignored, and plain text can never pause, resume or kill the bot.
- **Secrets stay out of git.** `.env` is ignored; CI scans every push and the full history (Gitleaks), and an optional pre-commit hook (`scripts/install-hooks.sh`) scans before you commit.
- **Automated security checks.** Dependency audit (`pip-audit`), CodeQL, an SBOM, weekly Dependabot, GitHub Actions pinned to commit SHAs, and `scripts/security_checks.py`, which fails the build on `eval`/`shell=True`/pickle, weak crypto, disabled TLS verification and any Telegram handler that skips the chat-ID check. See [SECURITY.md](SECURITY.md) for the policy, how to report a vulnerability privately, and the recommended repository settings.

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
  strategies/     sports, predictions, latency arb, perpetuals, BTC 5-min, Polymarket, stocks, crypto, rotation, Markov, options
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
scripts/          security checks, release tooling, git hooks
tests/            unit tests (all network calls mocked)
```

## Development

```bash
python -m unittest discover -s tests -p "test_*.py"
```

CI runs the tests on Python 3.12 with a coverage floor of 50%, plus a secret scan. External APIs are mocked in tests. Configuration is always through environment variables, never hard-coded. Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Please keep `PAPER_MODE=true` as the default and never commit credentials. Run `python scripts/security_checks.py` before opening a pull request.

### Releases

Releases are automatic. Every pull request adds its entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md). Each Monday 09:00 UTC the [release workflow](.github/workflows/release.yml) turns those entries into a calendar-versioned release (`YYYY.M.N`, e.g. `2026.10.0`), prepends it to [RELEASE_NOTES.md](RELEASE_NOTES.md), updates the Release badge above, tags it and publishes a GitHub Release. A week with no entries makes no release. Preview with `python scripts/release.py cut --dry-run`, or run the workflow by hand from the Actions tab.

## Security

See [SECURITY.md](SECURITY.md) for what the project protects, the CI controls, the recommended GitHub settings and how to report a vulnerability privately. Please don't open public issues for security problems.

## License

Dual-licensed. See [LICENSE](LICENSE) for the full terms.

- **Open source:** [GNU AGPL v3](LICENSES/AGPL-3.0-only.txt). You may use, modify and distribute Zulubot under the AGPL. If you run a modified version as a network service, you must make your modifications available under the same license.
- **Commercial license** (Conteh Consulting LLC) for organizations that can't accept the AGPL's copyleft terms. Contact contehconsulting@gmail.com.

By contributing you agree to the [Contributor License Agreement](CLA.md). Signing is automated: the CLA bot (see [`.github/workflows/cla.yml`](.github/workflows/cla.yml)) asks you on your first pull request. Copyright (c) 2026 Conteh Consulting LLC. See [NOTICE](NOTICE).
