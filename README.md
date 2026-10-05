# ZuluBot — Multi-Strategy Trading Bot

An automated trading system combining value investing, crypto sentiment, altcoin rotation, prediction markets, and latency arbitrage — all orchestrated by a multi-analyst AI agent team. All modules share the same paper portfolio, persistent state, and risk controls.

## Architecture

The diagram below shows how `__main__.py` wires together the scheduler, strategies, brokers, market-data clients, AI analyst team, and shared infrastructure.

```mermaid
flowchart TB
    MAIN["__main__.py"] --> SVC["TradingBotService"]
    MAIN --> SCHED["SchedulerService"] --> SVC

    subgraph Strategies
        direction LR
        STRAT["Strategy Hub"]
        STK["Stock Value"]
        CRY["Crypto Sentiment"]
        ROT["Altcoin Rotation"]
        PRED["Kalshi Predictions"]
        POLY["Polymarket"]
        LARB["Latency Arb"]
        BTC5["BTC 5-Min Spread"]
    end

    SVC --> STRAT
    STRAT --> STK
    STRAT --> CRY
    STRAT --> ROT
    STRAT --> PRED
    STRAT --> POLY
    STRAT --> LARB
    STRAT --> BTC5

    subgraph Brokers
        direction LR
        PB["PaperBroker"]
        CB["CoinbaseBroker"]
        PMB["PolymarketBroker"]
    end

    STK --> PB
    CRY --> CB
    ROT --> CB
    PRED --> PB
    POLY --> PMB
    LARB --> PB
    BTC5 --> CB
    BTC5 --> PMB

    subgraph Data["Market Data"]
        direction LR
        D_CB["Coinbase · Binance · AlternativeMe"]
        D_FMP["FMP · MarketIntel"]
        D_KAL["Kalshi"]
        D_POLY["Polymarket CLOB · Gamma"]
    end

    CRY --> D_CB
    STK --> D_FMP
    PRED --> D_KAL
    LARB --> D_KAL
    POLY --> D_POLY
    BTC5 --> D_CB
    BTC5 --> D_KAL
    BTC5 --> D_POLY

    subgraph AI["AI Analyst Team"]
        direction LR
        TAS["TradingAgentsService"] --> PER["Analyst Personas"]
    end

    Strategies -.-> AI

    subgraph Infra["Infrastructure"]
        direction LR
        INFRA["Infra Hub"]
        STATE["StateStore"]
        RISK["RiskManager"]
        VOL["VolatilityEngine"]
        TG["Telegram"]
        FEED["PriceFeed"]
    end

    SVC --> INFRA
    INFRA --> STATE
    INFRA --> RISK
    RISK --> VOL
    INFRA --> TG
    INFRA --> FEED
```

## Multi-Analyst AI System

Inspired by [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund), ZuluBot uses a **team-of-analysts** approach where multiple AI personas independently evaluate signals and aggregate their views via confidence-weighted voting.

The diagram below shows how market data flows through each analyst persona and is aggregated into a final trading verdict.

```mermaid
graph LR
    DATA["Market Data"]

    subgraph "Analyst Team"
        V["Value Investor"]
        G["Growth Investor"]
        C["Contrarian"]
        T["Technical Analyst"]
        S["Sentiment Analyst"]
        R["Risk Analyst"]
    end

    AGG["Signal Aggregation"]
    VERDICT["Final Verdict"]

    DATA --> V
    DATA --> G
    DATA --> C
    DATA --> T
    DATA --> S
    DATA --> R

    V --> AGG
    G --> AGG
    C --> AGG
    T --> AGG
    S --> AGG
    R --> AGG

    AGG --> VERDICT
```

Each analyst produces an independent signal:

```
AnalystSignal(persona_key, signal="bullish"|"bearish"|"neutral", confidence=0.0-1.0, reasoning="...")
```

Signals are aggregated via weighted voting:
- Each signal is scored: bullish=+1, neutral=0, bearish=−1
- Weighted by confidence
- Final score > 0.2 → **buy**, < −0.2 → **sell**, else → **hold**

### AI Confirmation Modes

| Mode | Behavior |
|------|----------|
| `boost` | AI verdict adjusts position size (1.5× if agree, 0.5× if disagree) |
| `require` | AI must agree before trade executes |
| `log_only` | AI verdict logged for analysis, not used for decisions |

### Cost-Optimized LLM Architecture

| Strategy | Model | Cache TTL | Purpose |
|----------|-------|-----------|---------|
| Stocks | Haiku → Sonnet | 6 h | Validate fundamentals screen |
| Crypto | Haiku → Sonnet | 6 h | Validate F&G + funding signal |
| Kalshi predictions | Haiku → Sonnet | 6 h | Estimate fair probability |
| Latency arb — entry | Haiku only | 10 min | Filter noise from momentum signals |
| Latency arb — exit | Haiku only | 5 min | Veto premature stop-loss / take-profit exits |
| Latency arb — regime | Haiku only | 1 h | Adjust momentum threshold |
| Altcoin rotation | Haiku only | 30 min | Check target coin for macro red flags |

When multi-analyst mode is enabled (`TRADING_AGENTS_MULTI_ANALYST=true`), stock and crypto evaluations run N×Haiku calls (~$0.001 each) instead of the standard Haiku→Sonnet pipeline (~$0.03).

---

## Strategies

### 1. Value Investing — Stocks
Screens S&P 500 constituents weekly:
- Requires P/E < 20, P/B < 3, debt-to-equity < 0.5, positive FCF
- Buys top 3 ranked names only when VIX > 25
- Sells on next weekly review if any fundamental breaches threshold

### 2. Crypto Sentiment + Funding
Trades BTC and ETH on Coinbase:
- **Buy**: Fear & Greed < 25 AND Binance funding rate < 0
- **Sell**: Fear & Greed > 75

### 3. Altcoin Rotation
Rotates capital into the best-performing altcoin vs BTC/ETH ratio:
- Tracks top 50 altcoins by volume
- Switches positions when a better ratio opportunity appears
- Uses 60% of total capital allocation

### 4. Prediction Markets — Kalshi
Trades binary prediction contracts directly via the Kalshi API:
- Scans crypto, economics, finance, politics, and weather categories
- **Buy YES** when contract price < 0.30
- **Buy NO** when contract price > 0.70
- Optional Fear & Greed confirmation for crypto-tagged markets
- Filters out markets below the configurable 24h volume threshold (`PREDICTION_MIN_VOLUME_USD`, default `5000`)
- Rejects markets when either side is missing a live bid/ask or when the spread exceeds the configurable limit (`PREDICTION_MAX_SPREAD_USD`, default `0.15`)
- Market discovery works without credentials; trading requires Kalshi API key

### 5. Latency Arbitrage — Kalshi
High-frequency strategy that exploits Kalshi contract mispricing:
- Polls Coinbase BTC-USD / ETH-USD spot prices every ~2 seconds
- Detects sharp momentum moves (≥ 0.30% in 30 seconds)
- When contract odds are stale relative to the price move, buys the correct side
- Uses fractional Kelly criterion (25% of full Kelly) for position sizing
- Manages exits via take-profit (20%) and stop-loss (15%)

### 6. Polymarket
Trades on Polymarket prediction markets:
- Scans for markets matching configured tags
- Evaluates YES/NO pricing against configurable thresholds

### 7. BTC 5-Min Spread Terminal
Monitors implied BTC prices across three venues every 5 minutes and trades significant mispricings:
- Fetches the **Coinbase spot** BTC-USD price in real time
- Computes an **implied median BTC price** from Kalshi KXBTC binary markets via linear interpolation
- Computes an **implied median BTC price** from Polymarket bitcoin prediction markets
- When any venue's implied price diverges from spot by more than `BTC5MIN_BUY_SPREAD_THRESHOLD` (default 2%):
  - **Implied > spot** (venue too bullish) → buys **NO** on that venue; optionally opens a spot BTC long
  - **Implied < spot** (venue too bearish) → buys **YES** on that venue; optionally closes any open spot BTC long
- Prediction positions are **automatically closed** when the spread normalizes back within the threshold
- Launches a live terminal dashboard (`python -m trading_bot btc-terminal`) showing all three prices, spreads, positions, and recent signals refreshing every 30 seconds

### Volatility Targeting

Position size adjusts automatically so that every trade carries the **same dollar-risk**, regardless of how volatile the underlying market is. The formula is simple:

```
leverage = target_annual_vol / realized_vol
position_size = clamp(leverage, min_allocation, max_leverage) × base_notional
```

where `base_notional = equity × max_position_fraction` (the configured per-trade budget).

- **High vol** (e.g. crypto during a spike) → leverage < 1 → smaller position
- **Low vol** (e.g. a quiet sideways market) → leverage > 1 → larger position
- Applies to crypto and stock positions sized via the shared `RiskManager`
- Kalshi prediction strategies keep their own budget-based sizing and do not use vol targeting
- Enabled via `VOL_TARGETING_ENABLED=true`; off by default (flat 2% sizing preserved)
- In EWMA mode (`VOL_USE_EWMA=true`, default), falls back to flat sizing for any symbol without 2+ days of recorded price history; in rolling-window mode (`VOL_USE_EWMA=false`), requires at least `VOL_WINDOW_DAYS+1` daily prices (21 by default) before vol-adjusted sizing activates

Vol is estimated with an **EWMA** (decay = 0.94, the RiskMetrics standard), which reacts faster to regime changes than a plain rolling standard deviation:

```
ewma_variance[t] = (1 − 0.94) × return[t]² + 0.94 × ewma_variance[t−1]
annual_vol = sqrt(ewma_variance × 252)
```

The following diagram shows how daily price snapshots flow through the volatility engine into a final position size:

```mermaid
graph TD
    PRICES["Daily Price Snapshots\n(recorded by PaperBroker)"]
    VOL_ENGINE["VolatilityEngine\nEWMA · decay=0.94"]
    PRICES --> VOL_ENGINE

    VOL_ENGINE --> VOL["Realized Vol"]
    TARGET["Target Annual Vol\ndefault 10%"]

    VOL --> FORMULA["leverage = target / realized"]
    TARGET --> FORMULA

    FORMULA --> CLAMP["clamp(leverage, 0.20, 2.0)"]
    CLAMP --> NOTIONAL["position_size = leverage × base_notional"]
```

### Risk Controls

The following diagram shows the risk-management pipeline that runs before and after every trade cycle:

```mermaid
graph TD
    EQUITY["Portfolio Equity"]

    EQUITY --> DD_DAILY["Daily Drawdown Check · default 5%"]
    EQUITY --> DD_TOTAL["Total Drawdown Check · default 15%"]

    DD_DAILY -->|"Breached"| HALT["Daily Halt · No new trades until tomorrow"]
    DD_TOTAL -->|"Breached"| KILL["Kill Switch · Close all positions"]

    EQUITY --> VOL_TARGET["Volatility Targeting\ntarget_vol / realized_vol × base_notional"]
    VOL_TARGET --> POS_SIZE["Vol-adjusted position size\n(falls back to 2% flat if disabled)"]

    HALT --> LOG["Telegram Alert + Log"]
    KILL --> LOG
```

- **Volatility-targeted position sizing** (opt-in) — scales each position so dollar-risk stays constant
- Configurable flat position size per trade (default 2% of equity, used when vol targeting is off or history is insufficient)
- 5% daily drawdown halt on new entries
- 15% peak-to-equity kill switch (closes all positions); use `/resume force` to rebase the high-water mark and clear a stuck kill switch
- `PAPER_MODE=true` by default — no real money until explicitly enabled
- All signals, orders, and risk events logged and Telegram-notified

---

## Setup

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure credentials:

```env
# Required for live Coinbase trading
COINBASE_API_KEY=your_key
COINBASE_API_SECRET=your_secret

# Required for Kalshi prediction markets + latency arb
KALSHI_KEY_ID=your_kalshi_key_id
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Risk and mode
PAPER_MODE=true
INITIAL_CASH=10000

# Volatility targeting (off by default; enable after ~20 trading days of history)
# VOL_TARGETING_ENABLED=true
# VOL_TARGET_ANNUAL=0.10
# VOL_MAX_LEVERAGE=2.0

# Enable latency arb
LATENCY_ARB_ENABLED=true
PREDICTION_ENABLED=true
PREDICTION_MARKET_CATEGORIES=crypto,economics,finance,politics,weather,sports,entertainment
PREDICTION_MIN_VOLUME_USD=5000
PREDICTION_MAX_SPREAD_USD=0.15
```

### Security Notes
- **Never commit `.env`** — it is gitignored
- Private keys are read from env vars only, never hardcoded
- `PAPER_MODE=true` is the safe default; real execution requires explicit opt-in
- Kalshi RSA private key should use `\n` for newlines when stored in `.env`
- All API credentials are stored as environment variables, not in config files
- The bot never logs or exposes credentials

---

## Usage

Run all strategies once:
```bash
python -m trading_bot all
```

Run individual modules:
```bash
python -m trading_bot stocks          # value stock screener
python -m trading_bot crypto          # BTC/ETH sentiment
python -m trading_bot rotation        # altcoin rotation
python -m trading_bot predictions     # Kalshi prediction markets
python -m trading_bot latency_arb     # single latency arb scan
python -m trading_bot kalshi_sports   # single sports scan (needs KALSHI_SPORTS_ENABLED=true for 'all'; explicit run always scans)
python -m trading_bot btc5min         # single BTC 5-min spread scan
```

Launch the live BTC spread dashboard (requires `rich>=13.0.0`):
```bash
python -m trading_bot btc-terminal
```

Show portfolio status (including Kalshi positions and balance):
```bash
python -m trading_bot status
```

Run as a continuous daemon (recommended):
```bash
python -m trading_bot run-daemon
```

The scheduler runs crypto every 10 minutes (`SCHEDULER_CRYPTO_INTERVAL_MINUTES`), BTC 5-min spread every 5 minutes (`BTC5MIN_INTERVAL_MINUTES`), predictions every 30 minutes, Polymarket every 20 minutes, and latency arb every 2 seconds when enabled. All intervals are configurable via environment variables.

---

## Telegram Bot

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured, the bot exposes interactive commands and sends automatic notifications.

The diagram below shows the outbound notification types and inbound command categories the bot handles:

```mermaid
graph LR
    subgraph Outbound
        TA["Trade Alerts"]
        HB["Heartbeat"]
        RA["Risk Alerts"]
        ER["Error Reports"]
        LA["Leads Alerts"]
    end

    subgraph "Inbound Commands"
        MON["/status /positions /risk /history /leads"]
        EXEC["/crypto /stocks /rotation /predict /polymarket /btc5min /backtest"]
        CTRL["/pause /resume /resume force /kill"]
        CFG["/config /ai"]
    end

    BOT["Telegram Bot"]
    SVC2["TradingBotService"]

    SVC2 --> BOT
    BOT --> TA
    BOT --> HB
    BOT --> RA
    BOT --> ER
    BOT --> LA

    MON --> BOT
    EXEC --> BOT
    CTRL --> BOT
    CFG --> BOT
    BOT --> SVC2
```

### Commands

| Command | Description |
|---------|-------------|
| `/status` | Portfolio dashboard (equity, P&L, Kalshi balance, positions) |
| `/positions` | Open positions grouped by asset type |
| `/risk` | Risk metrics (drawdown, kill switch state) |
| `/latency` | Show recent per-module latency stats |
| `/history` | Recent trade log |
| `/audit [count]` | Tail the persistent trade audit log (1-10 entries) |
| `/crypto` | Force a crypto sentiment cycle |
| `/rotation` | Force an altcoin rotation cycle |
| `/predict` | Force a Kalshi prediction scan |
| `/stocks` | Force a stock screen |
| `/polymarket` | Force a Polymarket scan |
| `/btc5min` | BTC 5-min spread signal card (Coinbase vs Kalshi vs Polymarket implied) |
| `/backtest [range]` | Run a Freqtrade backtest |
| `/leads` | Watchlist — skipped buys and top targets from the last strategy run |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
| `/resume force` | Resume trading **and** reset the peak-equity high-water mark to current equity, clearing a stuck kill switch |
| `/kill` | Emergency stop (kill switch) |
| `/config` | Show current configuration |
| `/ai` | AI analyst team status & budget |

### Notifications

- **Trade alerts** are sent after each scheduled cycle when trades are executed or errors occur.
- **No-signal cycles** (no trades, no errors) are suppressed to reduce spam.
- **Risk events** (kill switch, daily halt) always surface even when there are no trades.
- **Interactive commands** always return a completion message (including "no trades executed" when applicable).
- **Trade alerts** and **heartbeat** include stale-price warnings and recent module latency when available.
- **Heartbeat** shows equity, Kalshi balance, positions grouped by type, AI analyst status, and last scan dates.
- **Leads alerts** are automatically pushed after every successful module run with new watchlist entries (skipped buys and top targets). Deduped per-strategy per-day so the same lead doesn't re-fire. Controlled by `LEADS_NOTIFICATION_ENABLED`.

---

## Configuration

### BTC 5-Min Spread Strategy

| Env Var | Default | Description |
|---------|---------|-------------|
| `BTC5MIN_ENABLED` | `true` | Enable the strategy |
| `BTC5MIN_INTERVAL_MINUTES` | `5` | How often the scheduler fires this strategy (minutes). Also readable as `SCHEDULER_BTC5MIN_INTERVAL_MINUTES`. |
| `BTC5MIN_SYMBOL` | `BTCUSDT` | Spot BTC symbol (used for Coinbase price lookup and optional spot position) |
| `BTC5MIN_BUY_SPREAD_THRESHOLD` | `0.02` | Spread (as a fraction) that triggers a trade; 0.02 = 2% |
| `BTC5MIN_MAX_STRIKE_AGE_HOURS` | `48.0` | Only include prediction markets expiring within this window |
| `BTC5MIN_MIN_KALSHI_VOLUME_USD` | `3000.0` | Minimum 24h volume (USD) for a Kalshi market to be included |
| `BTC5MIN_MIN_POLY_VOLUME_USD` | `50000.0` | Minimum 24h volume (USD) for a Polymarket market to be included |
| `BTC5MIN_SPOT_TRADING_ENABLED` | `true` | Also open/close a spot BTC position when prediction-market signal fires |

### Kalshi Latency Arb

| Env Var | Default | Description |
|---------|---------|-------------|
| `LATENCY_ARB_ENABLED` | `false` | Enable the strategy |
| `LATENCY_ARB_SYMBOLS` | `BTC-USD,ETH-USD` | Coinbase spot symbols tracked for momentum |
| `LATENCY_ARB_MOMENTUM_WINDOW` | `30.0` | Seconds of spot history used for move detection |
| `LATENCY_ARB_MOMENTUM_THRESHOLD` | `0.30` | Min % spot move to trigger signal |
| `LATENCY_ARB_MIN_EDGE` | `0.05` | Min probability edge to enter |
| `LATENCY_ARB_KELLY_FRACTION` | `0.25` | Fraction of full Kelly to bet |
| `LATENCY_ARB_MAX_POSITION_PCT` | `0.08` | Max 8% of equity per trade |
| `LATENCY_ARB_MAX_OPEN_CONTRACTS` | `4` | Max simultaneous open positions |
| `LATENCY_ARB_SCAN_INTERVAL` | `2.0` | Seconds between scans |
| `LATENCY_ARB_PRICE_HISTORY` | `120.0` | Seconds of tick history kept in memory |
| `LATENCY_ARB_TAKE_PROFIT` | `0.20` | Exit at +20% price gain |
| `LATENCY_ARB_STOP_LOSS` | `0.15` | Exit at -15% price loss |
| `LATENCY_ARB_MIN_TRADE_USD` | `1.0` | Minimum position size |
| `LATENCY_ARB_AI_CONFIRMATION` | `boost` | AI confirmation mode |
| `LATENCY_ARB_MAX_PRICE_AGE_SECONDS` | `15.0` | Skip stale Coinbase prices before signaling |

### Leads Notifications

| Env Var | Default | Description |
|---------|---------|-------------|
| `LEADS_NOTIFICATION_ENABLED` | `true` | Push watchlist leads to Telegram after each module run |
| `LEADS_MAX_PER_MESSAGE` | `10` | Max leads shown per Telegram message |

A **lead** is a symbol the bot identified as tradeable but did not buy — e.g. because the position-size rounded to zero, the max-positions cap was reached, the AI vetoed it, or the symbol is in cooldown. Leads are deduped per-strategy per-day so the same entry doesn't re-fire until the next calendar day. Use `/leads` to see the current in-memory cache.

### Observability & Freshness

| Env Var | Default | Description |
|---------|---------|-------------|
| `PRICE_STALENESS_ENABLED` | `false` | Annotate stale prices in summaries and Telegram alerts |
| `PRICE_STALENESS_STOCK_MAX_AGE_SECONDS` | `900.0` | Max allowed age for stock marks |
| `PRICE_STALENESS_CRYPTO_MAX_AGE_SECONDS` | `180.0` | Max allowed age for crypto marks |
| `PRICE_STALENESS_PREDICTION_MAX_AGE_SECONDS` | `300.0` | Max allowed age for Kalshi/Polymarket marks |
| `PRICE_STALENESS_LATENCY_MAX_AGE_SECONDS` | `15.0` | Max allowed age for latency-arb feed marks |
| `OBS_TRADE_AUDIT_ENABLED` | `false` | Persist trade opens/closes to a JSONL audit log |
| `OBS_MODULE_LATENCY_WINDOW` | `50` | Number of module runs kept for latency stats |
| `OBS_MAX_STALE_SYMBOLS_IN_SUMMARY` | `10` | Cap stale-symbol list size in status summaries |
| `TRADE_AUDIT_PATH` | `data/trade_audit.jsonl` | Override the audit log file location |

### Volatility Targeting

| Env Var | Default | Description |
|---------|---------|-------------|
| `VOL_TARGETING_ENABLED` | `false` | Enable vol-adjusted position sizing |
| `VOL_TARGET_ANNUAL` | `0.10` | Target annual portfolio volatility (10%) |
| `VOL_WINDOW_DAYS` | `20` | Rolling window for realized vol (days) |
| `VOL_USE_EWMA` | `true` | Use EWMA instead of rolling std |
| `VOL_EWMA_DECAY` | `0.94` | EWMA decay factor (0.94 = RiskMetrics standard) |
| `VOL_MAX_LEVERAGE` | `2.0` | Hard leverage cap (never exceed 2× equity) |
| `VOL_MIN_ALLOCATION` | `0.20` | Minimum leverage floor (always allocate ≥ 20% of base notional) |
| `VOL_NO_TRADE_BAND` | `0.10` | Drift threshold for `should_rebalance()` (not yet wired into sizing) |
| `VOL_MAX_HISTORY_DAYS` | `60` | Max daily prices stored per symbol |

### AI Integration

| Env Var | Default | Description |
|---------|---------|-------------|
| `TRADING_AGENTS_ENABLED` | `false` | Enable AI analysis across all strategies |
| `ANTHROPIC_API_KEY` | | Anthropic API key (`sk-ant-...`) |
| `TRADING_AGENTS_CONFIRMATION_MODE` | `boost` | `boost`, `require`, or `log_only` |
| `TRADING_AGENTS_MAX_COST_PER_RUN` | `0.10` | Hard cap for a single AI request |
| `TRADING_AGENTS_MAX_COST_PER_MONTH` | `25.00` | Hard monthly spend cap in USD |
| `TRADING_AGENTS_CACHE_TTL_HOURS` | `6` | Cache window for repeated market evaluations |
| `TRADING_AGENTS_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `TRADING_AGENTS_MULTI_ANALYST` | `false` | Enable multi-analyst persona mode |
| `TRADING_AGENTS_ANALYSTS` | `value_investor,contrarian,sentiment_analyst` | Comma-separated persona keys |
| `TRADING_AGENTS_PROMPT_GUARD_ENABLED` | `false` | Strip and bound untrusted market text before prompting |
| `TRADING_AGENTS_CIRCUIT_BREAKER_ENABLED` | `false` | Temporarily pause AI calls after repeated failures |
| `TRADING_AGENTS_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failures before opening the breaker |
| `TRADING_AGENTS_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `300` | Cooldown before AI calls resume |
| `TRADING_AGENTS_STREAMING_ENABLED` | `false` | Use Anthropic streaming when the client supports it |
| `TRADING_AGENTS_TOOL_CALLING_ENABLED` | `false` | Allow registered local tools to participate in analysis |
| `TRADING_AGENTS_EXTENDED_THINKING_ENABLED` | `false` | Request Anthropic extended thinking blocks |
| `TRADING_AGENTS_THINKING_BUDGET_TOKENS` | `1024` | Token budget for extended thinking |
| `TRADING_AGENTS_THINKING_DISPLAY` | `omitted` | Thinking visibility mode sent to Anthropic |
| `TRADING_AGENTS_VISION_ENABLED` | `false` | Allow image URL inputs during market analysis |
| `TRADING_AGENTS_BOOST_AGREE_MULTIPLIER` | `1.50` | Position-size multiplier when AI agrees with a buy |
| `TRADING_AGENTS_BOOST_DISAGREE_MULTIPLIER` | `0.50` | Position-size multiplier when AI rejects a trade in boost mode |
| `TRADING_AGENTS_BOOST_MIN_CONFIDENCE` | `0.60` | Minimum confidence required before boost-mode size increases |
| `TRADING_AGENTS_REQUIRE_MIN_CONFIDENCE` | `0.50` | Minimum confidence required before require-mode allows a trade |
| `TRADING_AGENTS_ANALYST_BUY_THRESHOLD` | `0.20` | Consensus score above which multi-analyst mode emits `buy` |
| `TRADING_AGENTS_ANALYST_SELL_THRESHOLD` | `-0.20` | Consensus score below which multi-analyst mode emits `sell` |
| `TRADING_AGENTS_MODEL_PRICING_JSON` | built-in defaults | Override per-model input/output token pricing as JSON |

Available analyst personas: `value_investor`, `growth_investor`, `contrarian`, `technical_analyst`, `sentiment_analyst`, `risk_analyst`

Cost tracking uses **actual token counts** from the Anthropic response for precise monthly budget management.

Streaming, tool use, extended thinking, and vision are all opt-in. They only activate when the installed Anthropic client exposes the corresponding Messages API capability. When vision is enabled, Zulubot can render local candlestick charts as in-memory images for latency-arb analysis if `matplotlib` is installed via [requirements-tradingagents.txt](requirements-tradingagents.txt).

---

## Files

- State: `%LOCALAPPDATA%\TradingBot\data\state.json` (Windows), `~/.local/share/TradingBot/data/state.json` (Linux/macOS), or set `BOT_STORAGE_DIR` to override
- Trade audit: `%LOCALAPPDATA%\TradingBot\data\trade_audit.jsonl` (Windows), `~/.local/share/TradingBot/data/trade_audit.jsonl` (Linux/macOS), or set `TRADE_AUDIT_PATH` to override
- Logs: `%LOCALAPPDATA%\TradingBot\logs\` (Windows), `~/.local/share/TradingBot/logs/` (Linux/macOS), or set `BOT_LOG_DIR` to override

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Kalshi Sports Betting

`KalshiSportsStrategy` bets on Kalshi game-winner markets across NFL, college football (FBS and FCS), men's and women's college basketball, NBA, WNBA, MLB, NHL and soccer (EPL, La Liga, Bundesliga, Serie A, Ligue 1, MLS, Liga MX, Champions/Europa League; World Cup optional). Pick leagues with `KALSHI_SPORTS_LEAGUES`; leagues with no open Kalshi markets (offseason) are skipped without calling ESPN. Soccer uses three-way (home/draw/away) devigged odds and the separate "Tie" markets. Enable with `KALSHI_SPORTS_ENABLED=true`; it paper-trades unless `PAPER_MODE=false` and Kalshi credentials are set.

There are no fixed "minimum edge" cutoffs. Each cycle (default 30 min) it:
1. Pulls upcoming games from ESPN (schedule, DraftKings opening/closing moneylines, ESPN's matchup predictor, records) and matches open Kalshi markets to them.
2. Treats each input as a noisy measurement of the true win probability: the sportsbook closing line, Kalshi's own size-weighted order-book price (real-money flow), Polymarket's game price (a separate real-money crowd, public Gamma API, `KALSHI_SPORTS_USE_POLYMARKET`), ESPN's predictor, and a records model only when no line exists. They're fused by inverse-variance weighting in log-odds space.
3. **Learns** how reliable each source is: every forecast is logged, scored when Kalshi settles the game, and each source's uncertainty is updated (persisted in `sports_calibration.json` under the data dir). `/sports` and run results include `source_accuracy`.
4. Skips games where sources contradict each other beyond chance (a 99% chi-square test: usually a mismatched team or a stale quote) and drops a lone outlier source.
5. Values YES and NO **net of Kalshi's taker fee**, then sizes with Kelly shrunk by edge-vs-uncertainty, so noisy or thin edges get tiny stakes and strong, well-supported ones get larger ones. Stakes are capped by the size resting at the ask, a per-bet % of equity and a total exposure budget. One bet per game.
6. Holds to settlement and closes at Kalshi's settlement value ($1, $0, or $0.50 on a tie). Paper PnL includes the fee.
7. Every match must agree on kickoff time as well as teams, so a playoff series or home-and-home pair can't be priced off another day's game.

**Kill switch:** sports bets are independent, self-settling wagers, so by default (`KALSHI_SPORTS_IGNORE_GLOBAL_HALT=true`) a global kill switch or daily halt tripped by other strategies does not stop sports betting, does not close open sports bets, and does not leave the scheduler paused for sports. Sports keeps its own limits (per-bet cap, total exposure budget, max open bets). Set the flag to `false` to make sports obey the global switch again. A manual `/pause` still pauses everything, sports included.

**Parlays:** Kalshi combos are quoted on request (RFQ), not traded on an order book, so the bot never places them. Instead each scan suggests 2-4 leg parlays built only from legs it already finds profitable, from different games (independent), with the fair probability and the *maximum price worth paying*; place them in Kalshi's parlay builder. Betting the same legs separately with Kelly grows a bankroll faster than a parlay unless Kalshi's quote is below that price. `KALSHI_SPORTS_PARLAY_MAX_LEGS=0` turns suggestions off.

A sportsbook line is required by default (`KALSHI_SPORTS_REQUIRE_MARKET_SIGNAL`): ESPN's predictor and records alone are not sharper than Kalshi's price. Telegram: `/sports` runs a scan and shows picks with probability ± uncertainty, net edge after fees, and line movement.

Reality check from live testing (Oct 2026): DraftKings, Kalshi and ESPN agree to within a point or two on most games, so after the fee and spread there is usually no edge. Real edge needs more independent sharp sources (e.g. Pinnacle / multi-book lines, Polymarket prices) or catching Kalshi lagging a line move. Verified against live ESPN and Kalshi responses; if Kalshi renames a series, override it with `KALSHI_SPORTS_SERIES`. Soccer is unsupported (three-way outcomes). Sports betting carries real risk of loss; a positive modelled edge is not a guarantee.
