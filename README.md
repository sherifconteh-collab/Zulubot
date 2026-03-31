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
    end

    SVC --> STRAT
    STRAT --> STK
    STRAT --> CRY
    STRAT --> ROT
    STRAT --> PRED
    STRAT --> POLY
    STRAT --> LARB

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
- 15% peak-to-equity kill switch (closes all positions)
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
PREDICTION_MARKET_CATEGORIES=crypto,economics,finance,politics,weather
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
```

Show portfolio status (including Kalshi positions and balance):
```bash
python -m trading_bot status
```

Run as a continuous daemon (recommended):
```bash
python -m trading_bot run-daemon
```

The scheduler runs crypto every 15 minutes, predictions every 60 minutes, and latency arb every 2 seconds when enabled.

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
    end

    subgraph "Inbound Commands"
        MON["/status /positions /risk /history"]
        EXEC["/crypto /stocks /rotation /predict /polymarket /backtest"]
        CTRL["/pause /resume /kill"]
        CFG["/config /ai"]
    end

    BOT["Telegram Bot"]
    SVC2["TradingBotService"]

    SVC2 --> BOT
    BOT --> TA
    BOT --> HB
    BOT --> RA
    BOT --> ER

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
| `/backtest [range]` | Run a Freqtrade backtest |
| `/pause` | Pause trading |
| `/resume` | Resume trading |
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

---

## Configuration

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
