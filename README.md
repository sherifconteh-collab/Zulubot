# ZuluBot — Multi-Strategy Trading Bot

An automated trading system combining value investing, crypto sentiment, altcoin rotation, prediction markets, and latency arbitrage — all orchestrated by a multi-analyst AI agent team. All modules share the same paper portfolio, persistent state, and risk controls.

## Architecture

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
        TG["Telegram"]
        FEED["PriceFeed"]
    end

    SVC --> INFRA
    INFRA --> STATE
    INFRA --> RISK
    INFRA --> TG
    INFRA --> FEED
```

## Multi-Analyst AI System

Inspired by [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund), ZuluBot uses a **team-of-analysts** approach where multiple AI personas independently evaluate signals and aggregate their views via confidence-weighted voting.

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

### Risk Controls

```mermaid
graph TD
    EQUITY["Portfolio Equity"]

    EQUITY --> DD_DAILY["Daily Drawdown Check · default 5%"]
    EQUITY --> DD_TOTAL["Total Drawdown Check · default 15%"]

    DD_DAILY -->|"Breached"| HALT["Daily Halt · No new trades until tomorrow"]
    DD_TOTAL -->|"Breached"| KILL["Kill Switch · Close all positions"]

    EQUITY --> POS_SIZE["Position Sizing · default 2% of equity"]

    HALT --> LOG["Telegram Alert + Log"]
    KILL --> LOG
```

- Configurable position size per trade (default 2% of equity)
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
| `/history` | Recent trade log |
| `/crypto` | Force a crypto sentiment cycle |
| `/rotation` | Force an altcoin rotation cycle |
| `/predict` | Force a Kalshi prediction scan |
| `/stocks` | Force a stock screen |
| `/polymarket` | Force a Kalshi prediction scan (alias) |
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
- **Heartbeat** shows equity, Kalshi balance, positions grouped by type, AI analyst status, and last scan dates.

---

## Configuration

### Kalshi Latency Arb

| Env Var | Default | Description |
|---------|---------|-------------|
| `LATENCY_ARB_ENABLED` | `false` | Enable the strategy |
| `LATENCY_ARB_MOMENTUM_THRESHOLD` | `0.30` | Min % spot move to trigger signal |
| `LATENCY_ARB_MIN_EDGE` | `0.05` | Min probability edge to enter |
| `LATENCY_ARB_KELLY_FRACTION` | `0.25` | Fraction of full Kelly to bet |
| `LATENCY_ARB_MAX_POSITION_PCT` | `0.08` | Max 8% of equity per trade |
| `LATENCY_ARB_MAX_OPEN_CONTRACTS` | `4` | Max simultaneous open positions |
| `LATENCY_ARB_SCAN_INTERVAL` | `2.0` | Seconds between scans |
| `LATENCY_ARB_TAKE_PROFIT` | `0.20` | Exit at +20% price gain |
| `LATENCY_ARB_STOP_LOSS` | `0.15` | Exit at -15% price loss |
| `LATENCY_ARB_AI_CONFIRMATION` | `boost` | AI confirmation mode |

### AI Integration

| Env Var | Default | Description |
|---------|---------|-------------|
| `TRADING_AGENTS_ENABLED` | `false` | Enable AI analysis across all strategies |
| `ANTHROPIC_API_KEY` | | Anthropic API key (`sk-ant-...`) |
| `TRADING_AGENTS_CONFIRMATION_MODE` | `boost` | `boost`, `require`, or `log_only` |
| `TRADING_AGENTS_MAX_COST_PER_MONTH` | `10.00` | Hard monthly spend cap in USD |
| `TRADING_AGENTS_MULTI_ANALYST` | `false` | Enable multi-analyst persona mode |
| `TRADING_AGENTS_ANALYSTS` | `value_investor,contrarian,sentiment_analyst` | Comma-separated persona keys |

Available analyst personas: `value_investor`, `growth_investor`, `contrarian`, `technical_analyst`, `sentiment_analyst`, `risk_analyst`

Cost tracking uses **actual token counts** from the Anthropic response for precise monthly budget management.

---

## Files

- State: `%LOCALAPPDATA%\TradingBot\data\state.json` (Windows), `~/.local/share/TradingBot/data/state.json` (Linux/macOS), or set `BOT_STORAGE_DIR` to override
- Logs: `%LOCALAPPDATA%\TradingBot\logs\` (Windows), `~/.local/share/TradingBot/logs/` (Linux/macOS), or set `BOT_LOG_DIR` to override

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```
