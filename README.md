# ZuluBot — Multi-Strategy Trading Bot

An automated trading system combining value investing, crypto sentiment, altcoin rotation, prediction markets, and latency arbitrage. All modules share the same paper portfolio, persistent state, and risk controls.

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
- Scans crypto and economics categories
- **Buy YES** when contract price < 0.30
- **Buy NO** when contract price > 0.70
- Optional Fear & Greed confirmation for crypto-tagged markets
- Market discovery works without credentials; trading requires Kalshi API key

### 5. Latency Arbitrage — Kalshi (NEW)
High-frequency strategy that exploits Kalshi contract mispricing:
- Polls Coinbase BTC-USD / ETH-USD spot prices every ~2 seconds
- Detects sharp momentum moves (≥ 0.30% in 30 seconds)
- When contract odds are stale relative to the price move, buys the correct side
- Uses fractional Kelly criterion (25% of full Kelly) for position sizing
- Manages exits via take-profit (20%) and stop-loss (15%)
- Paper mode by default; goes live when `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY` are set and `LATENCY_ARB_ENABLED=true`

### Shared Risk Controls
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

Show portfolio status (including Kalshi positions):
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

### Commands

| Command | Description |
|---------|-------------|
| `/status` | Portfolio dashboard (equity, P&L, positions) |
| `/positions` | List open positions |
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
| `/ai` | AI cost & budget summary |

### Notifications

- **Trade alerts** are sent after each scheduled cycle when trades are executed or errors occur.
- **No-signal cycles** (no trades, no errors) are suppressed to reduce spam.
- **Risk events** (kill switch, daily halt) always surface even when there are no trades.
- **Interactive commands** always return a completion message (including "no trades executed" when applicable).
- **Heartbeat** runs on a configurable interval showing equity, positions, strategy status, and last scan dates.

---

## Kalshi Latency Arb — Configuration

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
| `LATENCY_ARB_AI_CONFIRMATION` | `boost` | AI confirmation mode: `boost`, `require`, or `log_only` |

---

## AI Integration

All strategies optionally use Claude (Haiku + Sonnet) for signal validation. Enable by setting `TRADING_AGENTS_ENABLED=true` and `ANTHROPIC_API_KEY`.

| Env Var | Default | Description |
|---------|---------|-------------|
| `TRADING_AGENTS_ENABLED` | `false` | Enable AI analysis across all strategies |
| `ANTHROPIC_API_KEY` | | Anthropic API key (`sk-ant-...`) |
| `TRADING_AGENTS_CONFIRMATION_MODE` | `boost` | `boost` (1.5×/0.5× sizing), `require` (AI must agree), `log_only` (advisory) |
| `TRADING_AGENTS_MAX_COST_PER_MONTH` | `10.00` | Hard monthly spend cap in USD |

### AI coverage per strategy

| Strategy | Model | Cache TTL | Purpose |
|----------|-------|-----------|---------|
| Stocks | Haiku → Sonnet | 6 h | Validate fundamentals screen |
| Crypto | Haiku → Sonnet | 6 h | Validate F&G + funding signal |
| Polymarket (legacy) | Sonnet | 6 h | Estimate fair probability |
| Kalshi predictions | Haiku → Sonnet | 6 h | Estimate fair probability |
| Latency arb — entry | Haiku only | 10 min | Filter noise from momentum signals |
| Latency arb — exit | Haiku only | 5 min | Veto premature stop-loss / take-profit exits |
| Latency arb — regime | Haiku only | 1 h | Adjust momentum threshold (volatile ↑ / bullish ↓) |
| Altcoin rotation | Haiku only | 30 min | Check target coin for macro red flags before swap |

Cost tracking uses **actual token counts** from the Anthropic response for precise monthly budget management.

---

## Files

- State: `%LOCALAPPDATA%\TradingBot\data\state.json` (Windows), `~/.local/share/TradingBot/data/state.json` (Linux/macOS), or set `BOT_STORAGE_DIR` to override
- Logs: `%LOCALAPPDATA%\TradingBot\logs\` (Windows), `~/.local/share/TradingBot/logs/` (Linux/macOS), or set `BOT_LOG_DIR` to override

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```
