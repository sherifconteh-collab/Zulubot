# Trading Bot

Two modules share the same paper portfolio, persistent state, and risk controls:

- `stocks`: value investing screener for S&P 500 names using Financial Modeling Prep stable endpoints.
- `crypto`: BTC/ETH sentiment plus Binance funding-rate bot using Alternative.me and Binance futures market data.

## What It Does

### Module 1: Value Investing Stocks

- Screens S&P 500 constituents.
- Requires:
  - P/E ratio below 20
  - P/B ratio below 3
  - debt-to-equity below 0.5
  - positive free-cash-flow signal
- Ranks qualified names with a composite value score built from P/E, P/B, debt-to-equity, and price-to-free-cash-flow.
- Buys the top 3 qualified names only when VIX is above 25.
- Sells held stock positions when any tracked fundamental breaches the threshold on the next weekly review.
- Self-throttles so the stock screen runs at most once every 7 days unless `--force` is used.

### Module 2: Crypto Sentiment And Funding

- Trades `BTCUSDT` and `ETHUSDT`.
- Buy signal:
  - Fear and Greed index below 25
  - Binance funding rate below 0
- Sell signal:
  - Fear and Greed index above 75
- The Alternative.me index is Bitcoin-centric, so this bot uses the same market-regime reading for both BTC and ETH and combines it with symbol-specific Binance funding data.

### Shared Risk Controls

- 2% position size per new trade
- 5% daily drawdown halt on new entries
- 15% peak-to-equity kill switch
- `PAPER_MODE=true` by default
- All runs, signals, order actions, and risk events are logged

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `FMP_API_KEY`.

## Usage

Run everything:

```powershell
python -m trading_bot all
```

Run only the weekly stock screen:

```powershell
python -m trading_bot stocks
```

Force the stock screen immediately:

```powershell
python -m trading_bot stocks --force
```

Run only the crypto module:

```powershell
python -m trading_bot crypto
```

Show current portfolio and risk state:

```powershell
python -m trading_bot status
```

## Files

- Default state path: `%LOCALAPPDATA%\TradingBot\data\state.json`
- Default log path: `%LOCALAPPDATA%\TradingBot\logs\trading_bot.log`
- Override storage with `BOT_STORAGE_DIR`, `BOT_DATA_DIR`, or `BOT_LOG_DIR`

## Scheduling

The simplest production-like setup on Windows is to schedule:

- `python -m trading_bot all`

to run hourly in Task Scheduler. The crypto module will evaluate every run, while the stock module will only execute once every 7 days unless forced.

## Testing

```powershell
python -m unittest discover -s tests -v
```

## Assumptions

- The stock ranking rule was not specified, so the bot uses a composite value rank across P/E, P/B, debt-to-equity, and price-to-free-cash-flow.
- Positive free cash flow is inferred from FMP free-cash-flow fields when present, otherwise from a positive price-to-free-cash-flow ratio.
- Live order routing is intentionally not implemented. Setting `PAPER_MODE=false` will raise an error instead of silently trading live.
