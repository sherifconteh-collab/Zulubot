# Zulubot

Paper trading bot with multi-strategy async architecture deployed on Railway.

## Tech Stack
- Python 3.12, asyncio
- unittest for testing
- Railway for deployment (persistent volume at /data)
- Telegram bot for commands and notifications
- Anthropic Claude API for optional AI analysis

## Project Structure
- `trading_bot/` — Main package
  - `strategies/` — 6 strategies: stocks, crypto, altcoin rotation, Kalshi predictions, latency arb, Polymarket
  - `integrations/` — AI analysis (trading_agents), analyst personas, Freqtrade backtesting
  - `service.py` — Core TradingBotService orchestrator
  - `scheduler.py` — SchedulerService daemon loop
  - `broker.py` — PaperBroker (simulated trading)
  - `coinbase_broker.py` — Live Coinbase trading
  - `polymarket_broker.py` — Polymarket trading
  - `risk.py` — RiskManager (position limits, drawdown halts, trailing stops, cooldowns)
  - `volatility.py` — EWMA-based volatility targeting engine
  - `config.py` — All configuration via frozen dataclasses + env vars
  - `telegram_bot.py` — Full Telegram command interface
  - `clients.py`, `clients_kalshi.py`, `clients_polymarket.py` — API clients
- `tests/` — 11 unit test files
- `.env.example` — All environment variables documented

## Code Standards
- All config via environment variables (never hardcode secrets)
- Frozen dataclasses in `config.py` for all settings
- Async/await patterns throughout strategies and services
- Mock external API calls in tests
- Strategies follow common pattern: setup → evaluate → execute

## Testing
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Key Commands
- `python -m trading_bot run-daemon` — Start scheduler daemon
- `python -m trading_bot run-once <strategy>` — Single strategy run
- Telegram: `/status`, `/positions`, `/crypto`, `/stocks`, `/pause`, `/resume`, `/kill`

## Important Notes
- Never commit `.env` files or API keys
- Risk management is critical: respect drawdown limits and position caps
- The `PAPER_MODE=true` default must not be changed without explicit intent
