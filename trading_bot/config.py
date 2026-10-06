from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_list(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return [part.strip().upper() for part in raw_value.split(",") if part.strip()]


def _env_list_lower(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return [part.strip().lower() for part in raw_value.split(",") if part.strip()]


def _env_float_map(name: str, default: dict[str, float]) -> dict[str, float]:
    """Parse "XRPUSDT:2.0,BTCUSDT:1.0" into {"XRPUSDT": 2.0, "BTCUSDT": 1.0}.

    Malformed entries are skipped; an empty/unset variable returns the default.
    """
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    parsed: dict[str, float] = {}
    for part in raw_value.split(","):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        key = key.strip().upper()
        if not key:
            continue
        try:
            parsed[key] = float(value.strip())
        except ValueError:
            continue
    return parsed if parsed else default


def _default_storage_root() -> Path:
    explicit_root = os.getenv("BOT_STORAGE_DIR")
    if explicit_root:
        return Path(explicit_root).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "TradingBot"
    return PROJECT_ROOT / ".runtime"


@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction: float = 0.02
    daily_halt_drawdown: float = 0.05
    kill_switch_drawdown: float = 0.15
    trailing_stop_pct: float = 0.0          # 0 = disabled; e.g. 0.10 → close when price drops 10% from peak
    max_open_positions: int = 0             # 0 = unlimited; portfolio-wide cap across all strategies
    cooldown_consecutive_losses: int = 0    # 0 = disabled; pause after N consecutive losing trades
    cooldown_periods: int = 3               # number of trade cycles to skip after cooldown triggers
    post_kill_cooldown_hours: float = 1.0   # hours to block new entries after kill switch is cleared (0 = no cooldown)


@dataclass(frozen=True)
class VolatilityConfig:
    enabled: bool = False            # opt-in; default preserves existing behaviour
    target_annual_vol: float = 0.10  # 10% annual vol target
    vol_window_days: int = 20        # rolling-std window (min history required)
    use_ewma: bool = True            # True=EWMA(decay), False=rolling std
    ewma_decay: float = 0.94         # RiskMetrics / professional-fund standard
    max_leverage: float = 2.0        # hard leverage cap (never exceed 2× capital)
    min_allocation: float = 0.20     # floor: never allocate < 20% of base notional (min leverage multiplier)
    no_trade_band: float = 0.10      # skip rebalance if allocation drift < 10%
    max_history_days: int = 60       # keep at most 60 daily prices per symbol


@dataclass(frozen=True)
class StockStrategyConfig:
    max_pe_ratio: float = 25.0
    max_pb_ratio: float = 3.5
    max_debt_to_equity: float = 0.75
    min_positive_fcf_signal: float = 0.0
    vix_buy_threshold: float = 20.0
    run_interval_days: int = 3
    max_positions: int = 5
    vix_symbol: str = "^VIX"
    # Buffett-style quality floor: return on equity. A common Buffett heuristic
    # is "companies that consistently earn high returns on equity" — he looks
    # for ROE >= 15%. Set to 0 to disable the filter.
    min_roe: float = 0.15
    # Minimum holding period for stock positions before the screener is
    # allowed to sell them due to drifting fundamentals. Protects against
    # churning good businesses on noisy readings ("our favorite holding
    # period is forever"). Hard risk events (stop-loss, kill switch, manual
    # /kill) still close positions immediately — this only gates the
    # screener's own sells.
    min_hold_days: int = 14


@dataclass(frozen=True)
class CryptoStrategyConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    buy_fear_threshold: int = 35
    sell_fear_threshold: int = 75
    use_coinbase_pricing: bool = True
    # If True, only buy when Binance perpetual funding rate is negative.
    # Default False → treat negative funding as an informational signal so the
    # bot trades on Fear & Greed alone; avoids the dual-gate filter that
    # previously kept the strategy idle for long stretches.
    require_negative_funding: bool = False
    # Autonomous exit rules so the bot sells as it sees fit, not only when
    # Fear & Greed climbs above ``sell_fear_threshold``. Both are fractions.
    # 0.0 disables the respective rule.
    take_profit_pct: float = 0.15   # +15% from entry → take profits
    stop_loss_pct: float = 0.10     # −10% from entry → cut loss
    # Per-symbol capital weights (e.g. {"XRPUSDT": 2.0}) scale the sized
    # notional for that symbol. Missing symbols default to 1.0.
    symbol_weights: dict[str, float] = field(default_factory=dict)
    # Per-symbol exit overrides; missing symbols fall back to the globals
    # above. A 0.0 override disables that rule for the symbol.
    take_profit_overrides: dict[str, float] = field(default_factory=dict)
    stop_loss_overrides: dict[str, float] = field(default_factory=dict)
    # Skip buys when the symbol's own price is more than ``trend_tolerance_pct``
    # below its ``trend_sma_days``-day SMA — avoids buying a coin in a steep
    # downtrend just because market-wide fear is high. Fails open when there is
    # insufficient price history.
    trend_filter_enabled: bool = True
    trend_sma_days: int = 20
    trend_tolerance_pct: float = 0.05


@dataclass(frozen=True)
class CoinbaseConfig:
    api_key: str = ""
    api_secret: str = ""
    # Coinbase is a price/data source by default. Orders are only sent when this is true
    # (and PAPER_MODE=false); otherwise the crypto/rotation strategies do not run live.
    trading_enabled: bool = False


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    # Routine alerts (cycle summaries, risk banners, leads, heartbeats) are batched into one
    # message every this many hours. 0 sends each alert immediately. Replies to commands
    # (/status, /sports, ...) and a kill switch firing are always immediate.
    digest_hours: float = 6.0

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class PriceStalenessConfig:
    enabled: bool = False
    stock_max_age_seconds: float = 900.0
    crypto_max_age_seconds: float = 180.0
    prediction_max_age_seconds: float = 300.0
    latency_max_age_seconds: float = 15.0


@dataclass(frozen=True)
class ObservabilityConfig:
    trade_audit_enabled: bool = False
    module_latency_window: int = 50
    max_stale_symbols_in_summary: int = 10


@dataclass(frozen=True)
class HttpRetryConfig:
    enabled: bool = False
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    backoff_multiplier: float = 2.0


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_m: float
    output_cost_per_m: float


def _default_model_pricing() -> dict[str, ModelPricing]:
    return {
        "claude-haiku-4-5-20251001": ModelPricing(input_cost_per_m=0.80, output_cost_per_m=4.00),
        "claude-sonnet-4-6": ModelPricing(input_cost_per_m=3.00, output_cost_per_m=15.00),
    }


def _env_model_pricing(name: str, default: dict[str, ModelPricing]) -> dict[str, ModelPricing]:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return dict(default)
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(payload, dict):
        return dict(default)

    parsed: dict[str, ModelPricing] = {}
    for model_name, values in payload.items():
        if not isinstance(model_name, str) or not isinstance(values, dict):
            continue
        try:
            parsed[model_name.strip()] = ModelPricing(
                input_cost_per_m=float(values["input_cost_per_m"]),
                output_cost_per_m=float(values["output_cost_per_m"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed or dict(default)


@dataclass(frozen=True)
class BTC5MinConfig:
    """Configuration for the 5-minute BTC spread monitor (spot vs Kalshi vs Polymarket)."""

    enabled: bool = True
    btc_symbol: str = "BTCUSDT"
    # Spread threshold (fraction) above which we act on a venue
    buy_spread_threshold: float = 0.02      # 2% spread → trade
    # Only use prediction markets expiring within this many hours
    max_strike_age_hours: float = 48.0
    min_kalshi_volume_usd: float = 3_000.0
    min_poly_volume_usd: float = 50_000.0
    # Whether to also trade spot BTC when spread is confirmed
    spot_trading_enabled: bool = True


@dataclass(frozen=True)
class LeadsConfig:
    """Telegram lead/watchlist push controls."""
    enabled: bool = True
    max_per_message: int = 10


@dataclass(frozen=True)
class LearningConfig:
    """Controls the per-symbol performance-memory learning layer.

    The bot persists realized P&L per ``(symbol)`` and, after
    ``loser_cooldown_losses`` consecutive losing closes, benches that symbol
    for ``loser_cooldown_cycles`` strategy cycles. This is a simple, auditable
    form of online learning — no black-box ML model. A winning close resets
    both counters.
    """
    enabled: bool = True
    loser_cooldown_losses: int = 3
    loser_cooldown_cycles: int = 10


@dataclass(frozen=True)
class MarkovStrategyConfig:
    """Configuration for the Markov Chain regime-switching strategy."""

    enabled: bool = False
    # Symbols to trade when the regime signal is bullish
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "XRPUSDT"])
    # Symbol whose price history drives regime detection (BTC is the market barometer)
    regime_symbol: str = "BTCUSDT"
    # Rolling return window (days) for state classification
    return_window: int = 20
    # Rolling return threshold to classify as Bull
    bull_threshold: float = 0.02
    # Rolling return threshold to classify as Bear
    bear_threshold: float = -0.02
    # signal = bull_prob - bear_prob; open long when above this
    entry_signal_threshold: float = 0.10
    # Close positions when signal drops below this
    exit_signal_threshold: float = 0.0
    # Autonomous stop-loss: close when price drops this fraction from entry
    stop_loss_pct: float = 0.10
    # Autonomous take-profit: close when price rises this fraction from entry
    take_profit_pct: float = 0.20
    # Minimum price history days before the strategy is allowed to trade
    min_history_days: int = 40
    # Laplace smoothing for transition matrix (0.0 = pure MLE)
    laplace_alpha: float = 0.1
    # Chapman-Kolmogorov forecast horizon (steps ahead; 1 = next period)
    forecast_steps: int = 1
    # Scheduler interval
    run_interval_minutes: int = 60
    # Use Coinbase pricing (True) or Binance mark price (False)
    use_coinbase_pricing: bool = True


@dataclass(frozen=True)
class SchedulerConfig:
    crypto_interval_minutes: int = 10
    btc5min_interval_minutes: int = 5
    stock_check_hour: int = 9
    polymarket_interval_minutes: int = 20
    heartbeat_interval_minutes: int = 60
    heartbeat_enabled: bool = False  # off by default — heartbeats were a spam source
    poll_interval_seconds: int = 45


@dataclass(frozen=True)
class TradingAgentsConfig:
    enabled: bool = False
    llm_provider: str = "anthropic"
    quick_model: str = "claude-haiku-4-5-20251001"  # cheap pre-screen
    deep_model: str = "claude-sonnet-4-6"           # deep analysis only when needed
    api_key: str = ""
    confirmation_mode: str = "boost"    # boost=1.5x agree / 0.5x disagree
    cache_ttl_hours: int = 6            # 6h cache avoids re-calling same market
    max_cost_per_run: float = 0.10      # hard cap per single AI call
    max_cost_per_month: float = 25.00   # hard monthly spend cap
    max_debate_rounds: int = 1
    timeout_seconds: int = 30           # short timeout to avoid blocking trades
    # Multi-analyst personas (inspired by virattt/ai-hedge-fund)
    multi_analyst_enabled: bool = False  # opt-in to multi-analyst evaluation
    selected_analysts: tuple[str, ...] = (
        "value_investor", "contrarian", "sentiment_analyst",
    )
    prompt_injection_protection_enabled: bool = False
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 300
    streaming_enabled: bool = False
    tool_calling_enabled: bool = False
    extended_thinking_enabled: bool = False
    thinking_budget_tokens: int = 1024
    thinking_display: str = "omitted"
    vision_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    boost_agree_multiplier: float = 1.5
    boost_disagree_multiplier: float = 0.5
    boost_min_confidence: float = 0.6
    require_min_confidence: float = 0.5
    analyst_buy_threshold: float = 0.2
    analyst_sell_threshold: float = -0.2
    default_input_cost_per_m: float = 3.0
    default_output_cost_per_m: float = 15.0
    estimated_input_tokens_per_call: int = 1500
    estimated_output_tokens_per_call: int = 300
    model_pricing: dict[str, ModelPricing] = field(default_factory=_default_model_pricing)


@dataclass(frozen=True)
class PolymarketStrategyConfig:
    enabled: bool = False
    market_tags: list[str] = field(default_factory=lambda: ["crypto", "finance"])
    min_volume: float = 50_000.0
    min_liquidity: float = 5_000.0
    buy_yes_threshold: float = 0.35
    buy_no_threshold: float = 0.65
    take_profit_threshold: float = 0.50
    stop_loss_threshold: float = 0.15
    max_positions: int = 5
    use_trading_agents: bool = False
    use_markov_regime_gate: bool = True  # block new entries when Markov is in bear regime


@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    private_key: str = ""


@dataclass(frozen=True)
class AltcoinRotationConfig:
    enabled: bool = True
    scout_multiplier: float = 3.0        # ratio must exceed fees × this to justify swap
    fee_pct: float = 0.006               # Coinbase taker fee ~0.6%
    check_interval_minutes: int = 30     # check every 30 minutes
    min_trade_usd: float = 1.0           # minimum $1 trade
    excluded_coins: list[str] = field(default_factory=list)  # coins to never trade
    max_coins_to_track: int = 50         # limit to top 50 by volume
    capital_allocation_pct: float = 0.60 # use 60% of total capital for rotation
    # "Never go backwards" memory expiry: remembered best coin quantities
    # older than this many cycles are ignored, so a coin crash can't lock the
    # bot out of re-entering forever. 0 = never expire (legacy behavior).
    qty_memory_cycles: int = 336         # ~1 week at 30-minute intervals


@dataclass(frozen=True)
class PredictionConfig:
    enabled: bool = False                # OFF by default
    kalshi_key_id: str = ""              # Kalshi API key ID
    kalshi_private_key: str = ""         # Kalshi RSA private key PEM
    buy_yes_threshold: float = 0.45      # Buy YES below this price (wider = more aggressive)
    buy_no_threshold: float = 0.55       # Buy NO above this price (wider = more aggressive)
    take_profit_threshold: float = 0.50  # Take profit at 0.50
    stop_loss_pct: float = 0.15          # Stop loss at 15% move against
    max_positions: int = 8
    check_interval_minutes: int = 30
    min_trade_usd: float = 1.0
    min_volume_usd: float = 3_000        # Minimum 24h volume to consider a market tradeable
    max_spread_usd: float = 0.20         # Maximum bid/ask spread allowed per side
    market_categories: list[str] = field(default_factory=lambda: ["crypto", "economics", "finance", "politics", "weather", "sports", "entertainment"])
    capital_allocation_pct: float = 0.40 # use 40% of total capital for predictions
    # Opt-in extra gate for crypto-category markets: require an extreme Fear &
    # Greed reading to align with the price threshold. Default False so crypto
    # predictions trade on price thresholds alone — previously this gate kept
    # the Kalshi strategy idle outside of extreme sentiment regimes.
    require_crypto_fear_greed: bool = False
    use_markov_regime_gate: bool = True  # block new entries when Markov is in bear regime
    # Market discovery breadth: pages of 50 markets fetched per series (the
    # API cursor used to be discarded, capping coverage at 20 markets/series).
    max_pages_per_series: int = 3
    # Opt-in: augment the hardcoded series lists with series discovered from
    # the Kalshi events API so new market series are picked up automatically.
    discover_series: bool = False


@dataclass(frozen=True)
class KalshiPerpetualsConfig:
    """Configuration for the Kalshi Crypto Perpetuals strategy."""

    enabled: bool = False
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"])
    max_leverage: float = 2.0
    take_profit_pct: float = 0.05
    stop_loss_pct: float = 0.03
    trailing_stop_pct: float = 0.0
    max_positions: int = 4
    capital_allocation_pct: float = 0.20
    check_interval_minutes: int = 15
    momentum_window_minutes: int = 60
    momentum_threshold_pct: float = 0.02
    min_volume_usd: float = 100_000.0
    min_trade_usd: float = 10.0


@dataclass(frozen=True)
class KalshiSportsConfig:
    """Configuration for the Kalshi sports (game-winner) strategy."""

    enabled: bool = True
    # League keys to scan (see clients_espn.LEAGUES); leagues with no open Kalshi markets are skipped.
    leagues: list[str] = field(default_factory=lambda: [
        "nfl", "ncaaf", "ncaaf-fcs", "ncaamb", "ncaawb", "nba", "wnba", "mlb", "nhl",
        "epl", "laliga", "bundesliga", "seriea", "ligue1", "mls", "ligamx", "ucl", "uel",
    ])
    # Extra Kalshi series per league, "key=SERIES" (e.g. "nfl=KXNFLGAME"); adds to the built-in ones.
    series_overrides: list[str] = field(default_factory=list)
    check_interval_minutes: int = 30
    max_hours_ahead: float = 48.0            # only bet games starting within this window
    min_minutes_to_start: float = 5.0        # don't enter right at kickoff
    # No fixed edge/price/volume cutoffs: stakes scale with net edge vs uncertainty
    # (see trading_bot/sports_model.py). These are risk limits and cost inputs only.
    kelly_fraction: float = 0.25             # fraction of (uncertainty-shrunk) Kelly to stake
    max_position_pct: float = 0.10           # per-bet cap, % of equity
    capital_allocation_pct: float = 0.50     # total cap on open sports exposure (% of bankroll)
    max_open_bets: int = 10
    min_trade_usd: float = 1.0               # smallest stake worth the fee/attention
    home_advantage: float = 0.55             # prior P(home wins) between equal teams (record model)
    fee_rate: float = 0.07                   # Kalshi taker fee: ceil(M * rate * C * p * (1 - p)); M is read per series
    # Require a sportsbook line for the game; ESPN's predictor/records alone can't beat Kalshi's price.
    require_market_signal: bool = True
    # Stakes are a share of the *current* wallet, so wins compound automatically. After a
    # bad stretch they shrink in proportion to today's drawdown from the day's peak
    # (never below 25% of normal). Drawdown at which scaling hits its floor: 0 disables.
    drawdown_throttle: float = 0.5
    # Sports bets are independent, self-settling wagers: a kill switch / daily halt tripped
    # by other strategies (crypto, predictions, ...) must not stop them or liquidate them.
    # Sports keeps its own limits: per-bet cap, exposure budget, max open bets.
    ignore_global_halt: bool = True
    use_polymarket: bool = True              # add Polymarket's real-money price as a signal
    # Parlay *suggestions* (never auto-placed): combine up to this many model-approved
    # legs from different games. 0 disables. Kalshi combos are quoted on request, not
    # via an order book, so they are placed manually in Kalshi's parlay builder.
    parlay_max_legs: int = 4
    hold_to_settlement: bool = True          # sports bets are held to the final whistle
    stop_loss_pct: float = 0.0               # only used when hold_to_settlement is False
    take_profit_pct: float = 0.0


@dataclass(frozen=True)
class FreqtradeConfig:
    enabled: bool = False
    api_server_port: int = 8080


@dataclass(frozen=True)
class LatencyArbConfig:
    """Configuration for the Kalshi latency arbitrage strategy."""

    enabled: bool = False
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    momentum_window_seconds: float = 30.0
    momentum_min_threshold_pct: float = 0.30  # min 0.3% spot move to signal
    min_edge_pct: float = 0.05                # min 5% edge over contract price
    kelly_fraction: float = 0.25             # fractional Kelly multiplier
    max_position_pct: float = 0.08           # max 8% of equity per contract
    max_open_contracts: int = 4
    scan_interval_seconds: float = 2.0
    price_history_seconds: float = 120.0
    take_profit_pct: float = 0.20            # exit when price rises 20%
    stop_loss_pct: float = 0.15
    min_trade_usd: float = 1.0               # minimum position size in USD
    ai_confirmation_mode: str = "boost"      # boost | require | log_only
    max_price_age_seconds: float = 15.0      # skip stale Coinbase feed prices


@dataclass(frozen=True)
class RobinhoodConfig:
    username: str = ""
    password: str = ""
    mfa_secret: str = ""  # Base32 TOTP secret; requires pyotp installed


@dataclass(frozen=True)
class OptionsStrategyConfig:
    enabled: bool = False
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "SPY", "QQQ", "TSLA", "NVDA"])
    min_dte: int = 21                    # minimum days to expiration
    max_dte: int = 60                    # maximum days to expiration
    # Only buy contracts where |delta| >= min_delta (deep in-the-money).
    # delta >= 0.70 → stock-like behavior, lower theta decay, high ITM probability.
    min_delta: float = 0.70
    max_positions: int = 5
    take_profit_pct: float = 0.50        # close when premium gains +50%
    stop_loss_pct: float = 0.50          # close when premium loses -50%
    close_at_dte: int = 7                # close when DTE drops below this
    buy_call_fear_threshold: int = 30    # buy calls when F&G < this
    buy_put_fear_threshold: int = 70     # buy puts when F&G > this
    capital_allocation_pct: float = 0.10 # fraction of total equity for options
    check_interval_minutes: int = 60
    min_volume: int = 100
    min_open_interest: int = 500
    max_iv: float = 1.50                 # skip contracts with IV > 150%


@dataclass(frozen=True)
class AppConfig:
    paper_mode: bool
    initial_cash: float
    timezone: ZoneInfo
    request_timeout_seconds: int
    fmp_api_key: str
    data_dir: Path
    log_dir: Path
    state_path: Path
    trade_audit_path: Path
    log_level: str
    http_retry: HttpRetryConfig
    price_staleness: PriceStalenessConfig
    observability: ObservabilityConfig
    risk: RiskConfig
    stock: StockStrategyConfig
    crypto: CryptoStrategyConfig
    coinbase: CoinbaseConfig
    telegram: TelegramConfig
    scheduler: SchedulerConfig
    polymarket: PolymarketConfig
    polymarket_strategy: PolymarketStrategyConfig
    trading_agents: TradingAgentsConfig
    altcoin_rotation: AltcoinRotationConfig
    predictions: PredictionConfig
    freqtrade: FreqtradeConfig
    latency_arb: LatencyArbConfig
    volatility: VolatilityConfig
    btc5min: BTC5MinConfig
    learning: LearningConfig
    leads: LeadsConfig
    markov: MarkovStrategyConfig
    robinhood: RobinhoodConfig
    options: OptionsStrategyConfig
    kalshi_perps: KalshiPerpetualsConfig
    kalshi_sports: KalshiSportsConfig
    protected_symbols: frozenset[str]

    @classmethod
    def load(cls) -> "AppConfig":
        storage_root = _default_storage_root()
        data_dir_env = os.getenv("BOT_DATA_DIR", "").strip()
        log_dir_env = os.getenv("BOT_LOG_DIR", "").strip()
        data_dir = Path(data_dir_env).expanduser() if data_dir_env else storage_root / "data"
        log_dir = Path(log_dir_env).expanduser() if log_dir_env else storage_root / "logs"
        trade_audit_path_env = os.getenv("TRADE_AUDIT_PATH", "").strip()
        trade_audit_path = Path(trade_audit_path_env).expanduser() if trade_audit_path_env else data_dir / "trade_audit.jsonl"
        return cls(
            paper_mode=_env_bool("PAPER_MODE", True),
            initial_cash=_env_float("INITIAL_CASH", 100_000.0),
            timezone=ZoneInfo(os.getenv("BOT_TIMEZONE", "America/New_York")),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 20),
            fmp_api_key=os.getenv("FMP_API_KEY", "").strip(),
            data_dir=data_dir,
            log_dir=log_dir,
            state_path=data_dir / "state.json",
            trade_audit_path=trade_audit_path,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            http_retry=HttpRetryConfig(
                enabled=_env_bool("HTTP_RETRY_ENABLED", False),
                max_attempts=_env_int("HTTP_RETRY_MAX_ATTEMPTS", 3),
                base_delay_seconds=_env_float("HTTP_RETRY_BASE_DELAY_SECONDS", 0.5),
                max_delay_seconds=_env_float("HTTP_RETRY_MAX_DELAY_SECONDS", 4.0),
                backoff_multiplier=_env_float("HTTP_RETRY_BACKOFF_MULTIPLIER", 2.0),
            ),
            price_staleness=PriceStalenessConfig(
                enabled=_env_bool("PRICE_STALENESS_ENABLED", False),
                stock_max_age_seconds=_env_float("PRICE_STALENESS_STOCK_MAX_AGE_SECONDS", 900.0),
                crypto_max_age_seconds=_env_float("PRICE_STALENESS_CRYPTO_MAX_AGE_SECONDS", 180.0),
                prediction_max_age_seconds=_env_float("PRICE_STALENESS_PREDICTION_MAX_AGE_SECONDS", 300.0),
                latency_max_age_seconds=_env_float("PRICE_STALENESS_LATENCY_MAX_AGE_SECONDS", 15.0),
            ),
            observability=ObservabilityConfig(
                trade_audit_enabled=_env_bool("OBS_TRADE_AUDIT_ENABLED", False),
                module_latency_window=_env_int("OBS_MODULE_LATENCY_WINDOW", 50),
                max_stale_symbols_in_summary=_env_int("OBS_MAX_STALE_SYMBOLS_IN_SUMMARY", 10),
            ),
            risk=RiskConfig(
                max_position_fraction=_env_float("MAX_POSITION_FRACTION", 0.02),
                daily_halt_drawdown=_env_float("DAILY_HALT_DRAWDOWN", 0.05),
                kill_switch_drawdown=_env_float("KILL_SWITCH_DRAWDOWN", 0.15),
                trailing_stop_pct=_env_float("TRAILING_STOP_PCT", 0.0),
                max_open_positions=_env_int("MAX_OPEN_POSITIONS", 0),
                cooldown_consecutive_losses=_env_int("COOLDOWN_CONSECUTIVE_LOSSES", 0),
                cooldown_periods=_env_int("COOLDOWN_PERIODS", 3),
                post_kill_cooldown_hours=_env_float("POST_KILL_COOLDOWN_HOURS", 1.0),
            ),
            stock=StockStrategyConfig(
                max_pe_ratio=_env_float("STOCK_MAX_PE", 25.0),
                max_pb_ratio=_env_float("STOCK_MAX_PB", 3.5),
                max_debt_to_equity=_env_float("STOCK_MAX_DEBT_TO_EQUITY", 0.75),
                min_positive_fcf_signal=_env_float("STOCK_MIN_POSITIVE_FCF_SIGNAL", 0.0),
                vix_buy_threshold=_env_float("STOCK_VIX_BUY_THRESHOLD", 20.0),
                run_interval_days=_env_int("STOCK_RUN_INTERVAL_DAYS", 3),
                max_positions=_env_int("STOCK_MAX_POSITIONS", 5),
                vix_symbol=os.getenv("STOCK_VIX_SYMBOL", "^VIX").strip() or "^VIX",
                min_roe=_env_float("STOCK_MIN_ROE", 0.15),
                min_hold_days=_env_int("STOCK_MIN_HOLD_DAYS", 14),
            ),
            crypto=CryptoStrategyConfig(
                symbols=_env_list("CRYPTO_SYMBOLS", ["BTCUSDT", "ETHUSDT", "XRPUSDT"]),
                buy_fear_threshold=_env_int("CRYPTO_BUY_FEAR_THRESHOLD", 35),
                sell_fear_threshold=_env_int("CRYPTO_SELL_FEAR_THRESHOLD", 75),
                use_coinbase_pricing=_env_bool("CRYPTO_USE_COINBASE_PRICING", True),
                require_negative_funding=_env_bool("CRYPTO_REQUIRE_NEGATIVE_FUNDING", False),
                take_profit_pct=_env_float("CRYPTO_TAKE_PROFIT_PCT", 0.15),
                stop_loss_pct=_env_float("CRYPTO_STOP_LOSS_PCT", 0.10),
                symbol_weights=_env_float_map("CRYPTO_SYMBOL_WEIGHTS", {}),
                take_profit_overrides=_env_float_map("CRYPTO_TAKE_PROFIT_OVERRIDES", {}),
                stop_loss_overrides=_env_float_map("CRYPTO_STOP_LOSS_OVERRIDES", {}),
                trend_filter_enabled=_env_bool("CRYPTO_TREND_FILTER_ENABLED", True),
                trend_sma_days=_env_int("CRYPTO_TREND_SMA_DAYS", 20),
                trend_tolerance_pct=_env_float("CRYPTO_TREND_TOLERANCE_PCT", 0.05),
            ),
            coinbase=CoinbaseConfig(
                api_key=os.getenv("COINBASE_API_KEY", "").strip(),
                api_secret=os.getenv("COINBASE_API_SECRET", "").strip(),
                trading_enabled=_env_bool("COINBASE_TRADING_ENABLED", False),
            ),
            telegram=TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
                digest_hours=_env_float("TELEGRAM_DIGEST_HOURS", 6.0),
            ),
            scheduler=SchedulerConfig(
                crypto_interval_minutes=_env_int("SCHEDULER_CRYPTO_INTERVAL_MINUTES", 10),
                btc5min_interval_minutes=int(os.getenv("BTC5MIN_INTERVAL_MINUTES") or os.getenv("SCHEDULER_BTC5MIN_INTERVAL_MINUTES") or "5"),
                stock_check_hour=_env_int("SCHEDULER_STOCK_CHECK_HOUR", 9),
                polymarket_interval_minutes=_env_int("SCHEDULER_POLYMARKET_INTERVAL_MINUTES", 20),
                heartbeat_interval_minutes=_env_int("SCHEDULER_HEARTBEAT_INTERVAL_MINUTES", 60),
                heartbeat_enabled=_env_bool("SCHEDULER_HEARTBEAT_ENABLED", False),
                poll_interval_seconds=_env_int("SCHEDULER_POLL_INTERVAL_SECONDS", 45),
            ),
            polymarket=PolymarketConfig(
                api_key=os.getenv("POLYMARKET_API_KEY", "").strip(),
                api_secret=os.getenv("POLYMARKET_API_SECRET", "").strip(),
                api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", "").strip(),
                private_key=os.getenv("POLYMARKET_PRIVATE_KEY", "").strip(),
            ),
            polymarket_strategy=PolymarketStrategyConfig(
                enabled=_env_bool("POLYMARKET_ENABLED", False),
                market_tags=_env_list_lower("POLYMARKET_MARKET_TAGS", ["crypto", "finance"]),
                min_volume=_env_float("POLYMARKET_MIN_VOLUME", 50_000.0),
                min_liquidity=_env_float("POLYMARKET_MIN_LIQUIDITY", 5_000.0),
                buy_yes_threshold=_env_float("POLYMARKET_BUY_YES_THRESHOLD", 0.35),
                buy_no_threshold=_env_float("POLYMARKET_BUY_NO_THRESHOLD", 0.65),
                take_profit_threshold=_env_float("POLYMARKET_TAKE_PROFIT", 0.50),
                stop_loss_threshold=_env_float("POLYMARKET_STOP_LOSS", 0.15),
                max_positions=_env_int("POLYMARKET_MAX_POSITIONS", 5),
                use_trading_agents=_env_bool("POLYMARKET_USE_TRADING_AGENTS", False),
                use_markov_regime_gate=_env_bool("POLYMARKET_MARKOV_GATE", True),
            ),
            trading_agents=TradingAgentsConfig(
                enabled=_env_bool("TRADING_AGENTS_ENABLED", False),
                llm_provider=os.getenv("TRADING_AGENTS_LLM_PROVIDER", "anthropic").strip(),
                quick_model=os.getenv("TRADING_AGENTS_QUICK_MODEL", "claude-haiku-4-5-20251001").strip(),
                deep_model=os.getenv("TRADING_AGENTS_DEEP_MODEL", "claude-sonnet-4-6").strip(),
                api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
                confirmation_mode=os.getenv("TRADING_AGENTS_CONFIRMATION_MODE", "boost").strip(),
                cache_ttl_hours=_env_int("TRADING_AGENTS_CACHE_TTL_HOURS", 6),
                max_cost_per_run=_env_float("TRADING_AGENTS_MAX_COST_PER_RUN", 0.10),
                max_cost_per_month=_env_float("TRADING_AGENTS_MAX_COST_PER_MONTH", 25.00),
                max_debate_rounds=_env_int("TRADING_AGENTS_MAX_DEBATE_ROUNDS", 1),
                timeout_seconds=_env_int("TRADING_AGENTS_TIMEOUT_SECONDS", 30),
                multi_analyst_enabled=_env_bool("TRADING_AGENTS_MULTI_ANALYST", False),
                selected_analysts=tuple(
                    _env_list_lower(
                        "TRADING_AGENTS_ANALYSTS",
                        ["value_investor", "contrarian", "sentiment_analyst"],
                    )
                ),
                prompt_injection_protection_enabled=_env_bool("TRADING_AGENTS_PROMPT_GUARD_ENABLED", False),
                circuit_breaker_enabled=_env_bool("TRADING_AGENTS_CIRCUIT_BREAKER_ENABLED", True),
                circuit_breaker_failure_threshold=_env_int("TRADING_AGENTS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3),
                circuit_breaker_cooldown_seconds=_env_int("TRADING_AGENTS_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 300),
                streaming_enabled=_env_bool("TRADING_AGENTS_STREAMING_ENABLED", False),
                tool_calling_enabled=_env_bool("TRADING_AGENTS_TOOL_CALLING_ENABLED", False),
                extended_thinking_enabled=_env_bool("TRADING_AGENTS_EXTENDED_THINKING_ENABLED", False),
                thinking_budget_tokens=_env_int("TRADING_AGENTS_THINKING_BUDGET_TOKENS", 1024),
                thinking_display=os.getenv("TRADING_AGENTS_THINKING_DISPLAY", "omitted").strip().lower(),
                vision_enabled=_env_bool("TRADING_AGENTS_VISION_ENABLED", False),
                boost_agree_multiplier=_env_float("TRADING_AGENTS_BOOST_AGREE_MULTIPLIER", 1.5),
                boost_disagree_multiplier=_env_float("TRADING_AGENTS_BOOST_DISAGREE_MULTIPLIER", 0.5),
                boost_min_confidence=_env_float("TRADING_AGENTS_BOOST_MIN_CONFIDENCE", 0.6),
                require_min_confidence=_env_float("TRADING_AGENTS_REQUIRE_MIN_CONFIDENCE", 0.5),
                analyst_buy_threshold=_env_float("TRADING_AGENTS_ANALYST_BUY_THRESHOLD", 0.2),
                analyst_sell_threshold=_env_float("TRADING_AGENTS_ANALYST_SELL_THRESHOLD", -0.2),
                default_input_cost_per_m=_env_float("TRADING_AGENTS_DEFAULT_INPUT_COST_PER_M", 3.0),
                default_output_cost_per_m=_env_float("TRADING_AGENTS_DEFAULT_OUTPUT_COST_PER_M", 15.0),
                estimated_input_tokens_per_call=_env_int("TRADING_AGENTS_ESTIMATED_INPUT_TOKENS_PER_CALL", 1500),
                estimated_output_tokens_per_call=_env_int("TRADING_AGENTS_ESTIMATED_OUTPUT_TOKENS_PER_CALL", 300),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
                model_pricing=_env_model_pricing("TRADING_AGENTS_MODEL_PRICING_JSON", _default_model_pricing()),
            ),
            altcoin_rotation=AltcoinRotationConfig(
                enabled=_env_bool("ROTATION_ENABLED", True),
                scout_multiplier=_env_float("ROTATION_SCOUT_MULTIPLIER", 3.0),
                fee_pct=_env_float("ROTATION_FEE_PCT", 0.006),
                check_interval_minutes=_env_int("ROTATION_CHECK_INTERVAL_MINUTES", 30),
                min_trade_usd=_env_float("ROTATION_MIN_TRADE_USD", 1.0),
                excluded_coins=_env_list("ROTATION_EXCLUDED_COINS", []),
                max_coins_to_track=_env_int("ROTATION_MAX_COINS_TO_TRACK", 50),
                capital_allocation_pct=_env_float("ROTATION_CAPITAL_PCT", 0.60),
                qty_memory_cycles=_env_int("ROTATION_QTY_MEMORY_CYCLES", 336),
            ),
            predictions=PredictionConfig(
                enabled=_env_bool("PREDICTION_ENABLED", False),
                kalshi_key_id=os.getenv("KALSHI_KEY_ID", "").strip(),
                kalshi_private_key=os.getenv("KALSHI_PRIVATE_KEY", "").strip(),
                buy_yes_threshold=_env_float("PREDICTION_BUY_YES_THRESHOLD", 0.45),
                buy_no_threshold=_env_float("PREDICTION_BUY_NO_THRESHOLD", 0.55),
                take_profit_threshold=_env_float("PREDICTION_TAKE_PROFIT", 0.50),
                stop_loss_pct=_env_float("PREDICTION_STOP_LOSS_PCT", 0.15),
                max_positions=_env_int("PREDICTION_MAX_POSITIONS", 8),
                check_interval_minutes=_env_int("PREDICTION_CHECK_INTERVAL_MINUTES", 30),
                min_trade_usd=_env_float("PREDICTION_MIN_TRADE_USD", 1.0),
                min_volume_usd=_env_float("PREDICTION_MIN_VOLUME_USD", 3_000),
                max_spread_usd=_env_float("PREDICTION_MAX_SPREAD_USD", 0.20),
                market_categories=_env_list_lower("PREDICTION_MARKET_CATEGORIES", ["crypto", "economics", "finance", "politics", "weather", "sports", "entertainment"]),
                capital_allocation_pct=_env_float("PREDICTION_CAPITAL_PCT", 0.40),
                require_crypto_fear_greed=_env_bool("PREDICTION_REQUIRE_CRYPTO_FNG", False),
                use_markov_regime_gate=_env_bool("PREDICTION_MARKOV_GATE", True),
                max_pages_per_series=_env_int("PREDICTION_MAX_PAGES_PER_SERIES", 3),
                discover_series=_env_bool("PREDICTION_DISCOVER_SERIES", False),
            ),
            freqtrade=FreqtradeConfig(
                enabled=_env_bool("FREQTRADE_ENABLED", False),
                api_server_port=_env_int("FREQTRADE_API_SERVER_PORT", 8080),
            ),
            latency_arb=LatencyArbConfig(
                enabled=_env_bool("LATENCY_ARB_ENABLED", False),
                symbols=_env_list("LATENCY_ARB_SYMBOLS", ["BTC-USD", "ETH-USD"]),
                momentum_window_seconds=_env_float("LATENCY_ARB_MOMENTUM_WINDOW", 30.0),
                momentum_min_threshold_pct=_env_float("LATENCY_ARB_MOMENTUM_THRESHOLD", 0.30),
                min_edge_pct=_env_float("LATENCY_ARB_MIN_EDGE", 0.05),
                kelly_fraction=_env_float("LATENCY_ARB_KELLY_FRACTION", 0.25),
                max_position_pct=_env_float("LATENCY_ARB_MAX_POSITION_PCT", 0.08),
                max_open_contracts=_env_int("LATENCY_ARB_MAX_OPEN_CONTRACTS", 4),
                scan_interval_seconds=_env_float("LATENCY_ARB_SCAN_INTERVAL", 2.0),
                price_history_seconds=_env_float("LATENCY_ARB_PRICE_HISTORY", 120.0),
                take_profit_pct=_env_float("LATENCY_ARB_TAKE_PROFIT", 0.20),
                stop_loss_pct=_env_float("LATENCY_ARB_STOP_LOSS", 0.15),
                min_trade_usd=_env_float("LATENCY_ARB_MIN_TRADE_USD", 1.0),
                ai_confirmation_mode=os.getenv("LATENCY_ARB_AI_CONFIRMATION", "boost").strip().lower(),
                max_price_age_seconds=_env_float("LATENCY_ARB_MAX_PRICE_AGE_SECONDS", 15.0),
            ),
            volatility=VolatilityConfig(
                enabled=_env_bool("VOL_TARGETING_ENABLED", False),
                target_annual_vol=_env_float("VOL_TARGET_ANNUAL", 0.10),
                vol_window_days=_env_int("VOL_WINDOW_DAYS", 20),
                use_ewma=_env_bool("VOL_USE_EWMA", True),
                ewma_decay=_env_float("VOL_EWMA_DECAY", 0.94),
                max_leverage=_env_float("VOL_MAX_LEVERAGE", 2.0),
                min_allocation=_env_float("VOL_MIN_ALLOCATION", 0.20),
                no_trade_band=_env_float("VOL_NO_TRADE_BAND", 0.10),
                max_history_days=_env_int("VOL_MAX_HISTORY_DAYS", 60),
            ),
            btc5min=BTC5MinConfig(
                enabled=_env_bool("BTC5MIN_ENABLED", True),
                btc_symbol=os.getenv("BTC5MIN_SYMBOL", "BTCUSDT").strip(),
                buy_spread_threshold=_env_float("BTC5MIN_BUY_SPREAD_THRESHOLD", 0.02),
                max_strike_age_hours=_env_float("BTC5MIN_MAX_STRIKE_AGE_HOURS", 48.0),
                min_kalshi_volume_usd=_env_float("BTC5MIN_MIN_KALSHI_VOLUME_USD", 3_000.0),
                min_poly_volume_usd=_env_float("BTC5MIN_MIN_POLY_VOLUME_USD", 50_000.0),
                spot_trading_enabled=_env_bool("BTC5MIN_SPOT_TRADING_ENABLED", True),
            ),
            learning=LearningConfig(
                enabled=_env_bool("LEARNING_ENABLED", True),
                loser_cooldown_losses=_env_int("LEARNING_LOSER_COOLDOWN_LOSSES", 3),
                loser_cooldown_cycles=_env_int("LEARNING_LOSER_COOLDOWN_CYCLES", 10),
            ),
            leads=LeadsConfig(
                enabled=_env_bool("LEADS_NOTIFICATION_ENABLED", True),
                max_per_message=_env_int("LEADS_MAX_PER_MESSAGE", 10),
            ),
            markov=MarkovStrategyConfig(
                enabled=_env_bool("MARKOV_ENABLED", True),
                symbols=_env_list("MARKOV_SYMBOLS", ["BTCUSDT", "ETHUSDT", "XRPUSDT"]),
                regime_symbol=os.getenv("MARKOV_REGIME_SYMBOL", "BTCUSDT").strip(),
                return_window=_env_int("MARKOV_RETURN_WINDOW", 20),
                bull_threshold=_env_float("MARKOV_BULL_THRESHOLD", 0.02),
                bear_threshold=_env_float("MARKOV_BEAR_THRESHOLD", -0.02),
                entry_signal_threshold=_env_float("MARKOV_ENTRY_SIGNAL_THRESHOLD", 0.10),
                exit_signal_threshold=_env_float("MARKOV_EXIT_SIGNAL_THRESHOLD", 0.0),
                stop_loss_pct=_env_float("MARKOV_STOP_LOSS_PCT", 0.10),
                take_profit_pct=_env_float("MARKOV_TAKE_PROFIT_PCT", 0.20),
                min_history_days=_env_int("MARKOV_MIN_HISTORY_DAYS", 40),
                laplace_alpha=_env_float("MARKOV_LAPLACE_ALPHA", 0.1),
                forecast_steps=_env_int("MARKOV_FORECAST_STEPS", 1),
                run_interval_minutes=_env_int("MARKOV_RUN_INTERVAL_MINUTES", 60),
                use_coinbase_pricing=_env_bool("MARKOV_USE_COINBASE_PRICING", True),
            ),
            robinhood=RobinhoodConfig(
                username=os.getenv("ROBINHOOD_USERNAME", "").strip(),
                password=os.getenv("ROBINHOOD_PASSWORD", "").strip(),
                mfa_secret=os.getenv("ROBINHOOD_MFA_SECRET", "").strip(),
            ),
            options=OptionsStrategyConfig(
                enabled=_env_bool("OPTIONS_ENABLED", False),
                symbols=_env_list("OPTIONS_SYMBOLS", ["AAPL", "SPY", "QQQ", "TSLA", "NVDA"]),
                min_dte=_env_int("OPTIONS_MIN_DTE", 21),
                max_dte=_env_int("OPTIONS_MAX_DTE", 60),
                min_delta=_env_float("OPTIONS_MIN_DELTA", 0.70),
                max_positions=_env_int("OPTIONS_MAX_POSITIONS", 5),
                take_profit_pct=_env_float("OPTIONS_TAKE_PROFIT_PCT", 0.50),
                stop_loss_pct=_env_float("OPTIONS_STOP_LOSS_PCT", 0.50),
                close_at_dte=_env_int("OPTIONS_CLOSE_AT_DTE", 7),
                buy_call_fear_threshold=_env_int("OPTIONS_BUY_CALL_FEAR_THRESHOLD", 30),
                buy_put_fear_threshold=_env_int("OPTIONS_BUY_PUT_FEAR_THRESHOLD", 70),
                capital_allocation_pct=_env_float("OPTIONS_CAPITAL_ALLOCATION_PCT", 0.10),
                check_interval_minutes=_env_int("OPTIONS_CHECK_INTERVAL_MINUTES", 60),
                min_volume=_env_int("OPTIONS_MIN_VOLUME", 100),
                min_open_interest=_env_int("OPTIONS_MIN_OPEN_INTEREST", 500),
                max_iv=_env_float("OPTIONS_MAX_IV", 1.50),
            ),
            kalshi_perps=KalshiPerpetualsConfig(
                enabled=_env_bool("KALSHI_PERP_ENABLED", False),
                symbols=_env_list("KALSHI_PERP_SYMBOLS", ["BTC", "ETH", "SOL", "XRP"]),
                max_leverage=_env_float("KALSHI_PERP_MAX_LEVERAGE", 2.0),
                take_profit_pct=_env_float("KALSHI_PERP_TAKE_PROFIT_PCT", 0.05),
                stop_loss_pct=_env_float("KALSHI_PERP_STOP_LOSS_PCT", 0.03),
                trailing_stop_pct=_env_float("KALSHI_PERP_TRAILING_STOP_PCT", 0.0),
                max_positions=_env_int("KALSHI_PERP_MAX_POSITIONS", 4),
                capital_allocation_pct=_env_float("KALSHI_PERP_CAPITAL_PCT", 0.20),
                check_interval_minutes=_env_int("KALSHI_PERP_INTERVAL_MINUTES", 15),
                momentum_window_minutes=_env_int("KALSHI_PERP_MOMENTUM_WINDOW_MINUTES", 60),
                momentum_threshold_pct=_env_float("KALSHI_PERP_MOMENTUM_THRESHOLD_PCT", 0.02),
                min_volume_usd=_env_float("KALSHI_PERP_MIN_VOLUME_USD", 100_000.0),
                min_trade_usd=_env_float("KALSHI_PERP_MIN_TRADE_USD", 10.0),
            ),
            kalshi_sports=KalshiSportsConfig(
                enabled=_env_bool("KALSHI_SPORTS_ENABLED", True),
                leagues=_env_list_lower("KALSHI_SPORTS_LEAGUES", [
                    "nfl", "ncaaf", "ncaaf-fcs", "ncaamb", "ncaawb", "nba", "wnba", "mlb", "nhl",
                    "epl", "laliga", "bundesliga", "seriea", "ligue1", "mls", "ligamx", "ucl", "uel"]),
                series_overrides=_env_list_lower("KALSHI_SPORTS_SERIES", []),
                check_interval_minutes=_env_int("KALSHI_SPORTS_INTERVAL_MINUTES", 30),
                max_hours_ahead=_env_float("KALSHI_SPORTS_MAX_HOURS_AHEAD", 48.0),
                min_minutes_to_start=_env_float("KALSHI_SPORTS_MIN_MINUTES_TO_START", 5.0),
                kelly_fraction=_env_float("KALSHI_SPORTS_KELLY_FRACTION", 0.25),
                max_position_pct=_env_float("KALSHI_SPORTS_MAX_POSITION_PCT", 0.10),
                capital_allocation_pct=_env_float("KALSHI_SPORTS_CAPITAL_PCT", 0.50),
                max_open_bets=_env_int("KALSHI_SPORTS_MAX_OPEN_BETS", 10),
                min_trade_usd=_env_float("KALSHI_SPORTS_MIN_TRADE_USD", 1.0),
                home_advantage=_env_float("KALSHI_SPORTS_HOME_ADVANTAGE", 0.55),
                fee_rate=_env_float("KALSHI_SPORTS_FEE_RATE", 0.07),
                require_market_signal=_env_bool("KALSHI_SPORTS_REQUIRE_MARKET_SIGNAL", True),
                drawdown_throttle=_env_float("KALSHI_SPORTS_DRAWDOWN_THROTTLE", 0.5),
                ignore_global_halt=_env_bool("KALSHI_SPORTS_IGNORE_GLOBAL_HALT", True),
                use_polymarket=_env_bool("KALSHI_SPORTS_USE_POLYMARKET", True),
                parlay_max_legs=_env_int("KALSHI_SPORTS_PARLAY_MAX_LEGS", 4),
                hold_to_settlement=_env_bool("KALSHI_SPORTS_HOLD_TO_SETTLEMENT", True),
                stop_loss_pct=_env_float("KALSHI_SPORTS_STOP_LOSS", 0.0),
                take_profit_pct=_env_float("KALSHI_SPORTS_TAKE_PROFIT", 0.0),
            ),
            protected_symbols=frozenset(
                s.strip().upper()
                for s in os.getenv("PROTECTED_SYMBOLS", "").split(",")
                if s.strip()
            ),
        )
