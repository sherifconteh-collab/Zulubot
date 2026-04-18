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


@dataclass(frozen=True)
class CoinbaseConfig:
    api_key: str = ""
    api_secret: str = ""


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""

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
    circuit_breaker_enabled: bool = False
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
    market_categories: list[str] = field(default_factory=lambda: ["crypto", "economics", "finance", "politics", "weather"])
    capital_allocation_pct: float = 0.40 # use 40% of total capital for predictions
    # Opt-in extra gate for crypto-category markets: require an extreme Fear &
    # Greed reading to align with the price threshold. Default False so crypto
    # predictions trade on price thresholds alone — previously this gate kept
    # the Kalshi strategy idle outside of extreme sentiment regimes.
    require_crypto_fear_greed: bool = False


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
                symbols=_env_list("CRYPTO_SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
                buy_fear_threshold=_env_int("CRYPTO_BUY_FEAR_THRESHOLD", 35),
                sell_fear_threshold=_env_int("CRYPTO_SELL_FEAR_THRESHOLD", 75),
                use_coinbase_pricing=_env_bool("CRYPTO_USE_COINBASE_PRICING", True),
                require_negative_funding=_env_bool("CRYPTO_REQUIRE_NEGATIVE_FUNDING", False),
            ),
            coinbase=CoinbaseConfig(
                api_key=os.getenv("COINBASE_API_KEY", "").strip(),
                api_secret=os.getenv("COINBASE_API_SECRET", "").strip(),
            ),
            telegram=TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
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
                circuit_breaker_enabled=_env_bool("TRADING_AGENTS_CIRCUIT_BREAKER_ENABLED", False),
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
                market_categories=_env_list_lower("PREDICTION_MARKET_CATEGORIES", ["crypto", "economics", "finance", "politics", "weather"]),
                capital_allocation_pct=_env_float("PREDICTION_CAPITAL_PCT", 0.40),
                require_crypto_fear_greed=_env_bool("PREDICTION_REQUIRE_CRYPTO_FNG", False),
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
        )
