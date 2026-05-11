"""Single-source-of-truth config for the MT5 trading bot.

Why a Python dataclass instead of YAML/env vars
-----------------------------------------------
1. **Editable in one file.**  You open ``my_demo_bot.py``, fill in your
   credentials, hit run.  No multi-file env-var dance.
2. **Type-checked by your IDE.**  PyCharm/VSCode autocomplete the field
   names.  Misspell ``risk_per_unit_pct`` → red squiggle.
3. **Validated at start.**  ``__post_init__`` rejects nonsense (negative
   risk, exit_period >= entry_period, empty symbol list) before the bot
   touches a broker.  Catches "I typed 15 instead of 0.015" type bugs.
4. **Single object passed to ``run_bot()``** — the function signature
   doesn't grow as we add knobs.

For people who prefer env vars: ``BotConfig.from_env()`` is also provided
(reads ``APEXFX_*`` vars).  Mix and match — pass a ``BotConfig`` with most
fields hard-coded and a few overridden from env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from apexfx.aggressive.risk.circuit_breaker import CircuitBreakerConfig
from apexfx.aggressive.strategies.donchian_turtle import TurtleConfig


# ---------------------------------------------------------------------------
# Sub-configs (re-exported so users can import everything from one place)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mt5LoginConfig:
    """MT5 login details — copy from your broker welcome email.

    To find these in an already-running MT5 terminal:
      * Tools → Options → Server tab shows the server name
      * The account number is in the title bar / Navigator panel
      * Password is the one you set during account creation
    """

    login: int                          # Account number (8-9 digit integer)
    password: str                       # Trading password (NOT investor password)
    server: str                         # Broker server name, e.g. "MetaQuotes-Demo"
    terminal_path: str | None = None    # Path to terminal64.exe; None ⇒ auto-detect

    def __post_init__(self) -> None:
        if self.login <= 0:
            raise ValueError(f"login must be positive int, got {self.login}")
        if not self.password:
            raise ValueError("password must be non-empty")
        if not self.server:
            raise ValueError("server must be non-empty")


@dataclass(frozen=True)
class TelegramSettings:
    """Telegram bot for live alerts — leave fields empty to disable."""

    bot_token: str = ""
    chat_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


# ---------------------------------------------------------------------------
# Main bot config
# ---------------------------------------------------------------------------


@dataclass
class BotConfig:
    """Everything the bot needs to run, in one object.

    Usage::

        config = BotConfig(
            mt5=Mt5LoginConfig(login=12345, password="x", server="Demo"),
            symbols=["EURUSD", "GBPUSD"],
            ...
        )
        run_bot(config)
    """

    # ----- MT5 connection -----
    mt5: Mt5LoginConfig

    # ----- Trading universe -----
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    timeframe: str = "H4"
    deposit_currency: str = "USD"

    # ----- Strategy parameters -----
    # Override individual fields rather than passing a full TurtleConfig
    # so the bot's surface stays one-level-flat for users.
    entry_period: int = 20
    exit_period: int = 10
    ema_period: int = 200
    use_trend_filter: bool = True
    atr_period: int = 20
    risk_per_unit_pct: float = 0.015
    stop_atr_mult: float = 2.0
    pyramid_atr_mult: float = 0.5
    max_units: int = 4

    # ----- Risk limits -----
    daily_loss_pct: float = 0.08
    weekly_loss_pct: float = 0.20
    monthly_dd_pct: float = 0.35
    max_consecutive_failed_orders: int = 3

    # ----- Execution -----
    magic_number: int = 770125          # MT5 magic to tag our orders
    deviation_points: int = 20          # Max slippage in points
    poll_interval_s: float = 60.0       # How often to check for new bars
    reconnect_interval_s: float = 30.0  # Wait before reconnect attempt

    # ----- Notifications -----
    telegram: TelegramSettings = field(default_factory=TelegramSettings)

    # ----- State persistence -----
    kill_switch_path: str = ".kill_switch"
    breaker_state_path: str = ".breaker_state.json"

    # ----- Logging -----
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must contain at least one symbol")
        if self.timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}:
            raise ValueError(f"timeframe {self.timeframe!r} not supported")
        if self.poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be positive, got {self.poll_interval_s}")
        if self.reconnect_interval_s <= 0:
            raise ValueError(f"reconnect_interval_s must be positive, got {self.reconnect_interval_s}")
        # Sub-config validation kicks in here too — TurtleConfig and
        # CircuitBreakerConfig raise on invalid inputs when we build them.
        # We force-build them now to fail fast.
        self.build_turtle_config()
        self.build_breaker_config()

    # ------------------------------------------------------------------
    # Builders for downstream components
    # ------------------------------------------------------------------

    def build_turtle_config(self) -> TurtleConfig:
        return TurtleConfig(
            entry_period=self.entry_period,
            exit_period=self.exit_period,
            ema_period=self.ema_period,
            use_trend_filter=self.use_trend_filter,
            atr_period=self.atr_period,
            risk_per_unit_pct=self.risk_per_unit_pct,
            stop_atr_mult=self.stop_atr_mult,
            pyramid_atr_mult=self.pyramid_atr_mult,
            max_units=self.max_units,
        )

    def build_breaker_config(self) -> CircuitBreakerConfig:
        return CircuitBreakerConfig(
            daily_loss_pct=self.daily_loss_pct,
            weekly_loss_pct=self.weekly_loss_pct,
            monthly_dd_pct=self.monthly_dd_pct,
            max_consecutive_failed_orders=self.max_consecutive_failed_orders,
        )

    # ------------------------------------------------------------------
    # Env-var compatibility (optional)
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, mt5_creds_from_env: bool = True) -> "BotConfig":
        """Populate from APEXFX_* env vars — for users who don't want their
        password in a Python file.  Fields not in env keep their defaults.

        Required when ``mt5_creds_from_env=True``:
          * APEXFX_MT5_LOGIN
          * APEXFX_MT5_PASSWORD
          * APEXFX_MT5_SERVER
        """
        if mt5_creds_from_env:
            login = int(os.environ["APEXFX_MT5_LOGIN"])
            password = os.environ["APEXFX_MT5_PASSWORD"]
            server = os.environ["APEXFX_MT5_SERVER"]
            mt5 = Mt5LoginConfig(login=login, password=password, server=server,
                                 terminal_path=os.environ.get("APEXFX_MT5_PATH") or None)
        else:
            # Caller will set mt5 separately; use a sentinel that fails fast
            raise ValueError(
                "from_env() requires MT5 credentials in env or pass mt5_creds_from_env=False "
                "and set .mt5 on the returned object yourself"
            )

        return cls(
            mt5=mt5,
            symbols=os.environ.get("APEXFX_SYMBOLS", "EURUSD,GBPUSD,USDJPY,AUDUSD").split(","),
            timeframe=os.environ.get("APEXFX_TIMEFRAME", "H4"),
            deposit_currency=os.environ.get("APEXFX_DEPOSIT_CCY", "USD"),
            risk_per_unit_pct=float(os.environ.get("APEXFX_RISK_PER_UNIT", "0.015")),
            max_units=int(os.environ.get("APEXFX_MAX_UNITS", "4")),
            daily_loss_pct=float(os.environ.get("APEXFX_BREAKER_DAILY_PCT", "0.08")),
            telegram=TelegramSettings(
                bot_token=os.environ.get("APEXFX_TELEGRAM_TOKEN", ""),
                chat_id=os.environ.get("APEXFX_TELEGRAM_CHAT_ID", ""),
            ),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary — printed at startup."""
        symbols_str = ", ".join(self.symbols)
        return (
            f"MT5 login={self.mt5.login}@{self.mt5.server} "
            f"timeframe={self.timeframe} symbols=[{symbols_str}] "
            f"risk={self.risk_per_unit_pct:.2%} max_units={self.max_units} "
            f"telegram={'on' if self.telegram.enabled else 'off'}"
        )
