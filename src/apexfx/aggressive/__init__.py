"""ApexFX aggressive trading stack — high-level public API.

The package's main entry-point for end users::

    from apexfx.aggressive import run_bot, BotConfig, Mt5LoginConfig, TelegramSettings

    config = BotConfig(
        mt5=Mt5LoginConfig(login=12345678, password="x", server="MetaQuotes-Demo"),
        symbols=["EURUSD", "GBPUSD"],
        risk_per_unit_pct=0.005,
        telegram=TelegramSettings(bot_token="...", chat_id="..."),
    )
    run_bot(config)

For library-level imports (custom orchestration, testing) reach into the
sub-packages directly:

  * ``apexfx.aggressive.exchanges`` — Mt5Client, OkxClient, base Exchange
  * ``apexfx.aggressive.strategies`` — DonchianTurtle, TurtleRunner
  * ``apexfx.aggressive.risk``      — KillSwitch, CircuitBreaker
  * ``apexfx.aggressive.alerts``    — TelegramNotifier
"""

from apexfx.aggressive.config import (
    BotConfig,
    Mt5LoginConfig,
    TelegramSettings,
)
from apexfx.aggressive.live.run_bot import run_bot

__all__ = [
    "BotConfig",
    "Mt5LoginConfig",
    "TelegramSettings",
    "run_bot",
]
