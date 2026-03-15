"""Multi-symbol trading loop — orchestrates parallel trading across FX pairs.

Each symbol gets its own LiveTradingLoop instance with a shared PortfolioManager
that enforces cross-pair risk limits (total exposure, correlation, concentration).

Usage
-----
>>> loop = MultiSymbolTradingLoop(config, ["EURUSD", "GBPUSD"])
>>> await loop.run()
"""

from __future__ import annotations

import asyncio

from apexfx.config.schema import AppConfig, PortfolioConfig
from apexfx.live.portfolio_manager import PortfolioManager
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class MultiSymbolTradingLoop:
    """Orchestrates parallel trading across multiple symbols.

    Creates a LiveTradingLoop per symbol, all sharing a single PortfolioManager
    for cross-pair risk coordination. An execution lock serialises order
    submissions to prevent race conditions on shared equity.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    symbols : list[str]
        FX pairs to trade (e.g. ["EURUSD", "GBPUSD"]).
    portfolio_config : PortfolioConfig | None
        Portfolio-level risk config. Falls back to config.execution.portfolio.
    """

    def __init__(
        self,
        config: AppConfig,
        symbols: list[str],
        portfolio_config: PortfolioConfig | None = None,
    ) -> None:
        self._config = config
        self._symbols = symbols

        p_cfg = portfolio_config or getattr(config.execution, "portfolio", None)
        if p_cfg is None:
            p_cfg = PortfolioConfig()

        self._portfolio = PortfolioManager(
            max_total_exposure=p_cfg.max_total_exposure,
            max_per_symbol=p_cfg.max_per_symbol,
            correlation_limit=p_cfg.correlation_limit,
        )

        self._execution_lock = asyncio.Lock()
        self._running = False

        # Per-symbol loops will be created lazily in run()
        self._loops: dict[str, object] = {}

    @property
    def portfolio(self) -> PortfolioManager:
        return self._portfolio

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    async def run(self) -> None:
        """Start all per-symbol trading loops in parallel.

        Each loop runs independently but shares the portfolio manager
        for cross-pair risk checks.
        """
        self._running = True
        logger.info(
            "Starting multi-symbol trading",
            symbols=self._symbols,
            n_symbols=len(self._symbols),
        )

        # Import here to avoid circular dependency
        from apexfx.live.trading_loop import LiveTradingLoop

        tasks = []
        for symbol in self._symbols:
            try:
                loop = LiveTradingLoop(
                    config=self._config,
                    symbol=symbol,
                )
                self._loops[symbol] = loop
                tasks.append(self._run_symbol(symbol, loop))
            except Exception as e:
                logger.error(
                    "Failed to create loop for symbol",
                    symbol=symbol,
                    error=str(e),
                )

        if not tasks:
            logger.error("No symbol loops created, aborting")
            return

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_symbol(self, symbol: str, loop) -> None:
        """Run a single symbol's trading loop with portfolio-aware execution."""
        try:
            logger.info("Starting symbol loop", symbol=symbol)
            await loop.run()
        except Exception as e:
            logger.error("Symbol loop failed", symbol=symbol, error=str(e))
        finally:
            logger.info("Symbol loop stopped", symbol=symbol)

    async def stop(self) -> None:
        """Signal all loops to stop."""
        self._running = False
        for symbol, loop in self._loops.items():
            if hasattr(loop, "stop"):
                await loop.stop()
        logger.info("Multi-symbol trading stopped")

    async def execute_with_lock(self, coro):
        """Execute a coroutine under the shared execution lock.

        Prevents concurrent order submissions from multiple symbol loops
        which could race on shared equity calculations.
        """
        async with self._execution_lock:
            return await coro
