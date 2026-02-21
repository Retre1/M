"""Launch the monitoring dashboard."""

from __future__ import annotations

import argparse

from apexfx.config.registry import init_config
from apexfx.dashboard.app import create_app
from apexfx.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ApexFX Quantum Dashboard")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = init_config(args.config_dir)
    setup_logging(level="INFO", fmt="console")

    app = create_app(
        state_file="data/portfolio_state.json",
        refresh_interval_s=config.dashboard.refresh_interval_s,
        theme=config.dashboard.theme,
    )

    host = args.host or config.dashboard.host
    port = args.port or config.dashboard.port

    print(f"\n  ApexFX Quantum Dashboard running at http://{host}:{port}\n")
    app.run(host=host, port=port, debug=config.dashboard.debug)


if __name__ == "__main__":
    main()
