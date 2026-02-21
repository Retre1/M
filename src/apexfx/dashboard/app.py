"""Dash application entry point — monitoring dashboard for ApexFX Quantum."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from apexfx.dashboard.callbacks_dash import register_callbacks
from apexfx.dashboard.layouts.overview import overview_layout
from apexfx.dashboard.layouts.risk import risk_layout
from apexfx.dashboard.layouts.signals import signals_layout
from apexfx.dashboard.layouts.training import training_layout


def create_app(
    state_file: str = "data/portfolio_state.json",
    refresh_interval_s: int = 5,
    theme: str = "darkly",
) -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[getattr(dbc.themes, theme.upper(), dbc.themes.DARKLY)],
        suppress_callback_exceptions=True,
    )

    app.title = "ApexFX Quantum Dashboard"

    # Navigation
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("ApexFX Quantum", className="ms-2", style={"fontSize": "1.4em"}),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Signals", href="/signals")),
                dbc.NavItem(dbc.NavLink("Risk", href="/risk")),
                dbc.NavItem(dbc.NavLink("Training", href="/training")),
            ], className="ms-auto", navbar=True),
        ]),
        color="dark",
        dark=True,
        className="mb-4",
    )

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="interval-component", interval=refresh_interval_s * 1000, n_intervals=0),
        dcc.Store(id="state-store"),
        navbar,
        dbc.Container(id="page-content", fluid=True),
    ])

    # Page routing
    @app.callback(
        dash.Output("page-content", "children"),
        dash.Input("url", "pathname"),
    )
    def display_page(pathname: str):
        if pathname == "/signals":
            return signals_layout()
        elif pathname == "/risk":
            return risk_layout()
        elif pathname == "/training":
            return training_layout()
        return overview_layout()

    # Register data update callbacks
    register_callbacks(app, state_file)

    return app
