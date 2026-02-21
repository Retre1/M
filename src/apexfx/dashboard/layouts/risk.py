"""Risk monitoring page layout."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def risk_layout() -> html.Div:
    return html.Div([
        # Risk gauges
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Daily VaR", className="card-subtitle text-muted"),
                    html.H3(id="var-value", children="0.0%"),
                    html.Small(id="var-limit", children="Limit: 2.0%", className="text-muted"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Current Drawdown", className="card-subtitle text-muted"),
                    html.H3(id="risk-drawdown", children="0.0%"),
                    html.Small(id="dd-limit", children="Limit: 5.0%", className="text-muted"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Cooldown Status", className="card-subtitle text-muted"),
                    html.H3(id="cooldown-status", children="Clear"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Position Size", className="card-subtitle text-muted"),
                    html.H3(id="position-size-risk", children="0.0%"),
                    html.Small("of portfolio", className="text-muted"),
                ])
            ], color="dark", outline=True), width=3),
        ], className="mb-4"),

        # Drawdown chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Drawdown History"),
                    dbc.CardBody([dcc.Graph(id="drawdown-chart")]),
                ], color="dark", outline=True),
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Spread Monitor"),
                    dbc.CardBody([dcc.Graph(id="spread-chart")]),
                ], color="dark", outline=True),
            ], width=4),
        ], className="mb-4"),

        # Execution quality
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Execution Quality"),
                    dbc.CardBody([html.Div(id="execution-quality")]),
                ], color="dark", outline=True),
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Events Log"),
                    dbc.CardBody([html.Div(id="risk-events-log")]),
                ], color="dark", outline=True),
            ], width=6),
        ]),
    ])
