"""Portfolio overview page layout."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def overview_layout() -> html.Div:
    return html.Div([
        # KPI Cards
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Equity", className="card-subtitle text-muted"),
                    html.H3(id="equity-value", children="$100,000"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Daily P&L", className="card-subtitle text-muted"),
                    html.H3(id="daily-pnl", children="$0"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Drawdown", className="card-subtitle text-muted"),
                    html.H3(id="drawdown-value", children="0.0%"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Win Rate", className="card-subtitle text-muted"),
                    html.H3(id="winrate-value", children="0%"),
                ])
            ], color="dark", outline=True), width=3),
        ], className="mb-4"),

        # Equity curve
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Equity Curve"),
                    dbc.CardBody([dcc.Graph(id="equity-chart")]),
                ], color="dark", outline=True),
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody([
                        html.Div(id="metrics-table"),
                    ]),
                ], color="dark", outline=True),
            ], width=4),
        ], className="mb-4"),

        # Current position & recent trades
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Position"),
                    dbc.CardBody([html.Div(id="current-position")]),
                ], color="dark", outline=True),
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Trades"),
                    dbc.CardBody([html.Div(id="recent-trades")]),
                ], color="dark", outline=True),
            ], width=8),
        ]),
    ])
