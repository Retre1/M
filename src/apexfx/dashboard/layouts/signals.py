"""Signal visualization page layout."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def signals_layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Price Chart with Signals"),
                    dbc.CardBody([dcc.Graph(id="price-chart")]),
                ], color="dark", outline=True),
            ], width=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Agent Actions"),
                    dbc.CardBody([dcc.Graph(id="agent-actions-chart")]),
                ], color="dark", outline=True),
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Gating Weights"),
                    dbc.CardBody([dcc.Graph(id="gating-weights-chart")]),
                ], color="dark", outline=True),
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Market Regime"),
                    dbc.CardBody([dcc.Graph(id="regime-chart")]),
                ], color="dark", outline=True),
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Variable Importance (TFT)"),
                    dbc.CardBody([dcc.Graph(id="variable-importance-chart")]),
                ], color="dark", outline=True),
            ], width=6),
        ]),
    ])
