"""Training progress page layout."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def training_layout() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Training Stage", className="card-subtitle text-muted"),
                    html.H3(id="training-stage", children="N/A"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total Timesteps", className="card-subtitle text-muted"),
                    html.H3(id="total-timesteps", children="0"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Mean Episode Return", className="card-subtitle text-muted"),
                    html.H3(id="mean-return", children="0.0"),
                ])
            ], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Best OOS Sharpe", className="card-subtitle text-muted"),
                    html.H3(id="best-sharpe", children="0.0"),
                ])
            ], color="dark", outline=True), width=3),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Episode Reward Curve"),
                    dbc.CardBody([dcc.Graph(id="reward-curve")]),
                ], color="dark", outline=True),
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Curriculum Progress"),
                    dbc.CardBody([html.Div(id="curriculum-progress")]),
                ], color="dark", outline=True),
            ], width=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Walk-Forward Fold Results"),
                    dbc.CardBody([html.Div(id="walkforward-results")]),
                ], color="dark", outline=True),
            ], width=12),
        ]),
    ])
