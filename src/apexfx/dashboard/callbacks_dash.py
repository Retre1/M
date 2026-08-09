"""Dash callback registrations for live data updates."""

from __future__ import annotations

import json
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
from dash import html

from apexfx.utils.metrics import compute_all_metrics


def register_callbacks(app: dash.Dash, state_file: str) -> None:
    """Register all dashboard callbacks."""

    def _load_state() -> dict:
        path = Path(state_file)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    @app.callback(
        [
            dash.Output("equity-value", "children"),
            dash.Output("daily-pnl", "children"),
            dash.Output("drawdown-value", "children"),
            dash.Output("winrate-value", "children"),
            dash.Output("equity-chart", "figure"),
            dash.Output("metrics-table", "children"),
            dash.Output("current-position", "children"),
            dash.Output("recent-trades", "children"),
        ],
        [dash.Input("interval-component", "n_intervals")],
    )
    def update_overview(_n):
        state = _load_state()
        if not state:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark", height=400)
            return ["$0", "$0", "0%", "0%", empty_fig, "", "", ""]

        equity = state.get("equity", 100_000)
        state.get("balance", 100_000)
        total_pnl = state.get("total_pnl", 0)
        max_dd = state.get("max_drawdown", 0)
        total_trades = state.get("total_trades", 0)
        winning = state.get("winning_trades", 0)

        winrate = f"{winning / total_trades * 100:.1f}%" if total_trades > 0 else "0%"

        # Equity chart
        eq_curve = state.get("equity_curve", [])
        fig = go.Figure()
        if eq_curve:
            fig.add_trace(go.Scatter(
                y=eq_curve, mode="lines",
                name="Equity", line=dict(color="#00d4aa", width=2),
            ))
        fig.update_layout(
            template="plotly_dark", height=400,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Bars", yaxis_title="Equity ($)",
        )

        # Metrics table
        daily_returns = state.get("daily_returns", [])
        if daily_returns:
            metrics = compute_all_metrics(np.array(daily_returns))
            metrics_html = html.Table([
                html.Tr([html.Td(k.replace("_", " ").title()), html.Td(f"{v:.4f}")])
                for k, v in metrics.items()
            ], className="table table-sm table-dark")
        else:
            metrics_html = html.P("No data yet", className="text-muted")

        # Current position
        pos_dir = state.get("current_position_direction", 0)
        if pos_dir != 0:
            pos_html = html.Div([
                html.P(f"Direction: {'LONG' if pos_dir > 0 else 'SHORT'}"),
                html.P(f"Volume: {state.get('current_position_volume', 0)} lots"),
                html.P(f"Entry: {state.get('current_position_entry_price', 0)}"),
                html.P(f"Unrealized P&L: ${state.get('unrealized_pnl', 0):.2f}"),
            ])
        else:
            pos_html = html.P("No open position", className="text-muted")

        # Recent trades
        trades = state.get("trade_history", [])[-10:]
        if trades:
            rows = []
            for t in reversed(trades):
                pnl_color = "text-success" if t.get("pnl", 0) > 0 else "text-danger"
                rows.append(html.Tr([
                    html.Td(t.get("direction", "")),
                    html.Td(f"{t.get('volume', 0)}"),
                    html.Td(f"{t.get('entry_price', 0):.5f}"),
                    html.Td(f"{t.get('exit_price', 0):.5f}"),
                    html.Td(f"${t.get('pnl', 0):.2f}", className=pnl_color),
                ]))
            trades_html = html.Table([
                html.Thead(html.Tr([
                    html.Th("Dir"), html.Th("Vol"), html.Th("Entry"),
                    html.Th("Exit"), html.Th("P&L"),
                ])),
                html.Tbody(rows),
            ], className="table table-sm table-dark")
        else:
            trades_html = html.P("No trades yet", className="text-muted")

        return [
            f"${equity:,.2f}",
            f"${total_pnl:,.2f}",
            f"{max_dd:.2%}",
            winrate,
            fig,
            metrics_html,
            pos_html,
            trades_html,
        ]

    @app.callback(
        [
            dash.Output("drawdown-chart", "figure"),
        ],
        [dash.Input("interval-component", "n_intervals")],
    )
    def update_risk(_n):
        state = _load_state()
        eq_curve = state.get("equity_curve", [])

        fig = go.Figure()
        if eq_curve and len(eq_curve) > 1:
            eq = np.array(eq_curve)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / peak * 100

            fig.add_trace(go.Scatter(
                y=dd, mode="lines", fill="tozeroy",
                name="Drawdown %", line=dict(color="#ff4444", width=1),
            ))
            fig.add_hline(y=5.0, line_dash="dash", line_color="yellow",
                         annotation_text="Max DD Limit (5%)")

        fig.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            yaxis_title="Drawdown %", yaxis_autorange="reversed",
        )

        return [fig]
