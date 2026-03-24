"""HTML report generator for backtest results.

Generates a self-contained HTML file with interactive charts:
- Equity curve + drawdown
- Monthly returns heatmap
- Trade distribution
- Risk rejection analysis
- Performance summary table
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from apexfx.backtest.result import BacktestResult
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def generate_html_report(
    result: BacktestResult,
    output_path: str | Path = "backtest_report.html",
    title: str = "ApexFX Quantum — Backtest Report",
) -> str:
    """Generate a self-contained HTML backtest report.

    Args:
        result: Completed BacktestResult with computed metrics.
        output_path: Where to save the HTML file.
        title: Report title.

    Returns:
        Path to the generated HTML file.
    """
    if not result.metrics:
        result.compute_metrics()

    m = result.metrics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    equity_data = _equity_json(result)
    drawdown_data = _drawdown_json(result)
    monthly_data = _monthly_returns_json(result)
    trade_pnl_data = _trade_pnl_json(result)
    exit_reasons = _exit_reasons_json(result)
    rejection_data = _rejection_json(result)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149; --orange: #d29922;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background:var(--bg); color:var(--text); }}
  .container {{ max-width:1200px; margin:0 auto; padding:20px; }}
  header {{ text-align:center; padding:30px 0 20px; border-bottom:1px solid var(--border); margin-bottom:24px; }}
  header h1 {{ color:var(--accent); font-size:1.8em; }}
  header .sub {{ color:var(--muted); font-size:0.9em; margin-top:6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; text-align:center; }}
  .kpi .val {{ font-size:1.8em; font-weight:700; }}
  .kpi .lbl {{ font-size:0.8em; color:var(--muted); margin-top:4px; }}
  .pos {{ color:var(--green); }}
  .neg {{ color:var(--red); }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:20px; margin-bottom:16px; }}
  .card h3 {{ color:var(--accent); margin-bottom:12px; }}
  canvas {{ max-height:300px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.85em; }}
  th,td {{ padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); }}
  th {{ color:var(--accent); background:rgba(88,166,255,0.08); }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:768px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  .summary-text {{ white-space:pre; font-family:'Courier New',monospace; font-size:0.82em; line-height:1.5; color:var(--muted); background:rgba(0,0,0,0.3); padding:16px; border-radius:6px; overflow-x:auto; }}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>{title}</h1>
  <div class="sub">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | {m.get('duration_days',0):.0f} days | {m.get('total_trades',0):.0f} trades</div>
</header>

<!-- KPIs -->
<div class="grid">
  <div class="kpi"><div class="val {'pos' if m.get('total_return_pct',0)>=0 else 'neg'}">{m.get('total_return_pct',0):+.1f}%</div><div class="lbl">Total Return</div></div>
  <div class="kpi"><div class="val">{m.get('sharpe_ratio',0):.2f}</div><div class="lbl">Sharpe Ratio</div></div>
  <div class="kpi"><div class="val">{m.get('sortino_ratio',0):.2f}</div><div class="lbl">Sortino Ratio</div></div>
  <div class="kpi"><div class="val neg">{m.get('max_drawdown_pct',0):.1f}%</div><div class="lbl">Max Drawdown</div></div>
  <div class="kpi"><div class="val">{m.get('profit_factor',0):.2f}</div><div class="lbl">Profit Factor</div></div>
  <div class="kpi"><div class="val">{m.get('win_rate',0):.0f}%</div><div class="lbl">Win Rate</div></div>
  <div class="kpi"><div class="val">{m.get('total_trades',0):.0f}</div><div class="lbl">Trades</div></div>
  <div class="kpi"><div class="val">{m.get('annual_return_pct',0):+.1f}%</div><div class="lbl">Annual Return</div></div>
</div>

<!-- Equity Curve -->
<div class="card">
  <h3>Equity Curve</h3>
  <canvas id="equityChart"></canvas>
</div>

<!-- Drawdown -->
<div class="card">
  <h3>Drawdown</h3>
  <canvas id="drawdownChart"></canvas>
</div>

<div class="grid2">
  <!-- Trade P&L Distribution -->
  <div class="card">
    <h3>Trade P&L Distribution</h3>
    <canvas id="pnlChart"></canvas>
  </div>
  <!-- Exit Reasons -->
  <div class="card">
    <h3>Exit Reasons</h3>
    <canvas id="exitChart"></canvas>
  </div>
</div>

<div class="grid2">
  <!-- Risk Rejections -->
  <div class="card">
    <h3>Risk Rejections ({m.get('risk_rejections',0):.0f} total, {m.get('risk_rejection_rate',0):.1f}%)</h3>
    <canvas id="rejectionChart"></canvas>
  </div>
  <!-- Trade Stats -->
  <div class="card">
    <h3>Trade Statistics</h3>
    <table>
      <tr><td>Avg Winner</td><td class="pos">${m.get('avg_winner',0):,.2f}</td></tr>
      <tr><td>Avg Loser</td><td class="neg">${m.get('avg_loser',0):,.2f}</td></tr>
      <tr><td>Largest Winner</td><td class="pos">${m.get('largest_winner',0):,.2f}</td></tr>
      <tr><td>Largest Loser</td><td class="neg">${m.get('largest_loser',0):,.2f}</td></tr>
      <tr><td>Expectancy</td><td>${m.get('expectancy',0):,.2f}</td></tr>
      <tr><td>Avg Bars Held</td><td>{m.get('avg_bars_held',0):.1f}</td></tr>
      <tr><td>Long / Short</td><td>{m.get('long_trades',0):.0f} / {m.get('short_trades',0):.0f}</td></tr>
      <tr><td>Time in Market</td><td>{m.get('time_in_market_pct',0):.1f}%</td></tr>
      <tr><td>Avg Exposure</td><td>{m.get('avg_exposure_pct',0):.1f}%</td></tr>
      <tr><td>Calmar Ratio</td><td>{m.get('calmar_ratio',0):.2f}</td></tr>
      <tr><td>Annual Volatility</td><td>{m.get('annual_volatility_pct',0):.1f}%</td></tr>
    </table>
  </div>
</div>

<!-- Full Summary -->
<div class="card">
  <h3>Full Summary</h3>
  <div class="summary-text">{result.summary()}</div>
</div>

</div>

<script>
const darkGrid = {{ color: 'rgba(48,54,61,0.5)' }};
const darkTick = {{ color: '#8b949e' }};
const defaultOpts = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
  scales: {{
    x: {{ grid: darkGrid, ticks: darkTick }},
    y: {{ grid: darkGrid, ticks: darkTick }}
  }}
}};

// Equity curve
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {equity_data['labels']},
    datasets: [{{ label: 'Equity', data: {equity_data['values']}, borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0, fill: false }}]
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, legend: {{ display: false }} }} }}
}});

// Drawdown
new Chart(document.getElementById('drawdownChart'), {{
  type: 'line',
  data: {{
    labels: {drawdown_data['labels']},
    datasets: [{{ label: 'Drawdown %', data: {drawdown_data['values']}, borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.1)', borderWidth: 1.5, pointRadius: 0, fill: true }}]
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, legend: {{ display: false }} }} }}
}});

// Trade P&L distribution
new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: {trade_pnl_data['labels']},
    datasets: [{{ label: 'P&L', data: {trade_pnl_data['values']}, backgroundColor: {trade_pnl_data['colors']} }}]
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, legend: {{ display: false }} }} }}
}});

// Exit reasons pie
new Chart(document.getElementById('exitChart'), {{
  type: 'doughnut',
  data: {{
    labels: {exit_reasons['labels']},
    datasets: [{{ data: {exit_reasons['values']}, backgroundColor: ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff'] }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }} }}
}});

// Risk rejections
new Chart(document.getElementById('rejectionChart'), {{
  type: 'bar',
  data: {{
    labels: {rejection_data['labels']},
    datasets: [{{ label: 'Rejections', data: {rejection_data['values']}, backgroundColor: '#d29922' }}]
  }},
  options: {{ ...defaultOpts, indexAxis: 'y', plugins: {{ ...defaultOpts.plugins, legend: {{ display: false }} }} }}
}});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Backtest report generated", path=str(output_path))
    return str(output_path)


# ------------------------------------------------------------------
# Data formatters for Chart.js
# ------------------------------------------------------------------

def _equity_json(result: BacktestResult) -> dict:
    step = max(1, len(result.equity_curve) // 500)
    labels = [f"'{e[0].strftime('%Y-%m-%d')}'" for e in result.equity_curve[::step]]
    values = [round(e[1], 2) for e in result.equity_curve[::step]]
    return {"labels": f"[{','.join(labels)}]", "values": str(values)}


def _drawdown_json(result: BacktestResult) -> dict:
    dd = result.drawdown_info.drawdown_series
    step = max(1, len(dd) // 500)
    labels = [f"'{result.equity_curve[i][0].strftime('%Y-%m-%d')}'" for i in range(0, len(dd), step) if i < len(result.equity_curve)]
    values = [round(-dd[i] * 100, 2) for i in range(0, len(dd), step)]
    return {"labels": f"[{','.join(labels)}]", "values": str(values)}


def _trade_pnl_json(result: BacktestResult) -> dict:
    if not result.trades:
        return {"labels": "[]", "values": "[]", "colors": "[]"}
    pnls = sorted([t.pnl for t in result.trades])
    # Bin into ~20 buckets
    if len(pnls) > 1:
        bins = np.linspace(min(pnls), max(pnls), min(20, len(pnls)))
        hist, edges = np.histogram(pnls, bins=bins)
        labels = [f"'{edges[i]:.0f}'" for i in range(len(hist))]
        values = hist.tolist()
        colors = [f"'{'#3fb950' if edges[i] >= 0 else '#f85149'}'" for i in range(len(hist))]
    else:
        labels = [f"'{pnls[0]:.0f}'"]
        values = [1]
        colors = ["'#3fb950'" if pnls[0] >= 0 else "'#f85149'"]
    return {"labels": f"[{','.join(labels)}]", "values": str(values), "colors": f"[{','.join(colors)}]"}


def _exit_reasons_json(result: BacktestResult) -> dict:
    if not result.trades:
        return {"labels": "[]", "values": "[]"}
    reasons: dict[str, int] = {}
    for t in result.trades:
        r = t.exit_reason or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    labels = [f"'{k}'" for k in reasons]
    values = list(reasons.values())
    return {"labels": f"[{','.join(labels)}]", "values": str(values)}


def _rejection_json(result: BacktestResult) -> dict:
    if not result.risk_rejection_reasons:
        return {"labels": "['None']", "values": "[0]"}
    # Top 10
    sorted_reasons = sorted(result.risk_rejection_reasons.items(), key=lambda x: -x[1])[:10]
    labels = [f"'{r[0][:30]}'" for r in sorted_reasons]
    values = [r[1] for r in sorted_reasons]
    return {"labels": f"[{','.join(labels)}]", "values": str(values)}


def _monthly_returns_json(result: BacktestResult) -> dict:
    if result.monthly_returns is None or result.monthly_returns.empty:
        return {"labels": "[]", "values": "[]"}
    labels = [f"'{idx.strftime('%Y-%m')}'" for idx in result.monthly_returns.index]
    values = [round(v * 100, 2) for v in result.monthly_returns["return"].values]
    return {"labels": f"[{','.join(labels)}]", "values": str(values)}
