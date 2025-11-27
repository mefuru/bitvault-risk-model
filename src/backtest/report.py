"""
Backtest report generator with visualizations.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.runner import BacktestResults, run_backtest
from src.config import get_project_root
from src.logging_config import get_logger

logger = get_logger("backtest.report")


def create_calibration_chart(results: BacktestResults) -> go.Figure:
    """Create calibration chart comparing predicted vs actual probabilities."""
    metrics = results.calculate_metrics()
    
    thresholds = []
    predicted = []
    actual = []
    
    for thresh, m in metrics['thresholds'].items():
        thresholds.append(f"≥{thresh}%")
        predicted.append(m['avg_predicted_prob'] * 100)
        actual.append(m['actual_frequency'] * 100)
    
    fig = go.Figure()
    
    # Predicted bars
    fig.add_trace(go.Bar(
        name='Predicted',
        x=thresholds,
        y=predicted,
        marker_color='steelblue',
        text=[f"{p:.1f}%" for p in predicted],
        textposition='outside'
    ))
    
    # Actual bars
    fig.add_trace(go.Bar(
        name='Actual',
        x=thresholds,
        y=actual,
        marker_color='coral',
        text=[f"{a:.1f}%" for a in actual],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Model Calibration: Predicted vs Actual Drop Probabilities",
        xaxis_title="Drop Threshold",
        yaxis_title="Probability (%)",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_calibration_scatter(results: BacktestResults) -> go.Figure:
    """Create scatter plot of predicted vs actual (ideal = 45° line)."""
    metrics = results.calculate_metrics()
    
    predicted = []
    actual = []
    labels = []
    
    for thresh, m in metrics['thresholds'].items():
        predicted.append(m['avg_predicted_prob'] * 100)
        actual.append(m['actual_frequency'] * 100)
        labels.append(f"≥{thresh}%")
    
    fig = go.Figure()
    
    # Perfect calibration line
    max_val = max(max(predicted), max(actual)) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))
    
    # Actual points
    fig.add_trace(go.Scatter(
        x=predicted,
        y=actual,
        mode='markers+text',
        name='Thresholds',
        marker=dict(size=15, color='steelblue'),
        text=labels,
        textposition='top center'
    ))
    
    fig.update_layout(
        title="Calibration Plot (Points on line = perfect calibration)",
        xaxis_title="Predicted Probability (%)",
        yaxis_title="Actual Frequency (%)",
        height=400,
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val])
    )
    
    return fig


def create_time_series_chart(results: BacktestResults) -> go.Figure:
    """Create time series of predictions and outcomes."""
    df = results.to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("BTC Price & Actual Drops", "Predicted P(≥20% drop)")
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['btc_price'],
            name='BTC Price',
            line=dict(color='steelblue')
        ),
        row=1, col=1
    )
    
    # Mark actual 20%+ drops
    drops_20 = df[df['actual_20pct'] == True]
    fig.add_trace(
        go.Scatter(
            x=drops_20['date'],
            y=drops_20['btc_price'],
            mode='markers',
            name='≥20% drop occurred',
            marker=dict(size=10, color='red', symbol='x')
        ),
        row=1, col=1
    )
    
    # Predicted probability
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['pred_20pct'] * 100,
            name='P(≥20% drop)',
            line=dict(color='coral'),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Backtest Time Series",
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
    
    return fig


def create_regime_analysis_chart(results: BacktestResults) -> go.Figure:
    """Analyze model performance by regime."""
    df = results.to_dataframe()
    
    # Split by regime
    normal = df[df['regime'] == 'normal']
    stress = df[df['regime'] == 'stress']
    
    thresholds = [5, 10, 15, 20, 25, 30]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Normal Regime", "Stress Regime")
    )
    
    for i, (regime_df, col) in enumerate([(normal, 1), (stress, 2)], 1):
        if len(regime_df) == 0:
            continue
        
        predicted = []
        actual = []
        
        for thresh in thresholds:
            pred_col = f'pred_{thresh}pct'
            actual_col = f'actual_{thresh}pct'
            
            if pred_col in regime_df.columns:
                predicted.append(regime_df[pred_col].mean() * 100)
                actual.append(regime_df[actual_col].mean() * 100)
        
        fig.add_trace(
            go.Bar(name='Predicted', x=[f"≥{t}%" for t in thresholds], y=predicted,
                   marker_color='steelblue', showlegend=(col == 1)),
            row=1, col=col
        )
        fig.add_trace(
            go.Bar(name='Actual', x=[f"≥{t}%" for t in thresholds], y=actual,
                   marker_color='coral', showlegend=(col == 1)),
            row=1, col=col
        )
    
    fig.update_layout(
        title="Calibration by Regime",
        barmode='group',
        height=400
    )
    
    return fig


def generate_report(
    results: BacktestResults,
    output_dir: str = None
) -> str:
    """
    Generate complete backtest report with charts.
    
    Args:
        results: BacktestResults from backtest run
        output_dir: Directory to save report (default: project/reports)
        
    Returns:
        Path to saved HTML report
    """
    if output_dir is None:
        output_dir = get_project_root() / "reports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    metrics = results.calculate_metrics()
    stress_df = results.get_stress_period_performance()
    
    # Generate charts
    cal_chart = create_calibration_chart(results)
    scatter_chart = create_calibration_scatter(results)
    ts_chart = create_time_series_chart(results)
    regime_chart = create_regime_analysis_chart(results)
    
    # Build HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BitVault Risk Model Backtest Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .summary-box {{ background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
            .metric-value {{ font-size: 28px; font-weight: bold; color: #4CAF50; }}
            .metric-label {{ font-size: 14px; color: #777; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .good {{ color: #4CAF50; }}
            .bad {{ color: #f44336; }}
            .warning {{ color: #ff9800; }}
            .chart {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 BitVault Risk Model Backtest Report</h1>
            <p>Generated: {results.run_date}</p>
            <p>Period: {results.config.start_date} to {results.config.end_date}</p>
            
            <div class="summary-box">
                <div class="metric">
                    <div class="metric-value">{metrics['n_points']}</div>
                    <div class="metric-label">Backtest Points</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['overall_brier_score']:.4f}</div>
                    <div class="metric-label">Brier Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'good' if abs(metrics['avg_calibration_error']) < 0.02 else 'warning' if abs(metrics['avg_calibration_error']) < 0.05 else 'bad'}">{metrics['avg_calibration_error']*100:+.2f}%</div>
                    <div class="metric-label">Calibration Error</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results.execution_time_seconds:.1f}s</div>
                    <div class="metric-label">Execution Time</div>
                </div>
            </div>
            
            <h2>📊 Assessment: {metrics['calibration_assessment']}</h2>
            
            <h2>Calibration by Threshold</h2>
            <table>
                <tr>
                    <th>Drop Threshold</th>
                    <th>Predicted Prob</th>
                    <th>Actual Frequency</th>
                    <th>Calibration Error</th>
                    <th>Brier Score</th>
                    <th>Actual Breaches</th>
                </tr>
    """
    
    for thresh, m in metrics['thresholds'].items():
        error_class = 'good' if abs(m['calibration_error']) < 0.02 else 'warning' if abs(m['calibration_error']) < 0.05 else 'bad'
        html += f"""
                <tr>
                    <td>≥{thresh}%</td>
                    <td>{m['avg_predicted_prob']*100:.2f}%</td>
                    <td>{m['actual_frequency']*100:.2f}%</td>
                    <td class="{error_class}">{m['calibration_error']*100:+.2f}%</td>
                    <td>{m['brier_score']:.4f}</td>
                    <td>{m['n_actual_breaches']}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>Calibration Charts</h2>
            <div class="chart" id="cal_chart"></div>
            <div class="chart" id="scatter_chart"></div>
            
            <h2>Time Series Analysis</h2>
            <div class="chart" id="ts_chart"></div>
            
            <h2>Performance by Regime</h2>
            <div class="chart" id="regime_chart"></div>
            
            <h2>Stress Period Analysis</h2>
            <table>
                <tr>
                    <th>Period</th>
                    <th>Dates</th>
                    <th>Points</th>
                    <th>Avg P(≥20%)</th>
                    <th>Actual ≥20%</th>
                    <th>Max Drop</th>
                </tr>
    """
    
    for _, row in stress_df.iterrows():
        pred_20 = f"{row['avg_predicted_20pct_drop']*100:.1f}%" if row['avg_predicted_20pct_drop'] else "N/A"
        actual_20 = f"{row['actual_20pct_drop_freq']*100:.1f}%" if row['actual_20pct_drop_freq'] else "N/A"
        html += f"""
                <tr>
                    <td>{row['period']}</td>
                    <td>{row['start']} to {row['end']}</td>
                    <td>{row['n_points']}</td>
                    <td>{pred_20}</td>
                    <td>{actual_20}</td>
                    <td>{row['max_actual_drop']:.1f}%</td>
                </tr>
        """
    
    html += f"""
            </table>
            
            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>Brier Score:</strong> Measures prediction accuracy (0 = perfect, 1 = worst). Values under 0.1 are good.</li>
                <li><strong>Calibration Error:</strong> Difference between predicted and actual. Positive = overestimates risk, Negative = underestimates risk.</li>
                <li><strong>Good calibration:</strong> Error within ±2%, meaning model probabilities are trustworthy.</li>
            </ul>
            
            <script>
                Plotly.newPlot('cal_chart', {cal_chart.to_json()}.data, {cal_chart.to_json()}.layout);
                Plotly.newPlot('scatter_chart', {scatter_chart.to_json()}.data, {scatter_chart.to_json()}.layout);
                Plotly.newPlot('ts_chart', {ts_chart.to_json()}.data, {ts_chart.to_json()}.layout);
                Plotly.newPlot('regime_chart', {regime_chart.to_json()}.data, {regime_chart.to_json()}.layout);
            </script>
        </div>
    </body>
    </html>
    """
    
    # Save report
    report_path = output_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path.write_text(html)
    
    logger.info(f"Report saved to {report_path}")
    
    return str(report_path)


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    start_date = "2022-01-01"
    end_date = "2025-10-01"
    step_days = 7
    
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            start_date = arg.split("=")[1]
        elif arg.startswith("--end="):
            end_date = arg.split("=")[1]
        elif arg.startswith("--step="):
            step_days = int(arg.split("=")[1])
    
    print("=" * 60)
    print("BitVault Risk Model Backtest")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Step: {step_days} days")
    print()
    
    # Run backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days
    )
    
    # Generate report
    report_path = generate_report(results)
    
    print(f"\n✅ Report saved to: {report_path}")
    print("\nOpen in browser to view charts and detailed analysis.")
