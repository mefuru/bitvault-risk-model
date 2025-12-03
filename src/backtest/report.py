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
    Generate complete backtest report with charts and detailed explanations.
    
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
    regime_breakdown = results.calculate_regime_breakdown()
    
    # Generate charts
    cal_chart = create_calibration_chart(results)
    scatter_chart = create_calibration_scatter(results)
    ts_chart = create_time_series_chart(results)
    regime_chart = create_regime_analysis_chart(results)
    
    # Calculate additional statistics for explanations
    df = results.to_dataframe()
    stress_count = sum(1 for p in results.points if p.regime == 'stress')
    normal_count = len(results.points) - stress_count
    
    # Determine overall assessment
    if metrics['avg_calibration_error'] > 0.10:
        bias_assessment = "significantly conservative"
        bias_implication = "The model provides a substantial safety margin. Probabilities can be interpreted as upper bounds."
    elif metrics['avg_calibration_error'] > 0.05:
        bias_assessment = "moderately conservative"
        bias_implication = "The model errs on the side of caution, which is appropriate for risk management."
    elif metrics['avg_calibration_error'] > 0.02:
        bias_assessment = "slightly conservative"
        bias_implication = "The model has a small safety margin built in."
    elif metrics['avg_calibration_error'] > -0.02:
        bias_assessment = "well-calibrated"
        bias_implication = "Model probabilities closely match historical outcomes."
    elif metrics['avg_calibration_error'] > -0.05:
        bias_assessment = "slightly aggressive"
        bias_implication = "The model may slightly underestimate risk. Consider adding a safety buffer."
    else:
        bias_assessment = "significantly aggressive"
        bias_implication = "⚠️ The model underestimates risk. Probabilities should be treated as lower bounds."
    
    # Brier score assessment
    if metrics['overall_brier_score'] < 0.05:
        brier_assessment = "Excellent"
        brier_color = "#4CAF50"
    elif metrics['overall_brier_score'] < 0.10:
        brier_assessment = "Good"
        brier_color = "#8BC34A"
    elif metrics['overall_brier_score'] < 0.15:
        brier_assessment = "Acceptable"
        brier_color = "#FF9800"
    else:
        brier_assessment = "Poor"
        brier_color = "#f44336"
    
    # Build HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BitVault Risk Model Backtest Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px; background: #f5f5f5; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 15px; margin-bottom: 30px; }}
            h2 {{ color: #444; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; }}
            h3 {{ color: #555; margin-top: 25px; }}
            .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin: 30px 0; color: white; }}
            .metric {{ display: inline-block; margin: 15px 30px; text-align: center; }}
            .metric-value {{ font-size: 32px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; font-weight: 600; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .good {{ color: #4CAF50; font-weight: bold; }}
            .bad {{ color: #f44336; font-weight: bold; }}
            .warning {{ color: #ff9800; font-weight: bold; }}
            .chart {{ margin: 30px 0; }}
            .explanation-box {{ background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 20px; margin: 20px 0; border-radius: 0 5px 5px 0; }}
            .warning-box {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 20px; margin: 20px 0; border-radius: 0 5px 5px 0; }}
            .info-box {{ background: #e3f2fd; border-left: 4px solid #2196F3; padding: 20px; margin: 20px 0; border-radius: 0 5px 5px 0; }}
            .key-finding {{ background: #f3e5f5; border-left: 4px solid #9c27b0; padding: 20px; margin: 20px 0; border-radius: 0 5px 5px 0; }}
            .toc {{ background: #fafafa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .toc ul {{ columns: 2; -webkit-columns: 2; -moz-columns: 2; }}
            .toc li {{ margin: 5px 0; }}
            .toc a {{ color: #4CAF50; text-decoration: none; }}
            .toc a:hover {{ text-decoration: underline; }}
            .exec-summary {{ font-size: 18px; color: #333; margin: 30px 0; }}
            code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
            .formula {{ background: #f9f9f9; padding: 15px; border-radius: 5px; font-family: 'Consolas', monospace; margin: 15px 0; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 BitVault Risk Model Backtest Report</h1>
            
            <p><strong>Generated:</strong> {results.run_date}</p>
            <p><strong>Backtest Period:</strong> {results.config.start_date} to {results.config.end_date}</p>
            <p><strong>Methodology:</strong> Rolling {results.config.calibration_years}-year calibration window, {results.config.horizon_days}-day forward simulation, {results.config.n_paths:,} Monte Carlo paths</p>
            
            <div class="toc">
                <h3>📋 Contents</h3>
                <ul>
                    <li><a href="#executive-summary">Executive Summary</a></li>
                    <li><a href="#key-metrics">Key Metrics Explained</a></li>
                    <li><a href="#calibration">Calibration Analysis</a></li>
                    <li><a href="#charts">Visual Analysis</a></li>
                    <li><a href="#regime">Regime Analysis</a></li>
                    <li><a href="#stress">Stress Period Performance</a></li>
                    <li><a href="#implications">Practical Implications</a></li>
                    <li><a href="#methodology">Methodology Details</a></li>
                </ul>
            </div>
            
            <div class="summary-box">
                <div class="metric">
                    <div class="metric-value">{metrics['n_points']}</div>
                    <div class="metric-label">Backtest Points</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['overall_brier_score']:.4f}</div>
                    <div class="metric-label">Brier Score ({brier_assessment})</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['avg_calibration_error']*100:+.1f}%</div>
                    <div class="metric-label">Calibration Error</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stress_count}/{len(results.points)}</div>
                    <div class="metric-label">Stress Periods</div>
                </div>
            </div>
            
            <h2 id="executive-summary">📊 Executive Summary</h2>
            
            <div class="exec-summary">
                <p>The backtest evaluated <strong>{metrics['n_points']} weekly predictions</strong> over approximately {(metrics['n_points'] * results.config.step_days) // 365} years. 
                The model is <strong>{bias_assessment}</strong> with a Brier score of <strong>{metrics['overall_brier_score']:.4f}</strong> ({brier_assessment}).</p>
            </div>
            
            <div class="key-finding">
                <h3>🎯 Key Finding: {metrics['calibration_assessment']}</h3>
                <p>{bias_implication}</p>
                <p>On average, when the model predicts an X% probability of a price drop, the actual frequency is approximately <strong>{max(0, (1 - metrics['avg_calibration_error']) * 100):.0f}%</strong> of that prediction.</p>
            </div>
            
            <h2 id="key-metrics">📈 Key Metrics Explained</h2>
            
            <h3>Brier Score: {metrics['overall_brier_score']:.4f}</h3>
            <div class="info-box">
                <p><strong>What it measures:</strong> Overall prediction accuracy, combining both calibration and discrimination.</p>
                <div class="formula">Brier Score = (1/N) × Σ(predicted probability - actual outcome)²</div>
                <p><strong>How to interpret:</strong></p>
                <ul>
                    <li><strong>0.00:</strong> Perfect predictions</li>
                    <li><strong>&lt; 0.05:</strong> Excellent accuracy</li>
                    <li><strong>0.05 - 0.10:</strong> Good accuracy ✓</li>
                    <li><strong>0.10 - 0.15:</strong> Acceptable accuracy</li>
                    <li><strong>&gt; 0.15:</strong> Poor accuracy, model needs improvement</li>
                </ul>
                <p><strong>Your result ({metrics['overall_brier_score']:.4f}):</strong> <span style="color: {brier_color}; font-weight: bold;">{brier_assessment}</span> - The model makes reasonably accurate probabilistic predictions.</p>
            </div>
            
            <h3>Calibration Error: {metrics['avg_calibration_error']*100:+.2f}%</h3>
            <div class="info-box">
                <p><strong>What it measures:</strong> The systematic bias in predictions (predicted probability minus actual frequency).</p>
                <div class="formula">Calibration Error = Average Predicted Probability - Actual Frequency</div>
                <p><strong>How to interpret:</strong></p>
                <ul>
                    <li><strong>Positive error (+):</strong> Model overestimates risk (conservative) - predicts higher probabilities than actually occur</li>
                    <li><strong>Negative error (-):</strong> Model underestimates risk (aggressive) - predicts lower probabilities than actually occur</li>
                    <li><strong>Within ±2%:</strong> Well-calibrated</li>
                    <li><strong>Within ±5%:</strong> Acceptable calibration</li>
                    <li><strong>Beyond ±10%:</strong> Significant bias</li>
                </ul>
                <p><strong>Your result ({metrics['avg_calibration_error']*100:+.2f}%):</strong> {bias_implication}</p>
            </div>
            
            <h2 id="calibration">📉 Calibration by Drop Threshold</h2>
            
            <div class="explanation-box">
                <p><strong>How to read this table:</strong> For each drop threshold (e.g., ≥10%), we compare what the model predicted on average versus how often that drop actually occurred. The calibration error shows the difference.</p>
                <p>For example, if the model predicted a 40% chance of a ≥10% drop, but it only happened 22% of the time, the calibration error is +18% (overestimate).</p>
            </div>
            
            <table>
                <tr>
                    <th>Drop Threshold</th>
                    <th>Model Predicted</th>
                    <th>Actually Occurred</th>
                    <th>Calibration Error</th>
                    <th>Brier Score</th>
                    <th>Times Occurred</th>
                    <th>Interpretation</th>
                </tr>
    """
    
    for thresh, m in metrics['thresholds'].items():
        error_class = 'good' if abs(m['calibration_error']) < 0.02 else 'warning' if abs(m['calibration_error']) < 0.10 else 'bad'
        
        # Generate interpretation for each threshold
        if m['calibration_error'] > 0.10:
            interpretation = f"Model significantly overestimates ≥{thresh}% drops"
        elif m['calibration_error'] > 0.02:
            interpretation = f"Model slightly overestimates ≥{thresh}% drops"
        elif m['calibration_error'] > -0.02:
            interpretation = "Well-calibrated for this threshold"
        elif m['calibration_error'] > -0.10:
            interpretation = f"Model slightly underestimates ≥{thresh}% drops"
        else:
            interpretation = f"⚠️ Model significantly underestimates ≥{thresh}% drops"
        
        html += f"""
                <tr>
                    <td><strong>≥{thresh}%</strong></td>
                    <td>{m['avg_predicted_prob']*100:.1f}%</td>
                    <td>{m['actual_frequency']*100:.1f}%</td>
                    <td class="{error_class}">{m['calibration_error']*100:+.1f}%</td>
                    <td>{m['brier_score']:.4f}</td>
                    <td>{m['n_actual_breaches']} / {metrics['n_points']}</td>
                    <td><em>{interpretation}</em></td>
                </tr>
        """
    
    html += f"""
            </table>
            
            <div class="{'warning-box' if metrics['avg_calibration_error'] > 0.05 else 'explanation-box'}">
                <h4>{'⚠️' if metrics['avg_calibration_error'] > 0.05 else '✓'} What This Means in Practice</h4>
                <p>When the model reports a probability, you can adjust your expectations as follows:</p>
                <table>
                    <tr><th>Model Says</th><th>Historical Reality (approx.)</th></tr>
                    <tr><td>60% chance of ≥5% drop</td><td>~{max(0, 60 * (1 - metrics['avg_calibration_error'])):.0f}% actual</td></tr>
                    <tr><td>40% chance of ≥10% drop</td><td>~{max(0, 40 * (1 - metrics['avg_calibration_error'])):.0f}% actual</td></tr>
                    <tr><td>20% chance of ≥15% drop</td><td>~{max(0, 20 * (1 - metrics['avg_calibration_error'])):.0f}% actual</td></tr>
                    <tr><td>10% chance of ≥20% drop</td><td>~{max(0, 10 * (1 - metrics['avg_calibration_error'])):.0f}% actual</td></tr>
                </table>
                <p><em>Note: These are approximations. Calibration varies by threshold.</em></p>
            </div>
            
            <h2 id="charts">📊 Visual Analysis</h2>
            
            <h3>Predicted vs Actual Drop Probabilities</h3>
            <div class="explanation-box">
                <p><strong>How to read:</strong> Blue bars show what the model predicted on average. Orange bars show how often drops actually occurred. If the model were perfect, the bars would be equal height.</p>
                <p>Bars where blue exceeds orange = model overestimates risk (conservative). Bars where orange exceeds blue = model underestimates risk (dangerous).</p>
            </div>
            <div class="chart" id="cal_chart"></div>
            
            <h3>Calibration Scatter Plot</h3>
            <div class="explanation-box">
                <p><strong>How to read:</strong> Each point represents a drop threshold. The diagonal dashed line represents perfect calibration.</p>
                <ul>
                    <li><strong>Points ON the line:</strong> Model is perfectly calibrated for that threshold</li>
                    <li><strong>Points BELOW the line:</strong> Model overestimates risk (your model shows this pattern)</li>
                    <li><strong>Points ABOVE the line:</strong> Model underestimates risk</li>
                </ul>
                <p>A well-calibrated model has all points close to the diagonal line.</p>
            </div>
            <div class="chart" id="scatter_chart"></div>
            
            <h3>Time Series: Predictions vs Outcomes</h3>
            <div class="explanation-box">
                <p><strong>How to read:</strong> The top panel shows BTC price over time with red X marks where ≥20% drops actually occurred within the next 30 days.</p>
                <p>The bottom panel shows the model's predicted probability of a ≥20% drop at each point in time. Higher values indicate the model saw more risk.</p>
                <p>Ideally, red X marks should appear when the probability line is elevated.</p>
            </div>
            <div class="chart" id="ts_chart"></div>
            
            <h2 id="regime">🔄 Performance by Market Regime</h2>
            
            <div class="info-box">
                <p><strong>What is regime detection?</strong> The model classifies market conditions as either "Normal" or "Stress" based on multiple indicators:</p>
                <ul>
                    <li><strong>VIX level:</strong> Above 30 indicates market fear</li>
                    <li><strong>BTC volatility:</strong> When recent volatility exceeds 1.5× the 90-day average</li>
                    <li><strong>BTC drawdown:</strong> When price is down more than 15% from 30-day high</li>
                    <li><strong>On-chain indicators:</strong> Exchange netflows, funding rates, SOPR (if available)</li>
                </ul>
                <p>During stress regimes, the model applies a 1.5× volatility multiplier and sets drift to zero, producing more conservative (higher) risk estimates.</p>
            </div>
            
            <h3>🔍 Regime Error Attribution: Where is the overestimation coming from?</h3>
    """
    
    # Add regime breakdown analysis
    normal_data = regime_breakdown.get('normal', {})
    stress_data = regime_breakdown.get('stress', {})
    error_attr = regime_breakdown.get('error_attribution', {})
    
    normal_error = normal_data.get('avg_calibration_error', 0) * 100
    stress_error = stress_data.get('avg_calibration_error', 0) * 100
    
    # Determine which regime is the problem
    if abs(normal_error) > abs(stress_error) and normal_error > 5:
        problem_regime = "NORMAL"
        problem_explanation = f"The model overestimates risk by {normal_error:.1f}% during normal market conditions. This suggests the base volatility assumptions are too high."
        fix_suggestion = "Consider reducing the base GARCH volatility forecast or adjusting the drift term upward during normal regimes."
    elif abs(stress_error) > abs(normal_error) and stress_error > 5:
        problem_regime = "STRESS"
        problem_explanation = f"The model overestimates risk by {stress_error:.1f}% during stress periods. The 1.5× volatility multiplier may be too aggressive."
        fix_suggestion = "Consider reducing the stress volatility multiplier from 1.5× to 1.2-1.3×, or making the stress regime trigger less sensitive."
    elif normal_error > 5 and stress_error > 5:
        problem_regime = "BOTH"
        problem_explanation = f"Both regimes contribute to overestimation (Normal: {normal_error:.1f}%, Stress: {stress_error:.1f}%). The entire model is too conservative."
        fix_suggestion = "Consider a global calibration adjustment or reducing volatility assumptions across the board."
    else:
        problem_regime = "NEITHER"
        problem_explanation = "Neither regime shows significant overestimation. The model is reasonably calibrated."
        fix_suggestion = "No major adjustments needed."
    
    html += f"""
            <div class="{'warning-box' if problem_regime != 'NEITHER' else 'explanation-box'}">
                <h4>{'⚠️' if problem_regime != 'NEITHER' else '✓'} Primary Error Source: {problem_regime} Regime</h4>
                <p>{problem_explanation}</p>
                <p><strong>Suggested fix:</strong> {fix_suggestion}</p>
            </div>
            
            <h3>Calibration Comparison by Regime</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Normal Regime</th>
                    <th>Stress Regime</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td><strong>Backtest Points</strong></td>
                    <td>{normal_data.get('n_points', 0)}</td>
                    <td>{stress_data.get('n_points', 0)}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td><strong>Average Calibration Error</strong></td>
                    <td class="{'bad' if normal_error > 10 else 'warning' if normal_error > 5 else 'good'}">{normal_error:+.1f}%</td>
                    <td class="{'bad' if stress_error > 10 else 'warning' if stress_error > 5 else 'good'}">{stress_error:+.1f}%</td>
                    <td>{normal_error - stress_error:+.1f}%</td>
                </tr>
                <tr>
                    <td><strong>Assessment</strong></td>
                    <td>{normal_data.get('assessment', 'N/A')}</td>
                    <td>{stress_data.get('assessment', 'N/A')}</td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>Detailed Breakdown by Drop Threshold</h3>
            <table>
                <tr>
                    <th rowspan="2">Threshold</th>
                    <th colspan="3" style="text-align: center; background: #66bb6a;">Normal Regime ({normal_data.get('n_points', 0)} points)</th>
                    <th colspan="3" style="text-align: center; background: #ef5350;">Stress Regime ({stress_data.get('n_points', 0)} points)</th>
                </tr>
                <tr>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Error</th>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Error</th>
                </tr>
    """
    
    for thresh in results.config.drop_thresholds:
        normal_thresh = normal_data.get('thresholds', {}).get(thresh, {})
        stress_thresh = stress_data.get('thresholds', {}).get(thresh, {})
        
        n_pred = normal_thresh.get('avg_predicted_prob', 0) * 100
        n_act = normal_thresh.get('actual_frequency', 0) * 100
        n_err = normal_thresh.get('calibration_error', 0) * 100
        
        s_pred = stress_thresh.get('avg_predicted_prob', 0) * 100
        s_act = stress_thresh.get('actual_frequency', 0) * 100
        s_err = stress_thresh.get('calibration_error', 0) * 100
        
        n_err_class = 'bad' if n_err > 15 else 'warning' if n_err > 5 else 'good'
        s_err_class = 'bad' if s_err > 15 else 'warning' if s_err > 5 else 'good'
        
        html += f"""
                <tr>
                    <td><strong>≥{thresh}%</strong></td>
                    <td>{n_pred:.1f}%</td>
                    <td>{n_act:.1f}%</td>
                    <td class="{n_err_class}">{n_err:+.1f}%</td>
                    <td>{s_pred:.1f}%</td>
                    <td>{s_act:.1f}%</td>
                    <td class="{s_err_class}">{s_err:+.1f}%</td>
                </tr>
        """
    
    html += """
            </table>
            
            <div class="info-box">
                <h4>📊 How to Interpret This Table</h4>
                <ul>
                    <li><strong>Green errors (±5%):</strong> Well-calibrated for this regime/threshold combination</li>
                    <li><strong>Yellow errors (5-15%):</strong> Moderate overestimation, acceptable for risk management</li>
                    <li><strong>Red errors (>15%):</strong> Significant overestimation, consider model adjustment</li>
                </ul>
                <p>Compare across regimes: If one regime consistently shows higher errors, that's where to focus calibration efforts.</p>
            </div>
            
            <div class="chart" id="regime_chart"></div>"""
    
    html += f"""
            
            <h2 id="stress">⚡ Stress Period Analysis</h2>
            
            <div class="explanation-box">
                <p><strong>Why this matters:</strong> A good risk model should perform well during crisis periods, not just calm markets. This section examines how the model performed during known historical stress events.</p>
            </div>
            
            <table>
                <tr>
                    <th>Period</th>
                    <th>Dates</th>
                    <th>Backtest Points</th>
                    <th>Avg P(≥20% drop)</th>
                    <th>Actual ≥20% Frequency</th>
                    <th>Max Actual Drop</th>
                </tr>
    """
    
    for _, row in stress_df.iterrows():
        pred_20 = f"{row['avg_predicted_20pct_drop']*100:.1f}%" if row['avg_predicted_20pct_drop'] else "N/A"
        actual_20 = f"{row['actual_20pct_drop_freq']*100:.1f}%" if row['actual_20pct_drop_freq'] else "N/A"
        html += f"""
                <tr>
                    <td><strong>{row['period']}</strong></td>
                    <td>{row['start']} to {row['end']}</td>
                    <td>{row['n_points']}</td>
                    <td>{pred_20}</td>
                    <td>{actual_20}</td>
                    <td>{row['max_actual_drop']:.1f}%</td>
                </tr>
        """
    
    html += f"""
            </table>
            
            <div class="warning-box">
                <h4>⚠️ Important Caveats</h4>
                <ul>
                    <li><strong>Limited stress data:</strong> Some stress periods may have few backtest points due to data availability or timing.</li>
                    <li><strong>Survivorship bias:</strong> The backtest uses data that survived these periods; real-time decisions would face more uncertainty.</li>
                    <li><strong>Regime timing:</strong> The model detects stress based on observable indicators, which may lag the actual onset of a crisis.</li>
                </ul>
            </div>
            
            <h2 id="implications">💼 Practical Implications for BitVault</h2>
            
            <div class="key-finding">
                <h3>For Risk Management Decisions</h3>
                <p>Given the model's {bias_assessment} nature:</p>
                <ul>
                    <li><strong>Margin call probability of 10%</strong> → Historical reality suggests approximately {max(0, 10 * (1 - metrics['avg_calibration_error'])):.0f}% actual risk</li>
                    <li><strong>Liquidation probability of 5%</strong> → Historical reality suggests approximately {max(0, 5 * (1 - metrics['avg_calibration_error'])):.0f}% actual risk</li>
                </ul>
                <p>{'This conservative bias is <strong>appropriate for a lending protocol</strong> where underestimating risk could lead to losses.' if metrics['avg_calibration_error'] > 0 else '<strong>⚠️ Consider adding a safety buffer to model outputs.</strong>'}</p>
            </div>
            
            <div class="info-box">
                <h3>Recommended Actions</h3>
                <ol>
                    <li><strong>Use model outputs as upper bounds:</strong> When communicating risk to stakeholders, note that actual probabilities are likely lower than model estimates.</li>
                    <li><strong>Monitor regime detection:</strong> Pay attention when the model shifts to "Stress" regime - this indicates elevated market risk.</li>
                    <li><strong>Regular recalibration:</strong> Re-run this backtest quarterly to ensure the model remains well-calibrated as market conditions evolve.</li>
                    <li><strong>Stress testing:</strong> For critical decisions, consider running scenarios with even higher volatility assumptions.</li>
                </ol>
            </div>
            
            <h2 id="methodology">🔬 Methodology Details</h2>
            
            <div class="info-box">
                <h3>Backtest Configuration</h3>
                <table>
                    <tr><td><strong>Backtest Period</strong></td><td>{results.config.start_date} to {results.config.end_date}</td></tr>
                    <tr><td><strong>Step Size</strong></td><td>Every {results.config.step_days} days</td></tr>
                    <tr><td><strong>Calibration Window</strong></td><td>{results.config.calibration_years} years of prior data</td></tr>
                    <tr><td><strong>Simulation Horizon</strong></td><td>{results.config.horizon_days} days forward</td></tr>
                    <tr><td><strong>Monte Carlo Paths</strong></td><td>{results.config.n_paths:,} per simulation</td></tr>
                    <tr><td><strong>Drop Thresholds</strong></td><td>{', '.join(f'≥{t}%' for t in results.config.drop_thresholds)}</td></tr>
                </table>
                
                <h3>Model Components</h3>
                <ul>
                    <li><strong>Volatility Model:</strong> GARCH(1,1) calibrated on rolling 3-year window</li>
                    <li><strong>Price Process:</strong> Geometric Brownian Motion with GARCH volatility</li>
                    <li><strong>Jump Component:</strong> Merton jump diffusion (calibrated from historical jumps)</li>
                    <li><strong>Regime Adjustment:</strong> 1.5× volatility multiplier in stress regime</li>
                </ul>
                
                <h3>Evaluation Metrics</h3>
                <ul>
                    <li><strong>Brier Score:</strong> Mean squared error between predicted probabilities and binary outcomes</li>
                    <li><strong>Calibration Error:</strong> Difference between average predicted probability and actual frequency</li>
                </ul>
            </div>
            
            <hr style="margin: 40px 0;">
            <p style="text-align: center; color: #777;">
                <em>Report generated by BitVault Risk Model v1.0</em><br>
                <em>Execution time: {results.execution_time_seconds:.1f} seconds</em>
            </p>
            
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
