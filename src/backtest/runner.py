"""
Backtesting framework for BitVault BTC Risk Model.

Validates model accuracy by running historical simulations and comparing
predicted probabilities to actual outcomes.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.database import get_db_path
from src.logging_config import get_logger

logger = get_logger("backtest")

# Suppress arch package warnings during backtesting
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    start_date: str = "2022-01-01"      # Start of backtest period
    end_date: str = "2025-10-01"        # End of backtest period
    calibration_years: int = 3           # Years of data for GARCH calibration
    horizon_days: int = 30               # Simulation horizon
    n_paths: int = 50_000                # Paths per simulation (reduced for speed)
    step_days: int = 7                   # Days between backtest points
    drop_thresholds: list = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    use_onchain_regime: bool = True      # Use on-chain data for regime detection
    regime_mode: str = "auto"            # "auto", "simple", "onchain_only", "market_only"


@dataclass
class BacktestPoint:
    """Results for a single backtest date."""
    date: str
    btc_price: float
    regime: str
    
    # Predicted probabilities for each drop threshold
    predicted_probs: dict[int, float]
    
    # Actual outcomes
    actual_min_price: float  # Minimum price in next 30 days
    actual_max_drop: float   # Maximum drawdown in next 30 days (as positive %)
    
    # Which thresholds were actually breached
    actual_breaches: dict[int, bool]
    
    # Regime indicator details (for analysis)
    regime_indicators: dict = field(default_factory=dict)


@dataclass 
class BacktestResults:
    """Complete backtest results."""
    config: BacktestConfig
    points: list[BacktestPoint]
    run_date: str
    execution_time_seconds: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        rows = []
        for point in self.points:
            row = {
                'date': point.date,
                'btc_price': point.btc_price,
                'regime': point.regime,
                'actual_max_drop': point.actual_max_drop,
            }
            # Add predicted probs
            for thresh, prob in point.predicted_probs.items():
                row[f'pred_{thresh}pct'] = prob
            # Add actual breaches
            for thresh, breached in point.actual_breaches.items():
                row[f'actual_{thresh}pct'] = breached
            # Add regime indicator details if available
            if point.regime_indicators:
                row['stress_count'] = point.regime_indicators.get('stress_count', 0)
                row['stress_signals'] = ','.join(point.regime_indicators.get('stress_signals', []))
                for indicator in ['vix', 'btc_volatility', 'btc_drawdown', 'exchange_netflow', 
                                  'funding_rate', 'sopr', 'mvrv']:
                    if indicator in point.regime_indicators:
                        row[f'ind_{indicator}'] = point.regime_indicators[indicator]
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def calculate_regime_breakdown(self) -> dict:
        """
        Calculate calibration metrics broken down by regime.
        
        Returns dict with 'normal' and 'stress' sub-dicts containing:
        - n_points: number of backtest points
        - thresholds: per-threshold metrics
        - avg_calibration_error: average error across thresholds
        - assessment: text description
        """
        df = self.to_dataframe()
        
        results = {}
        
        for regime in ['normal', 'stress']:
            regime_df = df[df['regime'] == regime]
            
            if len(regime_df) == 0:
                results[regime] = {'n_points': 0, 'thresholds': {}}
                continue
            
            regime_results = {
                'n_points': len(regime_df),
                'thresholds': {}
            }
            
            for thresh in self.config.drop_thresholds:
                pred_col = f'pred_{thresh}pct'
                actual_col = f'actual_{thresh}pct'
                
                if pred_col not in regime_df.columns:
                    continue
                
                predicted = regime_df[pred_col].values
                actual = regime_df[actual_col].astype(float).values
                
                avg_predicted = np.mean(predicted)
                actual_frequency = np.mean(actual)
                brier = np.mean((predicted - actual) ** 2)
                calibration_error = avg_predicted - actual_frequency
                
                regime_results['thresholds'][thresh] = {
                    'avg_predicted_prob': avg_predicted,
                    'actual_frequency': actual_frequency,
                    'calibration_error': calibration_error,
                    'brier_score': brier,
                    'n_actual_breaches': int(np.sum(actual)),
                }
            
            # Calculate average calibration error
            if regime_results['thresholds']:
                cal_errors = [m['calibration_error'] for m in regime_results['thresholds'].values()]
                regime_results['avg_calibration_error'] = np.mean(cal_errors)
                
                if regime_results['avg_calibration_error'] > 0.02:
                    regime_results['assessment'] = "Overestimates risk"
                elif regime_results['avg_calibration_error'] < -0.02:
                    regime_results['assessment'] = "Underestimates risk"
                else:
                    regime_results['assessment'] = "Well-calibrated"
            
            results[regime] = regime_results
        
        # Calculate which regime contributes more to overall error
        if results.get('normal', {}).get('avg_calibration_error') and results.get('stress', {}).get('avg_calibration_error'):
            normal_contribution = results['normal']['avg_calibration_error'] * results['normal']['n_points']
            stress_contribution = results['stress']['avg_calibration_error'] * results['stress']['n_points']
            total_points = results['normal']['n_points'] + results['stress']['n_points']
            
            results['error_attribution'] = {
                'normal_contribution': normal_contribution / total_points if total_points > 0 else 0,
                'stress_contribution': stress_contribution / total_points if total_points > 0 else 0,
                'primary_source': 'normal' if abs(normal_contribution) > abs(stress_contribution) else 'stress'
            }
        
        return results
    
    def calculate_metrics(self) -> dict:
        """Calculate backtest performance metrics."""
        df = self.to_dataframe()
        
        metrics = {
            'n_points': len(self.points),
            'date_range': f"{self.config.start_date} to {self.config.end_date}",
            'thresholds': {}
        }
        
        for thresh in self.config.drop_thresholds:
            pred_col = f'pred_{thresh}pct'
            actual_col = f'actual_{thresh}pct'
            
            if pred_col not in df.columns:
                continue
            
            predicted = df[pred_col].values
            actual = df[actual_col].astype(float).values
            
            # Calibration: average predicted vs actual frequency
            avg_predicted = np.mean(predicted)
            actual_frequency = np.mean(actual)
            
            # Brier score (lower is better)
            brier = np.mean((predicted - actual) ** 2)
            
            # Calibration error (predicted - actual)
            calibration_error = avg_predicted - actual_frequency
            
            metrics['thresholds'][thresh] = {
                'avg_predicted_prob': avg_predicted,
                'actual_frequency': actual_frequency,
                'calibration_error': calibration_error,
                'brier_score': brier,
                'n_actual_breaches': int(np.sum(actual)),
            }
        
        # Overall Brier score (average across thresholds)
        brier_scores = [m['brier_score'] for m in metrics['thresholds'].values()]
        metrics['overall_brier_score'] = np.mean(brier_scores)
        
        # Calibration assessment
        cal_errors = [m['calibration_error'] for m in metrics['thresholds'].values()]
        metrics['avg_calibration_error'] = np.mean(cal_errors)
        
        if metrics['avg_calibration_error'] > 0.02:
            metrics['calibration_assessment'] = "Model OVERESTIMATES risk"
        elif metrics['avg_calibration_error'] < -0.02:
            metrics['calibration_assessment'] = "Model UNDERESTIMATES risk"
        else:
            metrics['calibration_assessment'] = "Model is well-calibrated"
        
        return metrics
    
    def get_stress_period_performance(self) -> pd.DataFrame:
        """Analyze performance during known stress periods."""
        df = self.to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        
        stress_periods = [
            ("COVID Crash", "2020-02-15", "2020-04-15"),
            ("May 2021 Crash", "2021-04-15", "2021-06-15"),
            ("LUNA/3AC", "2022-04-15", "2022-07-15"),
            ("FTX Collapse", "2022-10-15", "2022-12-31"),
            ("2023 Recovery", "2023-01-01", "2023-03-31"),
            ("2024 Bull Run", "2024-01-01", "2024-04-30"),
        ]
        
        results = []
        for name, start, end in stress_periods:
            mask = (df['date'] >= start) & (df['date'] <= end)
            period_df = df[mask]
            
            if len(period_df) == 0:
                continue
            
            # Calculate metrics for this period
            avg_pred_20 = period_df['pred_20pct'].mean() if 'pred_20pct' in period_df else None
            actual_20 = period_df['actual_20pct'].mean() if 'actual_20pct' in period_df else None
            
            results.append({
                'period': name,
                'start': start,
                'end': end,
                'n_points': len(period_df),
                'avg_predicted_20pct_drop': avg_pred_20,
                'actual_20pct_drop_freq': actual_20,
                'max_actual_drop': period_df['actual_max_drop'].max(),
            })
        
        return pd.DataFrame(results)


class BacktestRunner:
    """Runs backtests on historical data."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.db_path = get_db_path()
    
    def _load_price_history(self) -> pd.DataFrame:
        """Load full BTC price history from database."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT date, close FROM btc_prices ORDER BY date",
            conn
        )
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        return df
    
    def _get_calibration_window(
        self, 
        prices: pd.DataFrame, 
        target_date: datetime
    ) -> pd.DataFrame:
        """Get price data for GARCH calibration (prior N years)."""
        start = target_date - timedelta(days=self.config.calibration_years * 365)
        end = target_date - timedelta(days=1)
        
        return prices.loc[start:end]
    
    def _get_forward_prices(
        self, 
        prices: pd.DataFrame, 
        target_date: datetime
    ) -> pd.DataFrame:
        """Get actual prices for the forward horizon."""
        end = target_date + timedelta(days=self.config.horizon_days)
        return prices.loc[target_date:end]
    
    def _calibrate_and_simulate(
        self,
        calibration_prices: pd.DataFrame,
        current_price: float,
        regime: str
    ) -> dict[int, float]:
        """Calibrate GARCH and run simulation, return drop probabilities."""
        from arch import arch_model
        
        # Calculate returns
        returns = np.log(
            calibration_prices['close'] / calibration_prices['close'].shift(1)
        ).dropna() * 100
        
        if len(returns) < 252:  # Need at least 1 year
            return {t: 0.0 for t in self.config.drop_thresholds}
        
        try:
            # Fit GARCH
            model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
            result = model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            omega = result.params.get('omega', 1.0)
            alpha = result.params.get('alpha[1]', 0.1)
            beta = result.params.get('beta[1]', 0.85)
            mu = result.params.get('mu', 0.0)
            
            # Get current variance
            forecasts = result.forecast(horizon=1)
            current_var = forecasts.variance.iloc[-1, 0]
            
            # Run Monte Carlo
            n_paths = self.config.n_paths
            n_days = self.config.horizon_days
            
            np.random.seed(int(datetime.now().timestamp()) % 2**31)
            
            prices_sim = np.zeros((n_paths, n_days + 1))
            prices_sim[:, 0] = current_price
            
            variance = np.full(n_paths, current_var)
            
            # Regime adjustments
            vol_mult = 1.5 if regime == "stress" else 1.0
            drift_adj = 0.0 if regime == "stress" else mu / 100
            
            for t in range(1, n_days + 1):
                z = np.random.standard_normal(n_paths)
                
                # GARCH variance update
                variance = omega + alpha * (z ** 2) * variance + beta * variance
                variance = np.maximum(variance, 0.01)
                
                # Daily returns
                daily_vol = np.sqrt(variance) / 100 * vol_mult
                returns_t = drift_adj + daily_vol * z
                
                prices_sim[:, t] = prices_sim[:, t-1] * np.exp(returns_t)
            
            # Calculate minimum prices and drop probabilities
            min_prices = prices_sim.min(axis=1)
            max_drops = (1 - min_prices / current_price) * 100
            
            probabilities = {}
            for threshold in self.config.drop_thresholds:
                probabilities[threshold] = float(np.mean(max_drops >= threshold))
            
            return probabilities
            
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return {t: 0.0 for t in self.config.drop_thresholds}
    
    def _detect_regime(
        self,
        prices: pd.DataFrame,
        target_date: datetime
    ) -> tuple[str, dict]:
        """
        Detect regime using available indicators.
        
        Returns:
            Tuple of (regime_string, indicator_dict)
        """
        indicators = {}
        stress_signals = []
        
        # Get last 30 days of prices
        start = target_date - timedelta(days=30)
        recent = prices.loc[start:target_date]
        
        if len(recent) < 10:
            return "normal", {"error": "insufficient data"}
        
        current_price = recent['close'].iloc[-1]
        
        # === MARKET INDICATORS ===
        
        # 1. BTC Volatility
        returns = np.log(recent['close'] / recent['close'].shift(1)).dropna()
        recent_vol = returns.std() * np.sqrt(365)
        
        start_90 = target_date - timedelta(days=90)
        longer = prices.loc[start_90:target_date]
        if len(longer) >= 30:
            returns_90 = np.log(longer['close'] / longer['close'].shift(1)).dropna()
            long_vol = returns_90.std() * np.sqrt(365)
            vol_multiple = recent_vol / long_vol if long_vol > 0 else 1.0
        else:
            vol_multiple = 1.0
        
        indicators['btc_volatility'] = vol_multiple
        if vol_multiple > 1.5:
            stress_signals.append('btc_volatility')
        
        # 2. BTC Drawdown
        high_30d = recent['close'].max()
        drawdown = (current_price / high_30d) - 1
        indicators['btc_drawdown'] = drawdown
        if drawdown < -0.15:
            stress_signals.append('btc_drawdown')
        
        # 3. VIX (if available)
        try:
            conn = sqlite3.connect(self.db_path)
            vix_df = pd.read_sql_query(
                f"""SELECT date, value FROM macro_data 
                    WHERE indicator = 'vix' 
                    AND date <= '{target_date.strftime('%Y-%m-%d')}'
                    ORDER BY date DESC LIMIT 5""",
                conn
            )
            if not vix_df.empty:
                vix_value = vix_df['value'].iloc[0]
                indicators['vix'] = vix_value
                if vix_value > 30:
                    stress_signals.append('vix')
            conn.close()
        except:
            pass
        
        # === ON-CHAIN INDICATORS (if enabled) ===
        if self.config.use_onchain_regime:
            conn = sqlite3.connect(self.db_path)
            target_str = target_date.strftime('%Y-%m-%d')
            
            # 4. Exchange Netflow
            try:
                netflow_df = pd.read_sql_query(
                    f"""SELECT date, net_flow FROM exchange_flows 
                        WHERE date <= '{target_str}'
                        ORDER BY date DESC LIMIT 7""",
                    conn
                )
                if not netflow_df.empty:
                    avg_netflow = netflow_df['net_flow'].mean()
                    indicators['exchange_netflow'] = avg_netflow
                    if avg_netflow > 10000:
                        stress_signals.append('exchange_netflow')
            except:
                pass
            
            # 5. Funding Rate
            try:
                funding_df = pd.read_sql_query(
                    f"""SELECT date, funding_rate FROM funding_rates 
                        WHERE date <= '{target_str}'
                        ORDER BY date DESC LIMIT 1""",
                    conn
                )
                if not funding_df.empty:
                    funding = funding_df['funding_rate'].iloc[0]
                    indicators['funding_rate'] = funding
                    if funding < -0.0001:  # -0.01%
                        stress_signals.append('funding_rate')
            except:
                pass
            
            # 6. SOPR
            try:
                sopr_df = pd.read_sql_query(
                    f"""SELECT date, value FROM onchain_metrics 
                        WHERE metric = 'sopr' AND date <= '{target_str}'
                        ORDER BY date DESC LIMIT 7""",
                    conn
                )
                if not sopr_df.empty:
                    avg_sopr = sopr_df['value'].mean()
                    indicators['sopr'] = avg_sopr
                    if avg_sopr < 0.98:
                        stress_signals.append('sopr')
            except:
                pass
            
            # 7. MVRV
            try:
                mvrv_df = pd.read_sql_query(
                    f"""SELECT date, value FROM onchain_metrics 
                        WHERE metric = 'mvrv' AND date <= '{target_str}'
                        ORDER BY date DESC LIMIT 1""",
                    conn
                )
                if not mvrv_df.empty:
                    mvrv = mvrv_df['value'].iloc[0]
                    indicators['mvrv'] = mvrv
                    if mvrv < 1.0 or mvrv > 3.5:
                        stress_signals.append('mvrv')
            except:
                pass
            
            conn.close()
        
        # Determine regime
        indicators['stress_signals'] = stress_signals
        indicators['stress_count'] = len(stress_signals)
        indicators['total_indicators'] = len([k for k in indicators.keys() 
                                              if k not in ['stress_signals', 'stress_count', 'total_indicators']])
        
        # STRESS if 2+ indicators trigger
        regime = "stress" if len(stress_signals) >= 2 else "normal"
        
        return regime, indicators
    
    def _detect_regime_simple(
        self,
        prices: pd.DataFrame,
        target_date: datetime
    ) -> tuple[str, dict]:
        """Simple regime detection based on recent volatility only (legacy method)."""
        # Get last 30 days
        start = target_date - timedelta(days=30)
        recent = prices.loc[start:target_date]
        
        if len(recent) < 10:
            return "normal", {}
        
        # Calculate recent volatility
        returns = np.log(recent['close'] / recent['close'].shift(1)).dropna()
        recent_vol = returns.std() * np.sqrt(365)
        
        # Get longer-term volatility (90 days)
        start_90 = target_date - timedelta(days=90)
        longer = prices.loc[start_90:target_date]
        
        if len(longer) < 30:
            return "normal", {}
        
        returns_90 = np.log(longer['close'] / longer['close'].shift(1)).dropna()
        long_vol = returns_90.std() * np.sqrt(365)
        
        # Check for drawdown
        high_30d = recent['close'].max()
        current = recent['close'].iloc[-1]
        drawdown = (current / high_30d) - 1
        
        indicators = {
            'recent_vol': recent_vol,
            'long_vol': long_vol,
            'vol_ratio': recent_vol / long_vol if long_vol > 0 else 1.0,
            'drawdown': drawdown
        }
        
        # Stress if vol elevated or significant drawdown
        if recent_vol > long_vol * 1.5 or drawdown < -0.15:
            return "stress", indicators
        
        return "normal", indicators
    
    def run(self, show_progress: bool = True) -> BacktestResults:
        """Run the full backtest."""
        start_time = datetime.now()
        
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Load price history
        prices = self._load_price_history()
        logger.info(f"Loaded {len(prices)} price records")
        
        # Generate backtest dates
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        
        # Need calibration_years of history before start
        min_history_date = prices.index.min()
        required_start = start - timedelta(days=self.config.calibration_years * 365)
        
        if required_start < min_history_date:
            start = min_history_date + timedelta(days=self.config.calibration_years * 365 + 30)
            logger.warning(f"Adjusted start date to {start.date()} due to data availability")
        
        # Need horizon_days of forward data
        max_date = prices.index.max() - timedelta(days=self.config.horizon_days)
        if end > max_date:
            end = max_date
            logger.warning(f"Adjusted end date to {end.date()} due to data availability")
        
        # Generate dates
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=self.config.step_days)
        
        logger.info(f"Running backtest on {len(dates)} dates")
        
        # Run backtest
        points = []
        iterator = tqdm(dates, desc="Backtesting") if show_progress else dates
        
        for target_date in iterator:
            try:
                # Get calibration data
                calib_prices = self._get_calibration_window(prices, target_date)
                
                if len(calib_prices) < 365:
                    continue
                
                # Get current price
                if target_date not in prices.index:
                    # Find nearest date
                    idx = prices.index.get_indexer([target_date], method='ffill')[0]
                    current_price = prices.iloc[idx]['close']
                else:
                    current_price = prices.loc[target_date, 'close']
                
                # Detect regime
                regime, regime_indicators = self._detect_regime(prices, target_date)
                
                # Run simulation
                predicted_probs = self._calibrate_and_simulate(
                    calib_prices, current_price, regime
                )
                
                # Get actual forward prices
                forward_prices = self._get_forward_prices(prices, target_date)
                
                if len(forward_prices) < self.config.horizon_days // 2:
                    continue
                
                actual_min = forward_prices['close'].min()
                actual_max_drop = (1 - actual_min / current_price) * 100
                
                # Check which thresholds were breached
                actual_breaches = {
                    t: actual_max_drop >= t 
                    for t in self.config.drop_thresholds
                }
                
                point = BacktestPoint(
                    date=target_date.strftime("%Y-%m-%d"),
                    btc_price=current_price,
                    regime=regime,
                    predicted_probs=predicted_probs,
                    actual_min_price=actual_min,
                    actual_max_drop=actual_max_drop,
                    actual_breaches=actual_breaches,
                    regime_indicators=regime_indicators
                )
                points.append(point)
                
            except Exception as e:
                logger.warning(f"Failed for {target_date}: {e}")
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Backtest complete: {len(points)} points in {execution_time:.1f}s")
        
        return BacktestResults(
            config=self.config,
            points=points,
            run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            execution_time_seconds=execution_time
        )


def run_backtest(
    start_date: str = "2022-01-01",
    end_date: str = "2025-10-01",
    step_days: int = 7,
    use_onchain: bool = True
) -> BacktestResults:
    """Convenience function to run a backtest."""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days,
        use_onchain_regime=use_onchain
    )
    runner = BacktestRunner(config)
    return runner.run()


def run_comparison_backtest(
    start_date: str = "2022-01-01",
    end_date: str = "2025-10-01",
    step_days: int = 7
) -> dict:
    """
    Run backtests with and without on-chain regime detection to compare.
    
    Returns dict with 'with_onchain' and 'without_onchain' results.
    """
    print("=" * 60)
    print("BACKTEST COMPARISON: On-Chain vs Market-Only Regime Detection")
    print("=" * 60)
    
    # Run with on-chain
    print("\n[1/2] Running backtest WITH on-chain indicators...")
    config_with = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days,
        use_onchain_regime=True
    )
    runner_with = BacktestRunner(config_with)
    results_with = runner_with.run()
    
    # Run without on-chain
    print("\n[2/2] Running backtest WITHOUT on-chain indicators...")
    config_without = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days,
        use_onchain_regime=False
    )
    runner_without = BacktestRunner(config_without)
    results_without = runner_without.run()
    
    # Calculate metrics for both
    metrics_with = results_with.calculate_metrics()
    metrics_without = results_without.calculate_metrics()
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'With On-Chain':>15} {'Without':>15} {'Diff':>10}")
    print("-" * 70)
    
    print(f"{'Backtest Points':<30} {len(results_with.points):>15} {len(results_without.points):>15}")
    print(f"{'Brier Score':<30} {metrics_with['overall_brier_score']:>15.4f} {metrics_without['overall_brier_score']:>15.4f} {metrics_with['overall_brier_score'] - metrics_without['overall_brier_score']:>+10.4f}")
    print(f"{'Calibration Error':<30} {metrics_with['avg_calibration_error']*100:>14.1f}% {metrics_without['avg_calibration_error']*100:>14.1f}% {(metrics_with['avg_calibration_error'] - metrics_without['avg_calibration_error'])*100:>+9.1f}%")
    
    # Regime breakdown
    stress_with = sum(1 for p in results_with.points if p.regime == 'stress')
    stress_without = sum(1 for p in results_without.points if p.regime == 'stress')
    
    print(f"\n{'Stress Regime Periods':<30} {stress_with:>15} {stress_without:>15}")
    print(f"{'Normal Regime Periods':<30} {len(results_with.points) - stress_with:>15} {len(results_without.points) - stress_without:>15}")
    
    # Better/worse assessment
    print("\n" + "-" * 60)
    if metrics_with['overall_brier_score'] < metrics_without['overall_brier_score']:
        print("✓ On-chain regime detection IMPROVES model accuracy")
        print(f"  Brier score improved by {(metrics_without['overall_brier_score'] - metrics_with['overall_brier_score']) * 100:.2f}%")
    else:
        print("✗ On-chain regime detection does NOT improve accuracy")
        print(f"  Brier score worsened by {(metrics_with['overall_brier_score'] - metrics_without['overall_brier_score']) * 100:.2f}%")
    
    if abs(metrics_with['avg_calibration_error']) < abs(metrics_without['avg_calibration_error']):
        print("✓ On-chain improves calibration (closer to actual probabilities)")
    else:
        print("✗ On-chain worsens calibration")
    
    return {
        'with_onchain': results_with,
        'without_onchain': results_without,
        'metrics_with': metrics_with,
        'metrics_without': metrics_without
    }


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    start_date = "2022-01-01"
    end_date = "2025-10-01"
    step_days = 7
    compare_mode = False
    use_onchain = True
    
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            start_date = arg.split("=")[1]
        elif arg.startswith("--end="):
            end_date = arg.split("=")[1]
        elif arg.startswith("--step="):
            step_days = int(arg.split("=")[1])
        elif arg == "--compare":
            compare_mode = True
        elif arg == "--no-onchain":
            use_onchain = False
    
    if compare_mode:
        # Run comparison backtest
        comparison = run_comparison_backtest(
            start_date=start_date,
            end_date=end_date,
            step_days=step_days
        )
    else:
        # Run single backtest
        print("=" * 60)
        print("BitVault Risk Model Backtest")
        print(f"On-chain regime detection: {'ENABLED' if use_onchain else 'DISABLED'}")
        print("=" * 60)
        
        results = run_backtest(
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            use_onchain=use_onchain
        )
        
        print(f"\nCompleted {len(results.points)} backtest points")
        print(f"Execution time: {results.execution_time_seconds:.1f}s")
        
        # Regime breakdown
        stress_count = sum(1 for p in results.points if p.regime == 'stress')
        print(f"\nRegime breakdown:")
        print(f"  Stress: {stress_count} periods ({stress_count/len(results.points)*100:.1f}%)")
        print(f"  Normal: {len(results.points) - stress_count} periods ({(len(results.points) - stress_count)/len(results.points)*100:.1f}%)")
        
        # Calculate and display metrics
        metrics = results.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("Calibration Results")
        print("=" * 60)
        
        print(f"\nOverall Brier Score: {metrics['overall_brier_score']:.4f}")
        print(f"Average Calibration Error: {metrics['avg_calibration_error']*100:.2f}%")
        print(f"Assessment: {metrics['calibration_assessment']}")
        
        print("\nBy Threshold:")
        print("-" * 60)
        print(f"{'Drop':<8} {'Predicted':>12} {'Actual':>12} {'Error':>12} {'Brier':>10}")
        print("-" * 60)
        
        for thresh, m in metrics['thresholds'].items():
            print(
                f"≥{thresh}%{'':<4} "
                f"{m['avg_predicted_prob']*100:>11.2f}% "
                f"{m['actual_frequency']*100:>11.2f}% "
                f"{m['calibration_error']*100:>+11.2f}% "
                f"{m['brier_score']:>10.4f}"
            )
        
        # Stress period analysis
        print("\n" + "=" * 60)
        print("Stress Period Analysis")
        print("=" * 60)
        
        stress_df = results.get_stress_period_performance()
        if not stress_df.empty:
            for _, row in stress_df.iterrows():
                print(f"\n{row['period']} ({row['start']} to {row['end']})")
                print(f"  Points: {row['n_points']}")
                if row['avg_predicted_20pct_drop'] is not None:
                    print(f"  Avg predicted P(≥20% drop): {row['avg_predicted_20pct_drop']*100:.1f}%")
                    print(f"  Actual ≥20% drop frequency: {row['actual_20pct_drop_freq']*100:.1f}%")
                print(f"  Max actual drop: {row['max_actual_drop']:.1f}%")
