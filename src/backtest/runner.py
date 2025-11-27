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
            rows.append(row)
        
        return pd.DataFrame(rows)
    
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
    ) -> str:
        """Simple regime detection based on recent volatility."""
        # Get last 30 days
        start = target_date - timedelta(days=30)
        recent = prices.loc[start:target_date]
        
        if len(recent) < 10:
            return "normal"
        
        # Calculate recent volatility
        returns = np.log(recent['close'] / recent['close'].shift(1)).dropna()
        recent_vol = returns.std() * np.sqrt(365)
        
        # Get longer-term volatility (90 days)
        start_90 = target_date - timedelta(days=90)
        longer = prices.loc[start_90:target_date]
        
        if len(longer) < 30:
            return "normal"
        
        returns_90 = np.log(longer['close'] / longer['close'].shift(1)).dropna()
        long_vol = returns_90.std() * np.sqrt(365)
        
        # Check for drawdown
        high_30d = recent['close'].max()
        current = recent['close'].iloc[-1]
        drawdown = (current / high_30d) - 1
        
        # Stress if vol elevated or significant drawdown
        if recent_vol > long_vol * 1.5 or drawdown < -0.15:
            return "stress"
        
        return "normal"
    
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
                regime = self._detect_regime(prices, target_date)
                
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
                    actual_breaches=actual_breaches
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
    step_days: int = 7
) -> BacktestResults:
    """Convenience function to run a backtest."""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days
    )
    runner = BacktestRunner(config)
    return runner.run()


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
    
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        step_days=step_days
    )
    
    print(f"\nCompleted {len(results.points)} backtest points")
    print(f"Execution time: {results.execution_time_seconds:.1f}s")
    
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
