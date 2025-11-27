"""
GARCH(1,1) volatility model for BTC returns.

This module handles:
- Model fitting/calibration
- Volatility forecasting
- Parameter persistence
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.data.database import get_db_path
from src.data.prices import PriceFetcher


@dataclass
class GARCHParams:
    """GARCH(1,1) model parameters."""
    omega: float      # Long-run variance weight
    alpha: float      # Shock coefficient (reaction to recent returns)
    beta: float       # Persistence coefficient
    mu: float         # Mean return (drift)
    
    # Diagnostics
    log_likelihood: float
    persistence: float  # alpha + beta
    long_run_variance: float
    long_run_volatility: float  # Annualized
    
    # Calibration metadata
    calibration_date: str
    data_start_date: str
    data_end_date: str
    n_observations: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'mu': self.mu,
            'log_likelihood': self.log_likelihood,
            'persistence': self.persistence,
            'long_run_variance': self.long_run_variance,
            'long_run_volatility': self.long_run_volatility,
            'calibration_date': self.calibration_date,
            'data_start_date': self.data_start_date,
            'data_end_date': self.data_end_date,
            'n_observations': self.n_observations
        }
    
    def __str__(self) -> str:
        return (
            f"GARCH(1,1) Parameters (calibrated {self.calibration_date}):\n"
            f"  ω (omega)  = {self.omega:.2e}\n"
            f"  α (alpha)  = {self.alpha:.4f}\n"
            f"  β (beta)   = {self.beta:.4f}\n"
            f"  μ (drift)  = {self.mu:.4f}% daily\n"
            f"  Persistence (α+β) = {self.persistence:.4f}\n"
            f"  Long-run volatility = {self.long_run_volatility:.1f}% annualized\n"
            f"  Observations: {self.n_observations}"
        )


class GARCHModel:
    """GARCH(1,1) model for BTC volatility."""
    
    def __init__(self):
        self.db_path = get_db_path()
        self.params: Optional[GARCHParams] = None
        self._fitted_model: Optional[ARCHModelResult] = None
        self._returns: Optional[pd.Series] = None
    
    def _load_returns(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Load BTC returns from database.
        
        Returns:
            Series of daily log returns (in percentage points)
        """
        fetcher = PriceFetcher(use_cryptoquant=False)
        df = fetcher.load_from_db(start_date=start_date, end_date=end_date)
        
        if df.empty:
            raise ValueError("No price data in database. Run price backfill first.")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Calculate log returns in percentage points (required by arch package)
        returns = 100 * np.log(df['close'] / df['close'].shift(1))
        returns = returns.dropna()
        
        return returns
    
    def fit(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        returns: Optional[pd.Series] = None
    ) -> GARCHParams:
        """
        Fit GARCH(1,1) model to BTC returns.
        
        Args:
            start_date: Optional start date for training data
            end_date: Optional end date for training data
            returns: Optional pre-computed returns series
            
        Returns:
            Fitted GARCH parameters
        """
        if returns is not None:
            self._returns = returns
        else:
            self._returns = self._load_returns(start_date, end_date)
        
        if len(self._returns) < 100:
            raise ValueError(f"Insufficient data: {len(self._returns)} observations. Need at least 100.")
        
        print(f"Fitting GARCH(1,1) on {len(self._returns)} observations...")
        print(f"  Date range: {self._returns.index[0].date()} to {self._returns.index[-1].date()}")
        
        # Fit GARCH(1,1) with constant mean
        model = arch_model(
            self._returns,
            vol='Garch',
            p=1,  # GARCH lag
            q=1,  # ARCH lag
            mean='Constant',
            dist='normal'
        )
        
        self._fitted_model = model.fit(disp='off')
        
        # Extract parameters
        omega = self._fitted_model.params['omega']
        alpha = self._fitted_model.params['alpha[1]']
        beta = self._fitted_model.params['beta[1]']
        mu = self._fitted_model.params['mu']
        
        # Calculate derived quantities
        persistence = alpha + beta
        
        # Long-run variance: ω / (1 - α - β)
        if persistence < 1:
            long_run_var = omega / (1 - persistence)
        else:
            long_run_var = omega / 0.01  # Fallback for near-integrated process
        
        # Annualized volatility (returns are in %, so sqrt(252) * std)
        long_run_vol = np.sqrt(long_run_var) * np.sqrt(252)
        
        self.params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            log_likelihood=self._fitted_model.loglikelihood,
            persistence=persistence,
            long_run_variance=long_run_var,
            long_run_volatility=long_run_vol,
            calibration_date=datetime.now().strftime('%Y-%m-%d'),
            data_start_date=self._returns.index[0].strftime('%Y-%m-%d'),
            data_end_date=self._returns.index[-1].strftime('%Y-%m-%d'),
            n_observations=len(self._returns)
        )
        
        print(self.params)
        
        return self.params
    
    def get_current_volatility(self) -> float:
        """
        Get the current conditional volatility (annualized %).
        
        Returns:
            Current volatility estimate (annualized percentage)
        """
        if self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last conditional variance from the fitted model
        cond_var = self._fitted_model.conditional_volatility[-1] ** 2
        
        # Annualize: daily variance * 252, then sqrt
        annual_vol = np.sqrt(cond_var * 252)
        
        return annual_vol
    
    def forecast_volatility(self, horizon: int = 30) -> pd.DataFrame:
        """
        Forecast volatility for future periods.
        
        Args:
            horizon: Number of days to forecast
            
        Returns:
            DataFrame with forecasted volatility term structure
        """
        if self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate variance forecasts
        forecasts = self._fitted_model.forecast(horizon=horizon)
        
        # Extract variance forecasts (arch package returns variance, not vol)
        var_forecast = forecasts.variance.iloc[-1].values
        
        # Convert to annualized volatility
        vol_forecast = np.sqrt(var_forecast * 252)
        
        df = pd.DataFrame({
            'day': range(1, horizon + 1),
            'variance_daily': var_forecast,
            'volatility_annual': vol_forecast
        })
        
        return df
    
    def get_conditional_variance_series(self) -> pd.Series:
        """
        Get the full conditional variance series from fitted model.
        
        Returns:
            Series of conditional variances indexed by date
        """
        if self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._fitted_model.conditional_volatility ** 2
    
    def save_calibration(self) -> None:
        """Save current calibration to database."""
        if self.params is None:
            raise ValueError("No calibration to save. Call fit() first.")
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO model_calibrations
            (calibration_date, omega, alpha, beta, lambda_jump, mu_jump, sigma_jump,
             log_likelihood, data_start_date, data_end_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.params.calibration_date,
            self.params.omega,
            self.params.alpha,
            self.params.beta,
            0,  # lambda_jump - will be set by jump calibration
            0,  # mu_jump
            0,  # sigma_jump
            self.params.log_likelihood,
            self.params.data_start_date,
            self.params.data_end_date
        ))
        conn.commit()
        conn.close()
        
        print(f"Calibration saved for {self.params.calibration_date}")
    
    def load_latest_calibration(self) -> Optional[GARCHParams]:
        """
        Load the most recent calibration from database.
        
        Returns:
            GARCHParams or None if no calibration exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT calibration_date, omega, alpha, beta, log_likelihood,
                   data_start_date, data_end_date
            FROM model_calibrations
            ORDER BY calibration_date DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        cal_date, omega, alpha, beta, ll, start_date, end_date = row
        
        persistence = alpha + beta
        long_run_var = omega / (1 - persistence) if persistence < 1 else omega / 0.01
        long_run_vol = np.sqrt(long_run_var) * np.sqrt(252)
        
        self.params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=0,  # Not stored, assume 0
            log_likelihood=ll,
            persistence=persistence,
            long_run_variance=long_run_var,
            long_run_volatility=long_run_vol,
            calibration_date=cal_date,
            data_start_date=start_date,
            data_end_date=end_date,
            n_observations=0  # Not stored
        )
        
        return self.params


def calibrate_garch(save: bool = True) -> GARCHParams:
    """
    Run GARCH calibration on all available data.
    
    Args:
        save: Whether to save calibration to database
        
    Returns:
        Fitted GARCH parameters
    """
    model = GARCHModel()
    params = model.fit()
    
    if save:
        model.save_calibration()
    
    return params


if __name__ == "__main__":
    import sys
    
    save = "--save" in sys.argv
    
    print("=" * 60)
    print("BTC GARCH(1,1) Calibration")
    print("=" * 60)
    
    params = calibrate_garch(save=save)
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("=" * 60)
    
    # Interpret the results
    if params.persistence > 0.99:
        print("⚠️  Very high persistence - volatility shocks are extremely long-lasting")
    elif params.persistence > 0.95:
        print("📊 High persistence - volatility shocks decay slowly (typical for BTC)")
    else:
        print("📊 Moderate persistence - volatility mean-reverts relatively quickly")
    
    half_life = np.log(0.5) / np.log(params.persistence) if params.persistence < 1 else float('inf')
    print(f"   Volatility half-life: {half_life:.1f} days")
    
    print(f"\n📈 Long-run annualized volatility: {params.long_run_volatility:.1f}%")
    print(f"   (This is what volatility reverts to over time)")
    
    if save:
        print(f"\n✅ Calibration saved to database")
    else:
        print(f"\n💡 Run with --save to persist calibration")
