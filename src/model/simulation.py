"""
Monte Carlo simulation engine for BTC price paths.

Implements GARCH(1,1) volatility with Merton jump diffusion.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.model.garch import GARCHModel, GARCHParams
from src.data.prices import PriceFetcher


@dataclass
class JumpParams:
    """Merton jump diffusion parameters."""
    lambda_annual: float  # Expected jumps per year
    mu_jump: float        # Mean jump size (log terms, e.g., -0.07 = -7%)
    sigma_jump: float     # Jump size volatility
    
    def __str__(self) -> str:
        return (
            f"Jump Parameters:\n"
            f"  λ (intensity) = {self.lambda_annual:.1f} jumps/year\n"
            f"  μ_J (mean)    = {self.mu_jump * 100:.1f}%\n"
            f"  σ_J (vol)     = {self.sigma_jump * 100:.1f}%"
        )


@dataclass 
class SimulationParams:
    """Complete parameters for Monte Carlo simulation."""
    # Current state
    current_price: float
    current_variance: float  # Daily variance (not annualized)
    
    # GARCH parameters
    omega: float
    alpha: float
    beta: float
    mu: float  # Daily drift in percentage points
    
    # Jump parameters
    lambda_annual: float
    mu_jump: float
    sigma_jump: float
    
    # Simulation settings
    n_paths: int = 100_000
    horizon_days: int = 30
    random_seed: Optional[int] = None
    
    # Regime adjustment
    regime: str = "normal"  # "normal" or "stress"


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation."""
    # Raw outputs
    terminal_prices: np.ndarray
    all_paths: Optional[np.ndarray] = None  # Shape: (n_paths, horizon+1)
    
    # Summary statistics
    current_price: float = 0
    mean_price: float = 0
    median_price: float = 0
    std_price: float = 0
    
    # Percentiles
    percentile_5: float = 0
    percentile_25: float = 0
    percentile_75: float = 0
    percentile_95: float = 0
    
    # Probability table
    prob_drop_5: float = 0
    prob_drop_10: float = 0
    prob_drop_15: float = 0
    prob_drop_20: float = 0
    prob_drop_25: float = 0
    prob_drop_30: float = 0
    prob_drop_35: float = 0
    prob_drop_40: float = 0
    prob_drop_45: float = 0
    prob_drop_50: float = 0
    
    # VaR and CVaR (as price levels)
    var_1pct: float = 0   # 1% worst case price
    var_5pct: float = 0   # 5% worst case price
    var_10pct: float = 0  # 10% worst case price
    cvar_1pct: float = 0  # Expected price in worst 1%
    cvar_5pct: float = 0  # Expected price in worst 5%
    
    # Metadata
    n_paths: int = 0
    horizon_days: int = 0
    regime: str = "normal"
    execution_time_seconds: float = 0
    simulation_date: str = ""
    
    def get_probability_table(self) -> pd.DataFrame:
        """Return probability table as DataFrame."""
        return pd.DataFrame({
            'drop_pct': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'probability': [
                self.prob_drop_5, self.prob_drop_10, self.prob_drop_15,
                self.prob_drop_20, self.prob_drop_25, self.prob_drop_30,
                self.prob_drop_35, self.prob_drop_40, self.prob_drop_45,
                self.prob_drop_50
            ],
            'price_level': [
                self.current_price * (1 - pct/100) 
                for pct in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            ]
        })
    
    def get_percentile_paths(self) -> pd.DataFrame:
        """Get percentile paths over time (requires all_paths to be stored)."""
        if self.all_paths is None:
            raise ValueError("Paths not stored. Run simulation with store_paths=True")
        
        percentiles = [5, 25, 50, 75, 95]
        data = {}
        
        for p in percentiles:
            data[f'p{p}'] = np.percentile(self.all_paths, p, axis=0)
        
        df = pd.DataFrame(data)
        df['day'] = range(len(df))
        
        return df
    
    def __str__(self) -> str:
        return (
            f"Simulation Results ({self.simulation_date}, {self.regime} regime)\n"
            f"{'=' * 50}\n"
            f"Paths: {self.n_paths:,}, Horizon: {self.horizon_days} days\n"
            f"Execution time: {self.execution_time_seconds:.2f}s\n\n"
            f"Current price: ${self.current_price:,.0f}\n"
            f"Mean terminal:  ${self.mean_price:,.0f}\n"
            f"Median terminal: ${self.median_price:,.0f}\n\n"
            f"Percentiles (day {self.horizon_days}):\n"
            f"  5th:  ${self.percentile_5:,.0f} ({(self.percentile_5/self.current_price - 1)*100:+.1f}%)\n"
            f"  25th: ${self.percentile_25:,.0f} ({(self.percentile_25/self.current_price - 1)*100:+.1f}%)\n"
            f"  75th: ${self.percentile_75:,.0f} ({(self.percentile_75/self.current_price - 1)*100:+.1f}%)\n"
            f"  95th: ${self.percentile_95:,.0f} ({(self.percentile_95/self.current_price - 1)*100:+.1f}%)\n\n"
            f"VaR (price levels):\n"
            f"  1%:  ${self.var_1pct:,.0f}\n"
            f"  5%:  ${self.var_5pct:,.0f}\n"
            f"  10%: ${self.var_10pct:,.0f}\n"
        )


class MonteCarloEngine:
    """Monte Carlo simulation engine for BTC price paths."""
    
    def __init__(self):
        self.garch_model = GARCHModel()
        self._last_results: Optional[SimulationResults] = None
    
    def calibrate_jumps(self, returns: Optional[pd.Series] = None) -> JumpParams:
        """
        Calibrate jump parameters from historical returns.
        
        Identifies jumps as returns exceeding 3 standard deviations.
        
        Args:
            returns: Optional pre-loaded returns series
            
        Returns:
            Calibrated jump parameters
        """
        if returns is None:
            fetcher = PriceFetcher(use_cryptoquant=False)
            df = fetcher.load_from_db()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Identify jumps: returns beyond 3 standard deviations
        mean_ret = returns.mean()
        std_ret = returns.std()
        threshold = 3 * std_ret
        
        jumps = returns[np.abs(returns - mean_ret) > threshold]
        
        # Count jumps per year
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        lambda_annual = len(jumps) / years if years > 0 else 10
        
        # Jump size statistics (focus on negative jumps for risk)
        negative_jumps = jumps[jumps < 0]
        
        if len(negative_jumps) > 0:
            mu_jump = negative_jumps.mean()
            sigma_jump = negative_jumps.std() if len(negative_jumps) > 1 else abs(mu_jump) * 0.5
        else:
            # Conservative defaults if no negative jumps found
            mu_jump = -0.07
            sigma_jump = 0.05
        
        params = JumpParams(
            lambda_annual=lambda_annual,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump
        )
        
        print(f"Jump calibration from {len(returns)} observations:")
        print(f"  Found {len(jumps)} jumps ({len(negative_jumps)} negative)")
        print(params)
        
        return params
    
    def _get_current_price(self) -> float:
        """Get the latest BTC price from database."""
        fetcher = PriceFetcher(use_cryptoquant=False)
        df = fetcher.load_from_db()
        return df['close'].iloc[-1]
    
    def _apply_regime_adjustment(
        self, 
        params: SimulationParams
    ) -> SimulationParams:
        """
        Adjust parameters based on regime.
        
        In STRESS regime:
        - Increase jump intensity by 50%
        - Shift mean jump more negative by 2%
        - Set drift to zero
        """
        if params.regime == "stress":
            params.lambda_annual *= 1.5
            params.mu_jump -= 0.02  # More negative
            params.mu = 0  # No drift in stress
        
        return params
    
    def run(
        self,
        n_paths: int = 100_000,
        horizon_days: int = 30,
        regime: str = "normal",
        store_paths: bool = False,
        random_seed: Optional[int] = None,
        current_price: Optional[float] = None,
        garch_params: Optional[GARCHParams] = None,
        jump_params: Optional[JumpParams] = None
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_paths: Number of simulation paths
            horizon_days: Days to simulate
            regime: "normal" or "stress"
            store_paths: If True, store all paths (memory intensive)
            random_seed: Random seed for reproducibility
            current_price: Starting price (default: latest from DB)
            garch_params: Pre-fitted GARCH params (default: fit from data)
            jump_params: Pre-calibrated jump params (default: calibrate from data)
            
        Returns:
            SimulationResults object with all outputs
        """
        start_time = time.time()
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            # Use date-based seed for daily reproducibility
            np.random.seed(int(datetime.now().strftime('%Y%m%d')))
        
        # Get current price
        if current_price is None:
            current_price = self._get_current_price()
        
        print(f"Running Monte Carlo simulation...")
        print(f"  Paths: {n_paths:,}, Horizon: {horizon_days} days")
        print(f"  Current price: ${current_price:,.0f}")
        print(f"  Regime: {regime}")
        
        # Get GARCH parameters
        if garch_params is None:
            # Try to load from DB, otherwise fit
            garch_params = self.garch_model.load_latest_calibration()
            if garch_params is None:
                print("  Fitting GARCH model...")
                garch_params = self.garch_model.fit()
        
        # Get jump parameters
        if jump_params is None:
            jump_params = self.calibrate_jumps()
        
        # Get current variance from GARCH (need to refit to get conditional var)
        if self.garch_model._fitted_model is not None:
            current_var = self.garch_model._fitted_model.conditional_volatility.iloc[-1] ** 2
        else:
            # Use long-run variance as fallback
            current_var = garch_params.long_run_variance
        
        # Build simulation parameters
        sim_params = SimulationParams(
            current_price=current_price,
            current_variance=current_var,
            omega=garch_params.omega,
            alpha=garch_params.alpha,
            beta=garch_params.beta,
            mu=garch_params.mu / 100,  # Convert from percentage to decimal
            lambda_annual=jump_params.lambda_annual,
            mu_jump=jump_params.mu_jump,
            sigma_jump=jump_params.sigma_jump,
            n_paths=n_paths,
            horizon_days=horizon_days,
            regime=regime
        )
        
        # Apply regime adjustments
        sim_params = self._apply_regime_adjustment(sim_params)
        
        # Run vectorized simulation
        paths = self._simulate_paths(sim_params)
        
        # Extract terminal prices
        terminal_prices = paths[:, -1]
        
        # Calculate statistics
        results = self._compute_statistics(
            terminal_prices=terminal_prices,
            all_paths=paths if store_paths else None,
            current_price=current_price,
            n_paths=n_paths,
            horizon_days=horizon_days,
            regime=regime,
            start_time=start_time
        )
        
        self._last_results = results
        
        print(f"\nSimulation complete in {results.execution_time_seconds:.2f}s")
        
        return results
    
    def _simulate_paths(self, params: SimulationParams) -> np.ndarray:
        """
        Generate price paths using GARCH + jump diffusion.
        
        Vectorized implementation for performance.
        
        Returns:
            Array of shape (n_paths, horizon_days + 1)
        """
        n = params.n_paths
        T = params.horizon_days
        
        # Initialize arrays
        prices = np.zeros((n, T + 1))
        prices[:, 0] = params.current_price
        
        # Initialize variance at current level
        variance = np.full(n, params.current_variance)
        
        # Daily jump probability
        lambda_daily = params.lambda_annual / 365.25
        
        # Pre-generate all random numbers (faster)
        z_returns = np.random.standard_normal((n, T))
        z_jumps = np.random.standard_normal((n, T))
        jump_occurs = np.random.poisson(lambda_daily, (n, T))
        
        # Simulate day by day
        for t in range(T):
            # GARCH volatility (standard deviation)
            sigma = np.sqrt(variance)
            
            # Continuous return component (in log terms)
            r_continuous = params.mu + sigma * z_returns[:, t] / 100  # Convert back from %
            
            # Jump component
            jump_size = np.where(
                jump_occurs[:, t] > 0,
                params.mu_jump + params.sigma_jump * z_jumps[:, t],
                0
            )
            
            # Total log return
            log_return = r_continuous + jump_size
            
            # Update price
            prices[:, t + 1] = prices[:, t] * np.exp(log_return)
            
            # Update variance (GARCH dynamics)
            # Note: we use the standardized return (z) for GARCH update
            shock = z_returns[:, t] ** 2
            variance = (
                params.omega + 
                params.alpha * shock * variance + 
                params.beta * variance
            )
        
        return prices
    
    def _compute_statistics(
        self,
        terminal_prices: np.ndarray,
        all_paths: Optional[np.ndarray],
        current_price: float,
        n_paths: int,
        horizon_days: int,
        regime: str,
        start_time: float
    ) -> SimulationResults:
        """Compute all statistics from simulation results."""
        
        # Drop probabilities
        def prob_drop(pct):
            threshold = current_price * (1 - pct / 100)
            return np.mean(terminal_prices < threshold)
        
        # VaR (price at given percentile - lower is worse)
        var_1 = np.percentile(terminal_prices, 1)
        var_5 = np.percentile(terminal_prices, 5)
        var_10 = np.percentile(terminal_prices, 10)
        
        # CVaR (expected value below VaR)
        cvar_1 = np.mean(terminal_prices[terminal_prices <= var_1])
        cvar_5 = np.mean(terminal_prices[terminal_prices <= var_5])
        
        return SimulationResults(
            terminal_prices=terminal_prices,
            all_paths=all_paths,
            current_price=current_price,
            mean_price=np.mean(terminal_prices),
            median_price=np.median(terminal_prices),
            std_price=np.std(terminal_prices),
            percentile_5=np.percentile(terminal_prices, 5),
            percentile_25=np.percentile(terminal_prices, 25),
            percentile_75=np.percentile(terminal_prices, 75),
            percentile_95=np.percentile(terminal_prices, 95),
            prob_drop_5=prob_drop(5),
            prob_drop_10=prob_drop(10),
            prob_drop_15=prob_drop(15),
            prob_drop_20=prob_drop(20),
            prob_drop_25=prob_drop(25),
            prob_drop_30=prob_drop(30),
            prob_drop_35=prob_drop(35),
            prob_drop_40=prob_drop(40),
            prob_drop_45=prob_drop(45),
            prob_drop_50=prob_drop(50),
            var_1pct=var_1,
            var_5pct=var_5,
            var_10pct=var_10,
            cvar_1pct=cvar_1,
            cvar_5pct=cvar_5,
            n_paths=n_paths,
            horizon_days=horizon_days,
            regime=regime,
            execution_time_seconds=time.time() - start_time,
            simulation_date=datetime.now().strftime('%Y-%m-%d %H:%M')
        )


def run_simulation(
    n_paths: int = 100_000,
    horizon_days: int = 30,
    regime: str = "normal"
) -> SimulationResults:
    """
    Convenience function to run a simulation with defaults.
    
    Args:
        n_paths: Number of paths
        horizon_days: Simulation horizon
        regime: "normal" or "stress"
        
    Returns:
        SimulationResults
    """
    engine = MonteCarloEngine()
    return engine.run(
        n_paths=n_paths,
        horizon_days=horizon_days,
        regime=regime,
        store_paths=True
    )


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    n_paths = 100_000
    regime = "normal"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--paths="):
            n_paths = int(arg.split("=")[1])
        elif arg == "--stress":
            regime = "stress"
    
    print("=" * 60)
    print("BTC Monte Carlo Simulation")
    print("=" * 60)
    
    results = run_simulation(n_paths=n_paths, regime=regime)
    
    print("\n" + str(results))
    
    print("\nProbability of drops:")
    prob_table = results.get_probability_table()
    for _, row in prob_table.iterrows():
        print(f"  ≥{row['drop_pct']:2.0f}%: {row['probability']*100:5.2f}%  (${row['price_level']:,.0f})")
