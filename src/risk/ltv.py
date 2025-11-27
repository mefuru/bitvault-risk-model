"""
LTV (Loan-to-Value) risk calculations for BitVault.

Connects Monte Carlo price simulations to collateral management.

LTV Calculation:
    LTV = Total Debt / Collateral Value
    where Total Debt = Loan Amount + Accrued Interest
    and Collateral Value = BTC Quantity × BTC Price
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.model.simulation import MonteCarloEngine, SimulationResults


@dataclass
class PortfolioState:
    """Current state of a collateralized position."""
    btc_collateral: float      # BTC amount held as collateral
    loan_amount: float         # Initial bvUSD loan amount
    accrued_interest: float    # Accrued interest on loan
    btc_price: float           # Current BTC price
    
    @property
    def total_debt(self) -> float:
        """Total debt = loan + accrued interest."""
        return self.loan_amount + self.accrued_interest
    
    @property
    def collateral_value(self) -> float:
        """Current USD value of collateral."""
        return self.btc_collateral * self.btc_price
    
    @property
    def current_ltv(self) -> float:
        """Current LTV ratio (0-1 scale)."""
        if self.collateral_value == 0:
            return float('inf')
        return self.total_debt / self.collateral_value
    
    @property
    def current_ltv_pct(self) -> float:
        """Current LTV as percentage."""
        return self.current_ltv * 100
    
    def ltv_at_price(self, price: float) -> float:
        """Calculate LTV at a given BTC price."""
        collateral_value = self.btc_collateral * price
        if collateral_value == 0:
            return float('inf')
        return self.total_debt / collateral_value
    
    def price_at_ltv(self, ltv: float) -> float:
        """
        Calculate BTC price that would result in given LTV.
        
        Formula derivation:
            LTV = Total Debt / (BTC Quantity × Price)
            Price = Total Debt / (LTV × BTC Quantity)
        """
        if ltv == 0 or self.btc_collateral == 0:
            return float('inf')
        return self.total_debt / (ltv * self.btc_collateral)
    
    def collateral_value_at_ltv(self, ltv: float) -> float:
        """
        Calculate required collateral value at a given LTV.
        
        Formula: Collateral Value = Total Debt / LTV
        """
        if ltv == 0:
            return float('inf')
        return self.total_debt / ltv
    
    def drop_to_ltv(self, target_ltv: float) -> float:
        """
        Calculate percentage price drop required to reach target LTV.
        
        Returns:
            Percentage drop (e.g., 0.25 for 25% drop)
        """
        target_price = self.price_at_ltv(target_ltv)
        if self.btc_price == 0:
            return 0
        return 1 - (target_price / self.btc_price)
    
    def __str__(self) -> str:
        return (
            f"Portfolio State:\n"
            f"  BTC Collateral: {self.btc_collateral:.4f} BTC\n"
            f"  Loan Amount: ${self.loan_amount:,.2f}\n"
            f"  Accrued Interest: ${self.accrued_interest:,.2f}\n"
            f"  Total Debt: ${self.total_debt:,.2f}\n"
            f"  BTC Price: ${self.btc_price:,.2f}\n"
            f"  Collateral Value: ${self.collateral_value:,.2f}\n"
            f"  Current LTV: {self.current_ltv_pct:.1f}%"
        )


@dataclass
class LTVThresholds:
    """LTV thresholds for risk management."""
    margin_call: float = 0.85    # 85% LTV triggers margin call
    liquidation: float = 0.95    # 95% LTV triggers liquidation
    
    def __str__(self) -> str:
        return (
            f"LTV Thresholds:\n"
            f"  Margin Call: {self.margin_call * 100:.0f}%\n"
            f"  Liquidation: {self.liquidation * 100:.0f}%"
        )


@dataclass
class LTVRiskMetrics:
    """Risk metrics for LTV breaches."""
    # Current state
    current_ltv: float
    current_price: float
    total_debt: float
    
    # Threshold prices
    margin_call_price: float
    liquidation_price: float
    
    # Collateral values at thresholds
    margin_call_collateral: float
    liquidation_collateral: float
    
    # Required drops to hit thresholds
    drop_to_margin_call: float  # Percentage drop required (0-1)
    drop_to_liquidation: float
    
    # Probabilities from simulation
    prob_margin_call: float     # P(LTV >= 85%) within horizon
    prob_liquidation: float     # P(LTV >= 95%) within horizon
    
    # Buffer analysis
    margin_call_buffer: float   # How much price can drop before margin call
    liquidation_buffer: float   # How much price can drop before liquidation
    
    # Time analysis (if paths stored)
    expected_days_to_margin_call: Optional[float] = None
    expected_days_to_liquidation: Optional[float] = None
    
    # Simulation metadata
    horizon_days: int = 30
    n_paths: int = 100_000
    regime: str = "normal"
    
    def __str__(self) -> str:
        return (
            f"LTV Risk Metrics ({self.horizon_days}-day horizon, {self.regime} regime)\n"
            f"{'=' * 55}\n"
            f"Current State:\n"
            f"  LTV: {self.current_ltv * 100:.1f}%\n"
            f"  BTC Price: ${self.current_price:,.0f}\n"
            f"  Total Debt: ${self.total_debt:,.0f}\n\n"
            f"Threshold Analysis:\n"
            f"  Margin Call (85% LTV):\n"
            f"    Trigger price: ${self.margin_call_price:,.0f}\n"
            f"    Required drop: {self.drop_to_margin_call * 100:.1f}%\n"
            f"    Buffer: ${self.margin_call_buffer:,.0f}\n"
            f"    Probability: {self.prob_margin_call * 100:.2f}%\n\n"
            f"  Liquidation (95% LTV):\n"
            f"    Trigger price: ${self.liquidation_price:,.0f}\n"
            f"    Required drop: {self.drop_to_liquidation * 100:.1f}%\n"
            f"    Buffer: ${self.liquidation_buffer:,.0f}\n"
            f"    Probability: {self.prob_liquidation * 100:.2f}%\n"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for dashboard/API."""
        return {
            'current_ltv': self.current_ltv,
            'current_price': self.current_price,
            'total_debt': self.total_debt,
            'margin_call_price': self.margin_call_price,
            'liquidation_price': self.liquidation_price,
            'margin_call_collateral': self.margin_call_collateral,
            'liquidation_collateral': self.liquidation_collateral,
            'drop_to_margin_call': self.drop_to_margin_call,
            'drop_to_liquidation': self.drop_to_liquidation,
            'prob_margin_call': self.prob_margin_call,
            'prob_liquidation': self.prob_liquidation,
            'margin_call_buffer': self.margin_call_buffer,
            'liquidation_buffer': self.liquidation_buffer,
            'horizon_days': self.horizon_days,
            'regime': self.regime
        }


class LTVRiskCalculator:
    """Calculates LTV risk metrics from Monte Carlo simulations."""
    
    def __init__(self, thresholds: Optional[LTVThresholds] = None):
        """
        Initialize calculator.
        
        Args:
            thresholds: LTV thresholds (default: 85% margin call, 95% liquidation)
        """
        self.thresholds = thresholds or LTVThresholds()
        self.engine = MonteCarloEngine()
    
    def calculate_threshold_prices(
        self, 
        portfolio: PortfolioState
    ) -> tuple[float, float]:
        """
        Calculate BTC prices that trigger margin call and liquidation.
        
        Returns:
            Tuple of (margin_call_price, liquidation_price)
        """
        margin_call_price = portfolio.price_at_ltv(self.thresholds.margin_call)
        liquidation_price = portfolio.price_at_ltv(self.thresholds.liquidation)
        
        return margin_call_price, liquidation_price
    
    def calculate_risk(
        self,
        portfolio: PortfolioState,
        simulation_results: Optional[SimulationResults] = None,
        n_paths: int = 100_000,
        horizon_days: int = 30,
        regime: str = "normal"
    ) -> LTVRiskMetrics:
        """
        Calculate comprehensive LTV risk metrics.
        
        Args:
            portfolio: Current portfolio state
            simulation_results: Pre-computed simulation (or None to run new)
            n_paths: Simulation paths (if running new simulation)
            horizon_days: Simulation horizon
            regime: "normal" or "stress"
            
        Returns:
            LTVRiskMetrics with all calculated values
        """
        # Run simulation if not provided
        if simulation_results is None:
            simulation_results = self.engine.run(
                n_paths=n_paths,
                horizon_days=horizon_days,
                regime=regime,
                current_price=portfolio.btc_price,
                store_paths=True
            )
        
        # Calculate threshold prices
        margin_call_price, liquidation_price = self.calculate_threshold_prices(portfolio)
        
        # Calculate collateral values at thresholds
        margin_call_collateral = portfolio.collateral_value_at_ltv(self.thresholds.margin_call)
        liquidation_collateral = portfolio.collateral_value_at_ltv(self.thresholds.liquidation)
        
        # Calculate required drops
        drop_to_mc = portfolio.drop_to_ltv(self.thresholds.margin_call)
        drop_to_liq = portfolio.drop_to_ltv(self.thresholds.liquidation)
        
        # Calculate probabilities from terminal prices
        terminal_prices = simulation_results.terminal_prices
        
        prob_margin_call = np.mean(terminal_prices <= margin_call_price)
        prob_liquidation = np.mean(terminal_prices <= liquidation_price)
        
        # Calculate buffers
        margin_call_buffer = portfolio.btc_price - margin_call_price
        liquidation_buffer = portfolio.btc_price - liquidation_price
        
        # Time analysis if paths are stored
        expected_days_mc = None
        expected_days_liq = None
        
        if simulation_results.all_paths is not None:
            expected_days_mc = self._expected_days_to_breach(
                simulation_results.all_paths,
                margin_call_price
            )
            expected_days_liq = self._expected_days_to_breach(
                simulation_results.all_paths,
                liquidation_price
            )
        
        return LTVRiskMetrics(
            current_ltv=portfolio.current_ltv,
            current_price=portfolio.btc_price,
            total_debt=portfolio.total_debt,
            margin_call_price=margin_call_price,
            liquidation_price=liquidation_price,
            margin_call_collateral=margin_call_collateral,
            liquidation_collateral=liquidation_collateral,
            drop_to_margin_call=drop_to_mc,
            drop_to_liquidation=drop_to_liq,
            prob_margin_call=prob_margin_call,
            prob_liquidation=prob_liquidation,
            margin_call_buffer=margin_call_buffer,
            liquidation_buffer=liquidation_buffer,
            expected_days_to_margin_call=expected_days_mc,
            expected_days_to_liquidation=expected_days_liq,
            horizon_days=horizon_days,
            n_paths=n_paths,
            regime=regime
        )
    
    def _expected_days_to_breach(
        self, 
        paths: np.ndarray, 
        threshold_price: float
    ) -> Optional[float]:
        """
        Calculate expected days until price breaches threshold.
        
        Only considers paths that actually breach.
        
        Returns:
            Expected days to breach, or None if no breaches
        """
        # Find first day each path breaches threshold
        breached = paths <= threshold_price
        
        # For each path, find first True (if any)
        first_breach_day = np.argmax(breached, axis=1)
        
        # argmax returns 0 if no True found, so filter those out
        actually_breached = breached.any(axis=1)
        
        if not actually_breached.any():
            return None
        
        breach_days = first_breach_day[actually_breached]
        
        return float(np.mean(breach_days))
    
    def generate_ltv_scenarios(
        self,
        portfolio: PortfolioState,
        price_drops: list[float] = None
    ) -> pd.DataFrame:
        """
        Generate table of LTV values at various price drops.
        
        Args:
            portfolio: Current portfolio state
            price_drops: List of drop percentages (default: 0 to 50 by 5)
            
        Returns:
            DataFrame with columns: drop_pct, price, ltv, status
        """
        if price_drops is None:
            price_drops = list(range(0, 55, 5))
        
        rows = []
        for drop in price_drops:
            price = portfolio.btc_price * (1 - drop / 100)
            ltv = portfolio.ltv_at_price(price)
            
            if ltv >= self.thresholds.liquidation:
                status = "LIQUIDATION"
            elif ltv >= self.thresholds.margin_call:
                status = "MARGIN CALL"
            else:
                status = "OK"
            
            rows.append({
                'drop_pct': drop,
                'price': price,
                'ltv': ltv,
                'ltv_pct': ltv * 100,
                'status': status
            })
        
        return pd.DataFrame(rows)


def analyze_portfolio(
    btc_collateral: float,
    loan_amount: float,
    accrued_interest: float = 0,
    btc_price: Optional[float] = None,
    regime: str = "normal"
) -> tuple[PortfolioState, LTVRiskMetrics]:
    """
    Convenience function to analyze a portfolio.
    
    Args:
        btc_collateral: BTC amount
        loan_amount: Initial loan amount
        accrued_interest: Accrued interest (default 0)
        btc_price: Current BTC price (default: fetch from DB)
        regime: "normal" or "stress"
        
    Returns:
        Tuple of (PortfolioState, LTVRiskMetrics)
    """
    from src.data.prices import PriceFetcher
    
    # Get current price if not provided
    if btc_price is None:
        fetcher = PriceFetcher(use_cryptoquant=False)
        df = fetcher.load_from_db()
        btc_price = df['close'].iloc[-1]
    
    portfolio = PortfolioState(
        btc_collateral=btc_collateral,
        loan_amount=loan_amount,
        accrued_interest=accrued_interest,
        btc_price=btc_price
    )
    
    calculator = LTVRiskCalculator()
    risk_metrics = calculator.calculate_risk(portfolio, regime=regime)
    
    return portfolio, risk_metrics


if __name__ == "__main__":
    import sys
    
    # Default example values from spreadsheet
    btc_collateral = 100.0  # 100 BTC
    loan_amount = 5_350_000  # $5.35M loan
    accrued_interest = 20_858.15  # ~$21k accrued
    regime = "normal"
    
    # Parse arguments
    for arg in sys.argv[1:]:
        if arg.startswith("--btc="):
            btc_collateral = float(arg.split("=")[1])
        elif arg.startswith("--loan="):
            loan_amount = float(arg.split("=")[1])
        elif arg.startswith("--interest="):
            accrued_interest = float(arg.split("=")[1])
        elif arg == "--stress":
            regime = "stress"
    
    print("=" * 60)
    print("BitVault LTV Risk Analysis")
    print("=" * 60)
    
    portfolio, risk_metrics = analyze_portfolio(
        btc_collateral=btc_collateral,
        loan_amount=loan_amount,
        accrued_interest=accrued_interest,
        regime=regime
    )
    
    print("\n" + str(portfolio))
    print("\n" + str(risk_metrics))
    
    # Scenario table
    print("\nLTV Scenario Analysis:")
    print("-" * 55)
    
    calculator = LTVRiskCalculator()
    scenarios = calculator.generate_ltv_scenarios(portfolio)
    
    for _, row in scenarios.iterrows():
        status_icon = "✓" if row['status'] == "OK" else "⚠️" if row['status'] == "MARGIN CALL" else "🔴"
        print(
            f"  {row['drop_pct']:3.0f}% drop → "
            f"${row['price']:>10,.0f} → "
            f"LTV {row['ltv_pct']:5.1f}%  {status_icon} {row['status']}"
        )
