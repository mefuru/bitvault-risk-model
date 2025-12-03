"""
Regime classification using market and on-chain data.

Classifies market as NORMAL or STRESS based on:

Market Indicators:
- VIX level
- BTC realized volatility
- BTC drawdown from recent high
- BTC-S&P 500 correlation

On-Chain Indicators (CryptoQuant):
- Exchange netflow (selling pressure)
- Funding rates (market sentiment)
- SOPR (realized profit/loss)
- MVRV (market vs realized value)
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.database import get_db_path
from src.data.prices import PriceFetcher
from src.data.macro import MacroFetcher
from src.logging_config import get_logger

logger = get_logger("regime")


@dataclass
class RegimeIndicator:
    """Single regime indicator with current value and threshold."""
    name: str
    value: float
    threshold: float
    is_stress: bool
    description: str
    
    def __str__(self) -> str:
        status = "🔴 STRESS" if self.is_stress else "🟢 Normal"
        return f"{self.name}: {self.value:.2f} (threshold: {self.threshold}) → {status}"


@dataclass
class RegimeClassification:
    """Complete regime classification result."""
    regime: str  # "normal" or "stress"
    indicators: list[RegimeIndicator]
    stress_count: int
    total_indicators: int
    classification_date: str
    
    @property
    def stress_ratio(self) -> float:
        """Proportion of indicators signaling stress."""
        return self.stress_count / self.total_indicators if self.total_indicators > 0 else 0
    
    def __str__(self) -> str:
        header = f"{'🔴 STRESS' if self.regime == 'stress' else '🟢 NORMAL'} REGIME"
        lines = [
            "=" * 55,
            f"Regime Classification: {header}",
            f"Date: {self.classification_date}",
            f"Stress indicators: {self.stress_count}/{self.total_indicators}",
            "=" * 55,
            ""
        ]
        
        for indicator in self.indicators:
            lines.append(str(indicator))
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/API."""
        return {
            "regime": self.regime,
            "stress_count": self.stress_count,
            "total_indicators": self.total_indicators,
            "stress_ratio": self.stress_ratio,
            "classification_date": self.classification_date,
            "indicators": {
                ind.name: {
                    "value": ind.value,
                    "threshold": ind.threshold,
                    "is_stress": ind.is_stress
                }
                for ind in self.indicators
            }
        }


class RegimeClassifier:
    """Classifies market regime based on available indicators."""
    
    # Thresholds for stress signals
    THRESHOLDS = {
        # Market indicators
        "vix": 30.0,                    # VIX above 30 = fear
        "btc_volatility_multiple": 1.5,  # Current vol > 1.5x average = elevated
        "btc_drawdown": -0.15,          # Down 15% from 30-day high
        "correlation_stress": 0.7,       # High BTC-SPX correlation during down move
        
        # On-chain indicators (CryptoQuant)
        "netflow_stress": 10000,         # >10k BTC flowing to exchanges = selling pressure
        "funding_rate_stress": -0.01,    # Negative funding = bearish sentiment
        "sopr_stress": 0.98,             # SOPR < 0.98 = selling at loss (capitulation)
        "mvrv_low": 1.0,                 # MVRV < 1 = historically oversold
        "mvrv_high": 3.5,                # MVRV > 3.5 = historically overbought
    }
    
    def __init__(self, use_onchain: bool = True):
        """
        Initialize classifier.
        
        Args:
            use_onchain: Whether to use CryptoQuant on-chain indicators
        """
        self.db_path = get_db_path()
        self.price_fetcher = PriceFetcher(use_cryptoquant=False)
        self.macro_fetcher = MacroFetcher()
        self.use_onchain = use_onchain
    
    def _load_btc_prices(self, days: int = 90) -> pd.DataFrame:
        """Load recent BTC prices."""
        df = self.price_fetcher.load_from_db()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df.tail(days)
    
    def _load_macro_data(self, indicator: str, days: int = 90) -> pd.Series:
        """Load recent macro indicator data."""
        df = self.macro_fetcher.load_from_db(indicator=indicator)
        if df.empty:
            return pd.Series(dtype=float)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['value'].tail(days)
    
    def _calculate_vix_indicator(self) -> RegimeIndicator:
        """Check if VIX is above stress threshold."""
        vix = self._load_macro_data('vix', days=5)
        
        if vix.empty:
            # Default to non-stress if no data
            return RegimeIndicator(
                name="VIX",
                value=0,
                threshold=self.THRESHOLDS["vix"],
                is_stress=False,
                description="No VIX data available"
            )
        
        current_vix = vix.iloc[-1]
        is_stress = current_vix > self.THRESHOLDS["vix"]
        
        return RegimeIndicator(
            name="VIX",
            value=current_vix,
            threshold=self.THRESHOLDS["vix"],
            is_stress=is_stress,
            description=f"VIX at {current_vix:.1f}, threshold {self.THRESHOLDS['vix']}"
        )
    
    def _calculate_volatility_indicator(self) -> RegimeIndicator:
        """Check if BTC realized volatility is elevated."""
        prices = self._load_btc_prices(days=90)
        
        if len(prices) < 30:
            return RegimeIndicator(
                name="BTC Volatility",
                value=0,
                threshold=self.THRESHOLDS["btc_volatility_multiple"],
                is_stress=False,
                description="Insufficient price data"
            )
        
        # Calculate log returns
        returns = np.log(prices['close'] / prices['close'].shift(1)).dropna()
        
        # Current volatility (last 7 days, annualized)
        current_vol = returns.tail(7).std() * np.sqrt(365) * 100
        
        # Average volatility (full period)
        avg_vol = returns.std() * np.sqrt(365) * 100
        
        # Multiple of average
        vol_multiple = current_vol / avg_vol if avg_vol > 0 else 1
        
        is_stress = vol_multiple > self.THRESHOLDS["btc_volatility_multiple"]
        
        return RegimeIndicator(
            name="BTC Volatility",
            value=vol_multiple,
            threshold=self.THRESHOLDS["btc_volatility_multiple"],
            is_stress=is_stress,
            description=f"Current vol {current_vol:.1f}% is {vol_multiple:.2f}x average ({avg_vol:.1f}%)"
        )
    
    def _calculate_drawdown_indicator(self) -> RegimeIndicator:
        """Check if BTC is in significant drawdown."""
        prices = self._load_btc_prices(days=30)
        
        if len(prices) < 7:
            return RegimeIndicator(
                name="BTC Drawdown",
                value=0,
                threshold=self.THRESHOLDS["btc_drawdown"],
                is_stress=False,
                description="Insufficient price data"
            )
        
        # Current price vs 30-day high
        current_price = prices['close'].iloc[-1]
        high_30d = prices['close'].max()
        
        drawdown = (current_price / high_30d) - 1  # Negative number
        
        is_stress = drawdown < self.THRESHOLDS["btc_drawdown"]
        
        return RegimeIndicator(
            name="BTC Drawdown",
            value=drawdown * 100,  # As percentage
            threshold=self.THRESHOLDS["btc_drawdown"] * 100,
            is_stress=is_stress,
            description=f"Current ${current_price:,.0f} vs 30d high ${high_30d:,.0f}"
        )
    
    def _calculate_correlation_indicator(self) -> RegimeIndicator:
        """Check BTC-SPX correlation (high correlation in down market = stress)."""
        btc_prices = self._load_btc_prices(days=30)
        spx = self._load_macro_data('sp500', days=30)
        
        if len(btc_prices) < 14 or len(spx) < 14:
            return RegimeIndicator(
                name="BTC-SPX Correlation",
                value=0,
                threshold=self.THRESHOLDS["correlation_stress"],
                is_stress=False,
                description="Insufficient data for correlation"
            )
        
        # Align dates
        btc_returns = np.log(btc_prices['close'] / btc_prices['close'].shift(1)).dropna()
        spx_returns = np.log(spx / spx.shift(1)).dropna()
        
        # Get common dates
        common_dates = btc_returns.index.intersection(spx_returns.index)
        
        if len(common_dates) < 10:
            return RegimeIndicator(
                name="BTC-SPX Correlation",
                value=0,
                threshold=self.THRESHOLDS["correlation_stress"],
                is_stress=False,
                description="Insufficient overlapping data"
            )
        
        btc_aligned = btc_returns.loc[common_dates]
        spx_aligned = spx_returns.loc[common_dates]
        
        correlation = btc_aligned.corr(spx_aligned)
        
        # Check if SPX is down (risk-off environment)
        spx_return_total = (spx.iloc[-1] / spx.iloc[0]) - 1 if len(spx) > 0 else 0
        spx_down = spx_return_total < -0.02  # SPX down >2%
        
        # Stress = high correlation AND SPX down (correlated selling)
        is_stress = correlation > self.THRESHOLDS["correlation_stress"] and spx_down
        
        return RegimeIndicator(
            name="BTC-SPX Correlation",
            value=correlation,
            threshold=self.THRESHOLDS["correlation_stress"],
            is_stress=is_stress,
            description=f"Correlation {correlation:.2f}, SPX {'down' if spx_down else 'up/flat'} {spx_return_total*100:.1f}%"
        )
    
    # =========================================================================
    # ON-CHAIN INDICATORS (CryptoQuant)
    # =========================================================================
    
    def _load_onchain_metric(self, metric: str, days: int = 7) -> pd.Series:
        """Load on-chain metric from database."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(
                """
                SELECT date, value FROM onchain_metrics 
                WHERE metric = ? 
                ORDER BY date DESC 
                LIMIT ?
                """,
                conn,
                params=(metric, days)
            )
            
            if df.empty:
                return pd.Series(dtype=float)
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df['value']
            
        except Exception as e:
            logger.warning(f"Error loading {metric}: {e}")
            return pd.Series(dtype=float)
        finally:
            conn.close()
    
    def _load_exchange_netflow(self, days: int = 7) -> pd.Series:
        """Load exchange netflow from database."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(
                """
                SELECT date, net_flow FROM exchange_flows 
                ORDER BY date DESC 
                LIMIT ?
                """,
                conn,
                params=(days,)
            )
            
            if df.empty:
                return pd.Series(dtype=float)
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df['net_flow']
            
        except Exception as e:
            logger.warning(f"Error loading netflow: {e}")
            return pd.Series(dtype=float)
        finally:
            conn.close()
    
    def _load_funding_rates(self, days: int = 7) -> pd.Series:
        """Load funding rates from database."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(
                """
                SELECT date, funding_rate FROM funding_rates 
                ORDER BY date DESC 
                LIMIT ?
                """,
                conn,
                params=(days,)
            )
            
            if df.empty:
                return pd.Series(dtype=float)
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            return df['funding_rate']
            
        except Exception as e:
            logger.warning(f"Error loading funding rates: {e}")
            return pd.Series(dtype=float)
        finally:
            conn.close()
    
    def _calculate_netflow_indicator(self) -> Optional[RegimeIndicator]:
        """Check exchange netflow (high inflow = selling pressure)."""
        netflow = self._load_exchange_netflow(days=7)
        
        if netflow.empty:
            return None
        
        # Use 7-day average
        avg_netflow = netflow.mean()
        
        is_stress = avg_netflow > self.THRESHOLDS["netflow_stress"]
        
        return RegimeIndicator(
            name="Exchange Netflow",
            value=avg_netflow,
            threshold=self.THRESHOLDS["netflow_stress"],
            is_stress=is_stress,
            description=f"7d avg netflow: {avg_netflow:,.0f} BTC ({'inflow' if avg_netflow > 0 else 'outflow'})"
        )
    
    def _calculate_funding_indicator(self) -> Optional[RegimeIndicator]:
        """Check funding rates (negative = bearish sentiment)."""
        funding = self._load_funding_rates(days=7)
        
        if funding.empty:
            return None
        
        # Use latest funding rate
        current_funding = funding.iloc[-1]
        
        is_stress = current_funding < self.THRESHOLDS["funding_rate_stress"]
        
        return RegimeIndicator(
            name="Funding Rate",
            value=current_funding,
            threshold=self.THRESHOLDS["funding_rate_stress"],
            is_stress=is_stress,
            description=f"Funding: {current_funding*100:.3f}% ({'bearish' if current_funding < 0 else 'bullish'})"
        )
    
    def _calculate_sopr_indicator(self) -> Optional[RegimeIndicator]:
        """Check SOPR (< 1 = selling at loss = capitulation)."""
        sopr = self._load_onchain_metric('sopr', days=7)
        
        if sopr.empty:
            return None
        
        # Use 7-day average
        avg_sopr = sopr.mean()
        
        is_stress = avg_sopr < self.THRESHOLDS["sopr_stress"]
        
        return RegimeIndicator(
            name="SOPR",
            value=avg_sopr,
            threshold=self.THRESHOLDS["sopr_stress"],
            is_stress=is_stress,
            description=f"SOPR: {avg_sopr:.3f} ({'loss' if avg_sopr < 1 else 'profit'})"
        )
    
    def _calculate_mvrv_indicator(self) -> Optional[RegimeIndicator]:
        """Check MVRV (extremes indicate overbought/oversold)."""
        mvrv = self._load_onchain_metric('mvrv', days=7)
        
        if mvrv.empty:
            return None
        
        current_mvrv = mvrv.iloc[-1]
        
        # Stress if extremely overbought OR oversold
        is_stress = (current_mvrv > self.THRESHOLDS["mvrv_high"] or 
                    current_mvrv < self.THRESHOLDS["mvrv_low"])
        
        if current_mvrv > self.THRESHOLDS["mvrv_high"]:
            status = "overbought"
        elif current_mvrv < self.THRESHOLDS["mvrv_low"]:
            status = "oversold"
        else:
            status = "neutral"
        
        return RegimeIndicator(
            name="MVRV",
            value=current_mvrv,
            threshold=self.THRESHOLDS["mvrv_high"],  # Show high threshold
            is_stress=is_stress,
            description=f"MVRV: {current_mvrv:.2f} ({status})"
        )
    
    def classify(self, stress_threshold: int = 2) -> RegimeClassification:
        """
        Classify current market regime.
        
        Args:
            stress_threshold: Number of stress indicators required for STRESS regime
            
        Returns:
            RegimeClassification with all indicator details
        """
        # Market indicators (always available)
        indicators = [
            self._calculate_vix_indicator(),
            self._calculate_volatility_indicator(),
            self._calculate_drawdown_indicator(),
            self._calculate_correlation_indicator(),
        ]
        
        # On-chain indicators (if enabled and available)
        if self.use_onchain:
            onchain_indicators = [
                self._calculate_netflow_indicator(),
                self._calculate_funding_indicator(),
                self._calculate_sopr_indicator(),
                self._calculate_mvrv_indicator(),
            ]
            
            # Only add indicators that have data
            for ind in onchain_indicators:
                if ind is not None:
                    indicators.append(ind)
        
        stress_count = sum(1 for ind in indicators if ind.is_stress)
        
        # STRESS if enough indicators trigger
        regime = "stress" if stress_count >= stress_threshold else "normal"
        
        logger.info(f"Regime: {regime} ({stress_count}/{len(indicators)} stress signals)")
        
        return RegimeClassification(
            regime=regime,
            indicators=indicators,
            stress_count=stress_count,
            total_indicators=len(indicators),
            classification_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    def get_regime(self) -> str:
        """Simple method to get just the regime string."""
        return self.classify().regime


def classify_regime(use_onchain: bool = True) -> RegimeClassification:
    """Convenience function to classify current regime."""
    classifier = RegimeClassifier(use_onchain=use_onchain)
    return classifier.classify()


def get_current_regime() -> str:
    """Get just the regime string ('normal' or 'stress')."""
    classifier = RegimeClassifier()
    return classifier.get_regime()


if __name__ == "__main__":
    print("=" * 55)
    print("Market Regime Classification")
    print("=" * 55)
    
    classification = classify_regime()
    print(classification)
    
    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    print(f"Regime: {classification.regime.upper()}")
    print(f"Stress signals: {classification.stress_count}/{classification.total_indicators}")
    
    if classification.regime == "stress":
        print("\n⚠️  STRESS regime detected - model will use elevated risk parameters")
    else:
        print("\n✓ Normal regime - model will use standard parameters")
