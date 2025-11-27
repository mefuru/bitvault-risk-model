"""
Validation utilities for BitVault Risk Model.

Handles:
- Input validation for portfolio parameters
- Data freshness checks
- Sanity checks on model outputs
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.logging_config import get_logger
from src.data.database import get_db_path

import sqlite3

logger = get_logger("validation")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    
    def __bool__(self):
        return self.is_valid
    
    def __str__(self):
        lines = []
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        if self.is_valid and not self.warnings:
            lines.append("✅ All validations passed")
        return "\n".join(lines)


class PortfolioValidator:
    """Validates portfolio inputs."""
    
    # Validation limits
    MIN_BTC_COLLATERAL = 0.001
    MAX_BTC_COLLATERAL = 100_000  # 100k BTC
    MIN_LOAN_AMOUNT = 100
    MAX_LOAN_AMOUNT = 10_000_000_000  # $10B
    MAX_STARTING_LTV = 0.90  # Don't allow starting LTV above 90%
    WARNING_LTV = 0.75  # Warn if starting LTV above 75%
    
    @classmethod
    def validate(
        cls,
        btc_collateral: float,
        loan_amount: float,
        accrued_interest: float,
        btc_price: float
    ) -> ValidationResult:
        """
        Validate portfolio inputs.
        
        Args:
            btc_collateral: BTC amount
            loan_amount: Loan principal
            accrued_interest: Accrued interest
            btc_price: Current BTC price
            
        Returns:
            ValidationResult with any errors/warnings
        """
        errors = []
        warnings = []
        
        # BTC Collateral checks
        if btc_collateral <= 0:
            errors.append(f"BTC collateral must be positive (got {btc_collateral})")
        elif btc_collateral < cls.MIN_BTC_COLLATERAL:
            errors.append(f"BTC collateral too small (min {cls.MIN_BTC_COLLATERAL})")
        elif btc_collateral > cls.MAX_BTC_COLLATERAL:
            warnings.append(f"BTC collateral unusually large ({btc_collateral:,.0f} BTC)")
        
        # Loan amount checks
        if loan_amount < 0:
            errors.append(f"Loan amount cannot be negative (got {loan_amount})")
        elif loan_amount < cls.MIN_LOAN_AMOUNT:
            errors.append(f"Loan amount too small (min ${cls.MIN_LOAN_AMOUNT})")
        elif loan_amount > cls.MAX_LOAN_AMOUNT:
            warnings.append(f"Loan amount unusually large (${loan_amount:,.0f})")
        
        # Accrued interest checks
        if accrued_interest < 0:
            errors.append(f"Accrued interest cannot be negative (got {accrued_interest})")
        if loan_amount > 0 and accrued_interest > loan_amount:
            warnings.append(f"Accrued interest exceeds loan principal")
        
        # BTC price checks
        if btc_price <= 0:
            errors.append(f"BTC price must be positive (got {btc_price})")
        elif btc_price < 1000:
            warnings.append(f"BTC price seems too low (${btc_price:,.0f})")
        elif btc_price > 1_000_000:
            warnings.append(f"BTC price seems too high (${btc_price:,.0f})")
        
        # LTV check (only if we have valid inputs)
        if btc_collateral > 0 and btc_price > 0 and loan_amount >= 0:
            total_debt = loan_amount + accrued_interest
            collateral_value = btc_collateral * btc_price
            ltv = total_debt / collateral_value
            
            if ltv > cls.MAX_STARTING_LTV:
                errors.append(
                    f"Starting LTV too high ({ltv*100:.1f}%). "
                    f"Max allowed is {cls.MAX_STARTING_LTV*100:.0f}%. "
                    f"Position may already be at risk."
                )
            elif ltv > cls.WARNING_LTV:
                warnings.append(
                    f"Starting LTV is elevated ({ltv*100:.1f}%). "
                    f"Consider reducing loan or adding collateral."
                )
            elif ltv < 0.1:
                warnings.append(
                    f"Starting LTV very low ({ltv*100:.1f}%). "
                    f"Position is heavily over-collateralized."
                )
        
        is_valid = len(errors) == 0
        
        if errors:
            logger.warning(f"Portfolio validation failed: {errors}")
        if warnings:
            logger.info(f"Portfolio validation warnings: {warnings}")
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


class DataFreshnessChecker:
    """Checks if data is fresh enough for reliable analysis."""
    
    # Freshness thresholds
    STALE_THRESHOLD_DAYS = 2  # Data older than this triggers warning
    CRITICAL_THRESHOLD_DAYS = 7  # Data older than this triggers error
    
    @classmethod
    def check_freshness(cls) -> ValidationResult:
        """
        Check if all data sources are fresh.
        
        Returns:
            ValidationResult with freshness status
        """
        errors = []
        warnings = []
        
        db_path = get_db_path()
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Check BTC prices
            cursor = conn.execute("SELECT MAX(date) FROM btc_prices")
            row = cursor.fetchone()
            
            if row and row[0]:
                btc_date = datetime.strptime(row[0], '%Y-%m-%d')
                days_old = (datetime.now() - btc_date).days
                
                if days_old > cls.CRITICAL_THRESHOLD_DAYS:
                    errors.append(
                        f"BTC price data is {days_old} days old (last: {row[0]}). "
                        f"Results may be unreliable. Please refresh data."
                    )
                elif days_old > cls.STALE_THRESHOLD_DAYS:
                    warnings.append(
                        f"BTC price data is {days_old} days old (last: {row[0]}). "
                        f"Consider refreshing data."
                    )
            else:
                errors.append("No BTC price data found. Please run data backfill.")
            
            # Check macro data (VIX)
            cursor = conn.execute(
                "SELECT MAX(date) FROM macro_data WHERE indicator = 'vix'"
            )
            row = cursor.fetchone()
            
            if row and row[0]:
                vix_date = datetime.strptime(row[0], '%Y-%m-%d')
                days_old = (datetime.now() - vix_date).days
                
                if days_old > cls.CRITICAL_THRESHOLD_DAYS:
                    warnings.append(
                        f"VIX data is {days_old} days old. "
                        f"Regime detection may be inaccurate."
                    )
            else:
                warnings.append("No VIX data found. Regime detection may be limited.")
            
            conn.close()
            
        except Exception as e:
            errors.append(f"Could not check data freshness: {str(e)}")
            logger.error(f"Data freshness check failed: {e}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    @classmethod
    def get_data_age_days(cls) -> Optional[int]:
        """
        Get the age of the most recent BTC price data in days.
        
        Returns:
            Number of days since last price update, or None if no data
        """
        db_path = get_db_path()
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT MAX(date) FROM btc_prices")
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                btc_date = datetime.strptime(row[0], '%Y-%m-%d')
                return (datetime.now() - btc_date).days
            return None
            
        except Exception:
            return None


class SimulationOutputValidator:
    """Validates simulation outputs for sanity."""
    
    @classmethod
    def validate(
        cls,
        prob_margin_call: float,
        prob_liquidation: float,
        current_ltv: float,
        drop_to_margin_call: float
    ) -> ValidationResult:
        """
        Validate simulation outputs make sense.
        
        Args:
            prob_margin_call: Probability of margin call
            prob_liquidation: Probability of liquidation
            current_ltv: Current LTV ratio
            drop_to_margin_call: Required drop to hit margin call
            
        Returns:
            ValidationResult with any anomalies detected
        """
        errors = []
        warnings = []
        
        # Probability sanity checks
        if prob_liquidation > prob_margin_call:
            errors.append(
                f"P(liquidation) > P(margin call) which is impossible. "
                f"Check model calibration."
            )
        
        if prob_margin_call > 0.5 and drop_to_margin_call > 0.3:
            warnings.append(
                f"High margin call probability ({prob_margin_call*100:.1f}%) "
                f"despite large buffer ({drop_to_margin_call*100:.1f}% drop needed). "
                f"Model may be overly pessimistic."
            )
        
        if prob_margin_call < 0.001 and drop_to_margin_call < 0.15:
            warnings.append(
                f"Very low margin call probability ({prob_margin_call*100:.2f}%) "
                f"with small buffer ({drop_to_margin_call*100:.1f}% drop needed). "
                f"Model may be overly optimistic."
            )
        
        # Extreme probability warnings
        if prob_margin_call > 0.25:
            warnings.append(
                f"High probability of margin call ({prob_margin_call*100:.1f}%). "
                f"Consider reducing position risk."
            )
        
        if prob_liquidation > 0.10:
            warnings.append(
                f"Elevated probability of liquidation ({prob_liquidation*100:.1f}%). "
                f"Immediate risk management action recommended."
            )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_all(
    btc_collateral: float,
    loan_amount: float,
    accrued_interest: float,
    btc_price: float
) -> ValidationResult:
    """
    Run all validations.
    
    Args:
        Portfolio parameters
        
    Returns:
        Combined ValidationResult
    """
    all_errors = []
    all_warnings = []
    
    # Portfolio validation
    portfolio_result = PortfolioValidator.validate(
        btc_collateral, loan_amount, accrued_interest, btc_price
    )
    all_errors.extend(portfolio_result.errors)
    all_warnings.extend(portfolio_result.warnings)
    
    # Data freshness
    freshness_result = DataFreshnessChecker.check_freshness()
    all_errors.extend(freshness_result.errors)
    all_warnings.extend(freshness_result.warnings)
    
    is_valid = len(all_errors) == 0
    
    return ValidationResult(is_valid=is_valid, errors=all_errors, warnings=all_warnings)
