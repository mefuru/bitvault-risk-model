"""
Data refresh utilities for updating price and macro data.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from src.data.database import get_db_path
from src.data.prices import PriceFetcher
from src.data.macro import MacroFetcher
from src.logging_config import get_logger

logger = get_logger("refresh")


def get_last_update_times() -> dict[str, Optional[str]]:
    """
    Get the last update time for each data source.
    
    Returns:
        Dictionary with source names and their latest dates
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    results = {}
    
    # BTC prices
    cursor = conn.execute("SELECT MAX(date) FROM btc_prices")
    row = cursor.fetchone()
    results['btc_prices'] = row[0] if row and row[0] else None
    
    # VIX
    cursor = conn.execute("SELECT MAX(date) FROM macro_data WHERE indicator = 'vix'")
    row = cursor.fetchone()
    results['vix'] = row[0] if row and row[0] else None
    
    # S&P 500
    cursor = conn.execute("SELECT MAX(date) FROM macro_data WHERE indicator = 'sp500'")
    row = cursor.fetchone()
    results['sp500'] = row[0] if row and row[0] else None
    
    # Fed Funds
    cursor = conn.execute("SELECT MAX(date) FROM macro_data WHERE indicator = 'fed_funds'")
    row = cursor.fetchone()
    results['fed_funds'] = row[0] if row and row[0] else None
    
    conn.close()
    
    return results


def get_data_freshness() -> tuple[str, bool]:
    """
    Check if data is fresh (updated today or yesterday).
    
    Returns:
        Tuple of (status message, is_fresh boolean)
    """
    updates = get_last_update_times()
    
    if not updates.get('btc_prices'):
        return "No data available", False
    
    latest_date = datetime.strptime(updates['btc_prices'], '%Y-%m-%d')
    today = datetime.now().date()
    days_old = (today - latest_date.date()).days
    
    if days_old == 0:
        return f"Data is current (as of {updates['btc_prices']})", True
    elif days_old == 1:
        return f"Data is 1 day old ({updates['btc_prices']})", True
    else:
        return f"Data is {days_old} days old ({updates['btc_prices']})", False


def refresh_all_data(days_back: int = 7) -> dict[str, int]:
    """
    Refresh all data sources with recent data.
    
    Args:
        days_back: Number of days to fetch (default 7 for safety)
        
    Returns:
        Dictionary with row counts for each source
    """
    results = {}
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    logger.info(f"Refreshing data from {start_date} to {end_date}")
    
    # Refresh BTC prices
    try:
        price_fetcher = PriceFetcher(use_cryptoquant=False)
        df = price_fetcher.fetch_prices(start_date, end_date)
        rows = price_fetcher.save_to_db(df)
        results['btc_prices'] = rows
        logger.info(f"BTC prices: {rows} rows updated")
    except Exception as e:
        results['btc_prices'] = f"Error: {str(e)[:50]}"
        logger.error(f"Failed to refresh BTC prices: {e}")
    
    # Refresh macro data
    try:
        macro_fetcher = MacroFetcher()
        df = macro_fetcher.fetch_all(start_date, end_date)
        if not df.empty:
            rows = macro_fetcher.save_to_db(df)
            results['macro'] = rows
            logger.info(f"Macro data: {rows} rows updated")
        else:
            results['macro'] = 0
            logger.warning("No macro data returned")
    except Exception as e:
        results['macro'] = f"Error: {str(e)[:50]}"
        logger.error(f"Failed to refresh macro data: {e}")
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("Data Freshness Check")
    print("=" * 50)
    
    updates = get_last_update_times()
    for source, date in updates.items():
        print(f"  {source}: {date or 'No data'}")
    
    status, is_fresh = get_data_freshness()
    print(f"\nStatus: {status}")
    print(f"Fresh: {'✓' if is_fresh else '✗'}")
    
    print("\n" + "=" * 50)
    print("Refreshing data...")
    print("=" * 50)
    
    results = refresh_all_data()
    for source, count in results.items():
        print(f"  {source}: {count} rows")
