"""
Backfill extended historical data for backtesting.

Fetches 6 years of BTC price data to enable backtesting from 2022 onwards
(need 3 years prior for GARCH calibration).
"""

from datetime import datetime, timedelta
from src.data.prices import PriceFetcher
from src.data.macro import MacroFetcher
from src.logging_config import get_logger

logger = get_logger("backfill")


def backfill_for_backtest():
    """Backfill 6 years of data to enable full backtesting."""
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=6*365)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("Backfilling Historical Data for Backtesting")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print()
    
    # BTC Prices
    print("Fetching BTC prices...")
    try:
        price_fetcher = PriceFetcher(use_cryptoquant=False)
        df = price_fetcher.fetch_prices(start_date, end_date)
        rows = price_fetcher.save_to_db(df)
        print(f"  ✓ BTC prices: {rows} rows saved")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        print(f"  ✗ BTC prices failed: {e}")
    
    # Macro data
    print("\nFetching macro data...")
    try:
        macro_fetcher = MacroFetcher()
        df = macro_fetcher.fetch_all(start_date, end_date)
        if not df.empty:
            rows = macro_fetcher.save_to_db(df)
            print(f"  ✓ Macro data: {rows} rows saved")
            for indicator in df['indicator'].unique():
                count = len(df[df['indicator'] == indicator])
                print(f"    {indicator}: {count} rows")
        else:
            print("  ⚠ No macro data returned")
    except Exception as e:
        print(f"  ✗ Macro data failed: {e}")
    
    print("\n" + "=" * 60)
    print("Backfill complete!")
    print("=" * 60)
    print("\nYou can now run the backtest:")
    print("  PYTHONPATH=. python -m src.backtest.report")


if __name__ == "__main__":
    backfill_for_backtest()
