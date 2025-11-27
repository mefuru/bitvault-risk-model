"""
Macro data fetching: Fed Funds Rate (FRED), VIX, S&P 500 (Yahoo Finance).
"""

import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import load_config, get_api_key
from src.data.database import get_db_path, init_database


class MacroFetcher:
    """Fetches macro indicators from FRED and Yahoo Finance."""

    def __init__(self):
        self.db_path = get_db_path()
        self.config = load_config()

    def _fetch_fed_funds(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Federal Funds Rate from FRED API.

        Args:
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD

        Returns:
            DataFrame with date, indicator, value, source
        """
        try:
            api_key = get_api_key("fred")
        except ValueError:
            print("FRED API key not configured, skipping Fed Funds fetch")
            return pd.DataFrame()

        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)

            # FEDFUNDS is the effective federal funds rate
            series = fred.get_series(
                'FEDFUNDS',
                observation_start=start_date,
                observation_end=end_date
            )

            if series.empty:
                print("No Fed Funds data returned")
                return pd.DataFrame()

            df = series.reset_index()
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['indicator'] = 'fed_funds'
            df['source'] = 'fred'

            return df[['date', 'indicator', 'value', 'source']]

        except Exception as e:
            print(f"Error fetching Fed Funds: {e}")
            return pd.DataFrame()

    def _fetch_yahoo_indicator(
        self,
        ticker: str,
        indicator_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch an indicator from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker (e.g., '^VIX', '^GSPC')
            indicator_name: Name to store in database
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD

        Returns:
            DataFrame with date, indicator, value, source
        """
        try:
            # Add buffer day because yfinance end is exclusive
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

            data = yf.Ticker(ticker)
            df = data.history(start=start_date, end=end_dt.strftime('%Y-%m-%d'))

            if df.empty:
                print(f"No data returned for {ticker}")
                return pd.DataFrame()

            df = df.reset_index()
            df = df[['Date', 'Close']].copy()
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['indicator'] = indicator_name
            df['source'] = 'yahoo'

            return df[['date', 'indicator', 'value', 'source']]

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch VIX (volatility index)."""
        print(f"Fetching VIX: {start_date} to {end_date}")
        return self._fetch_yahoo_indicator('^VIX', 'vix', start_date, end_date)

    def fetch_sp500(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch S&P 500 index."""
        print(f"Fetching S&P 500: {start_date} to {end_date}")
        return self._fetch_yahoo_indicator('^GSPC', 'sp500', start_date, end_date)

    def fetch_fed_funds(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Federal Funds Rate."""
        print(f"Fetching Fed Funds Rate: {start_date} to {end_date}")
        return self._fetch_fed_funds(start_date, end_date)

    def fetch_all(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all macro indicators.

        Returns:
            Combined DataFrame with all indicators
        """
        dfs = []

        # VIX
        df_vix = self.fetch_vix(start_date, end_date)
        if not df_vix.empty:
            dfs.append(df_vix)

        # S&P 500
        df_sp = self.fetch_sp500(start_date, end_date)
        if not df_sp.empty:
            dfs.append(df_sp)

        # Fed Funds
        df_ff = self.fetch_fed_funds(start_date, end_date)
        if not df_ff.empty:
            dfs.append(df_ff)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def save_to_db(self, df: pd.DataFrame) -> int:
        """
        Save macro data to SQLite database.

        Args:
            df: DataFrame with columns: date, indicator, value, source

        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0

        conn = sqlite3.connect(self.db_path)

        rows_affected = 0
        for _, row in df.iterrows():
            cursor = conn.execute("""
                INSERT OR REPLACE INTO macro_data
                (date, indicator, value, source, fetched_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (
                row['date'],
                row['indicator'],
                row['value'],
                row['source']
            ))
            rows_affected += cursor.rowcount

        conn.commit()
        conn.close()

        return rows_affected

    def load_from_db(
        self,
        indicator: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load macro data from database.

        Args:
            indicator: Optional filter for specific indicator
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with macro data
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT date, indicator, value, source FROM macro_data"
        conditions = []
        params = []

        if indicator:
            conditions.append("indicator = ?")
            params.append(indicator)
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date, indicator"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_latest(self, indicator: str) -> Optional[tuple[str, float]]:
        """
        Get the latest value for an indicator.

        Returns:
            Tuple of (date, value) or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT date, value FROM macro_data
            WHERE indicator = ?
            ORDER BY date DESC LIMIT 1
        """, (indicator,))

        result = cursor.fetchone()
        conn.close()

        return result if result else None


def backfill_macro(years: int = 3) -> None:
    """
    Backfill historical macro data.

    Args:
        years: Number of years of history to fetch
    """
    # Ensure database exists
    init_database()

    fetcher = MacroFetcher()

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    print(f"Backfilling macro data from {start_date} to {end_date}")

    df = fetcher.fetch_all(start_date, end_date)

    if not df.empty:
        rows = fetcher.save_to_db(df)
        print(f"Saved {rows} rows")

        # Summary by indicator
        for indicator in df['indicator'].unique():
            count = len(df[df['indicator'] == indicator])
            print(f"  {indicator}: {count} rows")
    else:
        print("No data fetched")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        years = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        backfill_macro(years=years)
    else:
        # Quick test - fetch last 30 days
        fetcher = MacroFetcher()

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        df = fetcher.fetch_all(start_date, end_date)

        if not df.empty:
            print("\nFetched data:")
            for indicator in df['indicator'].unique():
                subset = df[df['indicator'] == indicator]
                latest = subset.iloc[-1]
                print(f"  {indicator}: {len(subset)} rows, latest = {latest['value']:.2f} on {latest['date']}")
        else:
            print("No data fetched")
