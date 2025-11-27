"""
BTC price data fetching from CryptoQuant (primary) and Yahoo Finance (fallback).
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from src.config import load_config, get_api_key
from src.data.database import get_db_path


class PriceFetcher:
    """Fetches BTC price data from CryptoQuant or Yahoo Finance."""
    
    def __init__(self, use_cryptoquant: bool = True):
        """
        Initialize the price fetcher.
        
        Args:
            use_cryptoquant: If True, try CryptoQuant first. If False, use Yahoo Finance.
        """
        self.use_cryptoquant = use_cryptoquant
        self.config = load_config()
        self.db_path = get_db_path()
        
    def _fetch_from_cryptoquant(
        self, 
        start_date: str, 
        end_date: str,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from CryptoQuant API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_retries: Number of retry attempts
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            None if fetch fails
        """
        try:
            api_key = get_api_key("cryptoquant")
        except ValueError as e:
            print(f"CryptoQuant API key not configured: {e}")
            return None
            
        base_url = self.config["cryptoquant"]["base_url"]
        endpoint = f"{base_url}/v1/btc/market-data/price-ohlcv"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "window": "day",
            "from": start_date,
            "to": end_date,
            "limit": 1000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    endpoint, 
                    headers=headers, 
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "result" in data and "data" in data["result"]:
                        records = data["result"]["data"]
                        
                        df = pd.DataFrame(records)
                        df = df.rename(columns={
                            "datetime": "date",
                            "open": "open",
                            "high": "high", 
                            "low": "low",
                            "close": "close",
                            "volume": "volume"
                        })
                        
                        # Convert timestamp to date string
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                        df["source"] = "cryptoquant"
                        
                        return df[["date", "open", "high", "low", "close", "volume", "source"]]
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                else:
                    print(f"CryptoQuant API error: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def _fetch_from_yahoo(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch BTC price data from Yahoo Finance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Add one day to end_date because yfinance end is exclusive
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(start=start_date, end=end_dt.strftime("%Y-%m-%d"))
        
        if df.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {start_date} to {end_date}")
        
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["source"] = "yahoo"
        
        return df[["date", "open", "high", "low", "close", "volume", "source"]]
    
    def fetch_prices(
        self, 
        start_date: str, 
        end_date: str,
        fallback_to_yahoo: bool = True
    ) -> pd.DataFrame:
        """
        Fetch BTC price data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            fallback_to_yahoo: If True, use Yahoo Finance if CryptoQuant fails
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, source
        """
        df = None
        
        if self.use_cryptoquant:
            print(f"Fetching from CryptoQuant: {start_date} to {end_date}")
            df = self._fetch_from_cryptoquant(start_date, end_date)
            
        if df is None and fallback_to_yahoo:
            print(f"Fetching from Yahoo Finance: {start_date} to {end_date}")
            df = self._fetch_from_yahoo(start_date, end_date)
        elif df is None:
            raise ValueError("Failed to fetch price data and fallback disabled")
            
        return df
    
    def fetch_latest(self) -> pd.DataFrame:
        """Fetch the latest available price data (yesterday's close)."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        
        df = self.fetch_prices(start_date, end_date)
        return df.tail(1)
    
    def save_to_db(self, df: pd.DataFrame) -> int:
        """
        Save price data to SQLite database.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Number of rows inserted/updated
        """
        conn = sqlite3.connect(self.db_path)
        
        rows_affected = 0
        for _, row in df.iterrows():
            cursor = conn.execute("""
                INSERT OR REPLACE INTO btc_prices 
                (date, open, high, low, close, volume, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["source"]
            ))
            rows_affected += cursor.rowcount
            
        conn.commit()
        conn.close()
        
        return rows_affected
    
    def get_stored_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """Get the date range of stored price data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT MIN(date), MAX(date) FROM btc_prices
        """)
        result = cursor.fetchone()
        conn.close()
        
        return result[0], result[1]
    
    def get_missing_dates(self, start_date: str, end_date: str) -> list[str]:
        """
        Get list of dates missing from the database.
        
        Args:
            start_date: Start of range to check
            end_date: End of range to check
            
        Returns:
            List of missing date strings
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT date FROM btc_prices 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """, (start_date, end_date))
        
        stored_dates = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        # Generate all expected dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_dates = []
        current = start_dt
        while current <= end_dt:
            all_dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        missing = [d for d in all_dates if d not in stored_dates]
        return missing
    
    def load_from_db(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load price data from database.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with price data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT date, open, high, low, close, volume, source FROM btc_prices"
        conditions = []
        params = []
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df


def backfill_prices(years: int = 3, use_cryptoquant: bool = False) -> None:
    """
    Backfill historical price data.
    
    Args:
        years: Number of years of history to fetch
        use_cryptoquant: Whether to try CryptoQuant first
    """
    from src.data.database import init_database
    
    # Ensure database exists
    init_database()
    
    fetcher = PriceFetcher(use_cryptoquant=use_cryptoquant)
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    
    print(f"Backfilling prices from {start_date} to {end_date}")
    
    # Fetch in chunks to avoid API limits
    chunk_size = 365  # days
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_rows = 0
    
    while current_start < end_dt:
        chunk_end = min(current_start + timedelta(days=chunk_size), end_dt)
        
        try:
            df = fetcher.fetch_prices(
                current_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )
            
            rows = fetcher.save_to_db(df)
            total_rows += rows
            print(f"  Saved {rows} rows for {current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"  Error fetching {current_start} to {chunk_end}: {e}")
        
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Rate limiting
    
    print(f"Backfill complete. Total rows: {total_rows}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        years = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        backfill_prices(years=years, use_cryptoquant=False)
    else:
        # Quick test - fetch last 7 days
        fetcher = PriceFetcher(use_cryptoquant=False)
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        print(f"Fetching BTC prices from {start_date} to {end_date}")
        df = fetcher.fetch_prices(start_date, end_date)
        print(df)
        print(f"\nLatest close: ${df['close'].iloc[-1]:,.2f}")
