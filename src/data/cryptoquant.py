"""
CryptoQuant API fetcher for on-chain data.

Fetches:
- Exchange Netflow (Binance) - selling pressure indicator
- Funding Rates - market sentiment
- SOPR - profitability of spent outputs
- MVRV - market value vs realized value

Run with: PYTHONPATH=. python -m src.data.cryptoquant --backfill 365
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from src.config import get_api_key
from src.data.database import get_db_path, init_database
from src.logging_config import get_logger

logger = get_logger("cryptoquant")


class CryptoQuantFetcher:
    """Fetches on-chain data from CryptoQuant API."""
    
    BASE_URL = "https://api.cryptoquant.com/v1"
    
    def __init__(self):
        try:
            self.api_key = get_api_key("cryptoquant")
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            self.available = True
            logger.info("CryptoQuant API initialized")
        except ValueError as e:
            logger.warning(f"CryptoQuant API not configured: {e}")
            self.available = False
    
    def _fetch(self, endpoint: str, params: dict) -> Optional[pd.DataFrame]:
        """Make API request and return DataFrame."""
        if not self.available:
            logger.warning("CryptoQuant API not available")
            return None
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                return None
            
            data = response.json()
            
            if "result" not in data or "data" not in data["result"]:
                logger.error(f"Unexpected response format: {list(data.keys())}")
                return None
            
            rows = data["result"]["data"]
            if not rows:
                logger.warning(f"No data returned for {endpoint}")
                return None
            
            df = pd.DataFrame(rows)
            logger.info(f"Fetched {len(df)} rows from {endpoint}")
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
    
    def fetch_exchange_netflow(
        self, 
        start_date: str, 
        end_date: str,
        exchange: str = "binance"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch exchange netflow data.
        
        Positive netflow = coins flowing INTO exchange (selling pressure)
        Negative netflow = coins flowing OUT (accumulation)
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            exchange: Exchange name (binance, coinbase, etc.)
        """
        # Convert date format
        start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        params = {
            "window": "day",
            "from": start,
            "to": end,
            "exchange": exchange
        }
        
        df = self._fetch("/btc/exchange-flows/netflow", params)
        
        if df is not None:
            df = df.rename(columns={"netflow_total": "netflow"})
            df["exchange"] = exchange
        
        return df
    
    def fetch_funding_rates(
        self, 
        start_date: str, 
        end_date: str,
        exchange: str = "all_exchange"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch funding rates.
        
        Positive = longs pay shorts (bullish sentiment)
        Negative = shorts pay longs (bearish sentiment)
        
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
            exchange: Exchange or "all_exchange"
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        params = {
            "window": "day",
            "from": start,
            "to": end,
            "exchange": exchange
        }
        
        df = self._fetch("/btc/market-data/funding-rates", params)
        
        if df is not None:
            df = df.rename(columns={"funding_rates": "funding_rate"})
        
        return df
    
    def fetch_sopr(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch SOPR (Spent Output Profit Ratio).
        
        SOPR > 1 = coins sold at profit
        SOPR < 1 = coins sold at loss (capitulation)
        SOPR = 1 = break-even, often acts as support/resistance
        
        Also returns:
        - a_sopr: Adjusted SOPR (excludes coins < 1 hour old)
        - sth_sopr: Short-term holder SOPR
        - lth_sopr: Long-term holder SOPR
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        params = {
            "window": "day",
            "from": start,
            "to": end
        }
        
        return self._fetch("/btc/market-indicator/sopr", params)
    
    def fetch_mvrv(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch MVRV (Market Value to Realized Value).
        
        MVRV > 3.5 = historically overbought (top signal)
        MVRV < 1 = historically oversold (bottom signal)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        params = {
            "window": "day",
            "from": start,
            "to": end
        }
        
        return self._fetch("/btc/market-indicator/mvrv", params)
    
    def fetch_all(self, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """
        Fetch all available on-chain metrics.
        
        Returns dict with keys: netflow, funding_rate, sopr, mvrv
        """
        results = {}
        
        # Exchange netflow
        df = self.fetch_exchange_netflow(start_date, end_date)
        if df is not None:
            results["netflow"] = df
        
        # Funding rates
        df = self.fetch_funding_rates(start_date, end_date)
        if df is not None:
            results["funding_rate"] = df
        
        # SOPR
        df = self.fetch_sopr(start_date, end_date)
        if df is not None:
            results["sopr"] = df
        
        # MVRV
        df = self.fetch_mvrv(start_date, end_date)
        if df is not None:
            results["mvrv"] = df
        
        return results
    
    def save_to_db(self, data: dict[str, pd.DataFrame]) -> dict[str, int]:
        """
        Save fetched data to SQLite database.
        
        Args:
            data: Dict from fetch_all()
            
        Returns:
            Dict with row counts saved for each metric
        """
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        
        # Ensure tables exist
        self._create_tables(conn)
        
        counts = {}
        
        # Save netflow
        if "netflow" in data:
            df = data["netflow"]
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO exchange_flows (date, net_flow, source)
                    VALUES (?, ?, 'cryptoquant')
                """, (row["date"], row["netflow"]))
            counts["netflow"] = len(df)
        
        # Save funding rates
        if "funding_rate" in data:
            df = data["funding_rate"]
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO funding_rates (date, funding_rate, source)
                    VALUES (?, ?, 'cryptoquant')
                """, (row["date"], row["funding_rate"]))
            counts["funding_rate"] = len(df)
        
        # Save SOPR to onchain_metrics table
        if "sopr" in data:
            df = data["sopr"]
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO onchain_metrics 
                    (date, metric, value, source)
                    VALUES (?, 'sopr', ?, 'cryptoquant')
                """, (row["date"], row["sopr"]))
                # Also save adjusted SOPR
                if "a_sopr" in row:
                    conn.execute("""
                        INSERT OR REPLACE INTO onchain_metrics 
                        (date, metric, value, source)
                        VALUES (?, 'a_sopr', ?, 'cryptoquant')
                    """, (row["date"], row["a_sopr"]))
            counts["sopr"] = len(df)
        
        # Save MVRV
        if "mvrv" in data:
            df = data["mvrv"]
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO onchain_metrics 
                    (date, metric, value, source)
                    VALUES (?, 'mvrv', ?, 'cryptoquant')
                """, (row["date"], row["mvrv"]))
            counts["mvrv"] = len(df)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved to database: {counts}")
        return counts
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create tables if they don't exist."""
        # On-chain metrics table (for SOPR, MVRV, etc.)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS onchain_metrics (
                date TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                source TEXT DEFAULT 'cryptoquant',
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, metric)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_onchain_metric 
            ON onchain_metrics(metric)
        """)


def backfill_cryptoquant(days: int = 365, chunk_days: int = 90):
    """
    Backfill CryptoQuant data for the specified number of days.
    
    Args:
        days: Total days of history to fetch
        chunk_days: Days per API request (to avoid timeouts)
    """
    fetcher = CryptoQuantFetcher()
    
    if not fetcher.available:
        print("CryptoQuant API not configured. Set CRYPTOQUANT_API_KEY in .env")
        return
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Backfilling CryptoQuant data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Fetching in {chunk_days}-day chunks...")
    
    total_counts = {}
    
    # Fetch in chunks to avoid API limits
    current_end = end_date
    chunk_num = 1
    
    while current_end > start_date:
        current_start = max(current_end - timedelta(days=chunk_days), start_date)
        
        print(f"\nChunk {chunk_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        # Fetch all data for this chunk
        data = fetcher.fetch_all(
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        )
        
        if data:
            counts = fetcher.save_to_db(data)
            for metric, count in counts.items():
                total_counts[metric] = total_counts.get(metric, 0) + count
                print(f"  {metric}: {count} rows")
        else:
            print("  No data returned for this chunk")
        
        current_end = current_start - timedelta(days=1)
        chunk_num += 1
    
    print("\n" + "=" * 50)
    print("Backfill complete - Total rows:")
    for metric, count in total_counts.items():
        print(f"  {metric}: {count} rows")


if __name__ == "__main__":
    import sys
    
    days = 365
    for arg in sys.argv[1:]:
        if arg.startswith("--backfill"):
            if "=" in arg:
                days = int(arg.split("=")[1])
        elif arg.startswith("--days="):
            days = int(arg.split("=")[1])
    
    backfill_cryptoquant(days)
