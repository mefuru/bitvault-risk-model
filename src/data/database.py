"""
Database initialization and schema management.
"""

import sqlite3
from pathlib import Path


def get_db_path() -> Path:
    """Get the database path, creating directories if needed."""
    db_path = Path(__file__).parent.parent.parent / "data" / "btc_risk.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_database(db_path: Path = None) -> None:
    """Initialize the SQLite database with all required tables."""
    
    if db_path is None:
        db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # BTC Price data (daily OHLCV)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS btc_prices (
            date TEXT PRIMARY KEY,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            source TEXT DEFAULT 'cryptoquant',
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Exchange net flows
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchange_flows (
            date TEXT PRIMARY KEY,
            net_flow REAL NOT NULL,
            inflow REAL,
            outflow REAL,
            source TEXT DEFAULT 'cryptoquant',
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Whale movements
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS whale_movements (
            date TEXT PRIMARY KEY,
            large_tx_count INTEGER,
            whale_balance_change REAL,
            source TEXT DEFAULT 'cryptoquant',
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Funding rates (aggregated daily)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            date TEXT PRIMARY KEY,
            funding_rate REAL NOT NULL,
            source TEXT DEFAULT 'cryptoquant',
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Open interest
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            date TEXT PRIMARY KEY,
            open_interest REAL NOT NULL,
            source TEXT DEFAULT 'cryptoquant',
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Macro indicators
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            date TEXT NOT NULL,
            indicator TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, indicator)
        )
    """)
    
    # Model calibration history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_calibrations (
            calibration_date TEXT PRIMARY KEY,
            omega REAL NOT NULL,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            lambda_jump REAL NOT NULL,
            mu_jump REAL NOT NULL,
            sigma_jump REAL NOT NULL,
            log_likelihood REAL,
            data_start_date TEXT,
            data_end_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Daily simulation results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulation_results (
            run_date TEXT PRIMARY KEY,
            regime TEXT NOT NULL,
            current_price REAL NOT NULL,
            prob_drop_5 REAL,
            prob_drop_10 REAL,
            prob_drop_15 REAL,
            prob_drop_20 REAL,
            prob_drop_25 REAL,
            prob_drop_30 REAL,
            prob_drop_35 REAL,
            prob_drop_40 REAL,
            prob_drop_45 REAL,
            prob_drop_50 REAL,
            var_1pct REAL,
            var_5pct REAL,
            var_10pct REAL,
            cvar_1pct REAL,
            cvar_5pct REAL,
            n_paths INTEGER,
            execution_time_seconds REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Execution log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            status TEXT NOT NULL,
            step TEXT,
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_data(indicator)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_log_date ON execution_log(run_date)")
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    init_database()
