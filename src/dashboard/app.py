"""
BitVault BTC Risk Dashboard

Run with: streamlit run src/dashboard/app.py
"""

import os
import sys
import sqlite3
from datetime import datetime

# Add project root to path for cloud deployment
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.model.simulation import MonteCarloEngine, SimulationResults
from src.model.garch import GARCHModel
from src.risk.ltv import (
    PortfolioState, 
    LTVThresholds, 
    LTVRiskCalculator,
    LTVRiskMetrics
)
from src.data.prices import PriceFetcher
from src.data.refresh import get_last_update_times, get_data_freshness, refresh_all_data
from src.data.database import init_database, get_db_path
from src.regime.classifier import RegimeClassifier, RegimeClassification
from src.validation import (
    PortfolioValidator, 
    DataFreshnessChecker, 
    SimulationOutputValidator,
    validate_all
)
from src.logging_config import get_logger

logger = get_logger("dashboard")


# ============================================================================
# CLOUD DEPLOYMENT: Data Initialization
# ============================================================================
# On Streamlit Cloud, the filesystem resets on each deploy. This function
# ensures we have data available by fetching it on startup if needed.

@st.cache_resource(show_spinner=False)
def initialize_cloud_data():
    """
    Initialize database and fetch data if running in cloud environment.
    This runs once per app instance and is cached.
    """
    try:
        # Initialize database schema
        init_database()
        
        # Also create CryptoQuant tables if they don't exist
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        
        # Create all required tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchange_flows (
                date TEXT PRIMARY KEY,
                net_flow REAL,
                inflow REAL,
                outflow REAL,
                source TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                date TEXT PRIMARY KEY,
                funding_rate REAL,
                source TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS onchain_metrics (
                date TEXT,
                metric TEXT,
                value REAL,
                source TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, metric)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_onchain_metric ON onchain_metrics(metric)")
        
        conn.commit()
        
        # Check if we have sufficient data
        try:
            count = conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        except sqlite3.OperationalError:
            count = 0
        finally:
            conn.close()
        
        # If we have less than 100 rows, backfill data
        if count < 100:
            logger.info(f"Database has {count} rows, backfilling data...")
            
            # Show a message to the user
            with st.spinner("🔄 Initializing data (first run only, ~60 seconds)..."):
                from src.data.prices import PriceFetcher
                from src.data.macro import MacroFetcher
                from datetime import datetime, timedelta
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
                
                # Fetch BTC prices
                try:
                    price_fetcher = PriceFetcher(use_cryptoquant=False)
                    df = price_fetcher.fetch_prices(start_date, end_date)
                    price_fetcher.save_to_db(df)
                    logger.info(f"Backfilled {len(df)} BTC price rows")
                except Exception as e:
                    logger.error(f"Failed to backfill prices: {e}")
                
                # Fetch macro data (VIX, etc.)
                try:
                    macro_fetcher = MacroFetcher()
                    df = macro_fetcher.fetch_all(start_date, end_date)
                    if not df.empty:
                        macro_fetcher.save_to_db(df)
                        logger.info(f"Backfilled macro data")
                except Exception as e:
                    logger.error(f"Failed to backfill macro: {e}")
                
                # Fetch CryptoQuant data if API key is available
                try:
                    from src.data.cryptoquant import CryptoQuantFetcher
                    cq_fetcher = CryptoQuantFetcher()
                    if cq_fetcher.available:
                        # Fetch 1 year of on-chain data
                        cq_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                        data = cq_fetcher.fetch_all(cq_start, end_date)
                        if data:
                            counts = cq_fetcher.save_to_db(data)
                            logger.info(f"Backfilled CryptoQuant data: {counts}")
                    else:
                        logger.warning("CryptoQuant API key not configured, skipping on-chain data")
                except ImportError:
                    logger.warning("CryptoQuant module not available")
                except Exception as e:
                    logger.error(f"Failed to backfill CryptoQuant: {e}")
                
                # Calibrate GARCH model and save to DB for consistency
                try:
                    from src.model.garch import GARCHModel
                    garch = GARCHModel()
                    params = garch.fit()
                    garch.save_calibration(params)
                    logger.info(f"Calibrated GARCH model: vol={params.long_run_volatility:.1f}%")
                except Exception as e:
                    logger.error(f"Failed to calibrate GARCH: {e}")
            
            return "initialized"
        else:
            logger.info(f"Database has {count} rows, no backfill needed")
            return "ready"
            
    except Exception as e:
        logger.error(f"Cloud initialization failed: {e}")
        return f"error: {e}"


# Run initialization on app start
_init_status = initialize_cloud_data()


# Page config
st.set_page_config(
    page_title="BitVault Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None


def get_current_price(live: bool = True):
    """
    Fetch current BTC price.
    
    Args:
        live: If True, fetch from Yahoo Finance. If False, use database.
    """
    if live:
        try:
            import yfinance as yf
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period="1d")
            if not data.empty:
                price = data['Close'].iloc[-1]
                logger.info(f"Fetched live BTC price: ${price:,.0f}")
                return price
        except Exception as e:
            logger.warning(f"Failed to fetch live price: {e}")
    
    # Fallback to database
    try:
        fetcher = PriceFetcher(use_cryptoquant=False)
        df = fetcher.load_from_db()
        if df.empty:
            raise ValueError("No price data in database")
        price = df['close'].iloc[-1]
        logger.info(f"Using database BTC price: ${price:,.0f}")
        return price
    except Exception as e:
        logger.error(f"Failed to get BTC price: {e}")
        raise


@st.cache_data(ttl=3600)
def get_garch_params():
    """Load GARCH parameters (cached)."""
    model = GARCHModel()
    params = model.load_latest_calibration()
    return params


@st.cache_data(ttl=1800)
def get_regime_classification():
    """Get current regime classification (cached for 30 min)."""
    # Try with on-chain support, fall back to basic if not available
    try:
        classifier = RegimeClassifier(use_onchain=True)
        return classifier.classify()
    except TypeError:
        # Fallback for older classifier without use_onchain parameter
        classifier = RegimeClassifier()
        return classifier.classify()
    except Exception as e:
        logger.warning(f"Regime classification failed: {e}")
        # Return a default classification if everything fails
        return RegimeClassification(
            regime="normal",
            indicators=[],
            stress_count=0,
            total_indicators=0,
            classification_date=datetime.now().strftime("%Y-%m-%d")
        )


def run_analysis(btc_collateral: float, loan_amount: float, accrued_interest: float, regime: str):
    """Run the full analysis pipeline with error handling."""
    try:
        current_price = get_current_price()
        
        portfolio = PortfolioState(
            btc_collateral=btc_collateral,
            loan_amount=loan_amount,
            accrued_interest=accrued_interest,
            btc_price=current_price
        )
        
        logger.info(
            f"Running analysis: {btc_collateral} BTC, "
            f"${loan_amount:,.0f} loan, {regime} regime"
        )
        
        calculator = LTVRiskCalculator()
        
        # Run simulation with fixed seed for reproducibility
        engine = MonteCarloEngine()
        
        # Load GARCH params to ensure consistency
        garch_params = engine.garch_model.load_latest_calibration()
        
        sim_results = engine.run(
            n_paths=100_000,
            horizon_days=30,
            regime=regime,
            current_price=current_price,
            store_paths=True,
            random_seed=42,  # Fixed seed for reproducibility
            garch_params=garch_params
        )
        
        # Store debug info in session state
        st.session_state.debug_info = {
            'current_price': current_price,
            'garch_params': {
                'omega': garch_params.omega if garch_params else None,
                'alpha': garch_params.alpha if garch_params else None,
                'beta': garch_params.beta if garch_params else None,
                'mu': garch_params.mu if garch_params else None,
                'long_run_vol': garch_params.long_run_volatility if garch_params else None,
            } if garch_params else None,
            'regime': regime,
            'sim_mean': sim_results.mean_price,
            'sim_p5': sim_results.percentile_5,
            'sim_p95': sim_results.percentile_95,
        }
        
        # Calculate risk metrics
        risk_metrics = calculator.calculate_risk(
            portfolio=portfolio,
            simulation_results=sim_results,
            regime=regime
        )
        
        logger.info(
            f"Analysis complete: LTV={risk_metrics.current_ltv*100:.1f}%, "
            f"P(MC)={risk_metrics.prob_margin_call*100:.2f}%, "
            f"P(Liq)={risk_metrics.prob_liquidation*100:.2f}%"
        )
        
        return portfolio, sim_results, risk_metrics
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def create_price_paths_chart(
    sim_results: SimulationResults,
    risk_metrics: LTVRiskMetrics
) -> go.Figure:
    """Create interactive price paths chart."""
    
    if sim_results.all_paths is None:
        return None
    
    paths = sim_results.all_paths
    days = list(range(paths.shape[1]))
    
    # Calculate percentiles
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    fig = go.Figure()
    
    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p95) + list(p5)[::-1],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='5th-95th percentile',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p75) + list(p25)[::-1],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        name='25th-75th percentile',
        showlegend=True
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=days,
        y=p50,
        mode='lines',
        name='Median',
        line=dict(color='blue', width=2)
    ))
    
    # Threshold lines
    fig.add_hline(
        y=risk_metrics.margin_call_price,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Margin Call (${risk_metrics.margin_call_price:,.0f})",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=risk_metrics.liquidation_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Liquidation (${risk_metrics.liquidation_price:,.0f})",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Simulated BTC Price Paths (30 Days)",
        xaxis_title="Days",
        yaxis_title="BTC Price (USD)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_distribution_chart(
    sim_results: SimulationResults,
    risk_metrics: LTVRiskMetrics
) -> go.Figure:
    """Create terminal price distribution histogram."""
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=sim_results.terminal_prices,
        nbinsx=100,
        name='Terminal Price Distribution',
        marker_color='rgba(99, 110, 250, 0.7)'
    ))
    
    # Current price line
    fig.add_vline(
        x=sim_results.current_price,
        line_dash="solid",
        line_color="green",
        line_width=2,
        annotation_text=f"Current (${sim_results.current_price:,.0f})",
        annotation_position="top"
    )
    
    # Threshold lines
    fig.add_vline(
        x=risk_metrics.margin_call_price,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Margin Call",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=risk_metrics.liquidation_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Liquidation",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Terminal Price Distribution (Day 30)",
        xaxis_title="BTC Price (USD)",
        yaxis_title="Frequency",
        height=350
    )
    
    return fig


def create_regime_history_chart(days: int = 90) -> go.Figure:
    """
    Create historical chart showing regime indicators over time.
    Shows BTC price with regime indicator overlays.
    
    Args:
        days: Number of days of history to show
    """
    from src.data.database import get_db_path
    
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # Helper function to safely query tables (handles missing tables)
    def safe_query(query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.debug(f"Query failed (table may not exist): {e}")
            return pd.DataFrame()
    
    # Load BTC prices
    btc_df = safe_query(
        f"""SELECT date, close FROM btc_prices 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not btc_df.empty:
        btc_df['date'] = pd.to_datetime(btc_df['date'])
        btc_df = btc_df.set_index('date').sort_index()
    
    # Load VIX
    vix_df = safe_query(
        f"""SELECT date, value as vix FROM macro_data 
            WHERE indicator = 'vix' 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not vix_df.empty:
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        vix_df = vix_df.set_index('date').sort_index()
    
    # Load funding rates
    funding_df = safe_query(
        f"""SELECT date, funding_rate FROM funding_rates 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not funding_df.empty:
        funding_df['date'] = pd.to_datetime(funding_df['date'])
        funding_df = funding_df.set_index('date').sort_index()
    
    # Load SOPR
    sopr_df = safe_query(
        f"""SELECT date, value as sopr FROM onchain_metrics 
            WHERE metric = 'sopr' 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not sopr_df.empty:
        sopr_df['date'] = pd.to_datetime(sopr_df['date'])
        sopr_df = sopr_df.set_index('date').sort_index()
    
    # Load Exchange Netflow
    netflow_df = safe_query(
        f"""SELECT date, net_flow FROM exchange_flows 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not netflow_df.empty:
        netflow_df['date'] = pd.to_datetime(netflow_df['date'])
        netflow_df = netflow_df.set_index('date').sort_index()
    
    # Load MVRV
    mvrv_df = safe_query(
        f"""SELECT date, value as mvrv FROM onchain_metrics 
            WHERE metric = 'mvrv' 
            ORDER BY date DESC LIMIT {days}"""
    )
    if not mvrv_df.empty:
        mvrv_df['date'] = pd.to_datetime(mvrv_df['date'])
        mvrv_df = mvrv_df.set_index('date').sort_index()
    
    conn.close()
    
    # If we don't have BTC prices, we can't show the chart
    if btc_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price data available. Run data refresh to populate.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Historical Regime Indicators",
            height=300
        )
        return fig
    
    # Count how many subplots we need
    num_plots = 1  # BTC price always
    if not vix_df.empty: num_plots += 1
    if not funding_df.empty: num_plots += 1
    if not sopr_df.empty: num_plots += 1
    if not netflow_df.empty: num_plots += 1
    if not mvrv_df.empty: num_plots += 1
    
    # Build subplot titles
    titles = ["BTC Price"]
    if not vix_df.empty: titles.append("VIX (Stress > 30)")
    if not funding_df.empty: titles.append("Funding Rate % (Stress < -0.01)")
    if not sopr_df.empty: titles.append("SOPR (Stress < 0.98)")
    if not netflow_df.empty: titles.append("Exchange Netflow BTC (Stress > 10k)")
    if not mvrv_df.empty: titles.append("MVRV (Stress < 1 or > 3.5)")
    
    # Calculate row heights - give more space to BTC price
    heights = [0.3] + [0.7 / (num_plots - 1)] * (num_plots - 1) if num_plots > 1 else [1.0]
    
    fig = make_subplots(
        rows=num_plots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=heights,
        subplot_titles=titles
    )
    
    current_row = 1
    
    # 1. BTC Price (always show)
    if not btc_df.empty:
        fig.add_trace(
            go.Scatter(
                x=btc_df.index,
                y=btc_df['close'],
                name='BTC Price',
                line=dict(color='#F7931A', width=1.5),
                hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Price ($)", row=current_row, col=1)
    current_row += 1
    
    # 2. VIX
    if not vix_df.empty:
        fig.add_trace(
            go.Scatter(
                x=vix_df.index,
                y=vix_df['vix'],
                name='VIX',
                line=dict(color='purple', width=1),
                hovertemplate='%{x}<br>VIX: %{y:.1f}<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=30, line_dash="dash", line_color="red", 
                     annotation_text="Stress threshold", row=current_row, col=1)
        fig.update_yaxes(title_text="VIX", row=current_row, col=1)
        current_row += 1
    
    # 3. Funding Rate
    if not funding_df.empty:
        colors = ['#00CC00' if x >= 0 else '#CC0000' for x in funding_df['funding_rate']]
        fig.add_trace(
            go.Bar(
                x=funding_df.index,
                y=funding_df['funding_rate'] * 100,
                name='Funding Rate',
                marker_color=colors,
                hovertemplate='%{x}<br>%{y:.3f}%<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=-0.01, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="Rate %", row=current_row, col=1)
        current_row += 1
    
    # 4. SOPR
    if not sopr_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sopr_df.index,
                y=sopr_df['sopr'],
                name='SOPR',
                line=dict(color='#0066CC', width=1),
                fill='tozeroy',
                fillcolor='rgba(0, 102, 204, 0.1)',
                hovertemplate='%{x}<br>SOPR: %{y:.3f}<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=1.0, line_dash="solid", line_color="gray", 
                     annotation_text="Break-even", row=current_row, col=1)
        fig.add_hline(y=0.98, line_dash="dash", line_color="red",
                     annotation_text="Stress", row=current_row, col=1)
        fig.update_yaxes(title_text="SOPR", row=current_row, col=1)
        current_row += 1
    
    # 5. Exchange Netflow
    if not netflow_df.empty:
        colors = ['#CC0000' if x > 0 else '#00CC00' for x in netflow_df['net_flow']]
        fig.add_trace(
            go.Bar(
                x=netflow_df.index,
                y=netflow_df['net_flow'],
                name='Netflow',
                marker_color=colors,
                hovertemplate='%{x}<br>%{y:,.0f} BTC<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=10000, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(title_text="BTC", row=current_row, col=1)
        current_row += 1
    
    # 6. MVRV
    if not mvrv_df.empty:
        fig.add_trace(
            go.Scatter(
                x=mvrv_df.index,
                y=mvrv_df['mvrv'],
                name='MVRV',
                line=dict(color='#CC6600', width=1),
                fill='tozeroy',
                fillcolor='rgba(204, 102, 0, 0.1)',
                hovertemplate='%{x}<br>MVRV: %{y:.2f}<extra></extra>'
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                     annotation_text="Oversold", row=current_row, col=1)
        fig.add_hline(y=3.5, line_dash="dash", line_color="red",
                     annotation_text="Overbought", row=current_row, col=1)
        fig.update_yaxes(title_text="MVRV", row=current_row, col=1)
    
    fig.update_layout(
        height=150 + (num_plots * 120),
        showlegend=False,
        title_text="Historical Regime Indicators",
        title_x=0.5,
        hovermode='x unified'
    )
    
    return fig


def create_probability_chart(sim_results: SimulationResults) -> go.Figure:
    """Create probability of drop bar chart."""
    
    prob_table = sim_results.get_probability_table()
    
    colors = ['green' if p < 0.1 else 'orange' if p < 0.2 else 'red' 
              for p in prob_table['probability']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"≥{int(d)}%" for d in prob_table['drop_pct']],
        y=prob_table['probability'] * 100,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in prob_table['probability']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Probability of Price Drops (30 Days)",
        xaxis_title="Price Drop",
        yaxis_title="Probability (%)",
        height=350,
        yaxis=dict(range=[0, max(prob_table['probability'] * 100) * 1.2])
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

st.title("📊 BitVault BTC Risk Dashboard")

# Sidebar - Portfolio Inputs
st.sidebar.header("Portfolio Settings")

# Live BTC Price
try:
    live_price = get_current_price(live=True)
    st.sidebar.metric("Live BTC Price", f"${live_price:,.0f}")
except Exception as e:
    st.sidebar.warning(f"Could not fetch live price")

# Data freshness status
st.sidebar.subheader("📡 Data Status")
try:
    status_msg, is_fresh = get_data_freshness()
    if is_fresh:
        st.sidebar.success(status_msg)
    else:
        st.sidebar.warning(status_msg)
    
    # Show detailed update times in expander
    with st.sidebar.expander("Data Sources"):
        updates = get_last_update_times()
        for source, date in updates.items():
            st.write(f"**{source}**: {date or 'No data'}")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Fetching latest data..."):
            results = refresh_all_data(days_back=7)
            # Clear caches so new data is used
            st.cache_data.clear()
        
        # Show results
        success_count = sum(1 for v in results.values() if isinstance(v, int) and v > 0)
        if success_count > 0:
            st.sidebar.success(f"Refreshed {success_count} sources")
        else:
            st.sidebar.error("Refresh failed - check logs")
        st.rerun()

except Exception as e:
    st.sidebar.error(f"Could not check data status: {e}")

st.sidebar.divider()

btc_collateral = st.sidebar.number_input(
    "BTC Collateral",
    min_value=0.01,
    max_value=10000.0,
    value=100.0,
    step=1.0,
    help="Amount of BTC held as collateral"
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=1000.0,
    max_value=100_000_000.0,
    value=5_350_000.0,
    step=100_000.0,
    help="Initial bvUSD loan amount"
)

accrued_interest = st.sidebar.number_input(
    "Accrued Interest ($)",
    min_value=0.0,
    max_value=10_000_000.0,
    value=0.0,
    step=1000.0,
    help="Accrued interest on the loan"
)

# Show total debt
total_debt = loan_amount + accrued_interest
st.sidebar.caption(f"Total Debt: ${total_debt:,.2f}")

regime = st.sidebar.selectbox(
    "Market Regime",
    options=["auto", "normal", "stress"],
    index=0,
    help="Auto detects regime from market indicators. Manual override available."
)

# Show auto-detected regime
if regime == "auto":
    try:
        auto_classification = get_regime_classification()
        detected_regime = auto_classification.regime
        regime_icon = "🔴" if detected_regime == "stress" else "🟢"
        st.sidebar.info(f"Detected: {regime_icon} {detected_regime.upper()}")
        
        # Show indicator summary in expander
        with st.sidebar.expander("Regime Indicators", expanded=False):
            for ind in auto_classification.indicators:
                icon = "🔴" if ind.is_stress else "🟢"
                st.write(f"{icon} **{ind.name}**: {ind.value:.2f}")
            
            st.caption(f"Stress signals: {auto_classification.stress_count}/{auto_classification.total_indicators}")
        
        # Show indicator reference guide
        with st.sidebar.expander("📊 Indicator Guide", expanded=False):
            st.markdown("""
**Market Indicators:**

| Indicator | Stress When |
|-----------|-------------|
| VIX | > 30 |
| BTC Volatility | > 1.5× average |
| BTC Drawdown | > 15% from high |
| BTC-SPX Corr | > 0.7 + SPX down |

**On-Chain (CryptoQuant):**

| Indicator | Stress When |
|-----------|-------------|
| Exchange Netflow | > 10k BTC inflow |
| Funding Rate | < -0.01% |
| SOPR | < 0.98 |
| MVRV | < 1.0 or > 3.5 |

*Regime = STRESS if ≥2 indicators trigger*
            """)
            
    except Exception as e:
        st.sidebar.warning(f"Could not detect regime: {e}")
        detected_regime = "normal"
else:
    detected_regime = regime

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    # Get current price for validation
    try:
        current_price = get_current_price()
    except Exception as e:
        st.error(f"❌ Could not fetch BTC price: {e}")
        st.stop()
    
    # Validate inputs
    validation = validate_all(
        btc_collateral=btc_collateral,
        loan_amount=loan_amount,
        accrued_interest=accrued_interest,
        btc_price=current_price
    )
    
    # Show validation errors
    if validation.errors:
        for error in validation.errors:
            st.sidebar.error(f"❌ {error}")
        st.stop()
    
    # Show validation warnings
    if validation.warnings:
        for warning in validation.warnings:
            st.sidebar.warning(f"⚠️ {warning}")
    
    # Run simulation
    with st.spinner("Running Monte Carlo simulation (100,000 paths)..."):
        try:
            portfolio, sim_results, risk_metrics = run_analysis(
                btc_collateral=btc_collateral,
                loan_amount=loan_amount,
                accrued_interest=accrued_interest,
                regime=detected_regime
            )
            
            # Validate outputs
            output_validation = SimulationOutputValidator.validate(
                prob_margin_call=risk_metrics.prob_margin_call,
                prob_liquidation=risk_metrics.prob_liquidation,
                current_ltv=risk_metrics.current_ltv,
                drop_to_margin_call=risk_metrics.drop_to_margin_call
            )
            
            st.session_state.portfolio = portfolio
            st.session_state.simulation_results = sim_results
            st.session_state.risk_metrics = risk_metrics
            st.session_state.regime_used = detected_regime
            st.session_state.output_warnings = output_validation.warnings
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            logger.error(f"Simulation failed: {e}")
            st.stop()
    
    st.success("✅ Simulation complete!")

# =============================================================================
# MAIN CONTENT
# =============================================================================

if st.session_state.simulation_results is not None:
    sim_results = st.session_state.simulation_results
    risk_metrics = st.session_state.risk_metrics
    portfolio = st.session_state.portfolio
    
    # Row 1: Key Metrics
    st.header("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current BTC Price",
            f"${sim_results.current_price:,.0f}"
        )
    
    with col2:
        ltv_color = "normal" if risk_metrics.current_ltv < 0.7 else "off" if risk_metrics.current_ltv < 0.85 else "inverse"
        st.metric(
            "Current LTV",
            f"{risk_metrics.current_ltv * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            "P(Margin Call)",
            f"{risk_metrics.prob_margin_call * 100:.2f}%",
            help="Probability of LTV reaching 85% within 30 days"
        )
    
    with col4:
        st.metric(
            "P(Liquidation)",
            f"{risk_metrics.prob_liquidation * 100:.2f}%",
            help="Probability of LTV reaching 95% within 30 days"
        )
    
    # Regime indicator
    regime_color = "🟢" if regime == "normal" else "🔴"
    st.caption(f"Regime: {regime_color} {regime.upper()} | Simulation: {sim_results.n_paths:,} paths | Horizon: {sim_results.horizon_days} days")
    
    # Show any output warnings
    if st.session_state.get('output_warnings'):
        for warning in st.session_state.output_warnings:
            st.warning(f"⚠️ {warning}")
    
    st.divider()
    
    # Row 2: Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig_paths = create_price_paths_chart(sim_results, risk_metrics)
        if fig_paths:
            st.plotly_chart(fig_paths, use_container_width=True)
    
    with col_right:
        fig_dist = create_distribution_chart(sim_results, risk_metrics)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.divider()
    
    # Row 3: Probability Table and Threshold Analysis
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.subheader("Price Drop Probabilities")
        
        prob_table = sim_results.get_probability_table()
        
        # Style the dataframe
        def color_prob(val):
            if val < 5:
                return 'background-color: #90EE90'  # Light green
            elif val < 15:
                return 'background-color: #FFE4B5'  # Light orange
            else:
                return 'background-color: #FFB6C1'  # Light red
        
        display_df = prob_table.copy()
        display_df['probability'] = display_df['probability'] * 100
        display_df.columns = ['Drop %', 'Probability %', 'Price Level']
        display_df['Price Level'] = display_df['Price Level'].apply(lambda x: f"${x:,.0f}")
        display_df['Drop %'] = display_df['Drop %'].apply(lambda x: f"≥{x:.0f}%")
        display_df['Probability %'] = display_df['Probability %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with col_right2:
        st.subheader("Threshold Analysis")
        
        st.markdown(f"""
        **Position Summary**
        - Total Debt: **${risk_metrics.total_debt:,.0f}**
        - Current LTV: **{risk_metrics.current_ltv * 100:.1f}%**
        
        **Margin Call (85% LTV)**
        - Trigger Price: **${risk_metrics.margin_call_price:,.0f}**
        - Required Drop: **{risk_metrics.drop_to_margin_call * 100:.1f}%**
        - Price Buffer: ${risk_metrics.margin_call_buffer:,.0f}
        - 30-Day Probability: **{risk_metrics.prob_margin_call * 100:.2f}%**
        
        **Liquidation (95% LTV)**
        - Trigger Price: **${risk_metrics.liquidation_price:,.0f}**
        - Required Drop: **{risk_metrics.drop_to_liquidation * 100:.1f}%**
        - Price Buffer: ${risk_metrics.liquidation_buffer:,.0f}
        - 30-Day Probability: **{risk_metrics.prob_liquidation * 100:.2f}%**
        """)
    
    st.divider()
    
    # Row 4: VaR Analysis
    st.subheader("Value at Risk (VaR)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_1_drop = (1 - sim_results.var_1pct / sim_results.current_price) * 100
        st.metric(
            "1% VaR (Worst Case)",
            f"${sim_results.var_1pct:,.0f}",
            f"-{var_1_drop:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        var_5_drop = (1 - sim_results.var_5pct / sim_results.current_price) * 100
        st.metric(
            "5% VaR",
            f"${sim_results.var_5pct:,.0f}",
            f"-{var_5_drop:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        var_10_drop = (1 - sim_results.var_10pct / sim_results.current_price) * 100
        st.metric(
            "10% VaR",
            f"${sim_results.var_10pct:,.0f}",
            f"-{var_10_drop:.1f}%",
            delta_color="inverse"
        )
    
    st.divider()
    
    # Row 5: LTV Scenario Table
    st.subheader("LTV Scenario Analysis")
    
    calculator = LTVRiskCalculator()
    scenarios = calculator.generate_ltv_scenarios(portfolio)
    
    def style_status(val):
        if val == "OK":
            return 'background-color: #90EE90'
        elif val == "MARGIN CALL":
            return 'background-color: #FFE4B5'
        else:
            return 'background-color: #FFB6C1'
    
    display_scenarios = scenarios.copy()
    display_scenarios['price'] = display_scenarios['price'].apply(lambda x: f"${x:,.0f}")
    display_scenarios['ltv_pct'] = display_scenarios['ltv_pct'].apply(lambda x: f"{x:.1f}%")
    display_scenarios['drop_pct'] = display_scenarios['drop_pct'].apply(lambda x: f"{x:.0f}%")
    display_scenarios = display_scenarios[['drop_pct', 'price', 'ltv_pct', 'status']]
    display_scenarios.columns = ['Price Drop', 'BTC Price', 'LTV', 'Status']
    
    st.dataframe(
        display_scenarios.style.applymap(style_status, subset=['Status']),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Row 6: Historical Regime Indicators
    st.subheader("📊 Historical Regime Indicators")
    
    # Time range selector
    col_hist1, col_hist2 = st.columns([1, 3])
    with col_hist1:
        history_days = st.selectbox(
            "History Period",
            options=[30, 60, 90, 180, 365, 730, 1095],
            index=2,  # Default to 90 days
            format_func=lambda x: f"{x} days" if x < 365 else f"{x//365} year{'s' if x >= 730 else ''}"
        )
    
    try:
        fig_regime = create_regime_history_chart(days=history_days)
        st.plotly_chart(fig_regime, use_container_width=True)
        
        # Check if CryptoQuant data is available
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        try:
            onchain_count = conn.execute("SELECT COUNT(*) FROM onchain_metrics").fetchone()[0]
        except:
            onchain_count = 0
        conn.close()
        
        if onchain_count == 0:
            st.info("💡 **On-chain indicators not shown.** Add your CryptoQuant API key to Streamlit secrets to enable exchange netflow, funding rates, SOPR, and MVRV charts.")
            
    except Exception as e:
        st.warning(f"Could not load regime history: {e}")
        logger.error(f"Regime history chart failed: {e}")
    
    # Indicator explanations in expander
    with st.expander("📖 Understanding the Indicators", expanded=False):
        st.markdown("""
### Market Indicators

**VIX (Volatility Index)** — *The "Fear Gauge"*
> Measures expected S&P 500 volatility. When VIX > 30, markets are fearful and BTC often correlates with equities in a selloff. High VIX periods include COVID crash (March 2020: VIX hit 82), and banking crisis (March 2023: VIX ~30).

**BTC Volatility** — *Recent Price Swings*  
> Compares current 7-day volatility to 90-day average. When current vol exceeds 1.5× average, price is moving more than usual — often during crashes or sharp rallies.

**BTC Drawdown** — *Distance from Recent High*
> How far BTC has fallen from its 30-day high. Drawdowns > 15% suggest meaningful selling pressure. Major drawdowns: -50% in May 2021, -75% in Nov 2022.

**BTC-SPX Correlation** — *Risk Asset Behavior*
> When BTC moves with stocks during a selloff (correlation > 0.7 while SPX is down), it signals BTC is trading as a risk asset rather than a hedge. Common during macro fear events.

---

### On-Chain Indicators (CryptoQuant)

**Exchange Netflow** — *Selling Pressure Signal*
> Net BTC flowing into exchanges. **Positive = inflow** (coins moving to exchanges, likely to sell). **Negative = outflow** (coins leaving exchanges, accumulation). Large inflows (>10k BTC) often precede selling. FTX collapse saw massive inflows.

**Funding Rate** — *Derivatives Sentiment*
> Cost to hold leveraged positions. **Positive** = longs pay shorts (bullish bias). **Negative** = shorts pay longs (bearish bias). Deeply negative funding (-0.01% to -0.1%) signals extreme bearish sentiment, often near bottoms.

**SOPR (Spent Output Profit Ratio)** — *Realized Profit/Loss*
> Ratio of price sold vs price bought for coins moving on-chain. **SOPR > 1** = selling at profit. **SOPR < 1** = selling at loss (capitulation). SOPR persistently below 1 indicates holders are panic selling at a loss — often a bottom signal.

**MVRV (Market Value to Realized Value)** — *Over/Undervaluation*
> Compares market cap to realized cap (avg cost basis of all coins). **MVRV > 3.5** = historically overbought (cycle tops). **MVRV < 1** = market trading below aggregate cost basis (cycle bottoms, extreme fear).

---

### Historical Stress Periods

| Period | What Happened | Key Indicators |
|--------|--------------|----------------|
| **Mar 2020** | COVID crash, -50% in days | VIX 82, SOPR < 0.9, massive netflow |
| **May 2021** | China mining ban, -50% | High netflow, negative funding |
| **Nov 2022** | FTX collapse, -25% | Extreme netflow, SOPR < 0.95 |
| **Mar 2023** | Banking crisis (SVB) | VIX ~30, correlation spike |
        """)
    
    # Footer
    st.divider()
    
    # Debug info expander for troubleshooting differences
    with st.expander("🔧 Debug Info (for troubleshooting)", expanded=False):
        if 'debug_info' in st.session_state and st.session_state.debug_info:
            debug = st.session_state.debug_info
            st.markdown("### Simulation Inputs")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Price & Regime:**
                - Current Price: ${debug.get('current_price', 'N/A'):,.2f}
                - Regime: {debug.get('regime', 'N/A')}
                - Random Seed: 42 (fixed)
                """)
            with col2:
                if debug.get('garch_params'):
                    gp = debug['garch_params']
                    st.markdown(f"""
                    **GARCH Parameters:**
                    - ω (omega): {gp.get('omega', 'N/A'):.2e}
                    - α (alpha): {gp.get('alpha', 'N/A'):.4f}
                    - β (beta): {gp.get('beta', 'N/A'):.4f}
                    - μ (drift): {gp.get('mu', 'N/A'):.4f}%
                    - Long-run vol: {gp.get('long_run_vol', 'N/A'):.2f}%
                    """)
            
            st.markdown("### Simulation Outputs")
            st.markdown(f"""
            - Mean Terminal Price: ${debug.get('sim_mean', 'N/A'):,.2f}
            - 5th Percentile: ${debug.get('sim_p5', 'N/A'):,.2f}
            - 95th Percentile: ${debug.get('sim_p95', 'N/A'):,.2f}
            """)
            
            # Add copy button for easy comparison
            debug_str = str(debug)
            st.code(debug_str, language="python")
        else:
            st.info("Run a simulation to see debug info")
    
    st.caption(f"Last updated: {sim_results.simulation_date} | Execution time: {sim_results.execution_time_seconds:.2f}s")

else:
    # No simulation run yet
    st.info("👈 Configure your portfolio settings and click **Run Simulation** to start.")
    
    # Show current price
    try:
        current_price = get_current_price()
        st.metric("Current BTC Price", f"${current_price:,.0f}")
    except Exception as e:
        st.warning(f"Could not fetch current price: {e}")
    
    # Show GARCH calibration info
    try:
        garch_params = get_garch_params()
        if garch_params:
            st.subheader("Model Calibration")
            st.markdown(f"""
            - **Calibration Date:** {garch_params.calibration_date}
            - **Data Range:** {garch_params.data_start_date} to {garch_params.data_end_date}
            - **Long-run Volatility:** {garch_params.long_run_volatility:.1f}% annualized
            - **Persistence (α+β):** {garch_params.persistence:.3f}
            """)
    except Exception as e:
        st.warning(f"Could not load model calibration: {e}")
