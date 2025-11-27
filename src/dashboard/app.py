"""
BitVault BTC Risk Dashboard

Run with: streamlit run src/dashboard/app.py
"""

import os
import sys
import sqlite3

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
        
        # Check if we have sufficient data
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        
        try:
            count = conn.execute("SELECT COUNT(*) FROM btc_prices").fetchone()[0]
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            count = 0
        finally:
            conn.close()
        
        # If we have less than 100 rows, backfill data
        if count < 100:
            logger.info(f"Database has {count} rows, backfilling data...")
            
            # Show a message to the user
            with st.spinner("Initializing data (first run only, ~30 seconds)..."):
                # Backfill 3 years of price data
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
                
                # Fetch macro data
                try:
                    macro_fetcher = MacroFetcher()
                    df = macro_fetcher.fetch_all(start_date, end_date)
                    if not df.empty:
                        macro_fetcher.save_to_db(df)
                        logger.info(f"Backfilled macro data")
                except Exception as e:
                    logger.error(f"Failed to backfill macro: {e}")
            
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
    classifier = RegimeClassifier()
    return classifier.classify()


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
        
        # Run simulation
        engine = MonteCarloEngine()
        sim_results = engine.run(
            n_paths=100_000,
            horizon_days=30,
            regime=regime,
            current_price=current_price,
            store_paths=True
        )
        
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
        with st.sidebar.expander("Regime Indicators"):
            for ind in auto_classification.indicators:
                icon = "🔴" if ind.is_stress else "🟢"
                st.write(f"{icon} **{ind.name}**: {ind.value:.2f}")
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
    
    # Footer
    st.divider()
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