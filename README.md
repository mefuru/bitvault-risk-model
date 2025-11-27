# BitVault BTC Risk Model

A Monte Carlo simulation engine for managing Bitcoin-collateralized loan risk. Built for BitVault's bvUSD stablecoin protocol.

---

## Table of Contents

1. [What This Tool Does](#what-this-tool-does)
2. [Quick Start](#quick-start)
3. [Understanding the Model](#understanding-the-model)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Data Pipeline](#data-pipeline)
7. [Running the Dashboard](#running-the-dashboard)
8. [Using the Model](#using-the-model)
9. [Backtesting](#backtesting)
10. [Model Methodology](#model-methodology)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)
13. [Extending the Model](#extending-the-model)

---

## What This Tool Does

If you manage a Bitcoin-collateralized lending protocol, your primary risk is this: **what happens if BTC price drops and borrowers' collateral becomes insufficient?**

This tool answers that question quantitatively by:

1. **Simulating 100,000 possible BTC price paths** over the next 30 days
2. **Calculating the probability** that any given loan position hits margin call (85% LTV) or liquidation (95% LTV)
3. **Detecting market regime** (normal vs. stress) to adjust risk estimates accordingly
4. **Providing an interactive dashboard** for real-time risk monitoring

### Example Output

For a position with 100 BTC collateral and $5.35M loan at current BTC price of $91,000:

```
Current LTV: 62.1%
Margin Call Price: $63,187 (requires 30.6% drop)
Liquidation Price: $56,535 (requires 37.9% drop)

30-Day Probabilities:
  P(Margin Call): 1.58%
  P(Liquidation): 0.37%
```

This tells you: there's roughly a 1.6% chance this position will need attention in the next month.

---

## Quick Start

If you just want to get the dashboard running:

```bash
# 1. Clone/download the project
cd bitvault-risk-model

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Backfill historical data (first time only)
PYTHONPATH=. python -m src.data.prices --backfill 3
PYTHONPATH=. python -m src.data.macro --backfill 3

# 5. Calibrate the GARCH model (first time only)
PYTHONPATH=. python -m src.model.garch --save

# 6. Launch dashboard
PYTHONPATH=. streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. Enter your portfolio details and click "Run Simulation".

---

## Understanding the Model

### The Problem

You have a loan secured by Bitcoin:
- **Collateral**: 100 BTC
- **Loan Amount**: $5,350,000
- **Accrued Interest**: $20,858

Your **Loan-to-Value (LTV)** ratio is:

```
LTV = Total Debt / Collateral Value
    = ($5,350,000 + $20,858) / (100 × $91,000)
    = $5,370,858 / $9,100,000
    = 59.0%
```

If BTC price drops, your LTV increases. At certain thresholds:
- **85% LTV → Margin Call**: Borrower must add collateral or repay debt
- **95% LTV → Liquidation**: Protocol forcibly sells collateral

**The question**: What's the probability BTC drops enough to trigger these thresholds?

### The Solution

We can't predict BTC's future price, but we can model the **distribution of possible prices** based on:

1. **Historical volatility patterns** (using GARCH model)
2. **Jump risk** (sudden large moves, like the March 2020 COVID crash)
3. **Current market regime** (calm vs. stressed conditions)

By simulating 100,000 price paths, we get a probability distribution. If 1,580 paths out of 100,000 hit the margin call price, we estimate P(Margin Call) ≈ 1.58%.

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on Python 3.11)
- **macOS or Linux** (Windows may work but is untested)
- **~500MB disk space** for historical data and dependencies

### Step-by-Step Installation

#### 1. Create Project Directory

```bash
mkdir bitvault-risk-model
cd bitvault-risk-model
```

#### 2. Set Up Virtual Environment

We use Python's built-in `venv` to isolate dependencies:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (you'll need to do this each time you open a new terminal)
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
```

**Why a virtual environment?** It keeps this project's dependencies separate from your system Python, avoiding conflicts with other projects.

#### 3. Install Dependencies

```bash
pip install numpy scipy pandas arch requests streamlit plotly pyyaml pytest yfinance fredapi python-dotenv tqdm
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### 4. Create Required Directories

```bash
mkdir -p data logs reports notebooks
mkdir -p src/data src/model src/regime src/risk src/dashboard src/backtest
mkdir -p scripts
```

#### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following (get a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html):

```
FRED_API_KEY=your_fred_api_key_here
```

The FRED API provides macroeconomic data (Fed Funds Rate). The other data sources (Yahoo Finance for BTC prices, VIX, S&P 500) don't require API keys.

#### 6. Verify Installation

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Test that imports work
python -c "import numpy, pandas, arch, streamlit; print('All imports successful!')"
```

---

## Project Structure

```
bitvault-risk-model/
│
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                 # Configuration loading
│   ├── logging_config.py         # Logging setup
│   ├── validation.py             # Input validation
│   │
│   ├── data/                     # Data fetching and storage
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite schema and connections
│   │   ├── prices.py             # BTC price fetcher
│   │   ├── macro.py              # Macro indicators (VIX, S&P, Fed Funds)
│   │   └── refresh.py            # Data refresh utilities
│   │
│   ├── model/                    # Core modeling
│   │   ├── __init__.py
│   │   ├── garch.py              # GARCH(1,1) volatility model
│   │   └── simulation.py         # Monte Carlo simulation engine
│   │
│   ├── regime/                   # Market regime detection
│   │   ├── __init__.py
│   │   └── classifier.py         # Normal vs. stress regime classification
│   │
│   ├── risk/                     # Risk calculations
│   │   ├── __init__.py
│   │   └── ltv.py                # LTV calculations and thresholds
│   │
│   ├── dashboard/                # Streamlit web interface
│   │   ├── __init__.py
│   │   └── app.py                # Main dashboard application
│   │
│   └── backtest/                 # Model validation
│       ├── __init__.py
│       ├── runner.py             # Backtest execution
│       └── report.py             # Report generation
│
├── scripts/                      # Utility scripts
│   └── backfill_historical.py    # Extended data backfill for backtesting
│
├── data/                         # SQLite database (created automatically)
│   └── btc_risk.db
│
├── logs/                         # Application logs
│   └── risk_model_YYYYMMDD.log
│
├── reports/                      # Generated backtest reports
│   └── backtest_report_*.html
│
├── config/                       # Configuration files
│   └── config.yaml
│
├── .env                          # Environment variables (API keys)
├── .gitignore
├── requirements.txt
└── README.md                     # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `src/model/garch.py` | Fits a GARCH(1,1) model to historical BTC returns to estimate volatility dynamics |
| `src/model/simulation.py` | Runs Monte Carlo simulation with GARCH volatility and jump diffusion |
| `src/risk/ltv.py` | Calculates LTV ratios, threshold prices, and breach probabilities |
| `src/regime/classifier.py` | Detects whether market is in "normal" or "stress" regime |
| `src/dashboard/app.py` | Streamlit web application for interactive risk analysis |
| `src/backtest/runner.py` | Runs historical backtests to validate model accuracy |

---

## Data Pipeline

The model requires historical data to calibrate its parameters. Here's what we fetch and why:

### Data Sources

| Data | Source | Purpose | Update Frequency |
|------|--------|---------|------------------|
| BTC Price | Yahoo Finance | GARCH calibration, current price | Daily |
| VIX | Yahoo Finance | Regime detection (fear gauge) | Daily |
| S&P 500 | Yahoo Finance | Regime detection (correlation) | Daily |
| Fed Funds Rate | FRED API | Macro context | Monthly |

### Initial Data Backfill

Before using the model, you need historical data. Run these commands:

```bash
# Activate virtual environment
source venv/bin/activate

# Backfill 3 years of BTC prices (required for GARCH calibration)
PYTHONPATH=. python -m src.data.prices --backfill 3

# Backfill macro indicators
PYTHONPATH=. python -m src.data.macro --backfill 3
```

**What does `PYTHONPATH=.` mean?** It tells Python to look for modules in the current directory. Without it, `from src.data.prices import ...` would fail because Python wouldn't know where to find `src`.

### Verifying Data

Check that data was loaded correctly:

```bash
# Open SQLite database
sqlite3 data/btc_risk.db

# Check BTC prices
SELECT COUNT(*) as rows, MIN(date) as earliest, MAX(date) as latest FROM btc_prices;

# Check macro data
SELECT indicator, COUNT(*) as rows FROM macro_data GROUP BY indicator;

# Exit SQLite
.quit
```

You should see ~1,095 rows for BTC prices (3 years × 365 days) and similar counts for VIX and S&P 500.

### Refreshing Data

Data becomes stale over time. To fetch the latest:

**Option 1: From the dashboard**
Click the "🔄 Refresh Data" button in the sidebar.

**Option 2: From command line**
```bash
PYTHONPATH=. python -c "from src.data.refresh import refresh_all_data; refresh_all_data()"
```

---

## Running the Dashboard

The dashboard is the primary interface for using the model.

### Starting the Dashboard

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Launch Streamlit
PYTHONPATH=. streamlit run src/dashboard/app.py
```

This opens your browser to `http://localhost:8501`.

### Dashboard Features

#### Sidebar (Left Panel)

1. **Live BTC Price**: Fetched from Yahoo Finance on page load

2. **Data Status**: Shows when data was last updated
   - Green: Data is current
   - Yellow: Data is 2+ days old
   - Red: Data is 7+ days old (refresh recommended)

3. **Portfolio Settings**:
   - **BTC Collateral**: Number of BTC in the position
   - **Loan Amount**: Principal borrowed (in USD)
   - **Accrued Interest**: Interest accumulated (in USD)
   - **Total Debt**: Automatically calculated (Loan + Interest)

4. **Market Regime**: 
   - **Auto**: Automatically detects based on VIX, volatility, drawdown
   - **Normal**: Standard market conditions
   - **Stress**: Elevated tail risk assumptions

5. **Run Simulation**: Executes 100,000 Monte Carlo paths

#### Main Panel (Results)

After running a simulation:

1. **Executive Summary**: Key metrics at a glance
   - Current BTC Price
   - Current LTV
   - P(Margin Call) - probability of hitting 85% LTV
   - P(Liquidation) - probability of hitting 95% LTV

2. **Price Paths Chart**: Visualizes simulated price trajectories
   - Shaded bands show 5th-95th and 25th-75th percentiles
   - Horizontal lines mark margin call and liquidation prices

3. **Terminal Price Distribution**: Histogram of day-30 prices

4. **Price Drop Probabilities**: Table showing P(drop ≥ X%) for various thresholds

5. **Threshold Analysis**: Detailed breakdown of margin call and liquidation levels

6. **Value at Risk (VaR)**: 1%, 5%, and 10% worst-case price levels

7. **LTV Scenario Table**: Shows LTV at various price drops (0%, 5%, 10%, etc.)

### Interpreting Results

**Example interpretation:**

```
P(Margin Call) = 1.58%
P(Liquidation) = 0.37%
```

This means:
- Out of 100,000 simulated scenarios, 1,580 hit the margin call price within 30 days
- 370 scenarios hit the liquidation price
- Your position is relatively safe, but not risk-free

**When to worry:**
- P(Margin Call) > 10%: Consider reducing loan or adding collateral
- P(Liquidation) > 5%: Urgent action recommended

---

## Using the Model

### From Command Line

You can run analyses without the dashboard:

#### Run Monte Carlo Simulation
```bash
PYTHONPATH=. python -m src.model.simulation
```

Options:
```bash
# More paths for higher precision
PYTHONPATH=. python -m src.model.simulation --paths=200000

# Stress regime
PYTHONPATH=. python -m src.model.simulation --stress
```

#### Analyze a Portfolio
```bash
PYTHONPATH=. python -m src.risk.ltv --btc=100 --loan=5350000 --interest=20858
```

#### Check Market Regime
```bash
PYTHONPATH=. python -m src.regime.classifier
```

#### Calibrate GARCH Model
```bash
# Calibrate and save to database
PYTHONPATH=. python -m src.model.garch --save
```

### From Python Code

```python
from src.risk.ltv import analyze_portfolio

# Define your position
portfolio, risk_metrics = analyze_portfolio(
    btc_collateral=100.0,
    loan_amount=5_350_000,
    accrued_interest=20_858,
    regime="normal"
)

# Access results
print(f"Current LTV: {risk_metrics.current_ltv * 100:.1f}%")
print(f"P(Margin Call): {risk_metrics.prob_margin_call * 100:.2f}%")
print(f"Margin Call Price: ${risk_metrics.margin_call_price:,.0f}")
```

---

## Backtesting

Backtesting validates the model by checking if historical predictions were accurate.

### What Backtesting Does

1. Goes back to a historical date (e.g., January 1, 2023)
2. Uses only data available at that time to calibrate the model
3. Runs a simulation predicting the next 30 days
4. Compares predictions to what actually happened
5. Repeats for ~150 dates over 3 years

### Running a Backtest

#### Step 1: Ensure Sufficient Historical Data

You need 6+ years of data to backtest from 2022 onwards (3 years for calibration + 3 years to test):

```bash
PYTHONPATH=. python scripts/backfill_historical.py
```

#### Step 2: Run the Backtest

```bash
PYTHONPATH=. python -m src.backtest.report
```

This takes 5-10 minutes and generates an HTML report in `reports/`.

#### Step 3: View Results

Open the generated `reports/backtest_report_*.html` in your browser.

### Interpreting Backtest Results

#### Calibration Error

**Positive error** = model overestimates risk (predicts more drops than occur)
**Negative error** = model underestimates risk (predicts fewer drops than occur)

For risk management, **slight overestimation is preferred**. It's better to be cautious.

#### Brier Score

Measures prediction accuracy:
- **0.0** = Perfect predictions
- **0.25** = Random guessing
- **< 0.10** = Good
- **0.10 - 0.15** = Acceptable
- **> 0.15** = Poor

#### Example Results

| Drop | Predicted | Actual | Error |
|------|-----------|--------|-------|
| ≥5% | 61.7% | 53.1% | +8.6% |
| ≥10% | 39.9% | 22.1% | +17.8% |
| ≥20% | 13.0% | 0.7% | +12.4% |

**Interpretation**: The model predicted a 13% chance of ≥20% drops, but only 0.7% actually occurred. The model is conservative—it overestimates risk. For a risk management tool, this is acceptable because it provides a safety margin.

---

## Model Methodology

### GARCH(1,1) Volatility Model

Bitcoin volatility is not constant—it clusters. Periods of high volatility tend to be followed by more high volatility. GARCH captures this:

```
σ²(t) = ω + α·r²(t-1) + β·σ²(t-1)
```

Where:
- `σ²(t)` = today's variance
- `ω` = long-run variance component
- `α` = reaction to recent shocks
- `β` = persistence of volatility
- `r(t-1)` = yesterday's return

**Parameters from calibration:**
- `α ≈ 0.10`: Volatility reacts moderately to shocks
- `β ≈ 0.79`: Volatility is persistent (decays slowly)
- `α + β ≈ 0.89`: High persistence, volatility mean-reverts over ~6 days

### Jump Diffusion (Merton Model)

Large BTC moves (>3 standard deviations) happen more often than a normal distribution predicts. We add "jumps":

```
dS/S = μ·dt + σ·dW + J·dN
```

Where:
- `μ` = drift (average daily return)
- `σ` = volatility from GARCH
- `dW` = Brownian motion (normal randomness)
- `J` = jump size (typically -5% to -15%)
- `dN` = Poisson process (random jump timing)

**Calibrated from historical data:**
- ~6 jumps per year
- Average jump size: -7.9%
- This captures events like COVID crash, LUNA collapse, FTX bankruptcy

### Regime Detection

The model detects two regimes:

| Regime | Condition | Model Adjustment |
|--------|-----------|------------------|
| **Normal** | Low VIX, stable volatility, no drawdown | Standard parameters |
| **Stress** | VIX > 30, elevated volatility, or >15% drawdown | 1.5x volatility multiplier, zero drift |

Regime detection uses:
1. **VIX level**: Above 30 = fear in equity markets
2. **BTC volatility**: Current vs. 90-day average
3. **BTC drawdown**: Current price vs. 30-day high
4. **BTC-S&P correlation**: High correlation during selloffs = contagion

### LTV Calculations

```
LTV = Total Debt / Collateral Value

Where:
  Total Debt = Loan Amount + Accrued Interest
  Collateral Value = BTC Quantity × BTC Price
```

**Price at Target LTV:**
```
Price = Total Debt / (Target LTV × BTC Quantity)
```

**Example:**
- Total Debt: $5,370,858
- BTC Quantity: 100
- Target LTV: 85% (margin call)
- Margin Call Price = $5,370,858 / (0.85 × 100) = $63,187

---

## Configuration

### Environment Variables (.env)

```bash
# Required for FRED API (Fed Funds Rate data)
FRED_API_KEY=your_key_here

# Optional: CryptoQuant API (not currently used, for future on-chain data)
CRYPTOQUANT_API_KEY=your_key_here
```

### Configuration File (config/config.yaml)

```yaml
# Data settings
data:
  btc_source: yahoo  # yahoo or cryptoquant
  history_years: 3

# Model settings
model:
  n_paths: 100000
  horizon_days: 30
  
# Risk thresholds
risk:
  margin_call_ltv: 0.85
  liquidation_ltv: 0.95

# Regime detection
regime:
  vix_threshold: 30
  vol_multiplier_stress: 1.5
  drawdown_threshold: -0.15
```

### Modifying Thresholds

To change margin call and liquidation LTV levels, edit `src/risk/ltv.py`:

```python
@dataclass
class LTVThresholds:
    margin_call: float = 0.85    # Change this
    liquidation: float = 0.95    # Change this
```

---

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'src'"

**Cause**: Python doesn't know where to find the `src` package.

**Solution**: Always run with `PYTHONPATH=.`:
```bash
PYTHONPATH=. python -m src.model.simulation
```

Or add to your shell profile:
```bash
echo 'export PYTHONPATH=".:$PYTHONPATH"' >> ~/.zshrc
source ~/.zshrc
```

#### "No data in database"

**Cause**: Historical data hasn't been backfilled.

**Solution**: Run the backfill commands:
```bash
PYTHONPATH=. python -m src.data.prices --backfill 3
PYTHONPATH=. python -m src.data.macro --backfill 3
```

#### Dashboard shows "Data is X days old"

**Cause**: Price data is stale.

**Solution**: Click "🔄 Refresh Data" in the dashboard sidebar, or:
```bash
PYTHONPATH=. python -c "from src.data.refresh import refresh_all_data; refresh_all_data()"
```

#### "GARCH calibration failed"

**Cause**: Usually insufficient data or extreme values.

**Solution**: 
1. Check you have enough historical data (~1,000+ days)
2. Re-run calibration: `PYTHONPATH=. python -m src.model.garch --save`

#### Simulation runs but probabilities seem wrong

**Cause**: GARCH parameters may be stale.

**Solution**: Recalibrate:
```bash
PYTHONPATH=. python -m src.model.garch --save
```

#### "Starting LTV too high" error

**Cause**: Your position is already near margin call (>90% LTV).

**Solution**: This is a validation check. Either:
1. The position genuinely needs immediate attention
2. Double-check your input values

### Checking Logs

Logs are written to `logs/risk_model_YYYYMMDD.log`:

```bash
# View today's logs
cat logs/risk_model_$(date +%Y%m%d).log

# Watch logs in real-time
tail -f logs/risk_model_$(date +%Y%m%d).log
```

---

## Extending the Model

### Adding On-Chain Data

The model is designed to incorporate on-chain data when available. With a CryptoQuant API subscription, you could add:

- **Exchange netflows**: Large outflows often precede selling
- **Funding rates**: Negative rates = bearish sentiment
- **Open interest**: Sudden drops may indicate liquidations

Edit `src/regime/classifier.py` to add new indicators to regime detection.

### Adding New Risk Metrics

To add metrics beyond LTV:

1. Create new calculations in `src/risk/`
2. Add to `SimulationResults` class in `src/model/simulation.py`
3. Display in dashboard via `src/dashboard/app.py`

### Supporting Multiple Portfolios

The current model analyzes one position at a time. To track multiple:

1. Create a `portfolios.yaml` config file
2. Modify `src/risk/ltv.py` to iterate over portfolios
3. Update dashboard to show aggregate risk

### Alerting

To get notifications when risk exceeds thresholds:

1. Add email/Slack integration to `src/` 
2. Create a scheduled job that runs the simulation
3. Trigger alerts when P(Margin Call) exceeds your threshold

---

## Development Notes

### Running Tests

```bash
PYTHONPATH=. pytest tests/
```

### Code Style

The codebase follows these conventions:
- Type hints for function signatures
- Dataclasses for structured data
- Docstrings for public functions
- Logging instead of print statements

### Adding Dependencies

```bash
# Activate virtual environment first
source venv/bin/activate

# Install new package
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

---

## License

Internal use only - BitVault proprietary.

---

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Check `logs/` for error details
3. Contact the development team

---

## Changelog

### v1.0.0 (2024-11)
- Initial release
- GARCH(1,1) volatility model
- Monte Carlo simulation with jump diffusion
- Regime detection (VIX, volatility, drawdown)
- LTV risk calculations
- Streamlit dashboard
- Backtesting framework
- Input validation and error handling
