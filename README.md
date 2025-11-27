# BitVault BTC Risk Model

Monte Carlo simulation model for Bitcoin price risk, designed to protect BitVault's BTC-backed stablecoin protocol from adverse price movements.

## Overview

This model simulates 100,000 BTC price paths over a 30-day horizon using GARCH(1,1) volatility with Merton jump diffusion. It calculates the probability of LTV threshold breaches (margin call at 85%, liquidation at 95%) and provides a dashboard for risk monitoring.

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.template .env
cp config/config.template.yaml config/config.yaml
```

Edit `.env` with your API keys:
- **CryptoQuant**: Get from your CryptoQuant dashboard
- **FRED**: Register free at https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Initialize Database

```bash
python -m src.data.init_db
```

### 4. Backfill Historical Data

```bash
python -m src.data.backfill --years 3
```

### 5. Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Project Structure

```
bitvault-risk-model/
├── src/
│   ├── data/           # Data fetching and storage
│   ├── model/          # GARCH, jump diffusion, Monte Carlo
│   ├── regime/         # Market regime classification
│   ├── risk/           # LTV and risk metric calculations
│   └── dashboard/      # Streamlit application
├── tests/              # Test suite
├── notebooks/          # Exploration and validation
├── data/               # SQLite database (gitignored)
├── logs/               # Execution logs (gitignored)
├── config/
│   ├── config.template.yaml
│   └── config.yaml     # Your config (gitignored)
└── requirements.txt
```

## Daily Execution

```bash
python run_daily.py
```

This fetches latest data, runs the simulation, and updates the dashboard.

## Key Components

### Data Sources
- **BTC Price**: CryptoQuant OHLCV
- **On-chain**: Exchange flows, whale movements, funding rates, open interest
- **Macro**: Fed funds rate (FRED), S&P 500, VIX (Yahoo Finance)

### Model
- **Volatility**: GARCH(1,1) with weekly recalibration
- **Jumps**: Merton jump diffusion for tail events
- **Regimes**: Normal vs Stress, based on market indicators

### Outputs
- Probability distribution of 30-day price changes
- P(Margin Call), P(Liquidation) given current portfolio
- VaR and CVaR at multiple confidence levels
- Interactive dashboard for monitoring

## License

Internal use only - BitVault proprietary.
