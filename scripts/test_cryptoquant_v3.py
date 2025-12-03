"""
CryptoQuant API Test - Updated with correct parameter format

Based on working examples from their documentation.
Run with: PYTHONPATH=. python scripts/test_cryptoquant_v3.py
"""

import os
import sys
import requests
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_api_key


def test_endpoint(name: str, url: str, api_key: str) -> dict:
    """Test a single endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        result = {
            "name": name,
            "url": url,
            "status_code": response.status_code,
            "accessible": response.status_code == 200,
        }
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data and "data" in data["result"]:
                result["rows"] = len(data["result"]["data"])
                if data["result"]["data"]:
                    result["sample_keys"] = list(data["result"]["data"][0].keys())
            else:
                result["response_keys"] = list(data.keys()) if isinstance(data, dict) else "not a dict"
        else:
            # Get error details
            try:
                error_data = response.json()
                result["error"] = error_data.get("status", {}).get("message", response.text[:100])
            except:
                result["error"] = response.text[:100]
            
        return result
        
    except Exception as e:
        return {
            "name": name,
            "url": url,
            "status_code": None,
            "accessible": False,
            "error": str(e)
        }


def main():
    print("=" * 70)
    print("CryptoQuant API Test v3 - Correct Parameter Format")
    print("=" * 70)
    
    # Get API key
    try:
        api_key = get_api_key("cryptoquant")
        print(f"\n✓ API Key: {api_key[:10]}...{api_key[-6:]}")
    except ValueError as e:
        print(f"\n✗ API Key error: {e}")
        sys.exit(1)
    
    # Date parameters (using YYYYMMDD format as shown in docs)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Base URL
    base = "https://api.cryptoquant.com/v1"
    
    # Test endpoints with the documented parameter format
    # Format: window=day&from=YYYYMMDD&to=YYYYMMDD
    endpoints = [
        # === EXCHANGE FLOWS ===
        ("Exchange Netflow (all)", f"{base}/btc/exchange-flows/netflow?window=day&from={start_date}&to={end_date}"),
        ("Exchange Netflow (binance)", f"{base}/btc/exchange-flows/netflow?window=day&from={start_date}&to={end_date}&exchange=binance"),
        ("Exchange Inflow", f"{base}/btc/exchange-flows/inflow?window=day&from={start_date}&to={end_date}"),
        ("Exchange Outflow", f"{base}/btc/exchange-flows/outflow?window=day&from={start_date}&to={end_date}"),
        ("Exchange Reserve", f"{base}/btc/exchange-flows/reserve?window=day&from={start_date}&to={end_date}"),
        
        # === MARKET DATA ===
        ("Funding Rates (all)", f"{base}/btc/market-data/funding-rates?window=day&from={start_date}&exchange=all_exchange"),
        ("Funding Rates (binance)", f"{base}/btc/market-data/funding-rates?window=day&from={start_date}&exchange=binance"),
        ("Open Interest", f"{base}/btc/market-data/open-interest?window=day&from={start_date}&to={end_date}"),
        ("Price OHLCV", f"{base}/btc/market-data/price-ohlcv?window=day&from={start_date}&to={end_date}"),
        ("Liquidations", f"{base}/btc/market-data/liquidations?window=day&from={start_date}&to={end_date}"),
        
        # === FLOW INDICATORS ===
        ("Fund Flow Ratio", f"{base}/btc/flow-indicator/fund-flow-ratio?window=day&from={start_date}&to={end_date}"),
        ("Exchange Whale Ratio", f"{base}/btc/flow-indicator/exchange-whale-ratio?window=day&from={start_date}&to={end_date}"),
        
        # === MARKET INDICATORS ===
        ("SOPR", f"{base}/btc/market-indicator/sopr?window=day&from={start_date}&to={end_date}"),
        ("MVRV", f"{base}/btc/market-indicator/mvrv?window=day&from={start_date}&to={end_date}"),
        
        # === MINER FLOWS ===
        ("Miner Outflow", f"{base}/btc/miner-flows/outflow?window=day&from={start_date}&to={end_date}"),
        
        # === TRY WITHOUT PARAMS (maybe some endpoints don't need them) ===
        ("Netflow (no params)", f"{base}/btc/exchange-flows/netflow"),
        ("Status check", f"{base}/btc/status"),
    ]
    
    print("\n" + "-" * 70)
    print("Testing endpoints...")
    print("-" * 70)
    
    accessible = []
    not_accessible = []
    
    for name, url in endpoints:
        result = test_endpoint(name, url, api_key)
        
        if result["accessible"]:
            accessible.append(result)
            rows = result.get('rows', '?')
            print(f"  ✓ {name}")
            print(f"      Rows: {rows}")
            if result.get('sample_keys'):
                print(f"      Fields: {', '.join(result['sample_keys'][:5])}")
        else:
            not_accessible.append(result)
            error = result.get('error', 'Unknown')
            print(f"  ✗ {name} - {result['status_code']}: {error}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Accessible: {len(accessible)}")
    print(f"✗ Not accessible: {len(not_accessible)}")
    
    if accessible:
        print("\n" + "-" * 70)
        print("WORKING ENDPOINTS - Can use these for regime detection:")
        print("-" * 70)
        for r in accessible:
            print(f"\n  • {r['name']}")
            print(f"    URL: {r['url'][:80]}...")
            if r.get('sample_keys'):
                print(f"    Fields: {r['sample_keys']}")
    
    # If nothing works, provide troubleshooting
    if not accessible:
        print("\n" + "=" * 70)
        print("TROUBLESHOOTING")
        print("=" * 70)
        print("""
All endpoints returned errors. Please check:

1. API Key Permissions:
   - Log into cryptoquant.com
   - Go to your profile/settings
   - Find API section
   - Ensure your key has API access enabled
   - Try generating a NEW API key

2. Subscription Tier:
   - Professional plan SHOULD include API
   - But API might be a separate add-on
   - Check your subscription details
   - Contact CryptoQuant support if unclear

3. IP Restrictions:
   - Some accounts have IP whitelisting
   - Add your current IP to whitelist

4. Rate Limiting:
   - Wait a few minutes and try again
   - You might have hit a rate limit

5. Contact Support:
   - Email: support@cryptoquant.com
   - Tell them you have Professional plan
   - Ask why API returns 403/400 errors
""")

    return accessible


if __name__ == "__main__":
    accessible = main()
