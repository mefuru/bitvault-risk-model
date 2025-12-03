"""
CryptoQuant API Endpoint Tester

Tests which endpoints are accessible with your API key.
Run with: PYTHONPATH=. python scripts/test_cryptoquant_endpoints.py
"""

import os
import sys
import requests
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_api_key


def test_endpoint(name: str, endpoint: str, api_key: str, params: dict = None) -> dict:
    """Test a single endpoint and return results."""
    base_url = "https://api.cryptoquant.com"
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Default params: last 7 days
    if params is None:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        params = {
            "window": "DAY",
            "from": start_date,
            "to": end_date,
            "limit": 10
        }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        result = {
            "name": name,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "accessible": response.status_code == 200,
        }
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data and "data" in data["result"]:
                result["rows"] = len(data["result"]["data"])
                result["sample"] = data["result"]["data"][0] if data["result"]["data"] else None
            else:
                result["rows"] = 0
                result["sample"] = None
        else:
            result["error"] = response.text[:200]
            
        return result
        
    except Exception as e:
        return {
            "name": name,
            "endpoint": endpoint,
            "status_code": None,
            "accessible": False,
            "error": str(e)
        }


def main():
    print("=" * 70)
    print("CryptoQuant API Endpoint Tester")
    print("=" * 70)
    
    # Get API key
    try:
        api_key = get_api_key("cryptoquant")
        print(f"\n✓ API Key loaded: {api_key[:10]}...{api_key[-6:]}")
    except ValueError as e:
        print(f"\n✗ API Key error: {e}")
        print("\nMake sure CRYPTOQUANT_API_KEY is set in your .env file")
        sys.exit(1)
    
    # Define endpoints to test (organized by category)
    endpoints = [
        # === EXCHANGE FLOWS (Critical for regime detection) ===
        ("Exchange Netflow (All)", "/v1/btc/exchange-flows/netflow", {}),
        ("Exchange Inflow", "/v1/btc/exchange-flows/inflow", {}),
        ("Exchange Outflow", "/v1/btc/exchange-flows/outflow", {}),
        ("Exchange Reserve", "/v1/btc/exchange-flows/reserve", {}),
        
        # === MARKET DATA ===
        ("BTC Price OHLCV", "/v1/btc/market-data/price-ohlcv", {}),
        ("Funding Rates", "/v1/btc/market-data/funding-rates", {}),
        ("Open Interest", "/v1/btc/market-data/open-interest", {}),
        ("Estimated Leverage Ratio", "/v1/btc/market-data/estimated-leverage-ratio", {}),
        ("Liquidations", "/v1/btc/market-data/liquidations", {}),
        
        # === NETWORK DATA ===
        ("Active Addresses", "/v1/btc/network-data/active-addresses", {}),
        ("Transaction Count", "/v1/btc/network-data/transaction-count", {}),
        
        # === FLOW INDICATORS ===
        ("Fund Flow Ratio", "/v1/btc/flow-indicator/fund-flow-ratio", {}),
        ("Exchange Whale Ratio", "/v1/btc/flow-indicator/exchange-whale-ratio", {}),
        
        # === MARKET INDICATORS ===
        ("SOPR", "/v1/btc/market-indicator/sopr", {}),
        ("NUPL", "/v1/btc/market-indicator/nupl", {}),
        ("MVRV", "/v1/btc/market-indicator/mvrv", {}),
        
        # === MINER DATA ===
        ("Miner Outflow", "/v1/btc/miner-flows/outflow", {}),
    ]
    
    print("\n" + "-" * 70)
    print("Testing endpoints...")
    print("-" * 70)
    
    accessible = []
    not_accessible = []
    
    for name, endpoint, custom_params in endpoints:
        params = custom_params if custom_params else None
        result = test_endpoint(name, endpoint, api_key, params)
        
        if result["accessible"]:
            accessible.append(result)
            status = f"✓ {result.get('rows', '?')} rows"
        else:
            not_accessible.append(result)
            status = f"✗ {result['status_code']}"
        
        print(f"  {status:15} {name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n✓ Accessible: {len(accessible)} endpoints")
    print(f"✗ Not accessible: {len(not_accessible)} endpoints")
    
    if accessible:
        print("\n" + "-" * 70)
        print("ACCESSIBLE ENDPOINTS (can use these)")
        print("-" * 70)
        for r in accessible:
            print(f"  • {r['name']}")
            print(f"    Endpoint: {r['endpoint']}")
            if r.get('sample'):
                # Show sample keys
                keys = list(r['sample'].keys())[:5]
                print(f"    Fields: {', '.join(keys)}")
            print()
    
    if not_accessible:
        print("-" * 70)
        print("NOT ACCESSIBLE (would need plan upgrade)")
        print("-" * 70)
        for r in not_accessible:
            error_msg = r.get('error', 'Unknown error')[:80]
            print(f"  • {r['name']} - Status {r['status_code']}: {error_msg}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR REGIME DETECTION")
    print("=" * 70)
    
    # Check which critical endpoints are available
    critical_endpoints = {
        "Exchange Netflow (All)": "exchange_netflow",
        "Funding Rates": "funding_rates", 
        "Open Interest": "open_interest",
        "Exchange Whale Ratio": "whale_ratio",
        "SOPR": "sopr"
    }
    
    available_critical = []
    missing_critical = []
    
    for name, key in critical_endpoints.items():
        if any(r['name'] == name for r in accessible):
            available_critical.append(name)
        else:
            missing_critical.append(name)
    
    if available_critical:
        print("\n✓ Available for regime detection:")
        for name in available_critical:
            print(f"  • {name}")
    
    if missing_critical:
        print("\n✗ Not available (consider alternatives):")
        for name in missing_critical:
            print(f"  • {name}")
    
    print("\n" + "=" * 70)
    
    return accessible, not_accessible


if __name__ == "__main__":
    accessible, not_accessible = main()
