"""
Test CryptoQuant API access and available endpoints.
"""

import requests
from src.config import get_api_key, load_config


def test_endpoint(api_key: str, endpoint: str, params: dict = None) -> dict:
    """Test a CryptoQuant API endpoint."""
    base_url = "https://api.cryptoquant.com"
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        return {
            "endpoint": endpoint,
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text[:500]
        }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "status_code": None,
            "success": False,
            "response": str(e)
        }


def main():
    print("=" * 60)
    print("CryptoQuant API Access Test")
    print("=" * 60)
    
    try:
        api_key = get_api_key("cryptoquant")
        print(f"✓ API key loaded (ends with ...{api_key[-6:]})\n")
    except ValueError as e:
        print(f"✗ {e}")
        return
    
    # Endpoints to test (from CryptoQuant docs)
    endpoints = [
        # Market data
        ("/v1/btc/market-data/price-ohlcv", {"window": "day", "limit": 1}),
        
        # Exchange flows
        ("/v1/btc/exchange-flows/netflow", {"exchange": "all_exchange", "window": "day", "limit": 1}),
        ("/v1/btc/exchange-flows/inflow", {"exchange": "all_exchange", "window": "day", "limit": 1}),
        ("/v1/btc/exchange-flows/outflow", {"exchange": "all_exchange", "window": "day", "limit": 1}),
        
        # Fund flows (alternative endpoint structure)
        ("/v1/btc/fund-data/exchange-net-position-change", {"window": "day", "limit": 1}),
        
        # Derivatives / Funding rates
        ("/v1/btc/market-data/funding-rates", {"exchange": "binance", "window": "day", "limit": 1}),
        ("/v1/btc/derivatives/funding-rates", {"exchange": "all_exchange", "limit": 1}),
        
        # Open interest
        ("/v1/btc/market-data/open-interest", {"exchange": "all_exchange", "limit": 1}),
        ("/v1/btc/derivatives/open-interest", {"exchange": "all_exchange", "limit": 1}),
        
        # Whale / Large transactions
        ("/v1/btc/network-data/transactions-count-large", {"window": "day", "limit": 1}),
        ("/v1/btc/flow-indicator/whale-ratio", {"window": "day", "limit": 1}),
        
        # Miner flows
        ("/v1/btc/miner-flows/miner-to-exchange", {"miner": "all_miner", "exchange": "all_exchange", "window": "day", "limit": 1}),
    ]
    
    print("Testing endpoints...\n")
    
    accessible = []
    denied = []
    
    for endpoint, params in endpoints:
        result = test_endpoint(api_key, endpoint, params)
        
        if result["success"]:
            print(f"✓ {endpoint}")
            accessible.append(endpoint)
            # Show sample data
            if isinstance(result["response"], dict) and "result" in result["response"]:
                data = result["response"]["result"]
                if isinstance(data, dict) and "data" in data:
                    sample = data["data"][:1] if isinstance(data["data"], list) else data["data"]
                    print(f"    Sample: {str(sample)[:100]}...")
        else:
            print(f"✗ {endpoint} ({result['status_code']})")
            denied.append(endpoint)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accessible endpoints: {len(accessible)}")
    print(f"Denied/Error endpoints: {len(denied)}")
    
    if accessible:
        print("\n✓ Available for regime classification:")
        for ep in accessible:
            print(f"    {ep}")
    
    if denied:
        print("\n✗ Not available (may require higher tier):")
        for ep in denied:
            print(f"    {ep}")


if __name__ == "__main__":
    main()
