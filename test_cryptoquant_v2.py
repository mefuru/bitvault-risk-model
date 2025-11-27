"""
Test CryptoQuant API access - updated endpoint structure.
"""

import requests
from src.config import get_api_key


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
            "response": response.json() if response.status_code in [200, 400, 401, 403] else response.text[:500]
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
    print("CryptoQuant API Access Test (v2)")
    print("=" * 60)
    
    try:
        api_key = get_api_key("cryptoquant")
        print(f"✓ API key loaded (ends with ...{api_key[-6:]})\n")
    except ValueError as e:
        print(f"✗ {e}")
        return
    
    # Test different API structures
    test_cases = [
        # Try v1 with different param formats
        ("/v1/btc/market-data/price-ohlcv", {"window": "DAY", "limit": "1"}),
        ("/v1/btc/market-data/price-ohlcv", {"interval": "day", "limit": "1"}),
        ("/v1/btc/market-data/price-ohlcv", {}),
        
        # Try exchange flows with different formats
        ("/v1/btc/exchange-flows/netflow", {"exchange": "binance", "window": "DAY"}),
        ("/v1/btc/exchange-flows/netflow", {}),
        ("/v1/btc/exchange-flows/reserve", {"exchange": "all_exchange"}),
        
        # Try newer API paths
        ("/v3/btc/exchange-flows/netflow", {"exchange": "all_exchange", "window": "DAY"}),
        ("/v2/btc/exchange-flows/netflow", {}),
        
        # Status/account endpoint
        ("/v1/status", {}),
        ("/v1/me", {}),
        ("/v1/user", {}),
        
        # Alternative data endpoints
        ("/v1/btc/network-indicator/sopr", {"window": "DAY", "limit": "1"}),
        ("/v1/btc/network-indicator/nupl", {"window": "DAY", "limit": "1"}),
        ("/v1/btc/market-indicator/estimated-leverage-ratio", {"exchange": "all_exchange", "limit": "1"}),
        ("/v1/btc/market-indicator/funding-rates", {"exchange": "all_exchange", "limit": "1"}),
        ("/v1/btc/market-indicator/open-interest", {"exchange": "all_exchange", "limit": "1"}),
        
        # Flow indicator
        ("/v1/btc/flow-indicator/exchange-whale-ratio", {"exchange": "all_exchange", "limit": "1"}),
        ("/v1/btc/flow-indicator/fund-flow-ratio", {"limit": "1"}),
        
        # Entity data
        ("/v1/btc/entity/exchange-reserve", {"exchange": "all_exchange", "limit": "1"}),
    ]
    
    print("Testing endpoints...\n")
    
    accessible = []
    denied = []
    errors = []
    
    for endpoint, params in test_cases:
        result = test_endpoint(api_key, endpoint, params)
        status = result["status_code"]
        
        if result["success"]:
            print(f"✓ {endpoint}")
            print(f"    Params: {params}")
            accessible.append((endpoint, params))
            # Show sample response
            resp = result["response"]
            if isinstance(resp, dict):
                print(f"    Response keys: {list(resp.keys())[:5]}")
                if "result" in resp:
                    print(f"    Result preview: {str(resp['result'])[:150]}...")
        elif status == 400:
            print(f"⚠ {endpoint} (400 Bad Request)")
            resp = result["response"]
            if isinstance(resp, dict) and "error" in resp:
                print(f"    Error: {resp.get('error', {}).get('message', resp)[:100]}")
            errors.append((endpoint, "bad_request", result["response"]))
        elif status in [401, 403]:
            print(f"✗ {endpoint} ({status} Unauthorized/Forbidden)")
            denied.append(endpoint)
        elif status == 404:
            print(f"- {endpoint} (404 Not Found)")
        else:
            print(f"? {endpoint} ({status})")
            print(f"    Response: {str(result['response'])[:100]}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accessible: {len(accessible)}")
    print(f"Bad Request (param issue): {len(errors)}")
    print(f"Denied: {len(denied)}")
    
    if accessible:
        print("\n✓ Working endpoints:")
        for ep, params in accessible:
            print(f"    {ep}")
            print(f"        params: {params}")
    
    if errors:
        print("\n⚠ Bad Request errors (might work with different params):")
        for ep, err_type, resp in errors[:3]:
            print(f"    {ep}")
            if isinstance(resp, dict):
                msg = resp.get("error", {}).get("message", "") or resp.get("message", "")
                if msg:
                    print(f"        {msg[:80]}")


if __name__ == "__main__":
    main()
