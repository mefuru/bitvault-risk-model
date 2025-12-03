"""
CryptoQuant API Discovery

Uses the discovery endpoint to find all available API endpoints.
Run with: PYTHONPATH=. python scripts/discover_cryptoquant.py
"""

import os
import sys
import requests
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_api_key


def main():
    print("=" * 70)
    print("CryptoQuant API Discovery")
    print("=" * 70)
    
    # Get API key
    try:
        api_key = get_api_key("cryptoquant")
        print(f"\n✓ API Key loaded: {api_key[:10]}...{api_key[-6:]}")
    except ValueError as e:
        print(f"\n✗ API Key error: {e}")
        sys.exit(1)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Step 1: Try the discovery endpoint
    print("\n" + "-" * 70)
    print("Step 1: Checking discovery endpoint...")
    print("-" * 70)
    
    discovery_url = "https://api.cryptoquant.com/v1/discovery/endpoints"
    
    try:
        response = requests.get(discovery_url, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Save full response for analysis
            with open("cryptoquant_endpoints.json", "w") as f:
                json.dump(data, f, indent=2)
            print("✓ Full endpoint list saved to cryptoquant_endpoints.json")
            
            # Parse and display
            if "result" in data:
                endpoints = data["result"]
                if isinstance(endpoints, list):
                    print(f"\nFound {len(endpoints)} endpoints:")
                    
                    # Group by category
                    categories = {}
                    for ep in endpoints:
                        if isinstance(ep, dict):
                            cat = ep.get("category", "unknown")
                            if cat not in categories:
                                categories[cat] = []
                            categories[cat].append(ep)
                        elif isinstance(ep, str):
                            # Sometimes it's just a list of strings
                            print(f"  • {ep}")
                    
                    # Print by category
                    for cat, eps in sorted(categories.items()):
                        print(f"\n📁 {cat.upper()} ({len(eps)} endpoints)")
                        for ep in eps[:10]:  # Show first 10 per category
                            name = ep.get("name", ep.get("endpoint", "?"))
                            endpoint = ep.get("endpoint", "?")
                            print(f"   • {name}: {endpoint}")
                        if len(eps) > 10:
                            print(f"   ... and {len(eps) - 10} more")
                
                elif isinstance(endpoints, dict):
                    print(f"\nEndpoint categories: {list(endpoints.keys())}")
                    for cat, items in endpoints.items():
                        if isinstance(items, list):
                            print(f"\n📁 {cat} ({len(items)} items)")
                            for item in items[:5]:
                                print(f"   • {item}")
        else:
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Step 2: Try alternate endpoint formats
    print("\n" + "-" * 70)
    print("Step 2: Testing alternate endpoint formats...")
    print("-" * 70)
    
    # CryptoQuant may have changed their API structure
    alternate_urls = [
        # Try with different base paths
        ("BTC Exchange Flow (v1)", "https://api.cryptoquant.com/v1/btc/exchange-flows/netflow"),
        ("BTC Exchange Flow (alt)", "https://api.cryptoquant.com/v1/btc/exchange_flows/netflow"),
        ("BTC Flow (category)", "https://api.cryptoquant.com/v1/btc/flow/exchange-netflow"),
        
        # Try market data variations
        ("Market Data (v1)", "https://api.cryptoquant.com/v1/btc/market-data/price-ohlcv"),
        ("Market Data (alt)", "https://api.cryptoquant.com/v1/btc/market_data/ohlcv"),
        
        # Try status/health endpoints
        ("API Status", "https://api.cryptoquant.com/v1/status"),
        ("API Health", "https://api.cryptoquant.com/health"),
        ("API Info", "https://api.cryptoquant.com/v1/info"),
        
        # Try user/account info
        ("User Info", "https://api.cryptoquant.com/v1/user"),
        ("Account Info", "https://api.cryptoquant.com/v1/account"),
        ("Subscription", "https://api.cryptoquant.com/v1/subscription"),
    ]
    
    for name, url in alternate_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            status = "✓" if response.status_code == 200 else f"✗ {response.status_code}"
            print(f"  {status:12} {name}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Show first 100 chars of response
                    preview = json.dumps(data)[:100]
                    print(f"              Response: {preview}...")
                except:
                    print(f"              Response: {response.text[:100]}...")
                    
        except Exception as e:
            print(f"  ✗ Error    {name}: {e}")
    
    # Step 3: Check API documentation endpoint
    print("\n" + "-" * 70)
    print("Step 3: Looking for API documentation...")
    print("-" * 70)
    
    doc_urls = [
        "https://api.cryptoquant.com/v1/docs",
        "https://api.cryptoquant.com/docs",
        "https://api.cryptoquant.com/swagger",
        "https://api.cryptoquant.com/openapi.json",
    ]
    
    for url in doc_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"  ✓ Found: {url}")
                print(f"    Content-Type: {response.headers.get('content-type', 'unknown')}")
        except:
            pass
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
If the discovery endpoint worked, check cryptoquant_endpoints.json for the full list.

If you're still getting 403 errors, please:

1. Log into your CryptoQuant dashboard
2. Go to Account Settings → API
3. Check if API access is enabled
4. Look for any IP whitelist settings
5. Verify your subscription tier includes API access

You may also need to:
- Generate a new API key specifically for API access
- Enable API access in your subscription settings
- Contact CryptoQuant support if you have Professional but can't access API
""")


if __name__ == "__main__":
    main()
