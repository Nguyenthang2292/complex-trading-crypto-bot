#!/usr/bin/env python3
"""
Test script for Token Metrics API implementation
"""

import sys
import os
from pathlib import Path

# Add the token_metrics directory to the path
sys.path.append(str(Path(__file__).parent / "token_metrics"))

def test_api_connectivity():
    """Test basic API connectivity"""
    print("🔍 Testing Token Metrics API Connectivity...")
    print("=" * 50)
    
    try:
        from token_metrics import create_token_metrics_client, test_api_connection
        
        # Test the main implementation
        print("1. Testing main implementation...")
        if test_api_connection():
            print("✅ Main implementation works!")
            return True
        else:
            print("❌ Main implementation failed")
            
    except Exception as e:
        print(f"❌ Error testing main implementation: {str(e)}")
    
    return False

def test_fallback_implementation():
    """Test fallback implementation"""
    print("\n2. Testing fallback implementation...")
    
    try:
        from token_metrics_fallback import create_token_metrics_fallback_client, test_fallback_api_connection
        
        if test_fallback_api_connection():
            print("✅ Fallback implementation works!")
            return True
        else:
            print("❌ Fallback implementation failed")
            
    except Exception as e:
        print(f"❌ Error testing fallback implementation: {str(e)}")
    
    return False

def test_direct_rest_api():
    """Test direct REST API call"""
    print("\n3. Testing direct REST API call...")
    
    try:
        import requests
        
        # Load API key
        from token_metrics import load_api_key_from_config
        api_key = load_api_key_from_config()
        
        # Make direct API call
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        
        # Test basic endpoint without parameters
        response = requests.get(
            "https://api.tokenmetrics.com/v2/tokens",
            headers=headers,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Direct REST API call successful!")
            data = response.json()
            print(f"Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
            return True
        else:
            print(f"❌ Direct REST API call failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error in direct REST API call: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Token Metrics API Test Suite")
    print("=" * 50)
    
    # Test 1: Main implementation
    success1 = test_api_connectivity()
    
    # Test 2: Fallback implementation
    success2 = test_fallback_implementation()
    
    # Test 3: Direct REST API
    success3 = test_direct_rest_api()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Main Implementation: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Fallback Implementation: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Direct REST API: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if any([success1, success2, success3]):
        print("\n🎉 At least one implementation is working!")
    else:
        print("\n💥 All implementations failed. Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 