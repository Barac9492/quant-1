#!/usr/bin/env python3
"""
Simple test with basic stock symbols to verify Yahoo Finance connectivity
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_basic_symbols():
    """Test basic stock symbols that should always work"""
    print("🔍 Testing Yahoo Finance API connectivity...")
    
    basic_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    for symbol in basic_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')  # Just 5 days
            
            if not hist.empty:
                latest_price = hist['Close'].iloc[-1]
                print(f"✅ {symbol}: ${latest_price:.2f} ({len(hist)} days)")
            else:
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Test with a simple correlation
    try:
        print(f"\n📊 Testing correlation calculation...")
        aapl = yf.download('AAPL', period='1mo', progress=False)['Close']
        spy = yf.download('SPY', period='1mo', progress=False)['Close'] 
        
        if not aapl.empty and not spy.empty:
            correlation = aapl.corr(spy)
            print(f"✅ AAPL-SPY correlation: {correlation:.3f}")
            return True
        else:
            print(f"❌ Could not calculate correlation")
            return False
            
    except Exception as e:
        print(f"❌ Correlation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_symbols()
    
    if success:
        print(f"\n✅ Yahoo Finance API is working!")
        print(f"The issue might be with specific symbol names in the config.")
    else:
        print(f"\n❌ Yahoo Finance API connectivity issues.")
        print(f"Please check your internet connection.")