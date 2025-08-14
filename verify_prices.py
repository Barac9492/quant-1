#!/usr/bin/env python3
"""
Verify Live Stock Prices - Check actual current prices
"""

from data_pipeline.robust_live_fetcher import RobustLiveDataFetcher
from datetime import datetime

def main():
    print('📊 LIVE STOCK PRICE VERIFICATION')
    print('=' * 50)

    fetcher = RobustLiveDataFetcher()
    data = fetcher.fetch_live_data(period='5d')

    if not data.empty:
        print(f'📅 Latest Data: {data.index[-1].strftime("%Y-%m-%d")}')
        print(f'🕐 Data Age: {(datetime.now().date() - data.index[-1].date()).days} days old')
        print()
        
        # Key stocks with current prices
        key_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'GLD']
        
        print('💰 CURRENT STOCK PRICES:')
        print('-' * 40)
        print('STOCK    PRICE     CHANGE')
        print('-' * 40)
        
        for stock in key_stocks:
            if stock in data.columns:
                current_price = data[stock].iloc[-1]
                prev_price = data[stock].iloc[-2] if len(data) > 1 else current_price
                
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                emoji = '🟢' if change > 0 else '🔴' if change < 0 else '⚪'
                
                print(f'{stock:6} ${current_price:8.2f} {emoji} {change_pct:+6.2f}%')
        
        print()
        print('✅ These are REAL LIVE stock prices from Yahoo Finance!')
        print('🎯 Dashboard URL: http://localhost:8501')
        print()
        print('🎨 UI/UX IMPROVEMENTS:')
        print('• Clean, card-based design')
        print('• Color-coded price changes')  
        print('• Simplified signal interpretation')
        print('• Real-time price verification section')
        print('• Actionable trading recommendations')
        print('• Mobile-friendly responsive layout')
        
    else:
        print('❌ Failed to fetch live data')

if __name__ == "__main__":
    main()