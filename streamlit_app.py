#!/usr/bin/env python3
"""
Streamlit app entry point for deployment
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Global Trading Signals",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .buy-signal {
        background-color: #00C853;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .sell-signal {
        background-color: #FF3D00;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .hold-signal {
        background-color: #FFC107;
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-high {
        color: #00C853;
        font-weight: bold;
        font-size: 20px;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
        font-size: 20px;
    }
    .confidence-low {
        color: #FF3D00;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_market_data(tickers):
    """Fetch market data"""
    try:
        data = yf.download(tickers, period='5d', interval='1d', progress=False)
        if len(data.columns.levels) > 1:
            return data['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def calculate_signal(prices):
    """Calculate simple signal"""
    if len(prices) < 5:
        return "HOLD", "LOW", "Insufficient data"
    
    current = prices.iloc[-1]
    prev = prices.iloc[-2]
    change = (current - prev) / prev * 100
    
    if change > 2:
        return "BUY", "HIGH", f"Strong gain: +{change:.1f}%"
    elif change < -2:
        return "SELL", "HIGH", f"Strong decline: {change:.1f}%"
    else:
        return "HOLD", "MEDIUM", f"Stable: {change:+.1f}%"

def main():
    st.markdown("<h1>🌍 Global Trading Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2 = st.tabs(["🇺🇸 US Markets", "🇰🇷 Korean Markets"])
    
    with tab1:
        st.subheader("🇺🇸 US Markets")
        
        us_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
        us_data = fetch_market_data(us_tickers)
        
        if not us_data.empty:
            cols = st.columns(3)
            for idx, ticker in enumerate(us_tickers):
                col = cols[idx % 3]
                with col:
                    if ticker in us_data.columns:
                        prices = us_data[ticker].dropna()
                        if len(prices) > 0:
                            current_price = prices.iloc[-1]
                            prev_price = prices.iloc[-2] if len(prices) > 1 else current_price
                            price_change = (current_price - prev_price) / prev_price * 100
                            
                            st.markdown(f"### {ticker}")
                            st.metric(
                                label="Price",
                                value=f"${current_price:.2f}",
                                delta=f"{price_change:+.2f}%"
                            )
                            
                            signal, confidence, reason = calculate_signal(prices)
                            
                            if signal == "BUY":
                                st.markdown(f'<div class="buy-signal">BUY</div>', unsafe_allow_html=True)
                            elif signal == "SELL":
                                st.markdown(f'<div class="sell-signal">SELL</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="hold-signal">HOLD</div>', unsafe_allow_html=True)
                            
                            conf_class = f"confidence-{confidence.lower()}"
                            st.markdown(f'<p class="{conf_class}">Confidence: {confidence}</p>', unsafe_allow_html=True)
                            st.caption(reason)
        else:
            st.error("Unable to fetch US market data")
    
    with tab2:
        st.subheader("🇰🇷 Korean Markets")
        
        korean_tickers = ['005930.KS', '000660.KS', '035420.KS', '051910.KS']
        korean_names = {
            '005930.KS': 'Samsung Electronics (삼성전자)',
            '000660.KS': 'SK Hynix (SK하이닉스)',
            '035420.KS': 'NAVER (네이버)',
            '051910.KS': 'LG Chem (LG화학)',
        }
        
        korean_data = fetch_market_data(korean_tickers)
        
        if not korean_data.empty:
            cols = st.columns(2)
            for idx, (ticker, name) in enumerate(korean_names.items()):
                col = cols[idx % 2]
                with col:
                    if ticker in korean_data.columns:
                        prices = korean_data[ticker].dropna()
                        if len(prices) > 0:
                            current_price = prices.iloc[-1]
                            prev_price = prices.iloc[-2] if len(prices) > 1 else current_price
                            price_change = (current_price - prev_price) / prev_price * 100
                            
                            st.markdown(f"### {name}")
                            st.metric(
                                label="Price (KRW)",
                                value=f"₩{current_price:,.0f}",
                                delta=f"{price_change:+.2f}%"
                            )
                            
                            signal, confidence, reason = calculate_signal(prices)
                            
                            if signal == "BUY":
                                st.markdown(f'<div class="buy-signal">매수 (BUY)</div>', unsafe_allow_html=True)
                            elif signal == "SELL":
                                st.markdown(f'<div class="sell-signal">매도 (SELL)</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="hold-signal">보유 (HOLD)</div>', unsafe_allow_html=True)
                            
                            conf_class = f"confidence-{confidence.lower()}"
                            st.markdown(f'<p class="{conf_class}">신뢰도: {confidence}</p>', unsafe_allow_html=True)
                            st.caption(reason)
        else:
            st.error("Unable to fetch Korean market data")

if __name__ == "__main__":
    main()