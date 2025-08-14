#!/usr/bin/env python3
"""
Korean Markets Dashboard - Simple Trading Signals
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Korean Trading Signals",
    page_icon="🇰🇷",
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
def fetch_korean_data():
    """Fetch Korean market data"""
    try:
        korean_tickers = [
            '005930.KS',  # Samsung Electronics
            '000660.KS',  # SK Hynix
            '035420.KS',  # NAVER
            '051910.KS',  # LG Chem
            '006400.KS',  # Samsung SDI
            '035720.KS',  # Kakao
        ]
        
        data = yf.download(korean_tickers, period='5d', interval='1d', progress=False)
        if len(data.columns.levels) > 1:
            return data['Close']
        else:
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
    st.markdown("<h1>🇰🇷 Korean Markets Intelligence</h1>", unsafe_allow_html=True)
    
    korean_names = {
        '005930.KS': 'Samsung Electronics (삼성전자)',
        '000660.KS': 'SK Hynix (SK하이닉스)', 
        '035420.KS': 'NAVER (네이버)',
        '051910.KS': 'LG Chem (LG화학)',
        '006400.KS': 'Samsung SDI (삼성SDI)',
        '035720.KS': 'Kakao (카카오)',
    }
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col2:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')} KST")
    
    # Fetch data
    with st.spinner("Fetching Korean market data..."):
        data = fetch_korean_data()
    
    if data.empty:
        st.error("Unable to fetch Korean market data")
        st.write("Please check your internet connection or try again later.")
        return
    
    st.success(f"✅ Data loaded: {len(data.columns)} Korean stocks")
    
    # Display signals
    st.markdown("---")
    st.subheader("📊 Korean Stock Signals")
    
    cols = st.columns(2)
    
    for idx, (ticker, name) in enumerate(korean_names.items()):
        col = cols[idx % 2]
        
        with col:
            if ticker in data.columns:
                prices = data[ticker].dropna()
                
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
                    st.warning(f"No data available for {name}")
            else:
                st.warning(f"Unable to fetch {name}")
    
    # Market overview
    st.markdown("---")
    st.subheader("🏭 Korean Market Overview")
    
    if not data.empty:
        # Calculate market stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            winners = 0
            total = 0
            for ticker in data.columns:
                prices = data[ticker].dropna()
                if len(prices) >= 2:
                    total += 1
                    if prices.iloc[-1] > prices.iloc[-2]:
                        winners += 1
            
            if total > 0:
                win_rate = winners / total * 100
                st.metric("상승 종목", f"{winners}/{total}", f"{win_rate:.0f}%")
        
        with col2:
            # Average change
            changes = []
            for ticker in data.columns:
                prices = data[ticker].dropna()
                if len(prices) >= 2:
                    change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
                    changes.append(change)
            
            if changes:
                avg_change = sum(changes) / len(changes)
                st.metric("평균 변화율", f"{avg_change:+.2f}%", "상승세" if avg_change > 0 else "하락세")
        
        with col3:
            st.metric("분석 종목", f"{len(data.columns)}", "활성")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()