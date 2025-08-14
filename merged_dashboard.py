#!/usr/bin/env python3
"""
Global Trading Signals Dashboard - US & Korean Markets
Executive-level interface with brilliant cross-market intelligence
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import sys
import os

# Try to import brilliant connections, fallback if not available
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from strategies.brilliant_connections import BrilliantConnectionEngine
    BRILLIANT_AVAILABLE = True
except:
    BRILLIANT_AVAILABLE = False

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
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066CC;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
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
    .brilliance-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
    }
    .genius-insight {
        background-color: #1a1a2e;
        color: #ffd700;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffd700;
        margin: 10px 0;
    }
    .alpha-score {
        font-size: 24px;
        font-weight: bold;
        color: #00C853;
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
    .nav-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .nav-button {
        background-color: #f0f0f0;
        border: 2px solid #ddd;
        padding: 15px 30px;
        margin: 0 10px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: bold;
        font-size: 18px;
    }
    .nav-button-active {
        background-color: #0066CC;
        color: white;
        border-color: #0066CC;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'US Markets'

# Data fetching functions
@st.cache_data(ttl=60)
def fetch_us_data():
    """Fetch US market data"""
    try:
        us_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
        data = yf.download(us_tickers, period='5d', interval='5m', progress=False, auto_adjust=True)
        if len(data.columns.levels) > 1:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

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
            '207940.KS',  # Samsung Biologics
            '068270.KS',  # Celltrion
        ]
        
        data = yf.download(korean_tickers, period='5d', interval='1d', progress=False, auto_adjust=True)
        if len(data.columns.levels) > 1:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_brilliant_data():
    """Fetch extended data for brilliant connections analysis"""
    if not BRILLIANT_AVAILABLE:
        return pd.DataFrame()
    
    try:
        brilliant_tickers = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA',
            '^VIX', '^TNX', 'TLT', 'GLD', 'USO', 
            'XLF', 'XLI', 'XLY', 'HYG', 'LQD'
        ]
        data = yf.download(brilliant_tickers, period='6mo', progress=False, auto_adjust=True)
        if len(data.columns.levels) > 1:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

def get_brilliant_insights():
    """Get brilliant cross-market insights"""
    if not BRILLIANT_AVAILABLE:
        return None
    
    try:
        data = fetch_brilliant_data()
        if data.empty:
            return None
        
        engine = BrilliantConnectionEngine(min_lead_time=3)
        results = engine.discover_brilliant_connections(data)
        return results
    except:
        return None

def calculate_signal(data, ticker):
    """Calculate simple signal"""
    if ticker not in data.columns or len(data[ticker].dropna()) < 20:
        return "HOLD", "LOW", "Insufficient data"
    
    prices = data[ticker].dropna()
    current_price = prices.iloc[-1]
    sma_20 = prices.tail(20).mean()
    sma_5 = prices.tail(5).mean()
    momentum = (current_price - prices.iloc[-5]) / prices.iloc[-5] * 100
    
    if sma_5 > sma_20 * 1.02 and momentum > 2:
        confidence = "HIGH" if momentum > 3 else "MEDIUM"
        return "BUY", confidence, f"Strong upward momentum: +{momentum:.1f}%"
    elif sma_5 < sma_20 * 0.98 and momentum < -2:
        confidence = "HIGH" if momentum < -3 else "MEDIUM"
        return "SELL", confidence, f"Downward pressure: {momentum:.1f}%"
    else:
        return "HOLD", "MEDIUM", "No clear direction"

def calculate_korean_signal(prices):
    """Calculate signal for Korean stocks (simpler logic for daily data)"""
    if len(prices) < 5:
        return "HOLD", "LOW", "Insufficient data"
    
    current = prices.iloc[-1]
    prev = prices.iloc[-2]
    change = (current - prev) / prev * 100
    
    # Calculate short-term trend
    recent_avg = prices.tail(3).mean()
    older_avg = prices.tail(5).head(2).mean()
    trend = (recent_avg - older_avg) / older_avg * 100
    
    if change > 3 or trend > 2:
        confidence = "HIGH" if change > 5 else "MEDIUM"
        return "BUY", confidence, f"Strong momentum: +{change:.1f}%"
    elif change < -3 or trend < -2:
        confidence = "HIGH" if change < -5 else "MEDIUM"
        return "SELL", confidence, f"Downward pressure: {change:.1f}%"
    else:
        return "HOLD", "MEDIUM", f"Stable: {change:+.1f}%"

def render_navigation():
    """Render navigation buttons"""
    st.markdown("""
    <div class="nav-container">
        <div style="display: flex; gap: 20px;">
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🇺🇸 US Markets", key="nav_us", use_container_width=True):
            st.session_state.current_page = 'US Markets'
            st.rerun()
    
    with col2:
        if st.button("🇰🇷 Korean Markets", key="nav_korean", use_container_width=True):
            st.session_state.current_page = 'Korean Markets'
            st.rerun()
    
    with col3:
        if st.button("🌍 Global Overview", key="nav_global", use_container_width=True):
            st.session_state.current_page = 'Global Overview'
            st.rerun()

def render_us_markets():
    """Render US Markets page"""
    st.markdown("<h1>💎 US Trading Signals Dashboard</h1>", unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="us_refresh")
    with col2:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')} EST")
    
    # Fetch data
    with st.spinner("Fetching US market data..."):
        data = fetch_us_data()
    
    if data.empty:
        st.error("Unable to fetch US market data")
        return
    
    st.success(f"✅ Data loaded: {len(data.columns)} US stocks")
    
    # Main signals
    st.markdown("---")
    st.subheader("📊 Current US Signals")
    
    us_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
    cols = st.columns(3)
    
    for idx, ticker in enumerate(us_tickers):
        col = cols[idx % 3]
        with col:
            if ticker in data.columns:
                signal, confidence, reason = calculate_signal(data, ticker)
                
                current_price = data[ticker].iloc[-1]
                prev_price = data[ticker].iloc[-10] if len(data[ticker]) > 10 else data[ticker].iloc[0]
                price_change = (current_price - prev_price) / prev_price * 100
                
                st.markdown(f"### {ticker}")
                st.metric(
                    label="Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:+.2f}%"
                )
                
                if signal == "BUY":
                    st.markdown(f'<div class="buy-signal">BUY</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f'<div class="sell-signal">SELL</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hold-signal">HOLD</div>', unsafe_allow_html=True)
                
                conf_class = f"confidence-{confidence.lower()}"
                st.markdown(f'<p class="{conf_class}">Confidence: {confidence}</p>', unsafe_allow_html=True)
                st.caption(reason)
    
    # Brilliant insights
    if BRILLIANT_AVAILABLE:
        st.markdown("---")
        st.subheader("🧠 Brilliant Cross-Market Insights")
        
        with st.spinner("Analyzing brilliant connections..."):
            brilliant_results = get_brilliant_insights()
        
        if brilliant_results and brilliant_results.get('brilliance_score', 0) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="brilliance-box">
                    <h3>🌟 Market Intelligence Active</h3>
                    <p class="alpha-score">Brilliance Score: {brilliant_results['brilliance_score']:.2f}/1.0</p>
                    <p>Analyzing {len(brilliant_results.get('brilliant_connections', []))} sophisticated cross-market relationships</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                genius_trades = brilliant_results.get('genius_trades', [])
                if genius_trades:
                    st.metric("Genius Trades", len(genius_trades), "Active")
            
            # Show top brilliant connection
            connections = brilliant_results.get('brilliant_connections', [])
            if connections:
                top_connection = connections[0]
                
                st.markdown(f"""
                <div class="genius-insight">
                    <h4>💎 {top_connection['chain_name'].replace('_', ' ').title()}</h4>
                    <p><strong>Alpha Mechanism:</strong> {top_connection['alpha_mechanism']}</p>
                    <p><strong>Lead Time:</strong> {top_connection['optimal_lead_time']} days</p>
                    <p><strong>Alpha Potential:</strong> {top_connection['alpha_potential']:.1%}</p>
                    <p><strong>Current Signal:</strong> {top_connection['current_signal']['direction']} 
                    ({top_connection['current_signal']['confidence']})</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔗 View Transmission Path"):
                    for i, step in enumerate(top_connection['connections'], 1):
                        st.write(f"{i}. {step}")
        else:
            st.info("🔍 Scanning for brilliant opportunities...")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def render_korean_markets():
    """Render Korean Markets page"""
    st.markdown("<h1>🇰🇷 Korean Markets Intelligence</h1>", unsafe_allow_html=True)
    
    korean_names = {
        '005930.KS': 'Samsung Electronics (삼성전자)',
        '000660.KS': 'SK Hynix (SK하이닉스)', 
        '035420.KS': 'NAVER (네이버)',
        '051910.KS': 'LG Chem (LG화학)',
        '006400.KS': 'Samsung SDI (삼성SDI)',
        '035720.KS': 'Kakao (카카오)',
        '207940.KS': 'Samsung Biologics (삼성바이오로직스)',
        '068270.KS': 'Celltrion (셀트리온)',
    }
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="korean_refresh")
    with col2:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')} KST")
    
    # Fetch data
    with st.spinner("Fetching Korean market data..."):
        data = fetch_korean_data()
    
    if data.empty:
        st.error("Unable to fetch Korean market data")
        return
    
    st.success(f"✅ Data loaded: {len(data.columns)} Korean stocks")
    
    # Korean signals
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
                    
                    signal, confidence, reason = calculate_korean_signal(prices)
                    
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
                st.warning(f"Unable to fetch {name}")
    
    # Korean sector analysis
    st.markdown("---")
    st.subheader("🏭 Korean Sector Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tech_tickers = ['005930.KS', '000660.KS', '035420.KS']
        available_tech = [t for t in tech_tickers if t in data.columns]
        if available_tech:
            tech_changes = []
            for ticker in available_tech:
                prices = data[ticker].dropna()
                if len(prices) >= 2:
                    change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
                    tech_changes.append(change)
            
            if tech_changes:
                tech_avg = sum(tech_changes) / len(tech_changes)
                st.metric("기술주 (Tech)", f"{tech_avg:+.1f}%", "강세" if tech_avg > 1 else "약세")
    
    with col2:
        chem_tickers = ['051910.KS', '006400.KS']
        available_chem = [t for t in chem_tickers if t in data.columns]
        if available_chem:
            chem_changes = []
            for ticker in available_chem:
                prices = data[ticker].dropna()
                if len(prices) >= 2:
                    change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
                    chem_changes.append(change)
            
            if chem_changes:
                chem_avg = sum(chem_changes) / len(chem_changes)
                st.metric("배터리/화학 (Battery)", f"{chem_avg:+.1f}%", "상승" if chem_avg > 0 else "하락")
    
    with col3:
        bio_tickers = ['207940.KS', '068270.KS']
        available_bio = [t for t in bio_tickers if t in data.columns]
        if available_bio:
            bio_changes = []
            for ticker in available_bio:
                prices = data[ticker].dropna()
                if len(prices) >= 2:
                    change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
                    bio_changes.append(change)
            
            if bio_changes:
                bio_avg = sum(bio_changes) / len(bio_changes)
                st.metric("바이오 (Bio)", f"{bio_avg:+.1f}%", "성장" if bio_avg > 0 else "조정")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def render_global_overview():
    """Render Global Overview page"""
    st.markdown("<h1>🌍 Global Markets Overview</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🇺🇸 US Market Status")
        us_data = fetch_us_data()
        if not us_data.empty:
            # Calculate US market health
            us_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
            us_winners = 0
            us_total = 0
            
            for ticker in us_tickers:
                if ticker in us_data.columns:
                    prices = us_data[ticker].dropna()
                    if len(prices) >= 10:
                        us_total += 1
                        if prices.iloc[-1] > prices.iloc[-10]:
                            us_winners += 1
            
            if us_total > 0:
                us_win_rate = us_winners / us_total * 100
                st.metric("Winners", f"{us_winners}/{us_total}", f"{us_win_rate:.0f}%")
                
                # SPY performance
                if 'SPY' in us_data.columns:
                    spy_prices = us_data['SPY'].dropna()
                    if len(spy_prices) >= 10:
                        spy_change = (spy_prices.iloc[-1] - spy_prices.iloc[-10]) / spy_prices.iloc[-10] * 100
                        st.metric("S&P 500", f"{spy_change:+.2f}%", "Bullish" if spy_change > 0 else "Bearish")
        else:
            st.warning("US market data unavailable")
    
    with col2:
        st.subheader("🇰🇷 Korean Market Status")
        korean_data = fetch_korean_data()
        if not korean_data.empty:
            # Calculate Korean market health
            korean_winners = 0
            korean_total = 0
            
            for ticker in korean_data.columns:
                prices = korean_data[ticker].dropna()
                if len(prices) >= 2:
                    korean_total += 1
                    if prices.iloc[-1] > prices.iloc[-2]:
                        korean_winners += 1
            
            if korean_total > 0:
                korean_win_rate = korean_winners / korean_total * 100
                st.metric("Winners", f"{korean_winners}/{korean_total}", f"{korean_win_rate:.0f}%")
                
                # Samsung performance (market leader)
                if '005930.KS' in korean_data.columns:
                    samsung_prices = korean_data['005930.KS'].dropna()
                    if len(samsung_prices) >= 2:
                        samsung_change = (samsung_prices.iloc[-1] - samsung_prices.iloc[-2]) / samsung_prices.iloc[-2] * 100
                        st.metric("Samsung (KOSPI Leader)", f"{samsung_change:+.2f}%", "상승" if samsung_change > 0 else "하락")
        else:
            st.warning("Korean market data unavailable")
    
    # Global insights
    st.markdown("---")
    st.subheader("🔗 Cross-Market Insights")
    
    if BRILLIANT_AVAILABLE:
        brilliant_results = get_brilliant_insights()
        if brilliant_results:
            st.success("🧠 Brilliant cross-market analysis active")
            st.info(f"Monitoring {len(brilliant_results.get('brilliant_connections', []))} sophisticated relationships")
        else:
            st.info("🔍 Scanning for cross-market opportunities...")
    else:
        st.info("📊 Basic technical analysis active")
    
    # Market timing
    st.markdown("---")
    st.subheader("⏰ Market Timing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ny_time = datetime.now()  # Adjust for actual timezone if needed
        st.metric("New York", ny_time.strftime('%H:%M:%S'), "EST")
    
    with col2:
        # Korean time (typically +14 hours from EST)
        korean_time = datetime.now() + timedelta(hours=14)
        st.metric("Seoul", korean_time.strftime('%H:%M:%S'), "KST")
    
    with col3:
        if ny_time.hour >= 9 and ny_time.hour <= 16:
            market_status = "🟢 OPEN"
        else:
            market_status = "🔴 CLOSED"
        st.metric("US Markets", market_status, "Live Trading")

def main():
    """Main application"""
    # Title and navigation
    st.markdown("<h1 style='text-align: center;'>🌍 Global Trading Intelligence</h1>", unsafe_allow_html=True)
    
    # Navigation
    render_navigation()
    
    # Current page indicator
    st.markdown(f"### Current Page: {st.session_state.current_page}")
    
    # Render selected page
    if st.session_state.current_page == 'US Markets':
        render_us_markets()
    elif st.session_state.current_page == 'Korean Markets':
        render_korean_markets()
    elif st.session_state.current_page == 'Global Overview':
        render_global_overview()

if __name__ == "__main__":
    main()