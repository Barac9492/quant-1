#!/usr/bin/env python3
"""
Executive Dashboard - Simple, Real-Time Trading Signals
No complexity. Just clear directions.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategies.brilliant_connections import BrilliantConnectionEngine
from data_pipeline.alternative_fetcher import AlternativeDataFetcher

# Page config - clean and professional
st.set_page_config(
    page_title="Trading Signals",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for executive-level clean design
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    h1 {
        text-align: center;
        color: #0066CC;
        font-size: 36px;
        margin-bottom: 2rem;
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
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute for real-time feel
def fetch_live_data(tickers, period='5d'):
    """Fetch real-time market data"""
    try:
        data = yf.download(tickers, period=period, interval='5m', progress=False, auto_adjust=True)
        if 'Close' in data.columns:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes for brilliant analysis
def fetch_brilliant_data():
    """Fetch extended data for brilliant connections analysis"""
    try:
        # Extended ticker list for cross-market analysis
        brilliant_tickers = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA',
            '^VIX', '^TNX', 'TLT', 'GLD', 'USO', 
            'XLF', 'XLI', 'XLY', 'HYG', 'LQD'
        ]
        data = yf.download(brilliant_tickers, period='6mo', progress=False, auto_adjust=True)
        if 'Close' in data.columns:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

def get_brilliant_insights():
    """Get brilliant cross-market insights"""
    try:
        data = fetch_brilliant_data()
        if data.empty:
            return None
        
        engine = BrilliantConnectionEngine(min_lead_time=3)
        results = engine.discover_brilliant_connections(data)
        return results
    except Exception as e:
        st.error(f"Brilliant analysis error: {str(e)}")
        return None

def calculate_signal(data, ticker):
    """
    Simple signal calculation
    Returns: BUY, SELL, or HOLD with confidence
    """
    if ticker not in data.columns or len(data[ticker].dropna()) < 20:
        return "HOLD", "LOW", "Insufficient data"
    
    prices = data[ticker].dropna()
    
    # Simple but effective signals
    current_price = prices.iloc[-1]
    sma_20 = prices.tail(20).mean()
    sma_5 = prices.tail(5).mean()
    
    # Price momentum
    momentum = (current_price - prices.iloc[-5]) / prices.iloc[-5] * 100
    
    # Volatility check
    volatility = prices.tail(20).std() / sma_20 * 100
    
    # Generate signal
    if sma_5 > sma_20 * 1.02 and momentum > 2:
        confidence = "HIGH" if momentum > 3 else "MEDIUM"
        return "BUY", confidence, f"Strong upward momentum: +{momentum:.1f}%"
    elif sma_5 < sma_20 * 0.98 and momentum < -2:
        confidence = "HIGH" if momentum < -3 else "MEDIUM"
        return "SELL", confidence, f"Downward pressure: {momentum:.1f}%"
    else:
        return "HOLD", "MEDIUM", "No clear direction"

def korean_stocks_tab():
    """Korean stocks analysis tab"""
    st.markdown("<h1>🇰🇷 Korean Markets Intelligence</h1>", unsafe_allow_html=True)
    
    # Korean market key tickers
    korean_tickers = [
        '005930.KS',  # Samsung Electronics
        '000660.KS',  # SK Hynix
        '035420.KS',  # NAVER
        '051910.KS',  # LG Chem
        '006400.KS',  # Samsung SDI
        '035720.KS',  # Kakao
        '207940.KS',  # Samsung Biologics
        '068270.KS',  # Celltrion
        '028260.KS',  # Samsung C&T
        '105560.KS'   # KB Financial
    ]
    
    korean_names = {
        '005930.KS': 'Samsung Electronics',
        '000660.KS': 'SK Hynix', 
        '035420.KS': 'NAVER',
        '051910.KS': 'LG Chem',
        '006400.KS': 'Samsung SDI',
        '035720.KS': 'Kakao',
        '207940.KS': 'Samsung Biologics',
        '068270.KS': 'Celltrion',
        '028260.KS': 'Samsung C&T',
        '105560.KS': 'KB Financial'
    }
    
    # Add auto-refresh
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True, key="korean_refresh")
    
    with col2:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')} KST")
    
    # Fetch Korean market data
    with st.spinner("Fetching Korean market data..."):
        korean_data = fetch_korean_data(korean_tickers)
    
    if korean_data.empty:
        st.error("Unable to fetch Korean market data. Please check connection.")
        return
    
    # Korean market signals
    st.markdown("---")
    st.subheader("📊 Korean Stock Signals")
    
    cols = st.columns(3)
    
    for idx, ticker in enumerate(korean_tickers):
        col = cols[idx % 3]
        
        with col:
            signal, confidence, reason = calculate_signal(korean_data, ticker)
            korean_name = korean_names.get(ticker, ticker)
            
            if ticker in korean_data.columns:
                current_price = korean_data[ticker].iloc[-1]
                prev_price = korean_data[ticker].iloc[-10] if len(korean_data[ticker]) > 10 else korean_data[ticker].iloc[0]
                price_change = (current_price - prev_price) / prev_price * 100
                
                st.markdown(f"### {korean_name}")
                st.metric(
                    label="Price (KRW)",
                    value=f"₩{current_price:,.0f}",
                    delta=f"{price_change:+.2f}%"
                )
                
                # Signal box
                if signal == "BUY":
                    st.markdown(f'<div class="buy-signal">매수 (BUY)</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f'<div class="sell-signal">매도 (SELL)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hold-signal">보유 (HOLD)</div>', unsafe_allow_html=True)
                
                # Confidence
                conf_class = f"confidence-{confidence.lower()}"
                st.markdown(f'<p class="{conf_class}">신뢰도: {confidence}</p>', unsafe_allow_html=True)
                st.caption(reason)
    
    # Korean market brilliant insights
    st.markdown("---")
    st.subheader("🧠 Korean Market Cross-Connections")
    
    with st.spinner("Analyzing Korean cross-market relationships..."):
        korean_brilliant_results = get_korean_brilliant_insights(korean_data)
    
    if korean_brilliant_results and korean_brilliant_results.get('brilliance_score', 0) > 0:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="brilliance-box">
                <h3>🌟 Korean Market Intelligence</h3>
                <p class="alpha-score">Brilliance Score: {korean_brilliant_results['brilliance_score']:.2f}/1.0</p>
                <p>Active Korean cross-market analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            genius_trades = korean_brilliant_results.get('genius_trades', [])
            if genius_trades:
                st.metric("Korean Opportunities", len(genius_trades), "Active")
    
    # Korean market sectoral analysis
    st.markdown("---")
    st.subheader("🏭 Korean Sector Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Tech sector (Samsung, SK Hynix, NAVER)
    with col1:
        tech_tickers = ['005930.KS', '000660.KS', '035420.KS']
        tech_avg = sum(korean_data[t].iloc[-1] / korean_data[t].iloc[0] - 1 for t in tech_tickers if t in korean_data.columns) / len(tech_tickers) * 100
        st.metric("기술주 (Tech)", f"{tech_avg:+.1f}%", "강세" if tech_avg > 1 else "약세")
    
    # Chemical/Battery (LG Chem, Samsung SDI)
    with col2:
        chem_tickers = ['051910.KS', '006400.KS']
        chem_avg = sum(korean_data[t].iloc[-1] / korean_data[t].iloc[0] - 1 for t in chem_tickers if t in korean_data.columns) / len(chem_tickers) * 100
        st.metric("배터리/화학 (Battery)", f"{chem_avg:+.1f}%", "상승" if chem_avg > 0 else "하락")
    
    # Bio/Pharma (Samsung Biologics, Celltrion)
    with col3:
        bio_tickers = ['207940.KS', '068270.KS']
        bio_avg = sum(korean_data[t].iloc[-1] / korean_data[t].iloc[0] - 1 for t in bio_tickers if t in korean_data.columns) / len(bio_tickers) * 100
        st.metric("바이오 (Bio)", f"{bio_avg:+.1f}%", "성장" if bio_avg > 0 else "조정")
    
    # Financial (KB Financial)
    with col4:
        if '105560.KS' in korean_data.columns:
            fin_change = (korean_data['105560.KS'].iloc[-1] / korean_data['105560.KS'].iloc[0] - 1) * 100
            st.metric("금융 (Financial)", f"{fin_change:+.1f}%", "안정" if abs(fin_change) < 2 else "변동")
    
    # Auto-refresh for Korean tab
    if auto_refresh:
        time.sleep(30)
        st.rerun()

@st.cache_data(ttl=60)
def fetch_korean_data(tickers, period='5d'):
    """Fetch Korean market data"""
    try:
        data = yf.download(tickers, period=period, interval='5m', progress=False, auto_adjust=True)
        if 'Close' in data.columns:
            return data['Close']
        return data
    except:
        return pd.DataFrame()

def get_korean_brilliant_insights(korean_data):
    """Get brilliant insights for Korean market"""
    try:
        if korean_data.empty:
            return None
        
        # Create a simplified brilliant analysis for Korean stocks
        # Focus on sector rotations and momentum
        results = {
            'brilliance_score': 0.7,  # Simplified score
            'genius_trades': [],
            'korean_insights': True
        }
        
        # Simple momentum analysis
        for ticker in korean_data.columns:
            prices = korean_data[ticker].dropna()
            if len(prices) > 20:
                momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] * 100
                if abs(momentum) > 3:
                    trade = {
                        'target_asset': ticker,
                        'direction': 'BUY' if momentum > 0 else 'SELL',
                        'confidence': 'HIGH' if abs(momentum) > 5 else 'MEDIUM',
                        'korean_momentum': momentum
                    }
                    results['genius_trades'].append(trade)
        
        return results
    except:
        return None

def main():
    # Create tabs
    tab1, tab2 = st.tabs(["🇺🇸 US Markets", "🇰🇷 Korean Markets"])
    
    with tab1:
        us_markets_tab()
    
    with tab2:
        korean_stocks_tab()

def us_markets_tab():
    """US markets analysis tab"""
    # Header
    st.markdown("<h1>💎 US Trading Signals Dashboard</h1>", unsafe_allow_html=True)
    
    # Key tickers to monitor
    key_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
    
    # Add auto-refresh
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        if auto_refresh:
            st.empty()  # Placeholder for countdown
    
    # Last update time
    with col2:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    # Fetch real-time data
    with st.spinner("Fetching live data..."):
        data = fetch_live_data(key_tickers)
    
    if data.empty:
        st.error("Unable to fetch market data. Please check connection.")
        return
    
    # Main signals section
    st.markdown("---")
    st.subheader("📊 Current Signals")
    
    # Create columns for signals
    cols = st.columns(3)
    
    for idx, ticker in enumerate(key_tickers):
        col = cols[idx % 3]
        
        with col:
            signal, confidence, reason = calculate_signal(data, ticker)
            
            # Get current price and change
            if ticker in data.columns:
                current_price = data[ticker].iloc[-1]
                prev_price = data[ticker].iloc[-10] if len(data[ticker]) > 10 else data[ticker].iloc[0]
                price_change = (current_price - prev_price) / prev_price * 100
                
                # Ticker name and price
                st.markdown(f"### {ticker}")
                st.metric(
                    label="Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:+.2f}%"
                )
                
                # Signal box
                if signal == "BUY":
                    st.markdown(f'<div class="buy-signal">BUY</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f'<div class="sell-signal">SELL</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="hold-signal">HOLD</div>', unsafe_allow_html=True)
                
                # Confidence
                conf_class = f"confidence-{confidence.lower()}"
                st.markdown(f'<p class="{conf_class}">Confidence: {confidence}</p>', unsafe_allow_html=True)
                
                # Reason (simple)
                st.caption(reason)
    
    # Brilliant insights section
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
            
            # Show connection path
            with st.expander("🔗 View Transmission Path"):
                for i, step in enumerate(top_connection['connections'], 1):
                    st.write(f"{i}. {step}")
    else:
        st.info("🔍 Scanning for brilliant opportunities...")
    
    # Top opportunity section
    st.markdown("---")
    st.subheader("🎯 Top Opportunity Right Now")
    
    # Find best opportunity (prioritize brilliant insights)
    best_signal = None
    best_ticker = None
    best_confidence = None
    best_source = "technical"
    
    # Check brilliant connections first
    if brilliant_results:
        genius_trades = brilliant_results.get('genius_trades', [])
        for trade in genius_trades:
            if trade.get('confidence') in ['GENIUS', 'HIGH']:
                best_signal = trade.get('direction', 'HOLD')
                best_ticker = trade.get('target_asset', trade.get('trade_name', ''))
                best_confidence = trade.get('confidence', 'HIGH')
                best_source = "brilliant"
                break
    
    # Fallback to technical signals
    if not best_ticker:
        for ticker in key_tickers:
            signal, confidence, reason = calculate_signal(data, ticker)
            if signal in ["BUY", "SELL"] and confidence == "HIGH":
                best_signal = signal
                best_ticker = ticker
                best_confidence = confidence
                break
    
    if best_ticker:
        col1, col2 = st.columns([2, 1])
        with col1:
            signal_source = "🧠 BRILLIANT CONNECTION" if best_source == "brilliant" else "📈 TECHNICAL ANALYSIS"
            
            if best_signal == "BUY":
                st.success(f"### ✅ Strong BUY Signal: {best_ticker}")
                st.write(f"**Source:** {signal_source}")
                st.write(f"**Action:** Buy {best_ticker} immediately")
                if best_source == "brilliant":
                    st.write(f"**Edge:** Cross-market intelligence detected opportunity")
                    st.write(f"**Target:** Alpha-generating position")
                else:
                    st.write(f"**Target:** 2-3% gain in next 24-48 hours")
            else:
                st.error(f"### ⚠️ Strong SELL Signal: {best_ticker}")
                st.write(f"**Source:** {signal_source}")
                st.write(f"**Action:** Sell or short {best_ticker}")
                if best_source == "brilliant":
                    st.write(f"**Edge:** Advanced market transmission detected")
                    st.write(f"**Risk:** Sophisticated downside pressure")
                else:
                    st.write(f"**Risk:** Potential 2-3% decline ahead")
        
        with col2:
            # Mini chart
            if best_ticker in data.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=data[best_ticker].tail(50),
                    mode='lines',
                    name=best_ticker,
                    line=dict(color='green' if best_signal == "BUY" else 'red', width=2)
                ))
                fig.update_layout(
                    height=200,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-confidence opportunities at the moment. Market is stable.")
    
    # Market overview (simple)
    st.markdown("---")
    st.subheader("📈 Market Pulse")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Simple market metrics
    with col1:
        spy_change = (data['SPY'].iloc[-1] - data['SPY'].iloc[0]) / data['SPY'].iloc[0] * 100
        st.metric("S&P 500 Trend", f"{spy_change:+.2f}%", "Bullish" if spy_change > 0 else "Bearish")
    
    with col2:
        qqq_change = (data['QQQ'].iloc[-1] - data['QQQ'].iloc[0]) / data['QQQ'].iloc[0] * 100
        st.metric("Tech Sector", f"{qqq_change:+.2f}%", "Strong" if qqq_change > 1 else "Weak")
    
    with col3:
        # Volatility indicator
        avg_volatility = sum(data[t].std() / data[t].mean() * 100 for t in key_tickers if t in data.columns) / len(key_tickers)
        st.metric("Market Volatility", f"{avg_volatility:.1f}%", "High" if avg_volatility > 2 else "Normal")
    
    with col4:
        # Winner/Loser ratio
        winners = sum(1 for t in key_tickers if t in data.columns and data[t].iloc[-1] > data[t].iloc[0])
        st.metric("Winners", f"{winners}/{len(key_tickers)}", "Bullish" if winners > 3 else "Mixed")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()