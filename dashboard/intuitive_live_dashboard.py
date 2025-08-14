#!/usr/bin/env python3
"""
Intuitive Live Dashboard - Clean, easy-to-interpret UI with real stock prices
Built for clear decision-making with live market data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.robust_live_fetcher import RobustLiveDataFetcher
from strategies.cross_market_discovery import CrossMarketDiscoveryEngine

# Page configuration
st.set_page_config(
    page_title="🎯 Smart Trading Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}

.price-box {
    background: linear-gradient(90deg, #f8f9fa, #e9ecef);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
}

.signal-box-positive {
    background: linear-gradient(90deg, #d4edda, #c3e6cb);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #28a745;
    margin: 0.5rem 0;
}

.signal-box-negative {
    background: linear-gradient(90deg, #f8d7da, #f5c6cb);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #dc3545;
    margin: 0.5rem 0;
}

.signal-box-neutral {
    background: linear-gradient(90deg, #fff3cd, #ffeaa7);
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #ffc107;
    margin: 0.5rem 0;
}

.metric-large {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
}

.metric-small {
    font-size: 0.9rem;
    color: #7f8c8d;
}

.discovery-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border: 1px solid #e9ecef;
}

.alpha-highlight {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: bold;
    display: inline-block;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_live_market_data():
    """Load live market data with caching"""
    fetcher = RobustLiveDataFetcher()
    return fetcher.fetch_live_data(period='1y')

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_discovery_analysis(data_hash):
    """Run discovery analysis with caching"""
    data = load_live_market_data()
    
    engine = CrossMarketDiscoveryEngine()
    results = engine.discover_hidden_connections(
        data, 
        min_lead_days=1, 
        min_alpha_threshold=0.15
    )
    
    return results, data

def format_price_change(current, previous):
    """Format price change with color coding"""
    if previous == 0:
        return "0.00", "⚪"
    
    change = current - previous
    change_pct = (change / previous) * 100
    
    if change > 0:
        return f"+{change_pct:.2f}%", "🟢"
    elif change < 0:
        return f"{change_pct:.2f}%", "🔴"
    else:
        return "0.00%", "⚪"

def get_signal_color_class(signal_type):
    """Get CSS class based on signal type"""
    if signal_type in ['BUY', 'POSITIVE', 'BULLISH']:
        return "signal-box-positive"
    elif signal_type in ['SELL', 'NEGATIVE', 'BEARISH']:
        return "signal-box-negative"
    else:
        return "signal-box-neutral"

def main():
    # Header
    st.markdown('<div class="main-header">📈 Smart Trading Signals</div>', unsafe_allow_html=True)
    st.markdown("**Real-time cross-market intelligence for smart trading decisions**")
    
    # Load data
    with st.spinner("🔄 Loading live market data..."):
        data = load_live_market_data()
    
    if data.empty:
        st.error("❌ Unable to fetch live market data. Please refresh the page.")
        st.stop()
    
    # Data freshness indicator
    data_age = (datetime.now() - data.index[-1]).days
    if data_age == 0:
        freshness_indicator = "🟢 Today's Data"
    elif data_age == 1:
        freshness_indicator = "🟡 Yesterday's Data"
    else:
        freshness_indicator = f"🔴 {data_age} Days Old"
    
    st.markdown(f"**Data Status:** {freshness_indicator} | **Last Updated:** {data.index[-1].strftime('%Y-%m-%d')} | **Assets:** {len(data.columns)}")
    
    # ═══════════════════════════════════════════════════
    # SECTION 1: LIVE STOCK PRICES
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 💰 Live Stock Prices")
    
    # Key stocks to display
    key_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'GLD']
    available_stocks = [stock for stock in key_stocks if stock in data.columns]
    
    if available_stocks:
        cols = st.columns(min(4, len(available_stocks)))
        
        for i, stock in enumerate(available_stocks[:8]):
            with cols[i % 4]:
                if stock in data.columns:
                    current_price = data[stock].iloc[-1]
                    prev_price = data[stock].iloc[-2] if len(data) > 1 else current_price
                    
                    change_text, change_emoji = format_price_change(current_price, prev_price)
                    
                    st.markdown(f"""
                    <div class="price-box">
                        <div style="font-size: 1.1rem; font-weight: bold; color: #2c3e50;">{stock}</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: #1f77b4;">${current_price:.2f}</div>
                        <div style="font-size: 0.9rem;">{change_emoji} {change_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════
    # SECTION 2: SMART TRADING SIGNALS
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 🎯 Smart Trading Signals")
    
    # Run analysis
    data_hash = hash(str(data.values.tobytes()))
    with st.spinner("🧠 Analyzing cross-market patterns..."):
        results, analysis_data = run_discovery_analysis(data_hash)
    
    if 'error' in results:
        st.error(f"❌ Analysis Error: {results['error']}")
    else:
        # Signal Summary
        total_signals = len(results['cross_market_signals'])
        alpha_chains = len(results['alpha_generating_chains'])
        patterns = len(results['exotic_pattern_alerts'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔍 Cross-Market Signals",
                value=total_signals,
                help="Hidden connections between distant asset classes"
            )
        
        with col2:
            st.metric(
                label="💰 Alpha Opportunities",
                value=alpha_chains,
                help="Potential profit-generating relationships"
            )
        
        with col3:
            st.metric(
                label="🎨 Pattern Alerts",
                value=patterns,
                help="Sophisticated market patterns detected"
            )
        
        with col4:
            market_status = "🟢 OPPORTUNITY" if total_signals > 2 else "🟡 NORMAL" if total_signals > 0 else "🔴 QUIET"
            st.metric(
                label="📊 Market Status",
                value=market_status.split()[1],
                help="Overall market opportunity level"
            )
        
        # ═══════════════════════════════════════════════════
        # SECTION 3: KEY DISCOVERIES (Simplified)
        # ═══════════════════════════════════════════════════
        
        if results['cross_market_signals']:
            st.markdown("## 🔍 Key Market Discoveries")
            
            for i, signal in enumerate(results['cross_market_signals'][:3], 1):
                # Determine signal strength and direction
                alpha = signal.get('alpha_potential', 0)
                lead_days = signal.get('optimal_lead_days', 0)
                
                # Simplified interpretation
                if alpha > 0.4:
                    strength = "🔥 STRONG"
                    strength_color = "#28a745"
                elif alpha > 0.25:
                    strength = "⚡ MODERATE"
                    strength_color = "#ffc107"
                else:
                    strength = "📊 WEAK"
                    strength_color = "#6c757d"
                
                st.markdown(f"""
                <div class="discovery-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #2c3e50;">Discovery #{i}: {signal['chain_name'].replace('_', ' ').title()}</h4>
                        <div class="alpha-highlight">{strength} Signal</div>
                    </div>
                    
                    <div style="font-size: 1.1rem; color: #495057; margin-bottom: 1rem;">
                        📝 <strong>What it means:</strong> {signal['description']}
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">SOURCE SIGNAL</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #1f77b4;">
                                {' + '.join(signal['source_assets'])}
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">PREDICTION TARGET</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #e74c3c;">
                                {signal.get('target_asset', signal.get('best_target', 'Multiple Assets'))}
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">TIME HORIZON</div>
                            <div style="font-size: 1.1rem; font-weight: bold; color: #27ae60;">
                                {lead_days} days ahead
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                        <div style="font-size: 0.9rem; color: #6c757d;">PROFIT POTENTIAL</div>
                        <div style="font-size: 1.3rem; font-weight: bold; color: {strength_color};">
                            {alpha:.1%} Alpha Score
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════
        # SECTION 4: TRADING OPPORTUNITIES (Actionable)
        # ═══════════════════════════════════════════════════
        
        if results['alpha_generating_chains']:
            st.markdown("## 💎 Trading Opportunities")
            
            for i, opportunity in enumerate(results['alpha_generating_chains'][:2], 1):
                alpha_score = opportunity.get('alpha_score', 0)
                sharpe = opportunity.get('sharpe_equivalent', 0)
                
                # Actionable recommendation
                if alpha_score > 0.5:
                    action = "🚀 STRONG BUY SIGNAL"
                    action_class = "signal-box-positive"
                elif alpha_score > 0.3:
                    action = "⚡ MODERATE OPPORTUNITY"
                    action_class = "signal-box-neutral"
                else:
                    action = "📊 WATCH CLOSELY"
                    action_class = "signal-box-neutral"
                
                st.markdown(f"""
                <div class="{action_class}">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div>
                            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {action}
                            </div>
                            <div style="font-size: 1rem; margin-bottom: 1rem;">
                                📈 {opportunity['description']}
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                                <div>
                                    <div style="font-size: 0.8rem; color: #666;">ALPHA SCORE</div>
                                    <div style="font-size: 1.1rem; font-weight: bold;">{alpha_score:.3f}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #666;">SHARPE EQUIVALENT</div>
                                    <div style="font-size: 1.1rem; font-weight: bold;">{sharpe:.2f}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #666;">TARGET ASSET</div>
                                    <div style="font-size: 1.1rem; font-weight: bold;">{opportunity.get('target_asset', 'Various')}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════
        # SECTION 5: MARKET OVERVIEW CHART
        # ═══════════════════════════════════════════════════
        
        st.markdown("## 📊 Market Overview")
        
        # Create market overview chart
        chart_assets = ['SPY', 'QQQ', 'TLT', 'GLD']
        available_chart_assets = [asset for asset in chart_assets if asset in data.columns]
        
        if available_chart_assets:
            fig = go.Figure()
            
            for asset in available_chart_assets:
                # Normalize to percentage change
                normalized = (data[asset] / data[asset].iloc[0] - 1) * 100
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode='lines',
                    name=asset,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Performance Comparison (Normalized to %)",
                xaxis_title="Date",
                yaxis_title="Return %",
                height=400,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════
    # FOOTER: DATA VERIFICATION
    # ═══════════════════════════════════════════════════
    
    st.markdown("---")
    
    # Data verification section
    with st.expander("🔍 Data Verification & System Status"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Quality:**")
            completeness = (data.notna().sum() / len(data)).mean()
            st.write(f"• Data Completeness: {completeness:.1%}")
            st.write(f"• Total Assets Tracked: {len(data.columns)}")
            st.write(f"• Date Range: {len(data)} trading days")
            st.write(f"• Latest Data: {data.index[-1].strftime('%Y-%m-%d')}")
        
        with col2:
            st.markdown("**🔄 System Status:**")
            st.write("• Data Source: ✅ Yahoo Finance Live")
            st.write("• Analysis Engine: ✅ Cross-Market Discovery")
            st.write("• Signal Generation: ✅ Active")
            st.write(f"• Last Analysis: {datetime.now().strftime('%H:%M:%S')}")
        
        # Sample price verification
        if available_stocks:
            st.markdown("**💰 Current Prices (for verification):**")
            price_verification = {}
            for stock in available_stocks[:6]:
                if stock in data.columns:
                    price_verification[stock] = f"${data[stock].iloc[-1]:.2f}"
            
            st.json(price_verification)
    
    # Auto-refresh option
    st.sidebar.markdown("### ⚙️ Settings")
    
    auto_refresh = st.sidebar.selectbox(
        "Auto Refresh",
        ["Off", "30 seconds", "1 minute", "5 minutes"],
        index=0
    )
    
    if auto_refresh != "Off":
        refresh_seconds = {
            "30 seconds": 30,
            "1 minute": 60,
            "5 minutes": 300
        }[auto_refresh]
        
        st.sidebar.info(f"Page will refresh every {auto_refresh}")
        time.sleep(refresh_seconds)
        st.rerun()
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Help section
    with st.sidebar.expander("❓ How to Use"):
        st.markdown("""
        **📈 Live Prices:** Current stock prices with daily changes
        
        **🎯 Smart Signals:** AI-discovered cross-market relationships
        
        **💎 Trading Opportunities:** Actionable profit opportunities
        
        **📊 Market Overview:** Performance comparison chart
        
        **🔍 Data Verification:** Check data quality and prices
        """)

if __name__ == "__main__":
    main()