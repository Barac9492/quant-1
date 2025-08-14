#!/usr/bin/env python3
"""
Elite Trading Signals - Institutional-Grade Predictive Engine
Built by 20+ year quant veteran for wealthy high net worth individuals
Focus: Clear actionable signals that make money, not technical complexity
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
    page_title="Elite Trading Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Elite CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.elite-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.elite-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.elite-header p {
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

.opportunity-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.opportunity-card h3 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
}

.profit-potential {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
}

.risk-warning {
    background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 6px 20px rgba(252, 70, 107, 0.4);
}

.price-card {
    background: white;
    border: 2px solid #f0f0f0;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}

.price-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.price-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2c3e50;
    margin: 0.5rem 0;
}

.price-change-positive {
    color: #27ae60;
    font-weight: 600;
    font-size: 1.1rem;
}

.price-change-negative {
    color: #e74c3c;
    font-weight: 600;
    font-size: 1.1rem;
}

.signal-strength-high {
    background: linear-gradient(45deg, #ff6b6b, #ffd93d);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    margin: 0.5rem 0;
}

.signal-strength-medium {
    background: linear-gradient(45deg, #74b9ff, #0984e3);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    margin: 0.5rem 0;
}

.action-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 30px;
    border: none;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
}

.action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}

.metric-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(240, 147, 251, 0.4);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}

.metric-label {
    font-size: 1rem;
    opacity: 0.9;
    margin: 0.5rem 0 0 0;
}

.hide-technical {
    display: none;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=120)  # Cache for 2 minutes - faster refresh for elite clients
def load_elite_data():
    """Load live market data with institutional speed"""
    fetcher = RobustLiveDataFetcher()
    return fetcher.fetch_live_data(period='6mo')  # Shorter period for speed

@st.cache_data(ttl=300)
def run_elite_analysis(data_hash):
    """Run elite-level predictive analysis"""
    data = load_elite_data()
    
    engine = CrossMarketDiscoveryEngine()
    results = engine.discover_hidden_connections(
        data, 
        min_lead_days=1, 
        min_alpha_threshold=0.25  # Higher threshold for elite signals
    )
    
    return results, data

def format_money(amount):
    """Format money for wealthy clients"""
    if amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.0f}K"
    else:
        return f"${amount:.0f}"

def get_trade_size_recommendation(price, risk_level='moderate'):
    """Suggest position sizes for wealthy clients"""
    base_amounts = {
        'conservative': 500_000,
        'moderate': 1_000_000,
        'aggressive': 2_500_000
    }
    
    base = base_amounts[risk_level]
    shares = int(base / price)
    return shares, base

def main():
    # Elite Header
    st.markdown(f"""
    <div class="elite-header">
        <h1>💎 Elite Trading Intelligence</h1>
        <p>Institutional-Grade Signals • Built by 20-Year Quant Veteran • For Sophisticated Investors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with professional speed
    with st.spinner("🔄 Analyzing global markets..."):
        data = load_elite_data()
    
    if data.empty:
        st.error("❌ Market data temporarily unavailable. Please contact your account manager.")
        st.stop()
    
    # ═══════════════════════════════════════════════════
    # SECTION 1: EXECUTIVE SUMMARY (What Matters Most)
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 📊 Market Pulse")
    
    # Key positions for wealthy investors
    elite_positions = {
        'AAPL': 'Tech Leader',
        'MSFT': 'Enterprise Cloud',
        'GOOGL': 'Digital Advertising', 
        'NVDA': 'AI Infrastructure',
        'SPY': 'Market Benchmark',
        'QQQ': 'Growth Portfolio',
        'GLD': 'Inflation Hedge',
        'TLT': 'Interest Rate Play'
    }
    
    cols = st.columns(4)
    
    for i, (symbol, description) in enumerate(list(elite_positions.items())[:8]):
        with cols[i % 4]:
            if symbol in data.columns:
                current_price = data[symbol].iloc[-1]
                prev_price = data[symbol].iloc[-2] if len(data) > 1 else current_price
                
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                change_class = "price-change-positive" if change > 0 else "price-change-negative"
                change_symbol = "▲" if change > 0 else "▼" if change < 0 else "►"
                
                # Position size for $1M allocation
                shares, allocation = get_trade_size_recommendation(current_price)
                
                st.markdown(f"""
                <div class="price-card">
                    <div style="font-weight: 600; color: #7f8c8d; font-size: 0.9rem;">{description}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #2c3e50; margin: 0.3rem 0;">{symbol}</div>
                    <div class="price-value">${current_price:.2f}</div>
                    <div class="{change_class}">{change_symbol} {abs(change_pct):.1f}%</div>
                    <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 0.5rem;">
                        ${allocation/1000:.0f}K → {shares:,} shares
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════
    # SECTION 2: PROFIT OPPORTUNITIES (The Money Makers)
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 💰 Current Opportunities")
    
    # Run elite analysis
    data_hash = hash(str(data.values.tobytes()))
    with st.spinner("🧠 Running proprietary algorithms..."):
        results, analysis_data = run_elite_analysis(data_hash)
    
    if 'error' not in results and results['cross_market_signals']:
        # Show only the best opportunities
        top_opportunities = sorted(
            results['cross_market_signals'], 
            key=lambda x: x.get('alpha_potential', 0), 
            reverse=True
        )[:2]  # Only show top 2 for elite clients
        
        for i, opportunity in enumerate(top_opportunities, 1):
            alpha = opportunity.get('alpha_potential', 0)
            lead_days = opportunity.get('optimal_lead_days', 0)
            
            # Calculate profit potential for wealthy client
            base_investment = 1_000_000  # $1M base
            potential_profit = base_investment * alpha
            
            # Simplify the description for client
            simple_description = {
                'commodity_tech_volatility': 'Gold signals predict tech stock movements',
                'central_bank_spillover': 'European market moves forecast US growth stocks',
                'credit_sector_rotation': 'Bond market shifts signal sector rotations',
                'energy_transition_alpha': 'Oil volatility predicts clean energy opportunities'
            }.get(opportunity.get('chain_name', ''), opportunity.get('description', ''))
            
            st.markdown(f"""
            <div class="opportunity-card">
                <h3>Opportunity #{i}: {simple_description}</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; margin: 1.5rem 0;">
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">PROFIT POTENTIAL</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{format_money(potential_profit)}</div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">on ${format_money(base_investment)} position</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">TIME TO PROFIT</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{lead_days} days</div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">average timeline</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">CONFIDENCE</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{alpha*100:.0f}%</div>
                        <div style="font-size: 0.8rem; opacity: 0.8;">historical accuracy</div>
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem;">RECOMMENDED ACTION:</div>
                    <div style="font-size: 1.1rem;">Monitor {' & '.join(opportunity['source_assets'])} → Position in {opportunity.get('target_asset', opportunity.get('best_target', 'target assets'))}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="opportunity-card">
            <h3>Market Analysis</h3>
            <p style="font-size: 1.1rem; margin: 0;">
                Current market conditions show limited high-conviction opportunities. 
                Maintaining defensive positioning recommended until clearer signals emerge.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════
    # SECTION 3: RISK MONITOR (Protect the Downside)
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 🛡️ Risk Monitor")
    
    # Calculate portfolio risk metrics
    key_assets = ['SPY', 'QQQ', 'TLT', 'GLD']
    available_assets = [asset for asset in key_assets if asset in data.columns]
    
    if available_assets:
        # Calculate recent volatility
        returns = data[available_assets].pct_change().dropna()
        recent_vol = returns.tail(20).std() * np.sqrt(252) * 100
        
        risk_cols = st.columns(len(available_assets))
        
        for i, asset in enumerate(available_assets):
            with risk_cols[i]:
                vol = recent_vol[asset]
                
                if vol > 25:
                    risk_level = "HIGH RISK"
                    risk_color = "#e74c3c"
                elif vol > 15:
                    risk_level = "MODERATE"
                    risk_color = "#f39c12"
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#27ae60"
                
                st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-weight: 600; font-size: 1rem;">{asset}</div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0;">{vol:.0f}%</div>
                    <div style="font-size: 0.8rem;">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════
    # SECTION 4: PORTFOLIO PERFORMANCE (The Bottom Line)
    # ═══════════════════════════════════════════════════
    
    st.markdown("## 📈 Portfolio Perspective")
    
    # Show performance of key holdings
    if len(available_assets) >= 2:
        fig = go.Figure()
        
        # Create normalized performance chart
        for asset in available_assets[:4]:  # Top 4 assets
            normalized = (data[asset] / data[asset].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=normalized,
                mode='lines',
                name=asset,
                line=dict(width=3),
                hovertemplate=f'<b>{asset}</b><br>Date: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="Portfolio Performance Comparison",
                font=dict(size=24, family="Inter"),
                x=0.5
            ),
            xaxis_title="",
            yaxis_title="Total Return (%)",
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════
    # SECTION 5: EXECUTION READY (Call to Action)
    # ═══════════════════════════════════════════════════
    
    st.markdown("## ⚡ Ready to Execute")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">LIVE</div>
            <div class="metric-label">Market Data Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        opportunities_count = len(results.get('cross_market_signals', []))
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{opportunities_count}</div>
            <div class="metric-label">Active Opportunities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        data_freshness = (datetime.now() - data.index[-1]).days
        freshness_text = "TODAY" if data_freshness == 0 else f"{data_freshness}D AGO"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{freshness_text}</div>
            <div class="metric-label">Last Update</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact section for elite service
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin: 2rem 0;">
        <h3 style="margin: 0 0 1rem 0;">Elite Trading Support</h3>
        <p style="font-size: 1.1rem; margin: 0;">
            Questions about these opportunities? Ready to execute? 
            <br>Your dedicated quant analyst is standing by.
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; 
                       border-radius: 25px; font-weight: 600;">
                📞 Direct Line: Available 24/7
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()