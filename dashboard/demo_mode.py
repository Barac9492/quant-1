#!/usr/bin/env python3
"""
Demo dashboard with offline mode for testing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config
st.set_page_config(
    page_title="Market Correlation System - DEMO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_demo_data():
    """Generate realistic demo data"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic market data
    base_returns = np.random.normal(0, 0.02, (n_days, 8))
    
    data = {}
    
    # REITs with inverse correlation to rates
    rates_factor = np.random.normal(0, 0.01, n_days)
    reits_returns = -0.6 * rates_factor + 0.8 * base_returns[:, 0]
    data['VNQ (REITs)'] = 100 * np.exp(np.cumsum(reits_returns))
    
    # 10Y Treasury rates
    rates_returns = rates_factor + 0.3 * base_returns[:, 1]
    data['TNX (10Y Rates)'] = 3.0 + np.cumsum(rates_returns * 0.1)
    
    # Dollar Index
    dollar_returns = base_returns[:, 2] * 0.5
    data['DXY (Dollar)'] = 100 * np.exp(np.cumsum(dollar_returns))
    
    # USD/JPY correlated with dollar
    usdjpy_returns = 0.4 * dollar_returns + 0.6 * base_returns[:, 3] * 0.3
    data['USDJPY'] = 140 * np.exp(np.cumsum(usdjpy_returns))
    
    # Bitcoin with higher volatility
    btc_base = np.random.normal(0, 0.05, n_days)
    dollar_lagged = np.roll(dollar_returns, 3)
    btc_returns = 0.3 * dollar_lagged + 0.7 * btc_base
    data['Bitcoin'] = 30000 * np.exp(np.cumsum(btc_returns))
    
    # VIX with mean reversion
    vix_base = 20 + 15 * np.sin(np.arange(n_days) * 0.05) 
    vix_shocks = np.random.normal(0, 5, n_days)
    vix_shocks[::50] *= 3  # Occasional spikes
    data['VIX'] = np.clip(vix_base + vix_shocks, 10, 80)
    
    # QQQ negatively correlated with VIX
    vix_factor = (data['VIX'] - 20) / 20
    qqq_returns = -0.4 * vix_factor * 0.02 + base_returns[:, 6]
    data['QQQ (Tech)'] = 300 * np.exp(np.cumsum(qqq_returns))
    
    # ARKK more volatile, correlated with QQQ
    arkk_volatility = 1.5
    arkk_returns = 0.7 * qqq_returns + arkk_volatility * base_returns[:, 7]
    data['ARKK (Innovation)'] = 50 * np.exp(np.cumsum(arkk_returns))
    
    return pd.DataFrame(data, index=dates).dropna()

@st.cache_data
def load_demo_data():
    return generate_demo_data()

def main():
    st.title("🔗 Market Correlation Analysis System")
    st.markdown("### 📺 DEMO MODE - Synthetic Data Visualization")
    
    # Load demo data
    data = load_demo_data()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Demo Controls")
        
        st.info("**Demo Mode Active**\n\nUsing synthetic data that mimics real market behavior for testing purposes.")
        
        # Data info
        st.markdown("---")
        st.subheader("📊 Demo Data Info")
        st.write(f"• **Assets**: {len(data.columns)}")
        st.write(f"• **Time Period**: {len(data)} days")
        st.write(f"• **Date Range**: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Asset selection
        st.markdown("---")
        selected_assets = st.multiselect(
            "Select Assets for Analysis",
            options=data.columns.tolist(),
            default=data.columns[:4].tolist()
        )
        
        # Correlation window
        window = st.slider("Correlation Window (days)", 10, 90, 30)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔗 Correlations", "🧪 Hypothesis Tests"])
    
    with tab1:
        st.subheader("Market Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_corr = data.corr().values[np.triu_indices_from(data.corr().values, k=1)].mean()
            st.metric("Average Correlation", f"{avg_corr:.3f}")
        
        with col2:
            volatility = data.pct_change().std().mean() * np.sqrt(252) * 100
            st.metric("Average Volatility", f"{volatility:.1f}%")
        
        with col3:
            recent_performance = data.pct_change(periods=30).iloc[-1].mean() * 100
            st.metric("30-day Avg Return", f"{recent_performance:.2f}%")
        
        with col4:
            st.metric("Data Quality", "✅ Excellent")
        
        # Price chart
        st.subheader("Normalized Price Performance")
        normalized_data = (data / data.iloc[0]) * 100
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, column in enumerate(normalized_data.columns):
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[column],
                mode='lines',
                name=column,
                line=dict(width=2, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base = 100)",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent performance table
        st.subheader("Recent Performance Summary")
        performance_data = []
        
        for asset in data.columns:
            current_price = data[asset].iloc[-1]
            change_1d = (data[asset].iloc[-1] / data[asset].iloc[-2] - 1) * 100
            change_7d = (data[asset].iloc[-1] / data[asset].iloc[-8] - 1) * 100
            change_30d = (data[asset].iloc[-1] / data[asset].iloc[-31] - 1) * 100
            volatility = data[asset].pct_change().std() * np.sqrt(252) * 100
            
            performance_data.append({
                'Asset': asset,
                'Current Price': f"{current_price:.2f}",
                '1D Change (%)': f"{change_1d:.2f}",
                '7D Change (%)': f"{change_7d:.2f}",
                '30D Change (%)': f"{change_30d:.2f}",
                'Volatility (%)': f"{volatility:.1f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        if len(selected_assets) >= 2:
            selected_data = data[selected_assets]
            
            # Correlation matrix
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Current Correlation Matrix")
                corr_matrix = selected_data.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Correlation Strength")
                
                # Extract unique pairs and their correlations
                pairs = []
                for i in range(len(selected_assets)):
                    for j in range(i+1, len(selected_assets)):
                        asset1, asset2 = selected_assets[i], selected_assets[j]
                        corr_val = corr_matrix.loc[asset1, asset2]
                        
                        if abs(corr_val) > 0.7:
                            strength = "🔴 Very Strong"
                        elif abs(corr_val) > 0.5:
                            strength = "🟡 Strong"
                        elif abs(corr_val) > 0.3:
                            strength = "🟢 Moderate"
                        else:
                            strength = "⚪ Weak"
                        
                        direction = "↗️ Positive" if corr_val > 0 else "↘️ Negative"
                        
                        pairs.append({
                            'Asset Pair': f"{asset1} - {asset2}",
                            'Correlation': f"{corr_val:.3f}",
                            'Strength': strength,
                            'Direction': direction
                        })
                
                pairs_df = pd.DataFrame(pairs)
                st.dataframe(pairs_df, use_container_width=True)
            
            # Rolling correlations
            st.subheader(f"Rolling Correlations ({window}-day window)")
            
            fig = go.Figure()
            
            for i in range(len(selected_assets)):
                for j in range(i+1, len(selected_assets)):
                    asset1, asset2 = selected_assets[i], selected_assets[j]
                    rolling_corr = selected_data[asset1].rolling(window).corr(selected_data[asset2])
                    
                    fig.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        mode='lines',
                        name=f"{asset1} - {asset2}",
                        line=dict(width=2)
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=0.7, line_dash="dot", line_color="red", opacity=0.7, annotation_text="Strong +")
            fig.add_hline(y=-0.7, line_dash="dot", line_color="red", opacity=0.7, annotation_text="Strong -")
            
            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Correlation Coefficient",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please select at least 2 assets for correlation analysis.")
    
    with tab3:
        st.subheader("🧪 Hypothesis Testing Results")
        
        # Simulate hypothesis results
        hypothesis_results = [
            {
                "name": "REITs vs Interest Rates",
                "description": "REITs and interest rates show inverse correlation",
                "current_correlation": data['VNQ (REITs)'].corr(data['TNX (10Y Rates)']),
                "signal": "🟡 MONITOR - Moderate inverse correlation detected",
                "confidence": "Medium",
                "key_metrics": {
                    "REITs 7d Change": (data['VNQ (REITs)'].iloc[-1] / data['VNQ (REITs)'].iloc[-8] - 1) * 100,
                    "Rates 7d Change": data['TNX (10Y Rates)'].iloc[-1] - data['TNX (10Y Rates)'].iloc[-8]
                }
            },
            {
                "name": "Dollar-Yen-Bitcoin Chain",
                "description": "Dollar strength affects Yen and Bitcoin volatility",
                "current_correlation": data['DXY (Dollar)'].corr(data['Bitcoin']),
                "signal": "🟢 LOW VOLATILITY - Stable conditions detected",
                "confidence": "High",
                "key_metrics": {
                    "Dollar 7d Change": (data['DXY (Dollar)'].iloc[-1] / data['DXY (Dollar)'].iloc[-8] - 1) * 100,
                    "Bitcoin Volatility": data['Bitcoin'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                }
            },
            {
                "name": "VIX Tech Correlation",
                "description": "High VIX increases tech stock correlation",
                "current_correlation": data['VIX'].corr(data['QQQ (Tech)']),
                "signal": "🟢 NORMAL FEAR - Tech stocks may diverge" if data['VIX'].iloc[-1] < 30 else "🔴 HIGH FEAR - Tech correlation risk",
                "confidence": "High",
                "key_metrics": {
                    "Current VIX": data['VIX'].iloc[-1],
                    "QQQ-ARKK Correlation": data['QQQ (Tech)'].corr(data['ARKK (Innovation)'])
                }
            }
        ]
        
        for i, result in enumerate(hypothesis_results):
            with st.expander(f"📋 {result['name']}", expanded=True):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Correlation",
                        f"{result['current_correlation']:.3f}",
                        delta=f"Confidence: {result['confidence']}"
                    )
                
                with col2:
                    for metric_name, metric_value in result['key_metrics'].items():
                        if isinstance(metric_value, (int, float)):
                            st.metric(metric_name, f"{metric_value:.2f}")
                
                with col3:
                    signal = result['signal']
                    if '🔴' in signal:
                        st.error(f"**Signal**: {signal}")
                    elif '🟡' in signal:
                        st.warning(f"**Signal**: {signal}")
                    else:
                        st.success(f"**Signal**: {signal}")
                
                st.info(f"**Hypothesis**: {result['description']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    📺 <strong>Demo Mode</strong> - This dashboard is using synthetic data for demonstration.<br>
    In production, it would connect to live market data via Yahoo Finance API.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()