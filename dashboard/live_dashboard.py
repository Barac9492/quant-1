#!/usr/bin/env python3
"""
Live dashboard with real market data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.alternative_fetcher import AlternativeDataFetcher, generate_real_insights

# Page configuration
st.set_page_config(
    page_title="🔥 LIVE Market Correlation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_live_data():
    """Load real market data with caching"""
    fetcher = AlternativeDataFetcher()
    return fetcher.fetch_multiple_sources()

@st.cache_data(ttl=300)
def generate_insights_cached(data_hash):
    """Generate insights with caching"""
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    return generate_real_insights(data), data

def main():
    st.title("🔥 LIVE Market Correlation Analysis")
    st.markdown("### 📈 Real-Time Market Data & AI-Powered Insights")
    
    # Load data
    with st.spinner("🔄 Fetching live market data..."):
        data = load_live_data()
    
    if data.empty:
        st.error("❌ Unable to fetch real market data. Please check your internet connection.")
        st.stop()
    
    # Generate insights  
    data_hash = hash(str(data.values.tobytes()))
    insights, data = generate_insights_cached(data_hash)
    
    # Header metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    summary = insights["data_summary"]
    perf = insights["performance_insights"]
    vol = insights["volatility_insights"]
    corr = insights["correlation_insights"]
    
    with col1:
        st.metric(
            "📊 Assets Tracked",
            summary["assets_count"],
            delta=f"{summary['total_days']} days data"
        )
    
    with col2:
        st.metric(
            "🎯 Best Performer (1D)",
            perf["best_1d"]["asset"],
            delta=f"{perf['best_1d']['return']:+.2f}%"
        )
    
    with col3:
        st.metric(
            "⚡ Highest Volatility",
            vol["most_volatile"]["asset"],
            delta=f"{vol['most_volatile']['volatility']:.1f}%"
        )
    
    with col4:
        st.metric(
            "🔗 Avg Correlation",
            f"{corr['average_correlation']:.3f}",
            delta="Market Integration"
        )
    
    with col5:
        last_update = datetime.strptime(summary["last_update"], "%Y-%m-%d")
        st.metric(
            "🔄 Last Update",
            last_update.strftime("%m/%d"),
            delta="Live Data"
        )
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Live Insights", "📊 Correlations", "💹 Performance", "⚠️ Risk Alerts"])
    
    with tab1:
        st.subheader("🎯 AI-Powered Market Insights")
        
        # Actionable insights
        actionable = insights["actionable_insights"]
        
        if actionable:
            for i, insight in enumerate(actionable):
                insight_type = insight["type"]
                message = insight["message"]
                action = insight["action"]
                
                # Color coding based on type
                if insight_type in ["BULL_MARKET", "MOMENTUM"]:
                    st.success(f"**{insight_type}**: {message}")
                    st.info(f"💡 **Action**: {action}")
                elif insight_type in ["RISK_WARNING", "HIGH_VOLATILITY", "BEAR_MARKET"]:
                    st.error(f"**{insight_type}**: {message}")
                    st.warning(f"⚠️ **Action**: {action}")
                else:
                    st.info(f"**{insight_type}**: {message}")
                    st.success(f"✅ **Action**: {action}")
                
                st.markdown("---")
        
        # Key correlations with trading implications
        st.subheader("🔗 Critical Correlation Shifts")
        
        risk_insights = insights["risk_insights"]
        if risk_insights["correlation_regime_changes"]:
            
            regime_df = []
            for pair, change_info in risk_insights["correlation_regime_changes"].items():
                regime_df.append({
                    "Asset Pair": pair,
                    "Previous (90d)": f"{change_info['long_term_corr']:.3f}",
                    "Current (30d)": f"{change_info['short_term_corr']:.3f}",
                    "Change": f"{change_info['change']:+.3f}",
                    "Implication": "🔴 Risk Increase" if change_info['change'] > 0 else "🟢 Diversification Opportunity"
                })
            
            if regime_df:
                st.dataframe(pd.DataFrame(regime_df), use_container_width=True)
        
        # Live market heatmap
        st.subheader("🌡️ Live Correlation Heatmap")
        
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1, zmax=1
        )
        
        fig.update_layout(
            title="Real-Time Asset Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Detailed Correlation Analysis")
        
        # Top correlations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Strongest Positive Correlations")
            pos_corrs = corr["strongest_positive"][:5]
            
            for item in pos_corrs:
                pair = item["pair"]
                corr_val = item["correlation"]
                
                # Progress bar for correlation strength
                progress_val = (corr_val + 1) / 2  # Normalize to 0-1
                
                st.metric(pair, f"{corr_val:.3f}")
                st.progress(progress_val)
                
                if corr_val > 0.8:
                    st.warning("⚠️ Very high correlation - low diversification benefit")
                
                st.markdown("---")
        
        with col2:
            st.markdown("### 🔵 Strongest Negative Correlations")
            neg_corrs = corr["strongest_negative"][:5]
            
            for item in neg_corrs:
                pair = item["pair"]
                corr_val = item["correlation"]
                
                progress_val = abs(corr_val)
                
                st.metric(pair, f"{corr_val:.3f}")
                st.progress(progress_val)
                
                if corr_val < -0.5:
                    st.success("✅ Strong hedge potential")
                
                st.markdown("---")
        
        # Rolling correlation analysis
        st.subheader("📈 Rolling Correlation Trends")
        
        asset1 = st.selectbox("Select First Asset", data.columns, key="corr_asset1")
        asset2 = st.selectbox("Select Second Asset", data.columns, key="corr_asset2", index=1)
        window = st.slider("Rolling Window (days)", 10, 60, 30)
        
        if asset1 != asset2:
            rolling_corr = data[asset1].rolling(window).corr(data[asset2])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                name=f"{asset1} - {asset2}",
                line=dict(width=3)
            ))
            
            # Add correlation bands
            fig.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="High Correlation")
            fig.add_hline(y=-0.7, line_dash="dot", line_color="red", annotation_text="High Inverse Correlation") 
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f"Rolling {window}-Day Correlation: {asset1} vs {asset2}",
                xaxis_title="Date",
                yaxis_title="Correlation Coefficient",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current correlation value
            current_corr = rolling_corr.iloc[-1]
            if abs(current_corr) > 0.7:
                st.error(f"🚨 **High correlation detected**: {current_corr:.3f}")
            elif abs(current_corr) < 0.3:
                st.success(f"✅ **Good diversification**: {current_corr:.3f}")
            else:
                st.info(f"ℹ️ **Moderate correlation**: {current_corr:.3f}")
    
    with tab3:
        st.subheader("💹 Live Performance Dashboard")
        
        # Performance metrics
        perf = insights["performance_insights"]
        
        # Performance comparison chart
        periods = ["1d", "7d", "30d"]
        performance_data = []
        
        for period in periods:
            best = perf[f"best_{period}"]
            performance_data.append({
                "Period": f"{period.upper()}",
                "Best Asset": best["asset"],
                "Return (%)": best["return"]
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        fig = px.bar(
            perf_df,
            x="Period",
            y="Return (%)",
            color="Best Asset",
            title="Best Performing Assets by Time Period"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        st.subheader("📊 Returns Distribution (Last 30 Days)")
        
        returns_30d = data.pct_change().tail(30) * 100
        
        fig = go.Figure()
        
        for asset in returns_30d.columns:
            fig.add_trace(go.Box(
                y=returns_30d[asset],
                name=asset,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="30-Day Daily Returns Distribution",
            yaxis_title="Daily Return (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility analysis
        st.subheader("⚡ Volatility Analysis")
        
        vol_data = data.pct_change().std() * np.sqrt(252) * 100
        vol_df = pd.DataFrame({
            'Asset': vol_data.index,
            'Volatility (%)': vol_data.values
        }).sort_values('Volatility (%)', ascending=False)
        
        fig = px.bar(
            vol_df,
            x='Asset',
            y='Volatility (%)',
            title="Annualized Volatility by Asset",
            color='Volatility (%)',
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Real-Time Risk Alerts")
        
        # Risk scoring
        risk_score = 0
        risk_factors = []
        
        # High correlation risk
        if corr["average_correlation"] > 0.6:
            risk_score += 30
            risk_factors.append("🔴 High average correlation reduces diversification")
        
        # High volatility risk  
        if vol["average_volatility"] > 25:
            risk_score += 25
            risk_factors.append("⚡ Elevated market volatility")
        
        # Concentration risk
        top_performer_return = max([perf[f"best_{p}"]["return"] for p in ["1d", "7d", "30d"]])
        if top_performer_return > 20:
            risk_score += 20
            risk_factors.append("🚀 Extreme performance concentration")
        
        # Correlation regime change risk
        if len(risk_insights["correlation_regime_changes"]) > 2:
            risk_score += 25
            risk_factors.append("🔄 Multiple correlation regime shifts detected")
        
        # Risk gauge
        st.subheader(f"🎯 Portfolio Risk Score: {risk_score}/100")
        
        if risk_score > 70:
            st.error("🚨 **HIGH RISK**: Multiple risk factors present")
        elif risk_score > 40:
            st.warning("⚠️ **MODERATE RISK**: Monitor closely")
        else:
            st.success("✅ **LOW RISK**: Normal market conditions")
        
        # Risk factor details
        if risk_factors:
            st.markdown("### Risk Factors Identified:")
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        
        # Diversification score
        div_ratio = risk_insights["diversification_ratio"]
        st.metric(
            "🎨 Diversification Score", 
            f"{div_ratio:.2f}",
            delta="Higher is better"
        )
        
        if div_ratio < 0.4:
            st.error("🔴 Poor diversification - consider adding uncorrelated assets")
        elif div_ratio < 0.6:
            st.warning("🟡 Moderate diversification - room for improvement")
        else:
            st.success("🟢 Good diversification across assets")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Live Controls")
        
        # Auto refresh option
        st.markdown("### 🔄 Refresh Settings")
        
        refresh_interval = st.selectbox(
            "Auto Refresh Interval",
            ["Manual Only", "5 minutes", "15 minutes", "30 minutes"],
            index=0
        )
        
        if refresh_interval != "Manual Only":
            st.info(f"Page will refresh every {refresh_interval}")
            # Use JavaScript for proper timed refresh
            interval_map = {"5 minutes": 300, "15 minutes": 900, "30 minutes": 1800}
            seconds = interval_map[refresh_interval]
            
            st.markdown(f"""
            <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {seconds * 1000});
            </script>
            """, unsafe_allow_html=True)
        
        # Manual refresh
        if st.button("🔄 Refresh Data Now", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Data info
        st.subheader("📊 Live Data Status")
        st.success(f"✅ **Connected**")
        st.info(f"📈 **{summary['assets_count']} assets**")
        st.info(f"📅 **{summary['total_days']} days history**")
        st.info(f"🔄 **Updated: {summary['last_update']}**")
        
        st.markdown("---")
        
        # Quick insights
        st.subheader("⚡ Quick Insights")
        
        # Top 3 actionable insights
        for insight in actionable[:3]:
            with st.expander(insight["type"]):
                st.write(insight["message"])
                st.success(f"💡 {insight['action']}")

if __name__ == "__main__":
    main()