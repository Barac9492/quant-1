#!/usr/bin/env python3
"""
Enhanced Live Dashboard with Target Asset Predictor and Adaptive Early Signals
Built with 20+ years quant experience
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
from strategies.adaptive_early_signals import AdaptiveEarlySignalEngine
from strategies.target_asset_predictor import TargetAssetPredictor
from strategies.practical_early_indicators import PracticalEarlyIndicatorSystem

# Page configuration
st.set_page_config(
    page_title="🧠 AI Quant Signal Engine",
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
def run_adaptive_analysis(data_hash):
    """Run adaptive early signal analysis with caching"""
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    engine = AdaptiveEarlySignalEngine()
    adaptive_signals = engine.detect_adaptive_signals(data)
    
    insights = generate_real_insights(data)
    
    return adaptive_signals, insights, data

@st.cache_data(ttl=300)
def run_target_analysis(target_asset, data_hash):
    """Run target asset analysis with caching"""
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    predictor = TargetAssetPredictor()
    return predictor.find_early_indicators(target_asset, data)

@st.cache_data(ttl=300)
def run_practical_signals_analysis(data_hash):
    """Run practical early signals analysis with caching"""
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    system = PracticalEarlyIndicatorSystem()
    
    # Analyze key assets for practical signals
    key_assets = ['SPY', 'QQQ', 'AAPL', 'VNQ', 'GLD']
    practical_results = {}
    
    for asset in key_assets:
        if asset in data.columns:
            practical_results[asset] = system.analyze_early_signals(asset, data)
    
    return practical_results, data

def main():
    st.title("🧠 AI Quant Signal Engine")
    st.markdown("### 🚀 Professional-Grade Early Signal Detection & Target Asset Analysis")
    
    # Load data
    with st.spinner("🔄 Loading real-time market data..."):
        data = load_live_data()
    
    if data.empty:
        st.error("❌ Unable to fetch real market data. Please check your connection.")
        st.stop()
    
    # Generate data hash for caching
    data_hash = hash(str(data.values.tobytes()))
    
    # Run comprehensive analysis
    with st.spinner("🧠 Running AI analysis..."):
        adaptive_signals, insights, data = run_adaptive_analysis(data_hash)
    
    # Header metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    regime = adaptive_signals.get('market_regime', {})
    stability = adaptive_signals.get('correlation_stability', {})
    execution_priority = adaptive_signals.get('execution_priority', [])
    
    with col1:
        regime_emoji = {'crisis': '🚨', 'stress': '🔴', 'normal': '🟡', 'complacency': '😴'}[regime.get('regime_type', 'normal')]
        st.metric(
            "Market Regime",
            f"{regime_emoji} {regime.get('regime_type', 'unknown').title()}",
            delta=f"{regime.get('confidence', 'LOW')} confidence"
        )
    
    with col2:
        st.metric(
            "Correlation Stability",
            f"{stability.get('stability_score', 0.5):.2f}",
            delta=f"Regime shift risk: {stability.get('regime_shift_probability', 0.5):.1%}"
        )
    
    with col3:
        priority_count = len([x for x in execution_priority if x.get('priority_rank', 5) <= 2])
        st.metric(
            "🚨 High Priority Signals",
            priority_count,
            delta="Critical actions needed" if priority_count > 0 else "All clear"
        )
    
    with col4:
        avg_vol = regime.get('avg_volatility', 0.15)
        vol_status = "High" if avg_vol > 0.25 else "Normal" if avg_vol > 0.15 else "Low"
        st.metric(
            "Market Volatility",
            f"{avg_vol:.1%}",
            delta=f"{vol_status} regime"
        )
    
    with col5:
        last_update = data.index[-1].strftime("%m/%d")
        st.metric(
            "🔄 Data Status",
            "Live",
            delta=f"Updated {last_update}"
        )
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Target Asset Analyzer", 
        "⚡ Early Warning System",
        "🧠 Adaptive Signals", 
        "📊 Execution Priority", 
        "🌊 Market Insights", 
        "🔧 Advanced Settings"
    ])
    
    with tab1:
        st.subheader("🎯 Target Asset Early Signal Finder")
        st.markdown("**Enter any ticker to find the best early indicators for trading that specific asset**")
        
        # Asset selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_assets = list(data.columns)
            selected_target = st.selectbox(
                "Select Target Asset to Analyze:",
                available_assets,
                index=available_assets.index('AAPL') if 'AAPL' in available_assets else 0
            )
        
        with col2:
            min_correlation = st.slider("Minimum Correlation Threshold", 0.1, 0.8, 0.3, 0.05)
            max_lead_days = st.slider("Maximum Lead Time (days)", 1, 30, 15)
        
        if st.button("🔍 Find Early Indicators", type="primary"):
            with st.spinner(f"Analyzing early indicators for {selected_target}..."):
                target_results = run_target_analysis(selected_target, data_hash)
            
            if 'error' in target_results:
                st.error(f"❌ {target_results['error']}")
            else:
                # Display results
                indicators = target_results.get('leading_indicators', [])
                
                if indicators:
                    st.success(f"🚀 Found {len(indicators)} reliable early indicators for {selected_target}")
                    
                    # Top indicators table
                    st.subheader("📊 Top Early Indicators")
                    
                    indicator_data = []
                    for i, ind in enumerate(indicators[:5], 1):
                        lead_time = f"{ind['optimal_lag']}d" if ind['optimal_lag'] > 0 else "Concurrent"
                        direction_emoji = "📈" if ind['direction'] == 'positive' else "📉"
                        
                        indicator_data.append({
                            'Rank': i,
                            'Indicator': f"{direction_emoji} {ind['indicator']}",
                            'Lead Time': lead_time,
                            'Correlation': f"{ind['correlation']:+.3f}",
                            'Strength': ind['relationship_strength'].title(),
                            'Stability': f"{ind['stability_score']:.2f}",
                            'Predictive Score': f"{ind['predictive_score']:.3f}",
                            'Significance': ind['significance']
                        })
                    
                    indicator_df = pd.DataFrame(indicator_data)
                    st.dataframe(indicator_df, use_container_width=True)
                    
                    # Current predictions
                    predictions = target_results.get('prediction_signals', [])
                    if predictions:
                        st.subheader(f"🔮 Current Predictions for {selected_target}")
                        
                        for pred in predictions[:3]:
                            direction_emoji = "🚀" if pred['predicted_direction'] == 'up' else "💥"
                            
                            with st.container():
                                st.markdown(f"**{pred['indicator']} Signal:**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Recent Move", f"{pred['indicator_move']:+.2f}%")
                                
                                with col2:
                                    st.metric("Prediction", f"{direction_emoji} {pred['predicted_direction'].upper()}")
                                
                                with col3:
                                    st.metric("Timeframe", pred['expected_timeframe'])
                                
                                if pred['confidence'] == 'HIGH':
                                    st.success(f"🔥 HIGH CONFIDENCE: {pred['indicator']} suggests {selected_target} likely to move {pred['predicted_direction']}")
                                else:
                                    st.info(f"⚡ MEDIUM CONFIDENCE: Watch {pred['indicator']} for {selected_target} direction")
                                
                                st.markdown("---")
                    
                    # Trading recommendations
                    recommendations = target_results.get('trading_recommendations', {})
                    if recommendations.get('primary_indicators_to_watch'):
                        st.subheader("💡 Trading Recommendations")
                        
                        st.markdown("**Primary Indicators to Monitor:**")
                        for rec in recommendations['primary_indicators_to_watch']:
                            st.markdown(f"• **{rec['indicator']}** - {rec['relationship']} relationship, {rec['reliability']} reliability (leads by {rec['lead_time']})")
                        
                        if recommendations.get('entry_signals'):
                            st.markdown("**Current Entry Signals:**")
                            for signal in recommendations['entry_signals']:
                                st.markdown(f"• {signal['signal']} ({signal['confidence']} confidence)")
                    
                    # Risk warnings
                    warnings = target_results.get('risk_warnings', [])
                    if warnings:
                        st.subheader("⚠️ Risk Warnings")
                        for warning in warnings:
                            st.warning(warning)
                    
                else:
                    st.warning(f"No strong early indicators found for {selected_target} with current thresholds. Try lowering the correlation threshold.")
    
    with tab2:
        st.subheader("⚡ Early Warning System")
        st.markdown("**🎯 PRACTICAL EARLY INDICATORS - Actionable signals with meaningful lead times (1+ days)**")
        st.markdown("*Focus on known market relationships that actually lead market moves*")
        
        # Run practical signals analysis
        with st.spinner("🔍 Analyzing practical early warning signals..."):
            practical_results, practical_data = run_practical_signals_analysis(data_hash)
        
        if not practical_results:
            st.warning("No practical early warning signals available")
        else:
            # Overall risk dashboard
            st.markdown("### 🚨 SYSTEM RISK DASHBOARD")
            
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            # Calculate overall system risk
            high_risk_assets = []
            critical_warnings = []
            
            for asset, result in practical_results.items():
                if 'error' not in result:
                    if result.get('risk_level') == 'HIGH':
                        high_risk_assets.append(asset)
                    
                    critical_warnings.extend([
                        w for w in result.get('current_early_warnings', []) 
                        if w.get('urgency') == 'HIGH'
                    ])
            
            with risk_col1:
                risk_emoji = "🚨" if len(high_risk_assets) >= 2 else "⚠️" if len(high_risk_assets) == 1 else "✅"
                st.metric("System Risk Level", 
                         f"{risk_emoji} {'HIGH' if len(high_risk_assets) >= 2 else 'MEDIUM' if len(high_risk_assets) == 1 else 'LOW'}", 
                         f"{len(high_risk_assets)} assets at risk")
            
            with risk_col2:
                st.metric("Critical Warnings", 
                         len(critical_warnings),
                         "Action required" if critical_warnings else "All clear")
            
            with risk_col3:
                total_warnings = sum(len(result.get('current_early_warnings', [])) for result in practical_results.values() if 'error' not in result)
                st.metric("Total Early Warnings", total_warnings)
            
            with risk_col4:
                monitored_assets = len([r for r in practical_results.values() if 'error' not in r])
                st.metric("Assets Monitored", monitored_assets)
            
            st.markdown("---")
            
            # Critical alerts first
            if critical_warnings:
                st.markdown("### 🚨 IMMEDIATE ACTION REQUIRED")
                for warning in critical_warnings[:5]:
                    st.error(f"**HIGH URGENCY**: {warning['message']}")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Recommended Action**: {warning['recommended_action']}")
                    with col2:
                        st.metric("Timeframe", warning['timeframe'])
                        st.metric("Reliability", f"{warning['reliability']:.0%}")
                st.markdown("---")
            
            # Asset-by-asset analysis
            st.markdown("### 📊 ASSET-BY-ASSET EARLY WARNINGS")
            
            for asset, result in practical_results.items():
                if 'error' in result:
                    continue
                
                risk_level = result.get('risk_level', 'LOW')
                risk_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[risk_level]
                
                with st.expander(f"{risk_emoji} {asset} - {risk_level} Risk", expanded=(risk_level == 'HIGH')):
                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    
                    warnings = result.get('current_early_warnings', [])
                    watchlist = result.get('watchlist_indicators', [])
                    
                    with col1:
                        st.metric("Risk Level", risk_level)
                    
                    with col2:
                        st.metric("Active Warnings", len(warnings))
                    
                    with col3:
                        st.metric("Indicators Watched", len(watchlist))
                    
                    # Current warnings
                    if warnings:
                        st.markdown("**🔔 Current Early Warnings:**")
                        for i, warning in enumerate(warnings, 1):
                            urgency_color = "error" if warning['urgency'] == 'HIGH' else "warning"
                            message = f"**{i}.** {warning['message']}"
                            
                            if urgency_color == "error":
                                st.error(message)
                            else:
                                st.warning(message)
                            
                            st.markdown(f"*Timeframe: {warning['timeframe']} | Reliability: {warning['reliability']:.0%}*")
                            st.markdown(f"*Action: {warning['recommended_action']}*")
                            st.markdown("---")
                    
                    # Market relationship signals
                    market_signals = result.get('market_relationship_signals', [])
                    if market_signals:
                        st.markdown("**🔗 Market Relationship Signals:**")
                        for signal in market_signals:
                            signal_strength = signal.get('signal_strength', 'NONE')
                            if signal_strength != 'NONE':
                                st.info(f"**{signal['indicator']}**: {signal['expected_impact']} (Lead: {signal.get('lead_time', 'N/A')})")
                    
                    # Watchlist
                    if watchlist:
                        st.markdown("**👀 Key Indicators to Monitor:**")
                        for item in watchlist[:3]:  # Top 3
                            priority_emoji = "🔥" if item['priority'] == 'HIGH' else "📊"
                            st.markdown(f"{priority_emoji} **{item['indicator']}**: {item['current_value']}")
                            st.markdown(f"   Watch for: {item['watch_for']}")
                            st.markdown(f"   Impact: {item['expected_impact']}")
    
    with tab3:
        st.subheader("🧠 Adaptive Early Signal Detection")
        st.markdown("**Advanced correlation-based signals accounting for regime changes and non-stationarity**")
        
        # Market regime analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Current Market Regime")
            regime_info = f"""
            **Regime Type:** {regime.get('regime_type', 'unknown').title()} {regime_emoji}
            **Average Correlation:** {regime.get('avg_correlation', 0):.3f}
            **Volatility Level:** {regime.get('avg_volatility', 0):.1%}
            **Momentum:** {regime.get('avg_momentum', 0):.1%}
            **Regime Persistence:** {regime.get('regime_persistence', 0):.1%}
            **Confidence:** {regime.get('confidence', 'LOW')}
            """
            st.markdown(regime_info)
        
        with col2:
            st.markdown("### 📊 Correlation Stability")
            stability_info = f"""
            **Stability Score:** {stability.get('stability_score', 0.5):.3f}
            **Regime Shift Risk:** {stability.get('regime_shift_probability', 0.5):.1%}
            **Status:** {stability.get('interpretation', 'Unknown')}
            """
            st.markdown(stability_info)
        
        # Correlation breaks
        correlation_breaks = stability.get('correlation_breaks', [])
        if correlation_breaks:
            st.markdown("### 🚨 Recent Correlation Breaks")
            
            break_data = []
            for brk in correlation_breaks[:5]:
                break_data.append({
                    'Asset Pair': brk['assets'],
                    'Break Direction': brk['direction'].title(),
                    'Magnitude': f"{brk['break_magnitude']:.3f}",
                    'Current (15d)': f"{brk['short_term_corr']:.3f}",
                    'Previous (45d)': f"{brk['medium_term_corr']:.3f}",
                    'Significance': brk['significance']
                })
            
            if break_data:
                break_df = pd.DataFrame(break_data)
                st.dataframe(break_df, use_container_width=True)
        
        # Regime-adjusted signals
        regime_signals = adaptive_signals.get('regime_adjusted_signals', [])
        if regime_signals:
            st.markdown("### 🎯 Regime-Adjusted Signals")
            
            for i, signal in enumerate(regime_signals[:5], 1):
                signal_type_emoji = "🔴" if signal['signal_type'] == 'CORRELATION_SURGE' else "🟢"
                
                with st.expander(f"{signal_type_emoji} {signal['signal_type']}: {signal['assets']}", expanded=i<=2):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Correlation Change", f"{signal['correlation_change']:+.3f}")
                    
                    with col2:
                        st.metric("Signal Strength", f"{signal['signal_strength']:.3f}")
                    
                    with col3:
                        st.metric("Expected Lead", f"{signal['expected_lead_days']} days")
                    
                    st.markdown(f"**Implication:** {signal['implication']}")
                    st.markdown(f"**Action:** {signal['action']}")
                    st.markdown(f"**Confidence:** {signal['confidence']}")
        
        # Portfolio impact
        portfolio_impact = adaptive_signals.get('portfolio_impact_signals', [])
        if portfolio_impact:
            st.markdown("### 💼 Portfolio Impact Analysis")
            
            for impact in portfolio_impact:
                priority_color = "error" if impact.get('priority') == 'CRITICAL' else "warning" if impact.get('priority') == 'HIGH' else "info"
                
                if priority_color == "error":
                    st.error(f"🚨 **{impact['signal_type']}:** {impact['signal']}")
                elif priority_color == "warning":
                    st.warning(f"⚠️ **{impact['signal_type']}:** {impact['signal']}")
                else:
                    st.info(f"ℹ️ **{impact['signal_type']}:** {impact['signal']}")
                
                st.markdown(f"**Impact:** {impact['impact']}")
                st.markdown(f"**Action:** {impact['action']}")
                st.markdown("---")
    
    with tab4:
        st.subheader("⚡ Execution Priority List")
        st.markdown("**Prioritized actions based on signal strength, confidence, and urgency**")
        
        execution_priority = adaptive_signals.get('execution_priority', [])
        
        if execution_priority:
            for i, action in enumerate(execution_priority, 1):
                urgency_emoji = {
                    'IMMEDIATE': '🚨',
                    'HIGH': '🔴', 
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }[action['execution_urgency']]
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {urgency_emoji} {action['category']}")
                        st.markdown(f"**Signal:** {action['signal']}")
                        st.markdown(f"**Action:** {action['action']}")
                    
                    with col2:
                        st.metric("Timeframe", action['timeframe'])
                        st.metric("Confidence", action['confidence'])
                        st.metric("Urgency", action['execution_urgency'])
                    
                    if action['execution_urgency'] in ['IMMEDIATE', 'HIGH']:
                        st.error(f"⚠️ **{action['execution_urgency']} PRIORITY** - Action required within {action['timeframe']}")
                    
                    st.markdown("---")
        else:
            st.success("✅ No high-priority actions required at this time")
    
    with tab5:
        st.subheader("📊 Market Insights Dashboard")
        
        # Use existing insights
        perf = insights["performance_insights"]
        vol = insights["volatility_insights"]
        corr = insights["correlation_insights"]
        actionable = insights["actionable_insights"]
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best 1D Performer", perf["best_1d"]["asset"], f"{perf['best_1d']['return']:+.2f}%")
        
        with col2:
            st.metric("Most Volatile Asset", vol["most_volatile"]["asset"], f"{vol['most_volatile']['volatility']:.1f}%")
        
        with col3:
            st.metric("Average Correlation", f"{corr['average_correlation']:.3f}")
        
        with col4:
            diversification_score = len([c for c in corr['strongest_positive'] if abs(c['correlation']) < 0.5]) / len(corr['strongest_positive']) * 100
            st.metric("Market Diversification", f"{diversification_score:.0f}%")
        
        # Actionable insights
        if actionable:
            st.markdown("### 💡 AI-Generated Market Insights")
            
            for insight in actionable[:5]:
                insight_type = insight["type"]
                message = insight["message"]
                action = insight["action"]
                
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
        
        # Correlation heatmap
        st.subheader("🌡️ Real-Time Correlation Matrix")
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("🔧 Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Analysis Parameters")
            
            correlation_threshold = st.slider("Correlation Alert Threshold", 0.3, 0.9, 0.7, 0.05)
            volatility_threshold = st.slider("High Volatility Threshold", 0.15, 0.50, 0.25, 0.05)
            regime_persistence = st.slider("Regime Persistence Requirement", 0.3, 0.9, 0.6, 0.1)
            
            st.markdown("### 📊 Data Quality")
            st.metric("Data Points", len(data))
            st.metric("Assets Tracked", len(data.columns))
            st.metric("Date Range", f"{(data.index[-1] - data.index[0]).days} days")
        
        with col2:
            st.markdown("### 🔄 Refresh Settings")
            
            refresh_interval = st.selectbox(
                "Auto Refresh Interval",
                ["Manual Only", "5 minutes", "15 minutes", "30 minutes"],
                index=0
            )
            
            if refresh_interval != "Manual Only":
                st.info(f"Page will refresh every {refresh_interval}")
                interval_map = {"5 minutes": 300, "15 minutes": 900, "30 minutes": 1800}
                seconds = interval_map[refresh_interval]
                
                st.markdown(f"""
                <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {seconds * 1000});
                </script>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Refresh Data Now", type="primary"):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("### 📈 Asset Universe")
            st.write("Available assets:")
            st.write(", ".join(data.columns))
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 AI Quant Engine")
        st.markdown("**Built with 20+ years quant experience**")
        
        # Quick stats
        st.markdown("---")
        st.subheader("📊 System Status")
        
        regime_color = {
            'crisis': '🔴',
            'stress': '🟡', 
            'normal': '🟢',
            'complacency': '🔵'
        }[regime.get('regime_type', 'normal')]
        
        st.markdown(f"{regime_color} **Regime:** {regime.get('regime_type', 'unknown').title()}")
        st.markdown(f"⚡ **Volatility:** {regime.get('avg_volatility', 0):.1%}")
        st.markdown(f"🔗 **Avg Correlation:** {regime.get('avg_correlation', 0):.3f}")
        st.markdown(f"📊 **Stability:** {stability.get('stability_score', 0.5):.2f}")
        
        # Top priority actions
        st.markdown("---")
        st.subheader("⚡ Priority Actions")
        
        priority_actions = [x for x in execution_priority if x.get('priority_rank', 5) <= 2][:3]
        
        if priority_actions:
            for action in priority_actions:
                urgency_emoji = {'IMMEDIATE': '🚨', 'HIGH': '🔴', 'MEDIUM': '🟡'}[action['execution_urgency']]
                st.markdown(f"{urgency_emoji} **{action['category']}**")
                st.markdown(f"*{action['timeframe']}*")
                st.markdown("---")
        else:
            st.success("✅ All systems normal")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Analysis")
        if st.button("🔍 Analyze Custom Asset"):
            st.info("Use the Target Asset Analyzer tab above")
        
        st.markdown("### 📚 Resources")
        st.markdown("• Non-stationary correlations")
        st.markdown("• Regime-dependent relationships")
        st.markdown("• Time-varying lead-lag analysis")
        st.markdown("• Multi-timeframe confirmation")

if __name__ == "__main__":
    main()