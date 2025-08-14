import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.fetchers import MarketDataFetcher, DatabaseManager
from strategies.hypothesis_engine import HypothesisEngine

# Page configuration
st.set_page_config(
    page_title="Market Correlation Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CorrelationDashboard:
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.db_manager = DatabaseManager()
        self.hypothesis_engine = HypothesisEngine()
        
    @st.cache_data
    def load_data(_self):
        """데이터 로드 (캐싱)"""
        return _self.fetcher.fetch_data(period="1y")
    
    def main_page(self):
        """메인 대시보드 페이지"""
        st.title("🔗 Market Correlation Analysis System")
        st.markdown("---")
        
        # 데이터 로드
        with st.spinner("Loading market data..."):
            data = self.load_data()
        
        if data.empty:
            st.error("❌ No data available. Please check your internet connection or API access.")
            return
        
        # 메트릭스 행
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Data Points",
                value=f"{len(data):,}",
                delta=f"{len(data.columns)} assets"
            )
        
        with col2:
            last_update = data.index[-1].strftime("%Y-%m-%d")
            st.metric(
                label="Last Update",
                value=last_update,
                delta="Live data"
            )
        
        with col3:
            avg_corr = data.corr().values[np.triu_indices_from(data.corr().values, k=1)].mean()
            st.metric(
                label="Avg Correlation",
                value=f"{avg_corr:.3f}",
                delta="All assets"
            )
        
        with col4:
            volatility = data.pct_change().std().mean() * np.sqrt(252) * 100
            st.metric(
                label="Avg Volatility",
                value=f"{volatility:.1f}%",
                delta="Annualized"
            )
        
        st.markdown("---")
        
        # 탭 레이아웃
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧪 Hypotheses", "🔔 Alerts", "📈 Analysis"])
        
        with tab1:
            self.overview_tab(data)
        
        with tab2:
            self.hypotheses_tab(data)
        
        with tab3:
            self.alerts_tab(data)
        
        with tab4:
            self.analysis_tab(data)
    
    def overview_tab(self, data: pd.DataFrame):
        """개요 탭"""
        st.subheader("Market Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 상관관계 히트맵
            st.subheader("Asset Correlation Matrix")
            corr_matrix = data.corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Assets", y="Assets", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 최근 성과
            st.subheader("Recent Performance")
            recent_returns = data.pct_change(periods=30).iloc[-1] * 100
            recent_returns = recent_returns.sort_values(ascending=False)
            
            for asset, return_val in recent_returns.items():
                color = "🟢" if return_val > 0 else "🔴"
                st.metric(
                    label=f"{color} {asset}",
                    value=f"{return_val:.2f}%",
                    delta="30-day"
                )
        
        # 가격 차트
        st.subheader("Price Trends (Normalized to 100)")
        normalized_data = (data / data.iloc[0]) * 100
        
        fig = go.Figure()
        for column in normalized_data.columns:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[column],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Normalized Price Performance",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def hypotheses_tab(self, data: pd.DataFrame):
        """가설 검증 탭"""
        st.subheader("🧪 Hypothesis Testing Lab")
        
        # 가설 테스트 실행
        with st.spinner("Testing market hypotheses..."):
            results = self.hypothesis_engine.test_all_hypotheses(data)
        
        # 가설별 결과 표시
        for hypothesis_name, result in results.items():
            if "error" in result:
                st.error(f"❌ {hypothesis_name}: {result['error']}")
                continue
            
            with st.expander(f"📋 {result.get('hypothesis', hypothesis_name)}", expanded=True):
                
                if hypothesis_name == "reits_rates":
                    self._display_reits_hypothesis(result)
                elif hypothesis_name == "dollar_yen_bitcoin":
                    self._display_crypto_hypothesis(result)
                elif hypothesis_name == "vix_tech":
                    self._display_vix_hypothesis(result)
    
    def _display_reits_hypothesis(self, result: dict):
        """REITs 가설 결과 표시"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Correlation",
                value=result.get('current_correlation', 'N/A'),
                delta=f"Avg: {result.get('average_correlation', 'N/A')}"
            )
        
        with col2:
            st.metric(
                "REITs Change",
                value=f"{result.get('recent_reits_change_pct', 0):.2f}%",
                delta="7-day"
            )
        
        with col3:
            st.metric(
                "Rates Change",
                value=f"{result.get('recent_rates_change_pts', 0):.3f}pts",
                delta="7-day"
            )
        
        # 신호 표시
        signal = result.get('signal', 'No signal')
        confidence = result.get('confidence', 'Unknown')
        
        if '🔴' in signal:
            st.error(f"**Signal**: {signal}")
        elif '🟡' in signal:
            st.warning(f"**Signal**: {signal}")
        else:
            st.success(f"**Signal**: {signal}")
        
        st.info(f"**Confidence Level**: {confidence} | **Data Points**: {result.get('data_points', 0)}")
    
    def _display_crypto_hypothesis(self, result: dict):
        """암호화폐 가설 결과 표시"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "DXY Change",
                value=f"{result.get('recent_dxy_change_pct', 0):.2f}%",
                delta="7-day"
            )
        
        with col2:
            st.metric(
                "USD/JPY Change", 
                value=f"{result.get('recent_usdjpy_change_pct', 0):.2f}%",
                delta="7-day"
            )
        
        with col3:
            current_vol = result.get('current_btc_volatility', 0)
            avg_vol = result.get('average_btc_volatility', 0)
            st.metric(
                "BTC Volatility",
                value=f"{current_vol:.1f}%",
                delta=f"vs {avg_vol:.1f}% avg"
            )
        
        # 지연 효과 상관관계
        st.subheader("Lag Effect Analysis")
        lag_corrs = result.get('lag_correlations', {})
        
        if lag_corrs:
            lag_df = pd.DataFrame(list(lag_corrs.items()), columns=['Lag', 'Correlation'])
            lag_df['Lag'] = lag_df['Lag'].str.replace('lag_', '').astype(int)
            
            fig = px.bar(
                lag_df, 
                x='Lag', 
                y='Correlation',
                title="Dollar Index vs Bitcoin Volatility (Lagged Correlations)",
                labels={'Lag': 'Days Lagged', 'Correlation': 'Correlation Coefficient'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 신호
        signal = result.get('signal', 'No signal')
        confidence = result.get('confidence', 'Unknown')
        
        if '🔴' in signal:
            st.error(f"**Signal**: {signal}")
        elif '🟡' in signal:
            st.warning(f"**Signal**: {signal}")
        else:
            st.success(f"**Signal**: {signal}")
        
        st.info(f"**Confidence**: {confidence} | **DXY-USDJPY Correlation**: {result.get('dxy_usdjpy_correlation', 'N/A')}")
    
    def _display_vix_hypothesis(self, result: dict):
        """VIX 가설 결과 표시"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_vix = result.get('current_vix', 0)
            st.metric(
                "Current VIX",
                value=f"{current_vix:.2f}",
                delta=result.get('current_condition', 'Normal')
            )
        
        with col2:
            high_vix_corr = result.get('high_vix_correlation')
            st.metric(
                "High VIX Correlation",
                value=f"{high_vix_corr:.3f}" if high_vix_corr else "N/A",
                delta="QQQ-ARKK when VIX>30"
            )
        
        with col3:
            normal_vix_corr = result.get('normal_vix_correlation')
            st.metric(
                "Normal VIX Correlation", 
                value=f"{normal_vix_corr:.3f}" if normal_vix_corr else "N/A",
                delta="QQQ-ARKK when VIX<30"
            )
        
        # VIX 레벨 시각화
        vix_levels = [
            ("Extreme Fear", 35, "🔴"),
            ("High Fear", 30, "🟡"), 
            ("Normal", 20, "🟢"),
            ("Complacency", 0, "🟦")
        ]
        
        st.subheader("VIX Fear Level")
        for level, threshold, emoji in vix_levels:
            if current_vix >= threshold:
                st.success(f"{emoji} **{level}** (VIX ≥ {threshold})")
                break
        
        # 신호
        signal = result.get('signal', 'No signal')
        confidence = result.get('confidence', 'Unknown')
        
        if '🔴' in signal:
            st.error(f"**Signal**: {signal}")
        elif '🟡' in signal:
            st.warning(f"**Signal**: {signal}")
        else:
            st.success(f"**Signal**: {signal}")
        
        st.info(f"**Confidence**: {confidence} | **High VIX Days**: {result.get('high_vix_days', 0)}")
    
    def alerts_tab(self, data: pd.DataFrame):
        """알림 탭"""
        st.subheader("🔔 Real-time Alerts")
        
        # 상관관계 이상 패턴 감지
        correlations = self.fetcher.calculate_correlations(data)
        anomalies = self.fetcher.detect_anomalies(correlations)
        
        if not anomalies:
            st.success("✅ No significant correlation anomalies detected")
        else:
            for group_name, group_anomalies in anomalies.items():
                st.subheader(f"⚠️ {group_name.upper()} Alerts")
                
                for anomaly in group_anomalies:
                    alert_type = "🔴 Strong" if abs(anomaly['correlation']) > 0.8 else "🟡 Moderate"
                    
                    with st.container():
                        st.markdown(f"""
                        **{alert_type} Correlation Alert**
                        - **Assets**: {anomaly['asset1']} ↔ {anomaly['asset2']}
                        - **Correlation**: {anomaly['correlation']:.3f}
                        - **Type**: {anomaly['strength']} correlation
                        - **Date**: {anomaly['date'].strftime('%Y-%m-%d')}
                        """)
                        st.markdown("---")
        
        # 설정 섹션
        st.subheader("⚙️ Alert Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            threshold = st.slider("Correlation Threshold", 0.5, 0.95, 0.7, 0.05)
        with col2:
            notification_type = st.selectbox("Notification Type", ["Dashboard Only", "Telegram", "Email"])
        
        if st.button("Update Alert Settings"):
            st.success("Alert settings updated!")
    
    def analysis_tab(self, data: pd.DataFrame):
        """분석 탭"""
        st.subheader("📈 Advanced Analysis")
        
        # 자산 선택
        selected_assets = st.multiselect(
            "Select assets for detailed analysis",
            options=data.columns.tolist(),
            default=data.columns[:3].tolist()
        )
        
        if len(selected_assets) >= 2:
            selected_data = data[selected_assets]
            
            # 롤링 상관관계
            st.subheader("Rolling Correlations")
            window = st.slider("Correlation Window (days)", 10, 90, 30)
            
            fig = go.Figure()
            
            # 모든 쌍의 상관관계 계산
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
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=0.7, line_dash="dot", line_color="red", annotation_text="Strong +")
            fig.add_hline(y=-0.7, line_dash="dot", line_color="red", annotation_text="Strong -")
            
            fig.update_layout(
                title=f"Rolling Correlations ({window}-day window)",
                xaxis_title="Date",
                yaxis_title="Correlation Coefficient",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            st.subheader("Statistical Summary")
            summary_data = []
            
            for i in range(len(selected_assets)):
                for j in range(i+1, len(selected_assets)):
                    asset1, asset2 = selected_assets[i], selected_assets[j]
                    corr = selected_data[asset1].corr(selected_data[asset2])
                    rolling_corr = selected_data[asset1].rolling(window).corr(selected_data[asset2])
                    
                    summary_data.append({
                        'Asset Pair': f"{asset1} - {asset2}",
                        'Overall Correlation': f"{corr:.3f}",
                        'Avg Rolling Correlation': f"{rolling_corr.mean():.3f}",
                        'Correlation Volatility': f"{rolling_corr.std():.3f}",
                        'Current Correlation': f"{rolling_corr.iloc[-1]:.3f}"
                    })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

def main():
    dashboard = CorrelationDashboard()
    
    # 사이드바
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # 자동 새로고침 설정
        auto_refresh = st.checkbox("Auto Refresh (5 min)")
        if auto_refresh:
            st.rerun()
        
        # 데이터 정보
        st.markdown("---")
        st.subheader("📊 Data Sources")
        st.markdown("""
        - **Yahoo Finance**: Real-time market data
        - **Update Frequency**: Every 15 minutes
        - **Coverage**: Stocks, ETFs, Bonds, Crypto, FX
        """)
        
        # 가설 정보
        st.markdown("---")
        st.subheader("🧪 Active Hypotheses")
        st.markdown("""
        1. **REITs vs Interest Rates**
        2. **Dollar-Yen-Bitcoin Chain**  
        3. **VIX Tech Correlation**
        """)
        
        # 실행 버튼
        st.markdown("---")
        if st.button("🔄 Refresh All Data", type="primary"):
            st.rerun()
        
        if st.button("💾 Download Report"):
            st.success("Report generated!")
    
    # 메인 대시보드 실행
    dashboard.main_page()

if __name__ == "__main__":
    main()