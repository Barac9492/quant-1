import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import sqlite3
from telegram import Bot
import requests
import json

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.alerts_config = self.config.get('alerts', {})
        self.telegram_bot = self._setup_telegram() if self.alerts_config.get('enabled') else None
        
    def _setup_telegram(self) -> Optional[Bot]:
        """Telegram 봇 설정"""
        token = self.alerts_config.get('telegram_bot_token')
        if token:
            try:
                bot = Bot(token=token)
                logger.info("✓ Telegram bot initialized")
                return bot
            except Exception as e:
                logger.error(f"✗ Telegram bot setup failed: {e}")
        return None
    
    async def send_correlation_alert(self, anomalies: Dict[str, List[Dict]]) -> bool:
        """상관관계 이상 알림 발송"""
        if not anomalies:
            return True
        
        alert_messages = []
        
        for group_name, group_anomalies in anomalies.items():
            group_message = f"🚨 **{group_name.upper()} CORRELATION ALERT**\n"
            
            for anomaly in group_anomalies:
                correlation = anomaly['correlation']
                strength = "🔴 STRONG" if abs(correlation) > 0.8 else "🟡 MODERATE"
                direction = "📈 POSITIVE" if correlation > 0 else "📉 NEGATIVE"
                
                group_message += f"""
{strength} {direction} Correlation Detected
• Assets: {anomaly['asset1']} ↔ {anomaly['asset2']}
• Correlation: {correlation:.3f}
• Date: {anomaly['date'].strftime('%Y-%m-%d %H:%M')}
                """
            
            alert_messages.append(group_message)
        
        # 알림 발송
        success = True
        for message in alert_messages:
            success &= await self._send_message(message)
            success &= self._log_alert(message)
        
        return success
    
    async def send_hypothesis_alert(self, hypothesis_name: str, result: Dict) -> bool:
        """가설 기반 알림 발송"""
        signal = result.get('signal', '')
        confidence = result.get('confidence', 'Unknown')
        
        # 중요한 신호만 알림 발송
        if not any(emoji in signal for emoji in ['🔴', '🟡']):
            return True
        
        hypothesis_titles = {
            'reits_rates': 'REITs vs Interest Rates',
            'dollar_yen_bitcoin': 'Dollar-Yen-Bitcoin Chain',
            'vix_tech': 'VIX Tech Correlation'
        }
        
        title = hypothesis_titles.get(hypothesis_name, hypothesis_name)
        
        message = f"""
🧪 **HYPOTHESIS ALERT: {title}**

{signal}

**Confidence**: {confidence}
**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 **Key Metrics**:
        """
        
        # 가설별 핵심 메트릭 추가
        if hypothesis_name == 'reits_rates':
            message += f"""
• Current Correlation: {result.get('current_correlation', 'N/A')}
• REITs 7-day Change: {result.get('recent_reits_change_pct', 0):.2f}%
• Rates 7-day Change: {result.get('recent_rates_change_pts', 0):.3f}pts
            """
        elif hypothesis_name == 'dollar_yen_bitcoin':
            message += f"""
• DXY 7-day Change: {result.get('recent_dxy_change_pct', 0):.2f}%
• USD/JPY 7-day Change: {result.get('recent_usdjpy_change_pct', 0):.2f}%
• BTC Volatility: {result.get('current_btc_volatility', 0):.1f}%
            """
        elif hypothesis_name == 'vix_tech':
            message += f"""
• Current VIX: {result.get('current_vix', 0):.2f}
• Condition: {result.get('current_condition', 'Normal')}
• QQQ-ARKK Correlation: {result.get('recent_30d_correlation', 'N/A')}
            """
        
        return await self._send_message(message) and self._log_alert(message)
    
    async def send_daily_summary(self, data_summary: Dict) -> bool:
        """일일 요약 알림"""
        message = f"""
📊 **DAILY MARKET CORRELATION SUMMARY**
Date: {datetime.now().strftime('%Y-%m-%d')}

📈 **Market Performance**:
• Data Points: {data_summary.get('data_points', 0):,}
• Average Correlation: {data_summary.get('avg_correlation', 0):.3f}
• Market Volatility: {data_summary.get('avg_volatility', 0):.1f}%

🔍 **Active Alerts**: {data_summary.get('active_alerts', 0)}
🧪 **Hypothesis Status**: {data_summary.get('hypothesis_status', 'Normal')}

💡 **Today's Insight**:
{data_summary.get('insight', 'No significant patterns detected today.')}
        """
        
        return await self._send_message(message) and self._log_alert(message, alert_type="daily_summary")
    
    async def _send_message(self, message: str) -> bool:
        """메시지 발송 (Telegram)"""
        if not self.telegram_bot:
            logger.warning("Telegram bot not configured")
            return False
        
        chat_id = self.alerts_config.get('chat_id')
        if not chat_id:
            logger.warning("Telegram chat_id not configured")
            return False
        
        try:
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info("✓ Telegram alert sent")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to send Telegram alert: {e}")
            return False
    
    def _log_alert(self, message: str, alert_type: str = "correlation") -> bool:
        """알림 히스토리 데이터베이스에 저장"""
        try:
            db_path = self.config.get('database', {}).get('path', 'database/market_data.db')
            
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (timestamp, group_name, asset1, asset2, correlation, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    alert_type,
                    "N/A",
                    "N/A", 
                    0.0,
                    message
                ))
            
            logger.info("✓ Alert logged to database")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to log alert: {e}")
            return False
    
    def get_alert_history(self, days: int = 7) -> List[Dict]:
        """알림 히스토리 조회"""
        try:
            db_path = self.config.get('database', {}).get('path', 'database/market_data.db')
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, group_name, message 
                    FROM alerts 
                    WHERE timestamp >= date('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days))
                
                return [
                    {
                        'timestamp': row[0],
                        'group_name': row[1], 
                        'message': row[2]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to retrieve alert history: {e}")
            return []

class AlertMonitor:
    """알림 모니터링 및 자동 실행 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        from data_pipeline.fetchers import MarketDataFetcher
        from strategies.hypothesis_engine import HypothesisEngine
        
        self.fetcher = MarketDataFetcher(config_path)
        self.hypothesis_engine = HypothesisEngine()
        self.alert_system = AlertSystem(config_path)
        
    async def run_monitoring_cycle(self):
        """전체 모니터링 사이클 실행"""
        logger.info("🔄 Starting monitoring cycle...")
        
        try:
            # 1. 데이터 수집
            data = self.fetcher.fetch_data(period="3mo")  # 최근 3개월
            if data.empty:
                logger.error("No data collected")
                return
            
            # 2. 상관관계 이상 감지
            correlations = self.fetcher.calculate_correlations(data)
            anomalies = self.fetcher.detect_anomalies(correlations)
            
            if anomalies:
                await self.alert_system.send_correlation_alert(anomalies)
            
            # 3. 가설 테스트 및 알림
            hypothesis_results = self.hypothesis_engine.test_all_hypotheses(data)
            
            for hypothesis_name, result in hypothesis_results.items():
                if "error" not in result:
                    await self.alert_system.send_hypothesis_alert(hypothesis_name, result)
            
            # 4. 일일 요약 (선택적)
            now = datetime.now()
            if now.hour == 18 and now.minute < 5:  # 오후 6시경
                summary = self._generate_daily_summary(data, anomalies, hypothesis_results)
                await self.alert_system.send_daily_summary(summary)
            
            logger.info("✓ Monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"✗ Monitoring cycle failed: {e}")
            
            # 에러 알림 발송
            error_message = f"""
🚨 **SYSTEM ERROR ALERT**

Monitoring cycle failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Error: {str(e)}

Please check system status.
            """
            await self.alert_system._send_message(error_message)
    
    def _generate_daily_summary(self, data, anomalies, hypothesis_results) -> Dict:
        """일일 요약 데이터 생성"""
        import numpy as np
        
        # 기본 통계
        avg_corr = data.corr().values[np.triu_indices_from(data.corr().values, k=1)].mean()
        volatility = data.pct_change().std().mean() * np.sqrt(252) * 100
        
        # 활성 알림 수
        active_alerts = sum(len(group_anomalies) for group_anomalies in anomalies.values())
        
        # 가설 상태 요약
        hypothesis_status = "Normal"
        critical_signals = 0
        
        for result in hypothesis_results.values():
            if "error" not in result:
                signal = result.get('signal', '')
                if '🔴' in signal:
                    critical_signals += 1
        
        if critical_signals >= 2:
            hypothesis_status = "Critical"
        elif critical_signals >= 1:
            hypothesis_status = "Warning"
        
        # 인사이트 생성
        insight = self._generate_insight(data, anomalies, hypothesis_results)
        
        return {
            'data_points': len(data),
            'avg_correlation': avg_corr,
            'avg_volatility': volatility,
            'active_alerts': active_alerts,
            'hypothesis_status': hypothesis_status,
            'insight': insight
        }
    
    def _generate_insight(self, data, anomalies, hypothesis_results) -> str:
        """AI 인사이트 생성 (단순화된 버전)"""
        insights = []
        
        # 상관관계 이상 인사이트
        if anomalies:
            insights.append("Unusual correlation patterns detected across multiple asset classes.")
        
        # 가설 기반 인사이트
        for hypothesis_name, result in hypothesis_results.items():
            if "error" not in result:
                signal = result.get('signal', '')
                confidence = result.get('confidence', '')
                
                if '🔴' in signal and confidence == 'High':
                    if hypothesis_name == 'reits_rates':
                        insights.append("Strong inverse REITs-rates correlation suggests interest rate sensitivity.")
                    elif hypothesis_name == 'dollar_yen_bitcoin':
                        insights.append("Dollar strength chain reaction affecting crypto volatility.")
                    elif hypothesis_name == 'vix_tech':
                        insights.append("Fear-driven tech stock correlation increase observed.")
        
        if not insights:
            return "Market showing normal correlation patterns with no significant anomalies."
        
        return " | ".join(insights)

async def main():
    """메인 모니터링 실행"""
    monitor = AlertMonitor()
    
    # 단일 모니터링 사이클 실행
    await monitor.run_monitoring_cycle()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())