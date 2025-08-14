import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import sqlite3
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.assets = self._flatten_assets()
        
    def _flatten_assets(self) -> List[str]:
        """모든 자산 심볼을 하나의 리스트로 평탄화"""
        all_assets = []
        for category in self.config['assets'].values():
            all_assets.extend(category)
        return list(set(all_assets))
    
    def fetch_data(self, period: str = "2y") -> pd.DataFrame:
        """Yahoo Finance에서 데이터 수집"""
        logger.info(f"Fetching data for {len(self.assets)} assets...")
        
        data = {}
        for symbol in self.assets:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    data[symbol] = hist['Close']
                    logger.info(f"✓ {symbol}: {len(hist)} records")
                else:
                    logger.warning(f"✗ {symbol}: No data available")
            except Exception as e:
                logger.error(f"✗ {symbol}: {e}")
        
        if data:
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index)
            return df.dropna()
        return pd.DataFrame()
    
    def calculate_correlations(self, data: pd.DataFrame, window: int = 30) -> Dict[str, pd.DataFrame]:
        """그룹별 롤링 상관관계 계산"""
        correlations = {}
        
        for group_name, symbols in self.config['assets'].items():
            # 해당 그룹의 자산들만 선택
            group_data = data[symbols].dropna()
            
            if len(group_data.columns) >= 2:
                # 롤링 상관관계 계산
                rolling_corr = group_data.rolling(window=window).corr()
                correlations[group_name] = rolling_corr
                
                logger.info(f"Calculated correlations for {group_name}: {len(group_data)} days")
        
        return correlations
    
    def detect_anomalies(self, correlations: Dict[str, pd.DataFrame], threshold: float = 0.7) -> Dict[str, List[Dict]]:
        """상관관계 이상 패턴 감지"""
        anomalies = {}
        
        for group_name, corr_data in correlations.items():
            group_anomalies = []
            
            # 가장 최근 상관관계 값들 확인
            if not corr_data.empty:
                latest_corr = corr_data.iloc[-1]
                
                # 임계값을 넘는 강한 상관관계 또는 역상관관계 찾기
                for idx, corr_val in latest_corr.items():
                    if isinstance(idx, tuple) and len(idx) == 2:
                        asset1, asset2 = idx
                        if asset1 != asset2 and abs(corr_val) > threshold:
                            group_anomalies.append({
                                'date': corr_data.index[-1],
                                'asset1': asset1,
                                'asset2': asset2,
                                'correlation': corr_val,
                                'strength': 'Strong' if corr_val > 0 else 'Inverse'
                            })
            
            if group_anomalies:
                anomalies[group_name] = group_anomalies
        
        return anomalies

class DatabaseManager:
    def __init__(self, db_path: str = "database/market_data.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """필요한 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            # 가격 데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    date DATE,
                    symbol TEXT,
                    close_price REAL,
                    PRIMARY KEY (date, symbol)
                )
            """)
            
            # 상관관계 데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlations (
                    date DATE,
                    group_name TEXT,
                    asset1 TEXT,
                    asset2 TEXT,
                    correlation REAL,
                    window_days INTEGER,
                    PRIMARY KEY (date, group_name, asset1, asset2)
                )
            """)
            
            # 알림 히스토리 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    group_name TEXT,
                    asset1 TEXT,
                    asset2 TEXT,
                    correlation REAL,
                    message TEXT
                )
            """)
    
    def save_price_data(self, data: pd.DataFrame):
        """가격 데이터 저장"""
        with sqlite3.connect(self.db_path) as conn:
            for symbol in data.columns:
                symbol_data = data[symbol].dropna()
                records = [(date.strftime('%Y-%m-%d'), symbol, float(price)) 
                          for date, price in symbol_data.items()]
                
                conn.executemany(
                    "INSERT OR REPLACE INTO price_data (date, symbol, close_price) VALUES (?, ?, ?)",
                    records
                )
            logger.info(f"Saved price data for {len(data.columns)} symbols")
    
    def save_correlations(self, correlations: Dict[str, pd.DataFrame], window: int = 30):
        """상관관계 데이터 저장"""
        with sqlite3.connect(self.db_path) as conn:
            for group_name, corr_data in correlations.items():
                records = []
                
                for date_idx in corr_data.index.get_level_values(0).unique():
                    date_data = corr_data.loc[date_idx]
                    
                    for asset1 in date_data.index:
                        for asset2 in date_data.columns:
                            if asset1 != asset2:
                                corr_val = date_data.loc[asset1, asset2]
                                if pd.notna(corr_val):
                                    records.append((
                                        date_idx.strftime('%Y-%m-%d'),
                                        group_name,
                                        asset1,
                                        asset2,
                                        float(corr_val),
                                        window
                                    ))
                
                if records:
                    conn.executemany("""
                        INSERT OR REPLACE INTO correlations 
                        (date, group_name, asset1, asset2, correlation, window_days) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, records)
            
            logger.info(f"Saved correlations for {len(correlations)} groups")

def main():
    """메인 실행 함수 - 데이터 수집 및 분석"""
    fetcher = MarketDataFetcher()
    db_manager = DatabaseManager()
    
    # 데이터 수집
    logger.info("Starting data collection...")
    data = fetcher.fetch_data(period="2y")
    
    if data.empty:
        logger.error("No data collected. Exiting.")
        return
    
    # 데이터베이스 저장
    db_manager.save_price_data(data)
    
    # 상관관계 계산
    logger.info("Calculating correlations...")
    correlations = fetcher.calculate_correlations(data)
    db_manager.save_correlations(correlations)
    
    # 이상 패턴 감지
    logger.info("Detecting anomalies...")
    anomalies = fetcher.detect_anomalies(correlations)
    
    # 결과 출력
    for group_name, group_anomalies in anomalies.items():
        print(f"\n🔍 {group_name.upper()} Group Alerts:")
        for anomaly in group_anomalies:
            print(f"  • {anomaly['asset1']} ↔ {anomaly['asset2']}: "
                  f"{anomaly['correlation']:.3f} ({anomaly['strength']})")
    
    logger.info("Data pipeline completed successfully!")

if __name__ == "__main__":
    main()