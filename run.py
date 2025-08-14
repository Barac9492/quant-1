#!/usr/bin/env python3
"""
Market Correlation Analysis System
실행 스크립트
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """필요한 패키지 설치"""
    logger.info("Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        logger.info("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install dependencies: {e.stderr}")
        return False

def run_data_collection():
    """데이터 수집 실행"""
    logger.info("Running data collection...")
    try:
        from data_pipeline.fetchers import main as fetch_main
        fetch_main()
        logger.info("✓ Data collection completed")
        return True
    except Exception as e:
        logger.error(f"✗ Data collection failed: {e}")
        return False

def run_hypothesis_testing():
    """가설 테스트 실행"""
    logger.info("Running hypothesis testing...")
    try:
        from data_pipeline.fetchers import MarketDataFetcher
        from strategies.hypothesis_engine import HypothesisEngine
        
        fetcher = MarketDataFetcher()
        hypothesis_engine = HypothesisEngine()
        
        # 데이터 로드
        data = fetcher.fetch_data(period="1y")
        if data.empty:
            logger.error("No data available for hypothesis testing")
            return False
        
        # 가설 테스트 실행
        results = hypothesis_engine.test_all_hypotheses(data)
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 HYPOTHESIS TESTING RESULTS")
        print("="*60)
        
        for hypothesis_name, result in results.items():
            if "error" in result:
                print(f"\n❌ {hypothesis_name}: {result['error']}")
                continue
            
            print(f"\n🧪 {result.get('hypothesis', hypothesis_name)}")
            print(f"   Signal: {result.get('signal', 'No signal')}")
            print(f"   Confidence: {result.get('confidence', 'Unknown')}")
            
            # 추가 세부사항
            if hypothesis_name == "reits_rates":
                print(f"   Current Correlation: {result.get('current_correlation', 'N/A')}")
                print(f"   REITs 7d Change: {result.get('recent_reits_change_pct', 0):.2f}%")
            elif hypothesis_name == "dollar_yen_bitcoin":
                print(f"   Best Lag Correlation: {max(result.get('lag_correlations', {}).values(), default=0):.3f}")
                print(f"   BTC Volatility: {result.get('current_btc_volatility', 0):.1f}%")
            elif hypothesis_name == "vix_tech":
                print(f"   Current VIX: {result.get('current_vix', 0):.2f}")
                print(f"   Condition: {result.get('current_condition', 'Normal')}")
        
        logger.info("✓ Hypothesis testing completed")
        return True
    except Exception as e:
        logger.error(f"✗ Hypothesis testing failed: {e}")
        return False

def run_dashboard():
    """Streamlit 대시보드 실행"""
    logger.info("Starting dashboard...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/main_dashboard.py", 
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to start dashboard: {e}")
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")

def run_monitoring():
    """모니터링 시스템 실행"""
    logger.info("Running monitoring cycle...")
    try:
        import asyncio
        from data_pipeline.alert_system import AlertMonitor
        
        monitor = AlertMonitor()
        asyncio.run(monitor.run_monitoring_cycle())
        
        logger.info("✓ Monitoring cycle completed")
        return True
    except Exception as e:
        logger.error(f"✗ Monitoring failed: {e}")
        return False

def setup_project():
    """프로젝트 초기 설정"""
    logger.info("Setting up project...")
    
    # 디렉토리 생성
    directories = ["database", "logs", "exports"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # 의존성 설치
    if not install_dependencies():
        return False
    
    # 초기 데이터 수집
    if not run_data_collection():
        logger.warning("Initial data collection failed, but continuing...")
    
    logger.info("✓ Project setup completed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Market Correlation Analysis System")
    parser.add_argument("command", choices=[
        "setup", "collect", "test", "dashboard", "monitor", "all"
    ], help="Command to execute")
    parser.add_argument("--install", action="store_true", 
                       help="Install dependencies first")
    
    args = parser.parse_args()
    
    print("""
🔗 Market Correlation Analysis System
=====================================
    """)
    
    # 의존성 설치 (옵션)
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    # 명령 실행
    success = True
    
    if args.command == "setup":
        success = setup_project()
    
    elif args.command == "collect":
        success = run_data_collection()
    
    elif args.command == "test":
        success = run_hypothesis_testing()
    
    elif args.command == "dashboard":
        run_dashboard()  # 이 함수는 blocking이므로 success 체크 불필요
    
    elif args.command == "monitor":
        success = run_monitoring()
    
    elif args.command == "all":
        logger.info("Running full pipeline...")
        success &= setup_project()
        success &= run_data_collection() 
        success &= run_hypothesis_testing()
        
        if success:
            logger.info("🚀 Full pipeline completed! Starting dashboard...")
            run_dashboard()
    
    if not success:
        sys.exit(1)
    
    print("\n✅ Command completed successfully!")

if __name__ == "__main__":
    main()