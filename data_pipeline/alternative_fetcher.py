#!/usr/bin/env python3
"""
Alternative data fetcher using multiple sources for real market data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging

logger = logging.getLogger(__name__)

class AlternativeDataFetcher:
    def __init__(self):
        self.base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'fmp': 'https://financialmodelingprep.com/api/v3',
            'yahoo_scrape': 'https://query1.finance.yahoo.com/v8/finance/chart'
        }
        
    def fetch_yahoo_direct(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Direct Yahoo Finance API call"""
        try:
            # Convert period to seconds
            period_map = {'1d': 86400, '5d': 432000, '1mo': 2592000, '3mo': 7776000, '6mo': 15552000, '1y': 31536000, '2y': 63072000}
            period_seconds = period_map.get(period, 31536000)
            
            end_time = int(time.time())
            start_time = end_time - period_seconds
            
            url = f"{self.base_urls['yahoo_scrape']}/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div%7Csplit'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    
                    if 'timestamp' in result and 'indicators' in result:
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                        closes = quotes.get('close', [])
                        
                        if closes and len(dates) == len(closes):
                            df = pd.DataFrame({
                                'Close': closes
                            }, index=pd.to_datetime(dates))
                            
                            df = df.dropna()
                            logger.info(f"✓ {symbol}: {len(df)} records fetched")
                            return df
            
            logger.warning(f"✗ {symbol}: Failed to fetch data")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"✗ {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_sources(self) -> pd.DataFrame:
        """Fetch data from multiple sources"""
        logger.info("🔄 Fetching real market data from alternative sources...")
        
        # Key symbols with alternative tickers
        symbols_map = {
            # REITs and Rates
            'VNQ': 'VNQ',           # REITs ETF
            'IEF': 'IEF',           # 7-10 Year Treasury ETF (proxy for rates)
            
            # Major indices and tech
            'SPY': 'SPY',           # S&P 500
            'QQQ': 'QQQ',           # Nasdaq 100
            'VTI': 'VTI',           # Total Stock Market
            
            # Individual tech stocks
            'AAPL': 'AAPL',         # Apple
            'MSFT': 'MSFT',         # Microsoft
            'GOOGL': 'GOOGL',       # Google
            
            # Volatility and alternatives
            'GLD': 'GLD',           # Gold ETF
            'TLT': 'TLT',           # Long-term Treasury
        }
        
        data = {}
        
        for name, symbol in symbols_map.items():
            try:
                df = self.fetch_yahoo_direct(symbol)
                if not df.empty:
                    data[name] = df['Close']
                    time.sleep(0.5)  # Rate limiting
                else:
                    logger.warning(f"No data for {name} ({symbol})")
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
        
        if data:
            combined_df = pd.DataFrame(data)
            combined_df = combined_df.dropna()
            logger.info(f"✅ Successfully fetched data for {len(combined_df.columns)} assets")
            return combined_df
        else:
            logger.error("❌ No data fetched from any source")
            return pd.DataFrame()

def generate_real_insights(data: pd.DataFrame) -> dict:
    """Generate real market insights from actual data"""
    
    if data.empty:
        return {"error": "No data available for analysis"}
    
    insights = {
        "data_summary": {
            "assets_count": len(data.columns),
            "date_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            "total_days": len(data),
            "last_update": data.index[-1].strftime('%Y-%m-%d')
        },
        "correlation_insights": {},
        "performance_insights": {},
        "volatility_insights": {},
        "risk_insights": {}
    }
    
    # Correlation Analysis
    corr_matrix = data.corr()
    
    # Find strongest correlations
    correlations = []
    for i in range(len(data.columns)):
        for j in range(i+1, len(data.columns)):
            asset1, asset2 = data.columns[i], data.columns[j]
            corr_val = corr_matrix.loc[asset1, asset2]
            correlations.append({
                'pair': f"{asset1}-{asset2}",
                'correlation': corr_val,
                'strength': abs(corr_val)
            })
    
    # Sort by strength
    correlations.sort(key=lambda x: x['strength'], reverse=True)
    
    insights["correlation_insights"] = {
        "strongest_positive": [c for c in correlations if c['correlation'] > 0][:3],
        "strongest_negative": [c for c in correlations if c['correlation'] < 0][:3],
        "average_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    }
    
    # Performance Analysis
    returns_1d = data.pct_change().iloc[-1] * 100
    returns_7d = (data.iloc[-1] / data.iloc[-8] - 1) * 100
    returns_30d = (data.iloc[-1] / data.iloc[-31] - 1) * 100
    
    insights["performance_insights"] = {
        "best_1d": {"asset": returns_1d.idxmax(), "return": returns_1d.max()},
        "worst_1d": {"asset": returns_1d.idxmin(), "return": returns_1d.min()},
        "best_7d": {"asset": returns_7d.idxmax(), "return": returns_7d.max()},
        "worst_7d": {"asset": returns_7d.idxmin(), "return": returns_7d.min()},
        "best_30d": {"asset": returns_30d.idxmax(), "return": returns_30d.max()},
        "worst_30d": {"asset": returns_30d.idxmin(), "return": returns_30d.min()},
    }
    
    # Volatility Analysis
    volatility = data.pct_change().std() * np.sqrt(252) * 100  # Annualized
    
    insights["volatility_insights"] = {
        "most_volatile": {"asset": volatility.idxmax(), "volatility": volatility.max()},
        "least_volatile": {"asset": volatility.idxmin(), "volatility": volatility.min()},
        "average_volatility": volatility.mean()
    }
    
    # Risk Analysis
    # Calculate rolling correlations to detect regime changes
    rolling_corr_changes = {}
    for i, asset1 in enumerate(data.columns):
        for j, asset2 in enumerate(data.columns[i+1:], i+1):
            short_corr = data[asset1].tail(30).corr(data[asset2].tail(30))
            long_corr = data[asset1].tail(90).corr(data[asset2].tail(90))
            change = short_corr - long_corr
            
            if abs(change) > 0.3:  # Significant correlation change
                rolling_corr_changes[f"{asset1}-{asset2}"] = {
                    "short_term_corr": short_corr,
                    "long_term_corr": long_corr,
                    "change": change
                }
    
    insights["risk_insights"] = {
        "correlation_regime_changes": rolling_corr_changes,
        "diversification_ratio": len([c for c in correlations if abs(c['correlation']) < 0.5]) / len(correlations)
    }
    
    # Generate actionable insights
    insights["actionable_insights"] = generate_actionable_insights(insights, data)
    
    return insights

def generate_actionable_insights(insights: dict, data: pd.DataFrame) -> list:
    """Generate actionable trading/investment insights"""
    
    actionable = []
    
    # Correlation insights
    corr_insights = insights["correlation_insights"]
    
    if corr_insights["average_correlation"] > 0.7:
        actionable.append({
            "type": "RISK_WARNING",
            "message": f"⚠️ HIGH CORRELATION REGIME: Average correlation is {corr_insights['average_correlation']:.3f}. Diversification benefits reduced.",
            "action": "Consider alternative assets or hedging strategies"
        })
    
    # Performance insights
    perf = insights["performance_insights"]
    
    if abs(perf["best_1d"]["return"]) > 5:
        actionable.append({
            "type": "MOMENTUM",
            "message": f"🚀 STRONG MOMENTUM: {perf['best_1d']['asset']} up {perf['best_1d']['return']:.2f}% today",
            "action": "Monitor for continuation or reversal patterns"
        })
    
    if abs(perf["worst_1d"]["return"]) > 5:
        actionable.append({
            "type": "OVERSOLD",
            "message": f"📉 POTENTIAL OVERSOLD: {perf['worst_1d']['asset']} down {perf['worst_1d']['return']:.2f}% today",
            "action": "Check fundamentals for potential buying opportunity"
        })
    
    # Volatility insights
    vol = insights["volatility_insights"]
    
    if vol["average_volatility"] > 25:
        actionable.append({
            "type": "HIGH_VOLATILITY",
            "message": f"⚡ HIGH VOLATILITY ENVIRONMENT: Average volatility at {vol['average_volatility']:.1f}%",
            "action": "Consider reducing position sizes or using volatility strategies"
        })
    
    # Risk regime changes
    risk = insights["risk_insights"]
    
    if risk["correlation_regime_changes"]:
        for pair, change_info in list(risk["correlation_regime_changes"].items())[:3]:
            if change_info["change"] > 0.3:
                actionable.append({
                    "type": "CORRELATION_SHIFT",
                    "message": f"🔄 CORRELATION INCREASE: {pair} correlation jumped from {change_info['long_term_corr']:.3f} to {change_info['short_term_corr']:.3f}",
                    "action": "Reassess portfolio diversification"
                })
            elif change_info["change"] < -0.3:
                actionable.append({
                    "type": "CORRELATION_SHIFT", 
                    "message": f"📈 CORRELATION DECREASE: {pair} correlation dropped from {change_info['long_term_corr']:.3f} to {change_info['short_term_corr']:.3f}",
                    "action": "Potential diversification opportunity"
                })
    
    # Add market regime assessment
    recent_performance = [perf[f"best_{period}"]["return"] for period in ["1d", "7d", "30d"]]
    if all(ret > 0 for ret in recent_performance):
        actionable.append({
            "type": "BULL_MARKET",
            "message": "🐂 BULLISH MOMENTUM: Consistent positive performance across timeframes",
            "action": "Consider momentum strategies but watch for overextension"
        })
    elif all(ret < 0 for ret in recent_performance):
        actionable.append({
            "type": "BEAR_MARKET", 
            "message": "🐻 BEARISH TREND: Consistent negative performance across timeframes",
            "action": "Consider defensive positioning or contrarian opportunities"
        })
    
    return actionable

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Fetching Real Market Data & Generating Insights...")
    print("=" * 60)
    
    # Fetch real data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ Unable to fetch real market data")
        return
    
    # Generate insights
    insights = generate_real_insights(data)
    
    # Print insights
    print(f"\n📊 REAL MARKET INSIGHTS")
    print("=" * 60)
    
    # Data summary
    summary = insights["data_summary"]
    print(f"📈 Data Coverage: {summary['assets_count']} assets, {summary['total_days']} days")
    print(f"📅 Period: {summary['date_range']}")
    print(f"🔄 Last Update: {summary['last_update']}")
    
    # Correlation insights
    print(f"\n🔗 CORRELATION ANALYSIS")
    corr = insights["correlation_insights"]
    print(f"Average Market Correlation: {corr['average_correlation']:.3f}")
    
    print(f"\n🔴 Strongest Positive Correlations:")
    for item in corr["strongest_positive"][:3]:
        print(f"   {item['pair']}: {item['correlation']:.3f}")
    
    print(f"\n🔵 Strongest Negative Correlations:")
    for item in corr["strongest_negative"][:3]:
        print(f"   {item['pair']}: {item['correlation']:.3f}")
    
    # Performance insights
    print(f"\n📈 PERFORMANCE ANALYSIS")
    perf = insights["performance_insights"]
    print(f"Best 1D: {perf['best_1d']['asset']} (+{perf['best_1d']['return']:.2f}%)")
    print(f"Worst 1D: {perf['worst_1d']['asset']} ({perf['worst_1d']['return']:.2f}%)")
    print(f"Best 7D: {perf['best_7d']['asset']} (+{perf['best_7d']['return']:.2f}%)")
    print(f"Best 30D: {perf['best_30d']['asset']} (+{perf['best_30d']['return']:.2f}%)")
    
    # Volatility insights
    print(f"\n⚡ VOLATILITY ANALYSIS")
    vol = insights["volatility_insights"]
    print(f"Most Volatile: {vol['most_volatile']['asset']} ({vol['most_volatile']['volatility']:.1f}%)")
    print(f"Least Volatile: {vol['least_volatile']['asset']} ({vol['least_volatile']['volatility']:.1f}%)")
    print(f"Average Volatility: {vol['average_volatility']:.1f}%")
    
    # Actionable insights
    print(f"\n💡 ACTIONABLE INSIGHTS")
    print("=" * 60)
    
    for insight in insights["actionable_insights"]:
        print(f"\n{insight['type']}: {insight['message']}")
        print(f"   → Action: {insight['action']}")
    
    # Risk assessment
    risk = insights["risk_insights"]
    if risk["correlation_regime_changes"]:
        print(f"\n🚨 CORRELATION REGIME CHANGES DETECTED:")
        for pair, change in list(risk["correlation_regime_changes"].items())[:3]:
            print(f"   {pair}: {change['long_term_corr']:.3f} → {change['short_term_corr']:.3f} (Δ{change['change']:+.3f})")
    
    print(f"\nDiversification Ratio: {risk['diversification_ratio']:.2f} (Higher is better)")
    
    print(f"\n✅ Real market analysis complete!")

if __name__ == "__main__":
    main()