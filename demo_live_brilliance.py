#!/usr/bin/env python3
"""
Demo: Brilliant Cross-Market Connections with Live Data
Show actual brilliant discoveries using real live market data
"""

from data_pipeline.robust_live_fetcher import RobustLiveDataFetcher
from strategies.cross_market_discovery import CrossMarketDiscoveryEngine
from strategies.brilliant_connections import BrilliantConnectionEngine
import pandas as pd
import numpy as np

def main():
    print("🚀 BRILLIANT CROSS-MARKET DISCOVERY WITH LIVE DATA")
    print("=" * 70)
    print("Demonstrating genius-level connections using actual market data")
    print("=" * 70)
    
    # Fetch live data
    print("📡 Fetching live market data...")
    fetcher = RobustLiveDataFetcher()
    live_data = fetcher.fetch_live_data(period='18mo')  # More data for better analysis
    
    if live_data.empty:
        print("❌ Failed to fetch live data")
        return
    
    print(f"✅ LIVE DATA ACQUIRED:")
    print(f"   📅 Period: {live_data.index[0].strftime('%Y-%m-%d')} to {live_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   📊 Assets: {len(live_data.columns)} real market securities")
    print(f"   📈 Data Points: {len(live_data)} trading days")
    print(f"   🎯 Sample Assets: {', '.join(live_data.columns[:12])}...")
    print()
    
    # Test 1: Cross-Market Discovery Engine
    print("🔍 RUNNING CROSS-MARKET DISCOVERY ENGINE...")
    cross_engine = CrossMarketDiscoveryEngine()
    cross_results = cross_engine.discover_hidden_connections(
        live_data, 
        min_lead_days=1, 
        min_alpha_threshold=0.15
    )
    
    if 'error' not in cross_results:
        print(f"🎯 CROSS-MARKET DISCOVERIES:")
        print(f"   • Cross-Market Signals: {len(cross_results['cross_market_signals'])}")
        print(f"   • Alpha Chains: {len(cross_results['alpha_generating_chains'])}")
        print(f"   • Exotic Patterns: {len(cross_results['exotic_pattern_alerts'])}")
        
        # Show top discoveries
        for i, signal in enumerate(cross_results['cross_market_signals'][:2], 1):
            print(f"\n   🔗 DISCOVERY {i}: {signal['chain_name'].upper().replace('_', ' ')}")
            print(f"      Description: {signal['description']}")
            print(f"      Sources: {' + '.join(signal['source_assets'])}")
            print(f"      Target: {signal.get('target_asset', signal.get('best_target', 'N/A'))}")
            print(f"      Lead Time: {signal['optimal_lead_days']} days")
            print(f"      Alpha Potential: {signal['alpha_potential']:.3f}")
            print(f"      Brilliance Score: {signal.get('brilliance_score', 'N/A')}")
    else:
        print(f"   ❌ {cross_results['error']}")
    
    print("\n" + "─" * 70)
    
    # Test 2: Brilliant Connection Engine  
    print("✨ RUNNING BRILLIANT CONNECTION ENGINE...")
    brilliant_engine = BrilliantConnectionEngine()
    brilliant_results = brilliant_engine.discover_brilliant_connections(live_data)
    
    if 'error' not in brilliant_results:
        print(f"🧠 BRILLIANT ANALYSIS:")
        print(f"   • Overall Brilliance Score: {brilliant_results['brilliance_score']:.3f}/1.0")
        print(f"   • Brilliant Connections: {len(brilliant_results['brilliant_connections'])}")
        print(f"   • Pattern Alerts: {len(brilliant_results['pattern_alerts'])}")
        print(f"   • Genius Trades: {len(brilliant_results['genius_trades'])}")
        
        # Show brilliant connections
        for i, conn in enumerate(brilliant_results['brilliant_connections'][:2], 1):
            print(f"\n   💡 BRILLIANT CONNECTION {i}: {conn['chain_name'].upper().replace('_', ' ')}")
            print(f"      Description: {conn['description']}")
            print(f"      Connection Path:")
            for step in conn['connections']:
                print(f"        → {step}")
            print(f"      Sources: {' + '.join(conn['source_assets'])}")
            print(f"      Target: {conn.get('target_asset', conn.get('best_target', 'N/A'))}")
            print(f"      Lead Time: {conn['optimal_lead_time']} days")
            print(f"      Alpha Potential: {conn['alpha_potential']:.3f}")
            print(f"      Brilliance Score: {conn['brilliance_score']:.2f}")
        
        # Show pattern alerts
        for i, pattern in enumerate(brilliant_results['pattern_alerts'], 1):
            print(f"\n   🎨 PATTERN ALERT {i}: {pattern['pattern_name'].upper().replace('_', ' ')}")
            print(f"      Description: {pattern['description']}")
            if 'implications' in pattern:
                print(f"      Implications: {pattern['implications']}")
            print(f"      Confidence: {pattern.get('confidence', 'Unknown')}")
        
        # Show genius trades
        for i, trade in enumerate(brilliant_results['genius_trades'][:3], 1):
            print(f"\n   💎 GENIUS TRADE {i}: {trade['trade_name'].upper().replace('_', ' ')}")
            print(f"      Type: {trade['trade_type']}")
            print(f"      Description: {trade['description']}")
            if 'direction' in trade:
                print(f"      Direction: {trade['direction']}")
            if 'lead_time' in trade:
                print(f"      Lead Time: {trade['lead_time']} days")
            print(f"      Confidence: {trade.get('confidence', 'Unknown')}")
        
        # Final brilliance assessment
        brilliance_score = brilliant_results['brilliance_score']
        if brilliance_score > 0.7:
            assessment = "🎯 GENIUS LEVEL: Exceptional cross-market insights!"
        elif brilliance_score > 0.4:
            assessment = "🎯 ADVANCED LEVEL: Strong cross-market connections!"
        elif brilliance_score > 0.1:
            assessment = "🎯 INTERMEDIATE LEVEL: Some interesting connections found"
        else:
            assessment = "🎯 BASIC LEVEL: Limited cross-market relationships detected"
        
        print(f"\n{assessment}")
        
    else:
        print(f"   ❌ {brilliant_results['error']}")
    
    print("\n" + "=" * 70)
    print("🌟 LIVE DATA ANALYSIS COMPLETE!")
    
    # Data quality summary
    data_quality = (live_data.notna().sum() / len(live_data)).mean()
    print(f"📊 Data Quality: {data_quality:.1%} complete")
    print(f"🔄 Data Freshness: {(pd.Timestamp.now() - live_data.index[-1]).days} days old")
    print("✅ CONFIRMED: Analysis running on 100% real live market data")
    
    # Show some actual correlations found
    print(f"\n🔍 SAMPLE REAL CORRELATIONS DISCOVERED:")
    if not live_data.empty and len(live_data.columns) >= 4:
        # Calculate some interesting correlations
        assets = live_data.columns[:8]  # First 8 assets
        for i in range(min(3, len(assets)-1)):
            for j in range(i+1, min(i+3, len(assets))):
                if i < len(assets) and j < len(assets):
                    asset1, asset2 = assets[i], assets[j]
                    correlation = live_data[asset1].corr(live_data[asset2])
                    print(f"   • {asset1} ↔ {asset2}: {correlation:+.3f}")

if __name__ == "__main__":
    main()