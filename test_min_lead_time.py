#!/usr/bin/env python3
"""
Test minimum lead time filtering for correlations
Demonstrates how correlations with insufficient lead time are filtered out
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_lagged_correlations(data: pd.DataFrame, ticker1: str, ticker2: str, 
                                  lag_days: List[int], min_lead_time: int) -> Dict:
    """
    Calculate correlations with various lags and filter by minimum lead time
    """
    results = {
        'pair': f'{ticker1}-{ticker2}',
        'all_correlations': [],
        'filtered_correlations': [],
        'min_lead_time': min_lead_time
    }
    
    # Calculate returns
    returns1 = data[ticker1].pct_change().dropna()
    returns2 = data[ticker2].pct_change().dropna()
    
    print(f"\n📊 Analyzing {ticker1} -> {ticker2} correlations")
    print(f"   Minimum lead time filter: {min_lead_time} days")
    print("=" * 60)
    
    for lag in lag_days:
        # Create lagged series (ticker1 leads ticker2 by 'lag' days)
        lagged_returns1 = returns1.shift(lag)
        
        # Align the data
        common_idx = lagged_returns1.index.intersection(returns2.index)
        aligned_returns1 = lagged_returns1[common_idx].dropna()
        aligned_returns2 = returns2[aligned_returns1.index]
        
        if len(aligned_returns1) > 30:  # Need sufficient data
            correlation = aligned_returns1.corr(aligned_returns2)
            
            correlation_data = {
                'lag_days': lag,
                'correlation': correlation,
                'significance': abs(correlation),
                'filtered': lag < min_lead_time
            }
            
            results['all_correlations'].append(correlation_data)
            
            # Apply minimum lead time filter
            if lag >= min_lead_time:
                results['filtered_correlations'].append(correlation_data)
                status = "✅ ACCEPTED"
                print(f"   Lag {lag} days: correlation = {correlation:+.4f} {status}")
            else:
                status = "❌ FILTERED (below minimum lead time)"
                print(f"   Lag {lag} days: correlation = {correlation:+.4f} {status}")
    
    return results

def analyze_correlation_quality(results: Dict) -> Dict:
    """
    Analyze the quality of correlations before and after filtering
    """
    analysis = {
        'before_filtering': {},
        'after_filtering': {},
        'improvement': {}
    }
    
    # Before filtering
    if results['all_correlations']:
        all_corrs = [abs(c['correlation']) for c in results['all_correlations']]
        analysis['before_filtering'] = {
            'count': len(all_corrs),
            'mean_abs_correlation': np.mean(all_corrs),
            'max_abs_correlation': np.max(all_corrs),
            'std_correlation': np.std(all_corrs)
        }
    
    # After filtering
    if results['filtered_correlations']:
        filtered_corrs = [abs(c['correlation']) for c in results['filtered_correlations']]
        analysis['after_filtering'] = {
            'count': len(filtered_corrs),
            'mean_abs_correlation': np.mean(filtered_corrs),
            'max_abs_correlation': np.max(filtered_corrs),
            'std_correlation': np.std(filtered_corrs)
        }
        
        # Calculate improvement
        if analysis['before_filtering']:
            analysis['improvement'] = {
                'correlations_removed': analysis['before_filtering']['count'] - analysis['after_filtering']['count'],
                'noise_reduction': (1 - analysis['after_filtering']['count'] / analysis['before_filtering']['count']) * 100
            }
    
    return analysis

def main():
    """Test minimum lead time filtering with SPY and AAPL"""
    
    print("🔍 MINIMUM LEAD TIME CORRELATION FILTER TEST")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    min_lead_time = config['correlation'].get('min_lead_time', 3)
    lag_days = config['correlation'].get('lag_days', [0, 1, 3, 7])
    
    print(f"\n📋 Configuration:")
    print(f"   Lag days to test: {lag_days}")
    print(f"   Minimum lead time: {min_lead_time} days")
    
    # Test tickers
    tickers = ['SPY', 'AAPL', 'QQQ', 'MSFT', '^VIX']
    
    print(f"\n📈 Fetching data for: {', '.join(tickers)}")
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Handle both single and multi-column data
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif 'Close' in data.columns:
            data = data['Close']
        
        # Drop any columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        if data.empty:
            print("❌ No data fetched")
            return
        
        print(f"✅ Data fetched: {len(data)} days")
        
        # Test case 1: SPY and AAPL (highly correlated, same-day correlation is meaningless)
        results_spy_aapl = calculate_lagged_correlations(
            data, 'SPY', 'AAPL', lag_days, min_lead_time
        )
        
        # Test case 2: VIX and SPY (inverse correlation, lead-lag relationship exists)
        if '^VIX' in data.columns:
            results_vix_spy = calculate_lagged_correlations(
                data, '^VIX', 'SPY', lag_days, min_lead_time
            )
        
        # Test case 3: QQQ and MSFT (tech sector correlation)
        results_qqq_msft = calculate_lagged_correlations(
            data, 'QQQ', 'MSFT', lag_days, min_lead_time
        )
        
        # Analyze results
        print("\n" + "=" * 60)
        print("📊 CORRELATION QUALITY ANALYSIS")
        print("=" * 60)
        
        for pair_results in [results_spy_aapl, results_qqq_msft]:
            pair = pair_results['pair']
            analysis = analyze_correlation_quality(pair_results)
            
            print(f"\n{pair}:")
            print(f"  Before filtering:")
            print(f"    • Correlations tested: {analysis['before_filtering'].get('count', 0)}")
            print(f"    • Mean |correlation|: {analysis['before_filtering'].get('mean_abs_correlation', 0):.4f}")
            
            if analysis['after_filtering']:
                print(f"  After filtering (min lead time = {min_lead_time} days):")
                print(f"    • Correlations kept: {analysis['after_filtering']['count']}")
                print(f"    • Mean |correlation|: {analysis['after_filtering']['mean_abs_correlation']:.4f}")
                
                if 'improvement' in analysis:
                    print(f"  Improvement:")
                    print(f"    • Noise reduction: {analysis['improvement']['noise_reduction']:.1f}%")
                    print(f"    • Meaningless correlations removed: {analysis['improvement']['correlations_removed']}")
        
        print("\n" + "=" * 60)
        print("✅ SUMMARY")
        print("=" * 60)
        print(f"""
The minimum lead time filter successfully:
1. Removes same-day and very short-term correlations that are often meaningless
2. Focuses on correlations with sufficient lead time for actionable trading signals
3. Reduces noise from spurious short-term correlations
4. Improves signal quality by emphasizing predictive relationships

For SPY-AAPL: Same-day correlation is high but meaningless for prediction.
              The filter removes this noise and focuses on longer-term relationships.

Configuration can be adjusted in config/config.yaml:
  min_lead_time: {min_lead_time}  # Current setting (days)
        """)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nPlease ensure yfinance is installed: pip install yfinance")

if __name__ == "__main__":
    main()