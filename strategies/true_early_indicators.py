#!/usr/bin/env python3
"""
True Early Indicator Finder - Focus on meaningful leading relationships
Built to find actual predictive signals, not just correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class TrueEarlyIndicatorFinder:
    """
    Find true early indicators that actually lead target assets by meaningful time periods
    Focus on cross-asset class relationships and macro-driven signals
    """
    
    def __init__(self):
        # Known leading indicator patterns with typical lead times
        self.macro_indicators = {
            # Volatility indicators - typically lead equity moves by 1-5 days
            'VIX': {
                'leads': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'VTI'],
                'typical_leads': [1, 2, 3, 5],
                'relationship': 'inverse',
                'strength_threshold': 0.4,
                'description': 'Fear index leads equity sell-offs'
            },
            
            # Interest rate indicators - lead rate-sensitive sectors by 3-10 days
            'TLT': {
                'leads': ['VNQ', 'XLF', 'utilities', 'dividend_stocks'],
                'typical_leads': [3, 5, 7, 10],
                'relationship': 'varies',
                'strength_threshold': 0.35,
                'description': 'Bond moves signal rate expectations'
            },
            
            # Currency indicators - lead international and commodity exposure by 5-15 days
            'DXY': {
                'leads': ['GLD', 'EFA', 'EEM', 'commodities'],
                'typical_leads': [5, 7, 10, 15],
                'relationship': 'inverse_to_commodities',
                'strength_threshold': 0.3,
                'description': 'Dollar strength affects international assets'
            },
            
            # Credit indicators - lead risk assets by 2-7 days
            'credit_spreads': {
                'leads': ['SPY', 'QQQ', 'high_beta_stocks'],
                'typical_leads': [2, 3, 5, 7],
                'relationship': 'inverse',
                'strength_threshold': 0.4,
                'description': 'Credit stress precedes equity weakness'
            },
            
            # Sector rotation indicators - lead other sectors by 3-10 days
            'XLF': {
                'leads': ['SPY', 'cyclical_sectors'],
                'typical_leads': [3, 5, 7, 10],
                'relationship': 'positive',
                'strength_threshold': 0.35,
                'description': 'Financial sector leads economic cycle'
            }
        }
        
        # Cross-asset class leading patterns
        self.cross_asset_patterns = {
            'rates_to_reits': {
                'leader': ['TLT', 'IEF'],
                'follower': ['VNQ'],
                'expected_lead': [5, 7, 10],
                'relationship': 'inverse'
            },
            'vix_to_tech': {
                'leader': ['VIX'],
                'follower': ['QQQ', 'ARKK', 'XLK'],
                'expected_lead': [1, 2, 3],
                'relationship': 'inverse'
            },
            'dollar_to_gold': {
                'leader': ['DXY'],
                'follower': ['GLD', 'SLV'],
                'expected_lead': [7, 10, 15],
                'relationship': 'inverse'
            },
            'bonds_to_financials': {
                'leader': ['TLT'],
                'follower': ['XLF'],
                'expected_lead': [3, 5, 7],
                'relationship': 'inverse'
            }
        }
    
    def find_true_leading_indicators(self, target_asset: str, data: pd.DataFrame, 
                                   min_lead_days: int = 1, max_lead_days: int = 20,
                                   min_correlation: float = 0.25) -> Dict:
        """
        Find true leading indicators that actually predict target asset moves
        
        Args:
            target_asset: Asset to find predictors for
            data: Market data DataFrame
            min_lead_days: Minimum lead time required (default 1 day)
            max_lead_days: Maximum lead time to search
            min_correlation: Minimum correlation for significance
            
        Returns:
            Dictionary with leading indicators and their predictive power
        """
        
        if target_asset not in data.columns:
            return {'error': f'Target asset {target_asset} not found in data'}
        
        if len(data) < 60:
            return {'error': 'Need at least 60 days of data for reliable lead-lag analysis'}
        
        results = {
            'target_asset': target_asset,
            'analysis_timestamp': datetime.now().isoformat(),
            'macro_leading_indicators': [],
            'cross_asset_leading_indicators': [],
            'sector_leading_indicators': [],
            'volatility_leading_indicators': [],
            'predictive_summary': {},
            'trading_signals': []
        }
        
        # 1. Macro leading indicators
        macro_indicators = self._find_macro_leading_indicators(target_asset, data, min_lead_days, max_lead_days, min_correlation)
        results['macro_leading_indicators'] = macro_indicators
        
        # 2. Cross-asset class indicators
        cross_asset = self._find_cross_asset_indicators(target_asset, data, min_lead_days, max_lead_days, min_correlation)
        results['cross_asset_leading_indicators'] = cross_asset
        
        # 3. Sector leading indicators
        sector_indicators = self._find_sector_leading_indicators(target_asset, data, min_lead_days, max_lead_days, min_correlation)
        results['sector_leading_indicators'] = sector_indicators
        
        # 4. Volatility-based indicators
        vol_indicators = self._find_volatility_leading_indicators(target_asset, data, min_lead_days, max_lead_days)
        results['volatility_leading_indicators'] = vol_indicators
        
        # 5. Generate predictive summary
        results['predictive_summary'] = self._generate_predictive_summary(results)
        
        # 6. Current trading signals
        results['trading_signals'] = self._generate_current_trading_signals(target_asset, data, results)
        
        return results
    
    def _find_macro_leading_indicators(self, target: str, data: pd.DataFrame, 
                                     min_lead: int, max_lead: int, min_corr: float) -> List[Dict]:
        """Find macro indicators that lead the target asset"""
        
        macro_indicators = []
        
        for indicator_name, indicator_info in self.macro_indicators.items():
            if indicator_name not in data.columns:
                continue
            
            # Check if target is in the typical followers list
            if not any(follower in target for follower in indicator_info['leads']):
                continue
            
            # Test lead-lag relationship
            best_lead_result = self._test_lead_lag_relationship(
                data[indicator_name], data[target], 
                min_lead, max_lead, min_corr
            )
            
            if best_lead_result and best_lead_result['lead_days'] >= min_lead:
                # Calculate predictive power
                predictive_power = self._calculate_predictive_power(
                    data[indicator_name], data[target], best_lead_result['lead_days']
                )
                
                macro_indicators.append({
                    'indicator': indicator_name,
                    'target': target,
                    'lead_days': best_lead_result['lead_days'],
                    'correlation': best_lead_result['correlation'],
                    'p_value': best_lead_result['p_value'],
                    'relationship_type': indicator_info['relationship'],
                    'predictive_power': predictive_power,
                    'description': indicator_info['description'],
                    'significance': 'HIGH' if predictive_power > 0.6 else 'MEDIUM' if predictive_power > 0.4 else 'LOW'
                })
        
        # Sort by predictive power
        return sorted(macro_indicators, key=lambda x: x['predictive_power'], reverse=True)
    
    def _find_cross_asset_indicators(self, target: str, data: pd.DataFrame,
                                   min_lead: int, max_lead: int, min_corr: float) -> List[Dict]:
        """Find cross-asset class leading indicators"""
        
        cross_asset_indicators = []
        
        for pattern_name, pattern_info in self.cross_asset_patterns.items():
            # Check if target matches follower pattern
            if not any(follower in target for follower in pattern_info['follower']):
                continue
            
            # Test each leader in the pattern
            for leader_asset in pattern_info['leader']:
                if leader_asset not in data.columns:
                    continue
                
                # Focus on expected lead times for this pattern
                best_result = None
                best_power = 0
                
                for expected_lead in pattern_info['expected_lead']:
                    if expected_lead >= min_lead and expected_lead <= max_lead:
                        result = self._test_specific_lead_lag(
                            data[leader_asset], data[target], expected_lead, min_corr
                        )
                        
                        if result:
                            power = self._calculate_predictive_power(
                                data[leader_asset], data[target], expected_lead
                            )
                            
                            if power > best_power:
                                best_power = power
                                best_result = {
                                    'lead_days': expected_lead,
                                    'correlation': result['correlation'],
                                    'p_value': result['p_value']
                                }
                
                if best_result and best_power > 0.3:  # Minimum predictive threshold
                    cross_asset_indicators.append({
                        'indicator': leader_asset,
                        'target': target,
                        'pattern': pattern_name,
                        'lead_days': best_result['lead_days'],
                        'correlation': best_result['correlation'],
                        'p_value': best_result['p_value'],
                        'relationship_type': pattern_info['relationship'],
                        'predictive_power': best_power,
                        'description': f"{leader_asset} leads {target} in {pattern_name} pattern",
                        'significance': 'HIGH' if best_power > 0.5 else 'MEDIUM'
                    })
        
        return sorted(cross_asset_indicators, key=lambda x: x['predictive_power'], reverse=True)
    
    def _find_sector_leading_indicators(self, target: str, data: pd.DataFrame,
                                      min_lead: int, max_lead: int, min_corr: float) -> List[Dict]:
        """Find sector indicators that lead the target"""
        
        sector_indicators = []
        
        # Define sector ETFs and their leadership patterns
        sector_leaders = {
            'XLF': {'leads': ['SPY', 'QQQ', 'financials'], 'description': 'Financials lead market'},
            'XLE': {'leads': ['energy', 'commodities'], 'description': 'Energy leads commodity cycle'},
            'XLK': {'leads': ['tech_stocks', 'growth'], 'description': 'Tech leads growth'},
            'XLU': {'leads': ['defensive', 'dividends'], 'description': 'Utilities lead defensive rotation'}
        }
        
        for sector_etf, sector_info in sector_leaders.items():
            if sector_etf not in data.columns or sector_etf == target:
                continue
            
            # Test if this sector leads the target
            best_lead_result = self._test_lead_lag_relationship(
                data[sector_etf], data[target], min_lead, max_lead, min_corr
            )
            
            if best_lead_result and best_lead_result['lead_days'] >= min_lead:
                predictive_power = self._calculate_predictive_power(
                    data[sector_etf], data[target], best_lead_result['lead_days']
                )
                
                if predictive_power > 0.3:
                    sector_indicators.append({
                        'indicator': sector_etf,
                        'target': target,
                        'lead_days': best_lead_result['lead_days'],
                        'correlation': best_lead_result['correlation'],
                        'p_value': best_lead_result['p_value'],
                        'predictive_power': predictive_power,
                        'description': sector_info['description'],
                        'significance': 'HIGH' if predictive_power > 0.5 else 'MEDIUM'
                    })
        
        return sorted(sector_indicators, key=lambda x: x['predictive_power'], reverse=True)
    
    def _find_volatility_leading_indicators(self, target: str, data: pd.DataFrame,
                                          min_lead: int, max_lead: int) -> List[Dict]:
        """Find volatility-based leading indicators"""
        
        vol_indicators = []
        
        # Calculate rolling volatilities
        target_vol = data[target].pct_change().rolling(10).std()
        
        # Test VIX if available
        if 'VIX' in data.columns:
            vix_data = data['VIX']
            
            # Test VIX levels leading volatility changes
            for lead_days in range(min_lead, min(max_lead + 1, 10)):
                try:
                    vix_lagged = vix_data.shift(lead_days)
                    vol_current = target_vol
                    
                    common_idx = vix_lagged.index.intersection(vol_current.index)
                    if len(common_idx) > 30:
                        vix_aligned = vix_lagged[common_idx].dropna()
                        vol_aligned = vol_current[vix_aligned.index]
                        
                        if len(vix_aligned) > 20:
                            corr, p_val = stats.pearsonr(vix_aligned, vol_aligned)
                            
                            if not np.isnan(corr) and p_val < 0.05 and abs(corr) > 0.3:
                                vol_indicators.append({
                                    'indicator': 'VIX',
                                    'target': f"{target}_volatility",
                                    'lead_days': lead_days,
                                    'correlation': corr,
                                    'p_value': p_val,
                                    'predictive_power': abs(corr),
                                    'description': f"VIX levels predict {target} volatility changes",
                                    'significance': 'HIGH' if abs(corr) > 0.5 else 'MEDIUM'
                                })
                                break  # Take first significant relationship
                except:
                    continue
        
        return vol_indicators
    
    def _test_lead_lag_relationship(self, leader: pd.Series, follower: pd.Series,
                                  min_lead: int, max_lead: int, min_corr: float) -> Optional[Dict]:
        """Test lead-lag relationship between two series"""
        
        leader_returns = leader.pct_change().dropna()
        follower_returns = follower.pct_change().dropna()
        
        common_idx = leader_returns.index.intersection(follower_returns.index)
        if len(common_idx) < 30:
            return None
        
        leader_aligned = leader_returns[common_idx]
        follower_aligned = follower_returns[common_idx]
        
        best_result = None
        best_correlation = 0
        
        for lag in range(min_lead, min(max_lead + 1, len(leader_aligned) // 4)):
            if len(leader_aligned) - lag < 20:
                break
            
            try:
                leader_lagged = leader_aligned.shift(lag).dropna()
                follower_current = follower_aligned[leader_lagged.index]
                
                if len(leader_lagged) >= 20:
                    corr, p_val = stats.pearsonr(leader_lagged, follower_current)
                    
                    if not np.isnan(corr) and p_val < 0.05 and abs(corr) >= min_corr:
                        if abs(corr) > abs(best_correlation):
                            best_correlation = corr
                            best_result = {
                                'lead_days': lag,
                                'correlation': corr,
                                'p_value': p_val
                            }
            except:
                continue
        
        return best_result
    
    def _test_specific_lead_lag(self, leader: pd.Series, follower: pd.Series,
                               lead_days: int, min_corr: float) -> Optional[Dict]:
        """Test lead-lag relationship at a specific lead time"""
        
        leader_returns = leader.pct_change().dropna()
        follower_returns = follower.pct_change().dropna()
        
        common_idx = leader_returns.index.intersection(follower_returns.index)
        if len(common_idx) < 30:
            return None
        
        leader_aligned = leader_returns[common_idx]
        follower_aligned = follower_returns[common_idx]
        
        if len(leader_aligned) - lead_days < 20:
            return None
        
        try:
            leader_lagged = leader_aligned.shift(lead_days).dropna()
            follower_current = follower_aligned[leader_lagged.index]
            
            if len(leader_lagged) >= 20:
                corr, p_val = stats.pearsonr(leader_lagged, follower_current)
                
                if not np.isnan(corr) and p_val < 0.05 and abs(corr) >= min_corr:
                    return {
                        'correlation': corr,
                        'p_value': p_val
                    }
        except:
            pass
        
        return None
    
    def _calculate_predictive_power(self, leader: pd.Series, follower: pd.Series, lead_days: int) -> float:
        """Calculate the predictive power of the leading indicator"""
        
        try:
            leader_returns = leader.pct_change().dropna()
            follower_returns = follower.pct_change().dropna()
            
            common_idx = leader_returns.index.intersection(follower_returns.index)
            leader_aligned = leader_returns[common_idx]
            follower_aligned = follower_returns[common_idx]
            
            if len(leader_aligned) - lead_days < 20:
                return 0.0
            
            leader_lagged = leader_aligned.shift(lead_days).dropna()
            follower_current = follower_aligned[leader_lagged.index]
            
            if len(leader_lagged) < 20:
                return 0.0
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(leader_lagged, follower_current)
            
            # Calculate directional accuracy
            leader_direction = (leader_lagged > 0).astype(int)
            follower_direction = (follower_current > 0).astype(int)
            directional_accuracy = (leader_direction == follower_direction).mean()
            
            # Combine correlation strength and directional accuracy
            predictive_power = (abs(corr) * 0.7) + (directional_accuracy * 0.3)
            
            # Adjust for statistical significance
            significance_adjustment = max(0.5, 1 - p_val * 2)
            
            return min(1.0, predictive_power * significance_adjustment)
            
        except:
            return 0.0
    
    def _generate_predictive_summary(self, results: Dict) -> Dict:
        """Generate summary of predictive indicators"""
        
        all_indicators = (
            results['macro_leading_indicators'] + 
            results['cross_asset_leading_indicators'] + 
            results['sector_leading_indicators'] +
            results['volatility_leading_indicators']
        )
        
        if not all_indicators:
            return {
                'total_indicators': 0,
                'best_predictor': None,
                'average_lead_time': 0,
                'predictive_strength': 'NONE'
            }
        
        # Sort by predictive power
        sorted_indicators = sorted(all_indicators, key=lambda x: x['predictive_power'], reverse=True)
        
        best_predictor = sorted_indicators[0]
        avg_lead_time = np.mean([ind['lead_days'] for ind in all_indicators])
        avg_predictive_power = np.mean([ind['predictive_power'] for ind in all_indicators])
        
        strength = 'HIGH' if avg_predictive_power > 0.6 else 'MEDIUM' if avg_predictive_power > 0.4 else 'LOW'
        
        return {
            'total_indicators': len(all_indicators),
            'best_predictor': {
                'indicator': best_predictor['indicator'],
                'lead_days': best_predictor['lead_days'],
                'predictive_power': best_predictor['predictive_power'],
                'description': best_predictor['description']
            },
            'average_lead_time': round(avg_lead_time, 1),
            'predictive_strength': strength,
            'high_confidence_indicators': len([x for x in all_indicators if x['significance'] == 'HIGH'])
        }
    
    def _generate_current_trading_signals(self, target: str, data: pd.DataFrame, results: Dict) -> List[Dict]:
        """Generate current trading signals based on leading indicators"""
        
        signals = []
        
        all_indicators = (
            results['macro_leading_indicators'] + 
            results['cross_asset_leading_indicators'] + 
            results['sector_leading_indicators']
        )
        
        # Take top 3 indicators
        top_indicators = sorted(all_indicators, key=lambda x: x['predictive_power'], reverse=True)[:3]
        
        for indicator_info in top_indicators:
            indicator_name = indicator_info['indicator']
            lead_days = indicator_info['lead_days']
            correlation = indicator_info['correlation']
            
            if indicator_name not in data.columns:
                continue
            
            try:
                # Calculate recent move in the indicator
                recent_change = data[indicator_name].pct_change().tail(lead_days + 1).iloc[:-1].mean() * 100
                
                if abs(recent_change) > 0.5:  # Meaningful move
                    # Predict target direction based on correlation
                    if correlation > 0:
                        predicted_direction = 'UP' if recent_change > 0 else 'DOWN'
                    else:
                        predicted_direction = 'DOWN' if recent_change > 0 else 'UP'
                    
                    confidence = 'HIGH' if indicator_info['predictive_power'] > 0.6 else 'MEDIUM'
                    
                    signals.append({
                        'indicator': indicator_name,
                        'target': target,
                        'recent_indicator_move': recent_change,
                        'predicted_target_direction': predicted_direction,
                        'expected_timeframe': f"{lead_days} days",
                        'confidence': confidence,
                        'predictive_power': indicator_info['predictive_power'],
                        'signal_description': f"{indicator_name} moved {recent_change:+.2f}%, predicting {target} {predicted_direction} in {lead_days} days"
                    })
            except:
                continue
        
        return sorted(signals, key=lambda x: x['predictive_power'], reverse=True)

def main():
    """Test the true early indicator finder"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🔍 TRUE EARLY INDICATOR FINDER")
    print("=" * 60)
    print("Finding REAL leading relationships with meaningful lead times")
    print("=" * 60)
    
    # Load market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No market data available")
        return
    
    print(f"\n📊 Available assets: {', '.join(data.columns)}")
    
    # Test with popular assets
    test_assets = ['AAPL', 'QQQ', 'SPY']
    
    finder = TrueEarlyIndicatorFinder()
    
    for target_asset in test_assets:
        if target_asset not in data.columns:
            continue
        
        print(f"\n" + "="*80)
        print(f"🎯 FINDING TRUE EARLY INDICATORS FOR: {target_asset}")
        print("="*80)
        
        results = finder.find_true_leading_indicators(
            target_asset, data, 
            min_lead_days=1,  # Require at least 1-day lead
            max_lead_days=15,
            min_correlation=0.15  # Lower threshold to find more relationships
        )
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            continue
        
        # Display results
        summary = results['predictive_summary']
        print(f"\n📈 PREDICTIVE SUMMARY:")
        print(f"   Total Leading Indicators: {summary['total_indicators']}")
        print(f"   Average Lead Time: {summary['average_lead_time']} days")
        print(f"   Predictive Strength: {summary['predictive_strength']}")
        if 'high_confidence_indicators' in summary:
            print(f"   High Confidence Indicators: {summary['high_confidence_indicators']}")
        
        if summary['best_predictor']:
            best = summary['best_predictor']
            print(f"\n🏆 BEST PREDICTOR:")
            print(f"   Indicator: {best['indicator']}")
            print(f"   Lead Time: {best['lead_days']} days")
            print(f"   Predictive Power: {best['predictive_power']:.3f}")
            print(f"   Description: {best['description']}")
        
        # Display macro indicators
        macro_indicators = results['macro_leading_indicators']
        if macro_indicators:
            print(f"\n🌍 MACRO LEADING INDICATORS ({len(macro_indicators)} found):")
            for i, indicator in enumerate(macro_indicators[:3], 1):
                print(f"   {i}. {indicator['indicator']} (leads by {indicator['lead_days']} days)")
                print(f"      Correlation: {indicator['correlation']:+.3f}")
                print(f"      Predictive Power: {indicator['predictive_power']:.3f}")
                print(f"      Description: {indicator['description']}")
                print()
        
        # Display cross-asset indicators
        cross_asset = results['cross_asset_leading_indicators']
        if cross_asset:
            print(f"\n🔄 CROSS-ASSET INDICATORS ({len(cross_asset)} found):")
            for i, indicator in enumerate(cross_asset[:3], 1):
                print(f"   {i}. {indicator['indicator']} → {indicator['target']} (leads by {indicator['lead_days']} days)")
                print(f"      Pattern: {indicator['pattern']}")
                print(f"      Predictive Power: {indicator['predictive_power']:.3f}")
                print(f"      Description: {indicator['description']}")
                print()
        
        # Current trading signals
        signals = results['trading_signals']
        if signals:
            print(f"\n🚨 CURRENT TRADING SIGNALS:")
            for signal in signals[:3]:
                print(f"   📡 {signal['signal_description']}")
                print(f"      Confidence: {signal['confidence']}")
                print(f"      Predictive Power: {signal['predictive_power']:.3f}")
                print()
    
    print(f"\n✅ True early indicator analysis complete!")

if __name__ == "__main__":
    main()