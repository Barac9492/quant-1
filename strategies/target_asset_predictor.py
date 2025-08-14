#!/usr/bin/env python3
"""
Target Asset Early Signal Finder
Input: Any ticker/asset you want to trade
Output: Best early indicators and lead times for that specific asset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class TargetAssetPredictor:
    """
    Find the best early indicators for any target asset
    Built with quant expertise - handles regime changes and time-varying relationships
    """
    
    def __init__(self):
        # Extended universe of potential leading indicators
        self.indicator_universe = {
            # Market-wide indicators
            'VIX': {'type': 'fear', 'typical_leads': [1, 2, 3, 5], 'relationship': 'inverse'},
            'DXY': {'type': 'currency', 'typical_leads': [3, 5, 7, 10], 'relationship': 'varies'},
            'TNX': {'type': 'rates', 'typical_leads': [5, 7, 10, 15], 'relationship': 'varies'},
            
            # Bonds & Credit
            'TLT': {'type': 'long_bonds', 'typical_leads': [3, 7, 10], 'relationship': 'inverse_to_stocks'},
            'IEF': {'type': 'medium_bonds', 'typical_leads': [2, 5, 7], 'relationship': 'varies'},
            'LQD': {'type': 'investment_grade', 'typical_leads': [3, 5, 10], 'relationship': 'credit_risk'},
            'HYG': {'type': 'high_yield', 'typical_leads': [2, 3, 7], 'relationship': 'risk_on_off'},
            
            # Commodities
            'GLD': {'type': 'gold', 'typical_leads': [7, 10, 15], 'relationship': 'safe_haven'},
            'SLV': {'type': 'silver', 'typical_leads': [5, 7, 10], 'relationship': 'industrial'},
            'USO': {'type': 'oil', 'typical_leads': [3, 5, 7], 'relationship': 'economic_growth'},
            
            # Broad Markets
            'SPY': {'type': 'large_cap', 'typical_leads': [1, 2, 3], 'relationship': 'market_beta'},
            'QQQ': {'type': 'tech', 'typical_leads': [1, 2, 5], 'relationship': 'growth_risk'},
            'IWM': {'type': 'small_cap', 'typical_leads': [2, 3, 5], 'relationship': 'risk_appetite'},
            'VTI': {'type': 'total_market', 'typical_leads': [1, 2, 3], 'relationship': 'broad_market'},
            
            # Sectors
            'XLF': {'type': 'financials', 'typical_leads': [3, 5, 7], 'relationship': 'rate_sensitive'},
            'XLE': {'type': 'energy', 'typical_leads': [5, 7, 10], 'relationship': 'commodity_cycle'},
            'XLU': {'type': 'utilities', 'typical_leads': [7, 10, 15], 'relationship': 'defensive'},
            'XLK': {'type': 'technology', 'typical_leads': [2, 3, 5], 'relationship': 'growth_risk'},
            'XLV': {'type': 'healthcare', 'typical_leads': [5, 10, 15], 'relationship': 'defensive'},
            
            # REITs & Real Assets
            'VNQ': {'type': 'reits', 'typical_leads': [7, 10, 15], 'relationship': 'rate_sensitive'},
            
            # International
            'EFA': {'type': 'developed_intl', 'typical_leads': [3, 5, 10], 'relationship': 'dollar_sensitive'},
            'EEM': {'type': 'emerging_markets', 'typical_leads': [5, 7, 10], 'relationship': 'risk_sensitive'},
            
            # Crypto (if available)
            'BTC-USD': {'type': 'crypto', 'typical_leads': [1, 3, 7], 'relationship': 'risk_on_speculative'}
        }
    
    def find_early_indicators(self, target_asset: str, data: pd.DataFrame, 
                            min_correlation: float = 0.3, 
                            max_lead_days: int = 20) -> Dict:
        """
        Find the best early indicators for a specific target asset
        
        Args:
            target_asset: The asset you want to trade (e.g., 'AAPL', 'TSLA', 'SPY')
            data: DataFrame with market data including potential indicators
            min_correlation: Minimum correlation threshold for significance
            max_lead_days: Maximum lead time to search
        
        Returns:
            Comprehensive analysis of early indicators for the target asset
        """
        
        if target_asset not in data.columns:
            return {'error': f'Target asset {target_asset} not found in data'}
        
        if len(data) < 60:
            return {'error': 'Need at least 60 days of data for reliable analysis'}
        
        results = {
            'target_asset': target_asset,
            'analysis_date': datetime.now().isoformat(),
            'data_period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            'leading_indicators': [],
            'regime_analysis': {},
            'prediction_signals': [],
            'trading_recommendations': {},
            'risk_warnings': []
        }
        
        target_returns = data[target_asset].pct_change().dropna()
        
        # Analyze each potential indicator
        for indicator_name in data.columns:
            if indicator_name == target_asset:
                continue
                
            indicator_analysis = self._analyze_indicator_relationship(
                indicator_name, target_asset, data, min_correlation, max_lead_days
            )
            
            if indicator_analysis and indicator_analysis['significance'] == 'HIGH':
                results['leading_indicators'].append(indicator_analysis)
        
        # Sort by predictive power
        results['leading_indicators'].sort(
            key=lambda x: x['predictive_score'], reverse=True
        )
        
        # Regime-specific analysis
        results['regime_analysis'] = self._analyze_regime_dependent_relationships(
            target_asset, data, results['leading_indicators']
        )
        
        # Current prediction signals
        results['prediction_signals'] = self._generate_current_predictions(
            target_asset, data, results['leading_indicators']
        )
        
        # Trading recommendations
        results['trading_recommendations'] = self._generate_trading_recommendations(
            target_asset, results['leading_indicators'], results['prediction_signals']
        )
        
        # Risk warnings
        results['risk_warnings'] = self._generate_risk_warnings(
            target_asset, data, results['leading_indicators']
        )
        
        return results
    
    def _analyze_indicator_relationship(self, indicator: str, target: str, data: pd.DataFrame,
                                     min_correlation: float, max_lead_days: int) -> Optional[Dict]:
        """Analyze the relationship between an indicator and target asset"""
        
        if len(data) < 30:
            return None
        
        indicator_returns = data[indicator].pct_change().dropna()
        target_returns = data[target].pct_change().dropna()
        
        # Find common index
        common_idx = indicator_returns.index.intersection(target_returns.index)
        if len(common_idx) < 30:
            return None
        
        indicator_aligned = indicator_returns[common_idx]
        target_aligned = target_returns[common_idx]
        
        # Test different lead times
        best_results = {'correlation': 0, 'lag': 0, 'p_value': 1.0}
        lag_results = {}
        
        # Test lag 0 (contemporaneous)
        try:
            corr_0, p_val_0 = stats.pearsonr(indicator_aligned, target_aligned)
            if not np.isnan(corr_0):
                lag_results[0] = {'correlation': corr_0, 'p_value': p_val_0}
                if abs(corr_0) > abs(best_results['correlation']) and p_val_0 < 0.05:
                    best_results = {'correlation': corr_0, 'lag': 0, 'p_value': p_val_0}
        except:
            pass
        
        # Test leading relationships (indicator leads target)
        for lag in range(1, min(max_lead_days + 1, len(indicator_aligned) // 3)):
            if len(indicator_aligned) - lag < 20:
                break
                
            try:
                indicator_lagged = indicator_aligned.shift(lag).dropna()
                target_current = target_aligned[indicator_lagged.index]
                
                if len(indicator_lagged) >= 20:
                    corr, p_val = stats.pearsonr(indicator_lagged, target_current)
                    
                    if not np.isnan(corr) and abs(corr) >= min_correlation and p_val < 0.05:
                        lag_results[lag] = {'correlation': corr, 'p_value': p_val}
                        
                        if abs(corr) > abs(best_results['correlation']):
                            best_results = {'correlation': corr, 'lag': lag, 'p_value': p_val}
            except:
                continue
        
        # Only accept relationships with meaningful lead times (1+ days)
        if abs(best_results['correlation']) < min_correlation or best_results['lag'] == 0:
            return None
        
        # Calculate additional metrics
        relationship_strength = self._categorize_relationship_strength(abs(best_results['correlation']))
        direction = 'positive' if best_results['correlation'] > 0 else 'negative'
        
        # Calculate predictive score (combination of correlation strength and statistical significance)
        predictive_score = abs(best_results['correlation']) * (1 - best_results['p_value'])
        
        # Stability analysis across different time periods
        stability_score = self._calculate_relationship_stability(
            indicator, target, data, best_results['lag']
        )
        
        # Recent signal strength
        recent_signal = self._calculate_recent_signal_strength(
            indicator, target, data, best_results['lag']
        )
        
        return {
            'indicator': indicator,
            'target': target,
            'optimal_lag': best_results['lag'],
            'correlation': best_results['correlation'],
            'p_value': best_results['p_value'],
            'relationship_strength': relationship_strength,
            'direction': direction,
            'predictive_score': predictive_score,
            'stability_score': stability_score,
            'recent_signal_strength': recent_signal,
            'significance': self._determine_significance_level(predictive_score, stability_score),
            'all_lag_results': lag_results,
            'indicator_type': self.indicator_universe.get(indicator, {}).get('type', 'unknown')
        }
    
    def _categorize_relationship_strength(self, abs_correlation: float) -> str:
        """Categorize the strength of the relationship"""
        if abs_correlation >= 0.7:
            return 'very_strong'
        elif abs_correlation >= 0.5:
            return 'strong'
        elif abs_correlation >= 0.3:
            return 'moderate'
        else:
            return 'weak'
    
    def _calculate_relationship_stability(self, indicator: str, target: str, 
                                        data: pd.DataFrame, optimal_lag: int) -> float:
        """Calculate how stable the relationship is over time"""
        
        if len(data) < 90:
            return 0.5
        
        # Test relationship over different periods
        periods = [30, 60, 90] if len(data) >= 90 else [30, 60] if len(data) >= 60 else [30]
        correlations = []
        
        for period in periods:
            if len(data) >= period + optimal_lag:
                period_data = data.tail(period + optimal_lag)
                
                indicator_returns = period_data[indicator].pct_change().dropna()
                target_returns = period_data[target].pct_change().dropna()
                
                common_idx = indicator_returns.index.intersection(target_returns.index)
                
                if len(common_idx) >= 20:
                    indicator_aligned = indicator_returns[common_idx]
                    target_aligned = target_returns[common_idx]
                    
                    if optimal_lag > 0:
                        indicator_lagged = indicator_aligned.shift(optimal_lag).dropna()
                        target_current = target_aligned[indicator_lagged.index]
                    else:
                        indicator_lagged = indicator_aligned
                        target_current = target_aligned
                    
                    if len(indicator_lagged) >= 15:
                        try:
                            corr, p_val = stats.pearsonr(indicator_lagged, target_current)
                            if not np.isnan(corr) and p_val < 0.10:  # Slightly relaxed for stability test
                                correlations.append(corr)
                        except:
                            pass
        
        if len(correlations) < 2:
            return 0.3
        
        # Stability = 1 - coefficient of variation
        stability = 1 - (np.std(correlations) / (abs(np.mean(correlations)) + 1e-6))
        return max(0, min(1, stability))
    
    def _calculate_recent_signal_strength(self, indicator: str, target: str,
                                        data: pd.DataFrame, optimal_lag: int) -> float:
        """Calculate recent signal strength for the indicator"""
        
        recent_window = min(10, len(data) // 4)
        if recent_window < 5:
            return 0.5
        
        try:
            indicator_recent = data[indicator].pct_change().tail(recent_window + optimal_lag)
            
            if optimal_lag > 0 and len(indicator_recent) > optimal_lag:
                # Take the recent move in the indicator
                recent_move = indicator_recent.iloc[-optimal_lag-1:-1].mean()  # Average move over lead period
            else:
                recent_move = indicator_recent.tail(recent_window).mean()
            
            # Normalize by the indicator's typical volatility
            indicator_vol = data[indicator].pct_change().std()
            
            if indicator_vol > 0:
                signal_strength = abs(recent_move) / indicator_vol
                return min(2.0, signal_strength)  # Cap at 2.0 for very strong signals
            
        except:
            pass
        
        return 0.5
    
    def _determine_significance_level(self, predictive_score: float, stability_score: float) -> str:
        """Determine overall significance level"""
        
        combined_score = predictive_score * (0.7 + 0.3 * stability_score)
        
        if combined_score >= 0.6:
            return 'HIGH'
        elif combined_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _analyze_regime_dependent_relationships(self, target: str, data: pd.DataFrame,
                                             leading_indicators: List[Dict]) -> Dict:
        """Analyze how relationships change in different market regimes"""
        
        if len(leading_indicators) == 0 or len(data) < 60:
            return {'regime_sensitivity': 'unknown'}
        
        # Simple regime classification based on market volatility
        returns = data.pct_change().dropna()
        if target in returns.columns:
            target_vol = returns[target].rolling(20).std()
            
            # High vol regime when above 75th percentile
            high_vol_threshold = target_vol.quantile(0.75)
            high_vol_periods = target_vol > high_vol_threshold
            
            regime_analysis = {}
            
            for indicator_info in leading_indicators[:3]:  # Top 3 indicators
                indicator = indicator_info['indicator']
                lag = indicator_info['optimal_lag']
                
                # Analyze correlation in high vs low vol regimes
                high_vol_corr = self._calculate_regime_correlation(
                    data, indicator, target, lag, high_vol_periods
                )
                low_vol_corr = self._calculate_regime_correlation(
                    data, indicator, target, lag, ~high_vol_periods
                )
                
                regime_analysis[indicator] = {
                    'high_volatility_correlation': high_vol_corr,
                    'low_volatility_correlation': low_vol_corr,
                    'regime_sensitivity': abs(high_vol_corr - low_vol_corr) if (high_vol_corr and low_vol_corr) else 0
                }
            
            return regime_analysis
        
        return {'regime_sensitivity': 'unknown'}
    
    def _calculate_regime_correlation(self, data: pd.DataFrame, indicator: str, target: str,
                                    lag: int, regime_mask: pd.Series) -> Optional[float]:
        """Calculate correlation within a specific regime"""
        
        try:
            regime_data = data[regime_mask]
            if len(regime_data) < 15:  # Need minimum data points
                return None
            
            indicator_returns = regime_data[indicator].pct_change().dropna()
            target_returns = regime_data[target].pct_change().dropna()
            
            common_idx = indicator_returns.index.intersection(target_returns.index)
            if len(common_idx) < 10:
                return None
            
            if lag > 0:
                indicator_lagged = indicator_returns[common_idx].shift(lag).dropna()
                target_current = target_returns[indicator_lagged.index]
            else:
                indicator_lagged = indicator_returns[common_idx]
                target_current = target_returns[common_idx]
            
            if len(indicator_lagged) >= 10:
                corr, p_val = stats.pearsonr(indicator_lagged, target_current)
                return corr if p_val < 0.10 else None
                
        except:
            pass
        
        return None
    
    def _generate_current_predictions(self, target: str, data: pd.DataFrame,
                                    leading_indicators: List[Dict]) -> List[Dict]:
        """Generate current predictions based on leading indicators"""
        
        predictions = []
        
        for indicator_info in leading_indicators[:5]:  # Top 5 indicators
            indicator = indicator_info['indicator']
            lag = indicator_info['optimal_lag']
            correlation = indicator_info['correlation']
            recent_strength = indicator_info['recent_signal_strength']
            
            if recent_strength < 0.5:  # Skip weak recent signals
                continue
            
            # Get recent move in indicator
            try:
                indicator_recent_change = data[indicator].pct_change().tail(lag + 1).iloc[:-1].mean() * 100
                
                if abs(indicator_recent_change) > 0.1:  # Meaningful move
                    # Predict target direction
                    if correlation > 0:
                        predicted_direction = 'up' if indicator_recent_change > 0 else 'down'
                    else:
                        predicted_direction = 'down' if indicator_recent_change > 0 else 'up'
                    
                    confidence_level = 'HIGH' if recent_strength > 1.0 else 'MEDIUM'
                    
                    predictions.append({
                        'indicator': indicator,
                        'indicator_move': indicator_recent_change,
                        'predicted_direction': predicted_direction,
                        'expected_timeframe': f'{lag} days' if lag > 0 else 'concurrent',
                        'confidence': confidence_level,
                        'signal_strength': recent_strength,
                        'correlation_basis': correlation
                    })
            except:
                continue
        
        return sorted(predictions, key=lambda x: x['signal_strength'], reverse=True)
    
    def _generate_trading_recommendations(self, target: str, leading_indicators: List[Dict],
                                        predictions: List[Dict]) -> Dict:
        """Generate specific trading recommendations"""
        
        if not leading_indicators:
            return {'recommendation': 'No reliable leading indicators found'}
        
        recommendations = {
            'primary_indicators_to_watch': [],
            'entry_signals': [],
            'exit_signals': [],
            'risk_management': []
        }
        
        # Primary indicators to monitor
        for indicator_info in leading_indicators[:3]:
            recommendations['primary_indicators_to_watch'].append({
                'indicator': indicator_info['indicator'],
                'lead_time': f"{indicator_info['optimal_lag']} days",
                'relationship': indicator_info['direction'],
                'reliability': indicator_info['significance']
            })
        
        # Entry signals based on current predictions
        for pred in predictions[:3]:
            if pred['confidence'] == 'HIGH':
                recommendations['entry_signals'].append({
                    'signal': f"Watch {pred['indicator']} for {pred['predicted_direction']} signal in {target}",
                    'timeframe': pred['expected_timeframe'],
                    'confidence': pred['confidence'],
                    'basis': f"{pred['indicator']} moved {pred['indicator_move']:+.2f}%"
                })
        
        # Risk management
        top_indicator = leading_indicators[0]
        recommendations['risk_management'].append({
            'stop_loss_indicator': top_indicator['indicator'],
            'logic': f"If {top_indicator['indicator']} moves against prediction by >2 std devs, consider exit",
            'hedge_suggestion': f"Use inverse correlation with {top_indicator['indicator']} if available"
        })
        
        return recommendations
    
    def _generate_risk_warnings(self, target: str, data: pd.DataFrame,
                              leading_indicators: List[Dict]) -> List[str]:
        """Generate risk warnings for the analysis"""
        
        warnings = []
        
        if not leading_indicators:
            warnings.append("⚠️ No strong leading indicators found - trading signals may be unreliable")
            return warnings
        
        # Check if top indicators are highly correlated (redundant signals)
        if len(leading_indicators) >= 2:
            top_indicators = [info['indicator'] for info in leading_indicators[:3]]
            redundant_pairs = []
            
            for i, ind1 in enumerate(top_indicators):
                for j, ind2 in enumerate(top_indicators[i+1:], i+1):
                    if ind1 in data.columns and ind2 in data.columns:
                        try:
                            corr = data[ind1].corr(data[ind2])
                            if abs(corr) > 0.7:
                                redundant_pairs.append((ind1, ind2, corr))
                        except:
                            pass
            
            if redundant_pairs:
                warnings.append(f"⚠️ High correlation between indicators: {redundant_pairs[0][0]} & {redundant_pairs[0][1]} ({redundant_pairs[0][2]:.3f})")
        
        # Check stability of relationships
        unstable_indicators = [info['indicator'] for info in leading_indicators 
                             if info['stability_score'] < 0.4]
        
        if unstable_indicators:
            warnings.append(f"⚠️ Unstable relationships detected: {', '.join(unstable_indicators[:2])}")
        
        # Check data sufficiency
        if len(data) < 120:
            warnings.append("⚠️ Limited historical data - relationships may not be robust")
        
        return warnings

def main():
    """Interactive target asset predictor"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🎯 TARGET ASSET EARLY SIGNAL FINDER")
    print("=" * 60)
    print("Find the best early indicators for any asset you want to trade")
    print("=" * 60)
    
    # Load market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No market data available")
        return
    
    print(f"\n📊 Available assets: {', '.join(data.columns)}")
    
    # Demo with a popular stock (you can change this to any asset)
    target_assets = ['AAPL', 'QQQ', 'SPY', 'GLD']  # Test multiple assets
    
    predictor = TargetAssetPredictor()
    
    for target_asset in target_assets:
        if target_asset not in data.columns:
            continue
            
        print(f"\n" + "="*80)
        print(f"🎯 ANALYZING EARLY INDICATORS FOR: {target_asset}")
        print("="*80)
        
        results = predictor.find_early_indicators(target_asset, data)
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            continue
        
        # Display leading indicators
        indicators = results['leading_indicators']
        if indicators:
            print(f"\n🚀 TOP EARLY INDICATORS FOR {target_asset} ({len(indicators)} found):")
            print("-" * 60)
            
            for i, indicator in enumerate(indicators[:5], 1):
                lead_time = f"{indicator['optimal_lag']} days ahead" if indicator['optimal_lag'] > 0 else "concurrent"
                direction_emoji = "📈" if indicator['direction'] == 'positive' else "📉"
                
                print(f"\n{i}. {direction_emoji} {indicator['indicator']} ({indicator['indicator_type']})")
                print(f"   Lead Time: {lead_time}")
                print(f"   Correlation: {indicator['correlation']:+.3f}")
                print(f"   Strength: {indicator['relationship_strength']}")
                print(f"   Stability: {indicator['stability_score']:.2f}")
                print(f"   Predictive Score: {indicator['predictive_score']:.3f}")
                print(f"   Significance: {indicator['significance']}")
        
        # Display current predictions
        predictions = results['prediction_signals']
        if predictions:
            print(f"\n🔮 CURRENT PREDICTIONS FOR {target_asset}:")
            print("-" * 50)
            
            for pred in predictions[:3]:
                confidence_emoji = "🔥" if pred['confidence'] == 'HIGH' else "⚡"
                direction_emoji = "🚀" if pred['predicted_direction'] == 'up' else "💥"
                
                print(f"\n{confidence_emoji} {pred['indicator']} Signal:")
                print(f"   Recent Move: {pred['indicator_move']:+.2f}%")
                print(f"   {target_asset} Prediction: {direction_emoji} {pred['predicted_direction'].upper()}")
                print(f"   Timeframe: {pred['expected_timeframe']}")
                print(f"   Confidence: {pred['confidence']}")
        
        # Display trading recommendations
        recs = results['trading_recommendations']
        if recs.get('primary_indicators_to_watch'):
            print(f"\n💡 TRADING RECOMMENDATIONS FOR {target_asset}:")
            print("-" * 50)
            
            print("📊 Primary Indicators to Monitor:")
            for rec in recs['primary_indicators_to_watch']:
                print(f"   • {rec['indicator']} (leads by {rec['lead_time']}, {rec['relationship']} relationship)")
            
            if recs.get('entry_signals'):
                print("\n🎯 Current Entry Signals:")
                for signal in recs['entry_signals']:
                    print(f"   • {signal['signal']} ({signal['confidence']} confidence)")
        
        # Display risk warnings
        warnings = results['risk_warnings']
        if warnings:
            print(f"\n⚠️ RISK WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
    
    print(f"\n✅ Target asset analysis complete!")
    print("\n💡 Usage: You can modify the target_assets list to analyze any ticker you want to trade!")

if __name__ == "__main__":
    main()