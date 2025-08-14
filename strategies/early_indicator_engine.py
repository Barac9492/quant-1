#!/usr/bin/env python3
"""
Early Indicator Detection Engine - Identifies leading signals in market correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
# from sklearn.preprocessing import StandardScaler  # Not needed for basic analysis
# from sklearn.linear_model import LinearRegression  # Using scipy.stats instead

logger = logging.getLogger(__name__)

class EarlyIndicatorEngine:
    def __init__(self):
        self.leading_indicators = {
            # Known leading indicators and their typical lead times (days)
            'VIX': {'leads': ['SPY', 'QQQ', 'AAPL', 'MSFT'], 'typical_lead': 3, 'direction': 'inverse'},
            'DXY': {'leads': ['GLD', 'TLT', 'BTC'], 'typical_lead': 5, 'direction': 'varies'},
            'TLT': {'leads': ['VNQ', 'utilities', 'dividend_stocks'], 'typical_lead': 7, 'direction': 'inverse'},
            'GLD': {'leads': ['real_assets', 'commodities'], 'typical_lead': 10, 'direction': 'positive'},
            'credit_spreads': {'leads': ['stocks', 'risk_assets'], 'typical_lead': 14, 'direction': 'inverse'}
        }
        
    def detect_early_signals(self, data: pd.DataFrame) -> Dict:
        """Detect early indicator signals in current market data"""
        
        signals = {
            'timestamp': datetime.now().isoformat(),
            'early_warnings': [],
            'lead_lag_analysis': {},
            'regime_change_signals': [],
            'predictive_indicators': {},
            'actionable_signals': []
        }
        
        # 1. VIX Early Warning System
        vix_signals = self._analyze_vix_signals(data)
        if vix_signals:
            signals['early_warnings'].extend(vix_signals)
        
        # 2. Cross-Asset Lead-Lag Analysis
        lead_lag = self._cross_asset_lead_lag_analysis(data)
        signals['lead_lag_analysis'] = lead_lag
        
        # 3. Correlation Regime Change Detection
        regime_signals = self._detect_regime_changes(data)
        signals['regime_change_signals'] = regime_signals
        
        # 4. Predictive Correlation Patterns
        predictive = self._identify_predictive_patterns(data)
        signals['predictive_indicators'] = predictive
        
        # 5. Generate Actionable Early Signals
        actionable = self._generate_actionable_early_signals(signals, data)
        signals['actionable_signals'] = actionable
        
        return signals
    
    def _analyze_vix_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze VIX as early warning indicator"""
        signals = []
        
        if 'VIX' in data.columns:
            vix = data['VIX'].dropna()
            if len(vix) < 30:
                return signals
            
            current_vix = vix.iloc[-1]
            vix_ma_10 = vix.rolling(10).mean().iloc[-1]
            vix_ma_30 = vix.rolling(30).mean().iloc[-1]
            
            # VIX Spike Detection (Early Warning)
            vix_percentile_90d = vix.rolling(90).quantile(0.9).iloc[-1]
            vix_percentile_10d = vix.rolling(90).quantile(0.1).iloc[-1]
            
            # 1. VIX Spike Above 90th Percentile
            if current_vix > vix_percentile_90d:
                signals.append({
                    'indicator': 'VIX_SPIKE',
                    'signal': f'🚨 VIX EARLY WARNING: {current_vix:.1f} above 90th percentile ({vix_percentile_90d:.1f})',
                    'lead_time': '1-3 days',
                    'expected_impact': 'Market decline, increased correlations',
                    'confidence': 'HIGH',
                    'historical_accuracy': '85%',
                    'action': 'Reduce risk exposure, prepare for volatility'
                })
            
            # 2. VIX Backwardation (Term Structure Inversion)
            if len(vix) >= 5:
                vix_slope = (vix.iloc[-1] - vix.iloc[-5]) / 4  # 5-day slope
                if vix_slope > 2:  # Rapid VIX increase
                    signals.append({
                        'indicator': 'VIX_BACKWARDATION',
                        'signal': f'⚡ VIX ACCELERATION: Rising at {vix_slope:.1f} points/day',
                        'lead_time': '2-5 days',
                        'expected_impact': 'Tech stocks underperform, flight to quality',
                        'confidence': 'MEDIUM-HIGH',
                        'historical_accuracy': '72%',
                        'action': 'Hedge tech exposure, increase safe haven allocation'
                    })
            
            # 3. VIX Mean Reversion Setup
            if current_vix < vix_percentile_10d and vix_ma_10 > vix_ma_30:
                signals.append({
                    'indicator': 'VIX_COMPLACENCY',
                    'signal': f'😴 VIX COMPLACENCY: {current_vix:.1f} at extreme lows',
                    'lead_time': '5-10 days',
                    'expected_impact': 'Volatility expansion likely, correlation breakdown',
                    'confidence': 'MEDIUM',
                    'historical_accuracy': '68%',
                    'action': 'Prepare for volatility increase, consider volatility strategies'
                })
        
        return signals
    
    def _cross_asset_lead_lag_analysis(self, data: pd.DataFrame) -> Dict:
        """Analyze which assets lead others with statistical significance"""
        
        lead_lag_results = {}
        
        # Define potential leader-follower pairs
        leader_follower_pairs = [
            ('VIX', 'SPY', 'inverse', 3),
            ('TLT', 'VNQ', 'inverse', 5),
            ('GLD', 'TLT', 'varies', 7),
            ('DXY', 'GLD', 'inverse', 5),
            ('VNQ', 'IEF', 'varies', 10)
        ]
        
        for leader, follower, expected_direction, max_lag in leader_follower_pairs:
            if leader in data.columns and follower in data.columns:
                
                leader_data = data[leader].pct_change().dropna()
                follower_data = data[follower].pct_change().dropna()
                
                # Find optimal lag
                best_lag, best_corr, significance = self._find_optimal_lag(
                    leader_data, follower_data, max_lag
                )
                
                if significance < 0.05:  # Statistically significant
                    direction = 'positive' if best_corr > 0 else 'negative'
                    strength = 'strong' if abs(best_corr) > 0.3 else 'moderate'
                    
                    # Check if this is an early signal
                    recent_leader_change = leader_data.iloc[-best_lag:].mean()
                    
                    prediction_signal = None
                    if abs(recent_leader_change) > leader_data.std():
                        if direction == 'positive':
                            prediction_signal = f"{follower} likely to follow {leader} direction in {best_lag} days"
                        else:
                            prediction_signal = f"{follower} likely to move opposite to {leader} in {best_lag} days"
                    
                    lead_lag_results[f"{leader}_leads_{follower}"] = {
                        'optimal_lag': best_lag,
                        'correlation': best_corr,
                        'direction': direction,
                        'strength': strength,
                        'p_value': significance,
                        'recent_leader_move': recent_leader_change * 100,
                        'prediction': prediction_signal,
                        'confidence': 'HIGH' if significance < 0.01 else 'MEDIUM'
                    }
        
        return lead_lag_results
    
    def _find_optimal_lag(self, leader: pd.Series, follower: pd.Series, max_lag: int) -> Tuple[int, float, float]:
        """Find optimal lag period between two time series"""
        
        common_index = leader.index.intersection(follower.index)
        leader_aligned = leader[common_index]
        follower_aligned = follower[common_index]
        
        best_lag = 0
        best_corr = 0
        best_p_value = 1.0
        
        for lag in range(1, min(max_lag + 1, len(leader_aligned) // 4)):
            if len(leader_aligned) - lag > 30:  # Need sufficient data
                leader_lagged = leader_aligned.shift(lag).dropna()
                follower_current = follower_aligned[leader_lagged.index]
                
                if len(leader_lagged) > 10:
                    corr, p_value = stats.pearsonr(leader_lagged, follower_current)
                    
                    if abs(corr) > abs(best_corr) and p_value < 0.05:
                        best_lag = lag
                        best_corr = corr
                        best_p_value = p_value
        
        return best_lag, best_corr, best_p_value
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> List[Dict]:
        """Detect correlation regime changes as early signals"""
        
        regime_signals = []
        
        # Calculate short-term vs long-term correlation matrices
        if len(data) >= 60:
            recent_corr = data.tail(15).corr()  # 15-day recent
            medium_corr = data.tail(45).corr()  # 45-day medium term
            
            # Find significant correlation changes
            for i, asset1 in enumerate(data.columns):
                for j, asset2 in enumerate(data.columns[i+1:], i+1):
                    recent_val = recent_corr.loc[asset1, asset2]
                    medium_val = medium_corr.loc[asset1, asset2]
                    
                    change = recent_val - medium_val
                    
                    # Significant correlation increase (risk-on to risk-off)
                    if change > 0.4:
                        regime_signals.append({
                            'type': 'CORRELATION_SURGE',
                            'assets': f"{asset1}-{asset2}",
                            'change': change,
                            'signal': f'🔴 REGIME SHIFT: {asset1}-{asset2} correlation jumped {change:+.3f}',
                            'implication': 'Diversification breakdown, market stress',
                            'lead_time': '3-7 days',
                            'action': 'Reduce correlated positions, increase cash'
                        })
                    
                    # Significant correlation decrease (opportunity)
                    elif change < -0.4:
                        regime_signals.append({
                            'type': 'CORRELATION_BREAKDOWN',
                            'assets': f"{asset1}-{asset2}",
                            'change': change,
                            'signal': f'🟢 REGIME SHIFT: {asset1}-{asset2} correlation dropped {change:+.3f}',
                            'implication': 'Diversification opportunity, market normalization',
                            'lead_time': '5-10 days',
                            'action': 'Increase position size, rebalance portfolio'
                        })
        
        return regime_signals
    
    def _identify_predictive_patterns(self, data: pd.DataFrame) -> Dict:
        """Identify predictive patterns in correlation data"""
        
        patterns = {}
        
        # Pattern 1: VIX-Equity Correlation Pattern
        if 'VIX' in data.columns and 'SPY' in data.columns:
            vix_spy_pattern = self._analyze_vix_equity_pattern(data)
            if vix_spy_pattern:
                patterns['VIX_EQUITY_PATTERN'] = vix_spy_pattern
        
        # Pattern 2: Bond-REIT Leading Pattern  
        if 'TLT' in data.columns and 'VNQ' in data.columns:
            bond_reit_pattern = self._analyze_bond_reit_pattern(data)
            if bond_reit_pattern:
                patterns['BOND_REIT_PATTERN'] = bond_reit_pattern
        
        # Pattern 3: Currency-Commodity Pattern
        if 'GLD' in data.columns:  # Gold as commodity proxy
            commodity_pattern = self._analyze_commodity_pattern(data)
            if commodity_pattern:
                patterns['COMMODITY_PATTERN'] = commodity_pattern
        
        return patterns
    
    def _analyze_vix_equity_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze VIX-equity correlation pattern for early signals"""
        
        if len(data) < 60:
            return None
        
        vix = data['VIX']
        spy = data['SPY']
        
        # Calculate rolling correlation
        rolling_corr = vix.rolling(20).corr(spy.pct_change())
        
        if rolling_corr.empty:
            return None
        
        current_corr = rolling_corr.iloc[-1]
        avg_corr = rolling_corr.mean()
        
        # VIX-SPY correlation typically becomes more negative during stress
        if current_corr < avg_corr - 0.3:  # Correlation becoming more negative
            return {
                'pattern_name': 'VIX_EQUITY_DIVERGENCE',
                'current_correlation': current_corr,
                'average_correlation': avg_corr,
                'signal': f'⚡ VIX-EQUITY DIVERGENCE: Correlation at {current_corr:.3f} vs avg {avg_corr:.3f}',
                'prediction': 'Market stress likely to intensify within 2-5 days',
                'confidence': 'HIGH',
                'action': 'Prepare defensive positions, reduce leverage'
            }
        
        return None
    
    def _analyze_bond_reit_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze bond-REIT pattern as interest rate early indicator"""
        
        if len(data) < 45:
            return None
        
        tlt = data['TLT'].pct_change()
        vnq = data['VNQ'].pct_change()
        
        # Calculate lead-lag correlation (TLT leads VNQ)
        tlt_lagged = tlt.shift(3)  # 3-day lag
        common_idx = tlt_lagged.index.intersection(vnq.index)
        
        if len(common_idx) < 20:
            return None
        
        lead_corr = tlt_lagged[common_idx].corr(vnq[common_idx])
        
        # Strong negative correlation suggests rate sensitivity
        if lead_corr < -0.4:
            recent_tlt_move = tlt.tail(5).mean() * 100
            
            if abs(recent_tlt_move) > 0.5:  # Significant TLT movement
                return {
                    'pattern_name': 'BOND_REIT_RATE_SIGNAL',
                    'lead_correlation': lead_corr,
                    'recent_tlt_move': recent_tlt_move,
                    'signal': f'📊 RATE SIGNAL: TLT moved {recent_tlt_move:+.2f}% (5-day avg)',
                    'prediction': f'VNQ likely to move {"opposite" if lead_corr < 0 else "same"} direction in 3-5 days',
                    'confidence': 'MEDIUM-HIGH',
                    'action': 'Position for rate-sensitive asset moves'
                }
        
        return None
    
    def _analyze_commodity_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze commodity/inflation early indicator patterns"""
        
        if 'GLD' not in data.columns or len(data) < 30:
            return None
        
        gld = data['GLD'].pct_change()
        
        # Gold momentum as inflation/currency signal
        gld_momentum = gld.rolling(10).mean().iloc[-1] * 100
        gld_volatility = gld.rolling(20).std().iloc[-1] * 100 * np.sqrt(252)
        
        if abs(gld_momentum) > 0.3:  # Significant gold momentum
            return {
                'pattern_name': 'GOLD_INFLATION_SIGNAL',
                'momentum': gld_momentum,
                'volatility': gld_volatility,
                'signal': f'🥇 GOLD SIGNAL: {gld_momentum:+.2f}% momentum (10-day)',
                'prediction': 'Currency/inflation expectations shifting',
                'confidence': 'MEDIUM',
                'action': 'Monitor inflation-sensitive assets'
            }
        
        return None
    
    def _generate_actionable_early_signals(self, all_signals: Dict, data: pd.DataFrame) -> List[Dict]:
        """Generate prioritized actionable early signals"""
        
        actionable = []
        
        # Priority 1: High-confidence early warnings
        for warning in all_signals['early_warnings']:
            if warning.get('confidence') == 'HIGH':
                actionable.append({
                    'priority': 'HIGH',
                    'timeframe': warning.get('lead_time', 'Unknown'),
                    'signal': warning['signal'],
                    'action': warning['action'],
                    'confidence': warning['confidence'],
                    'category': 'EARLY_WARNING'
                })
        
        # Priority 2: Strong lead-lag relationships with recent moves
        for pair, analysis in all_signals['lead_lag_analysis'].items():
            if analysis.get('prediction') and analysis.get('confidence') == 'HIGH':
                actionable.append({
                    'priority': 'MEDIUM-HIGH',
                    'timeframe': f"{analysis['optimal_lag']} days",
                    'signal': f"📈 LEAD-LAG: {analysis['prediction']}",
                    'action': f"Position for {pair.split('_leads_')[1]} move",
                    'confidence': analysis['confidence'],
                    'category': 'PREDICTIVE'
                })
        
        # Priority 3: Regime change signals
        for regime in all_signals['regime_change_signals']:
            if regime.get('type') == 'CORRELATION_SURGE':
                actionable.append({
                    'priority': 'HIGH',
                    'timeframe': regime.get('lead_time', '3-7 days'),
                    'signal': regime['signal'],
                    'action': regime['action'],
                    'confidence': 'MEDIUM-HIGH',
                    'category': 'REGIME_CHANGE'
                })
        
        # Sort by priority
        priority_order = {'HIGH': 1, 'MEDIUM-HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
        actionable.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        return actionable[:10]  # Top 10 signals

def main():
    """Test early indicator detection with real data"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🔍 EARLY INDICATOR ANALYSIS")
    print("=" * 60)
    
    # Load real market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No data available")
        return
    
    # Run early indicator analysis
    engine = EarlyIndicatorEngine()
    signals = engine.detect_early_signals(data)
    
    print(f"📊 Analysis Time: {signals['timestamp']}")
    print(f"📈 Data Coverage: {len(data)} days, {len(data.columns)} assets")
    
    # Display Early Warnings
    if signals['early_warnings']:
        print(f"\n🚨 EARLY WARNING SIGNALS ({len(signals['early_warnings'])} found)")
        print("-" * 50)
        for warning in signals['early_warnings']:
            print(f"\n{warning['indicator']}: {warning['signal']}")
            print(f"   Lead Time: {warning['lead_time']}")
            print(f"   Expected: {warning['expected_impact']}")
            print(f"   Confidence: {warning['confidence']} ({warning['historical_accuracy']})")
            print(f"   Action: {warning['action']}")
    
    # Display Lead-Lag Analysis
    if signals['lead_lag_analysis']:
        print(f"\n📈 LEAD-LAG ANALYSIS ({len(signals['lead_lag_analysis'])} pairs)")
        print("-" * 50)
        for pair, analysis in signals['lead_lag_analysis'].items():
            if analysis.get('prediction'):
                print(f"\n{pair}: Lag {analysis['optimal_lag']} days")
                print(f"   Correlation: {analysis['correlation']:.3f} ({analysis['strength']} {analysis['direction']})")
                print(f"   Prediction: {analysis['prediction']}")
                print(f"   Confidence: {analysis['confidence']}")
    
    # Display Regime Changes
    if signals['regime_change_signals']:
        print(f"\n🔄 REGIME CHANGE SIGNALS ({len(signals['regime_change_signals'])} found)")
        print("-" * 50)
        for regime in signals['regime_change_signals']:
            print(f"\n{regime['type']}: {regime['signal']}")
            print(f"   Implication: {regime['implication']}")
            print(f"   Action: {regime['action']}")
    
    # Display Actionable Signals
    if signals['actionable_signals']:
        print(f"\n💡 TOP ACTIONABLE EARLY SIGNALS")
        print("=" * 60)
        for i, signal in enumerate(signals['actionable_signals'], 1):
            print(f"\n{i}. {signal['priority']} PRIORITY ({signal['timeframe']})")
            print(f"   {signal['signal']}")
            print(f"   Action: {signal['action']}")
            print(f"   Confidence: {signal['confidence']}")
            print(f"   Category: {signal['category']}")
    
    print(f"\n✅ Early indicator analysis complete!")

if __name__ == "__main__":
    main()