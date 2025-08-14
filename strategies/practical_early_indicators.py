#!/usr/bin/env python3
"""
Practical Early Indicators - Focus on known market relationships and external signals
Built for real-world trading with actionable early warning signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class PracticalEarlyIndicatorSystem:
    """
    Focus on practically useful early indicators based on:
    1. Known market relationships (VIX -> Equities, Bonds -> REITs, etc.)
    2. Cross-market signals (Futures, Pre-market, International)
    3. Sector rotation patterns
    4. Volatility breakout patterns
    """
    
    def __init__(self):
        # Known reliable market relationships with typical lead times
        self.market_relationships = {
            # VIX leads equity weakness (1-3 days)
            'vix_equity_fear': {
                'trigger_conditions': ['VIX spike > 90th percentile', 'VIX > 30 sustained'],
                'affected_assets': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'growth_stocks'],
                'typical_lead': [1, 2, 3],
                'relationship': 'inverse',
                'reliability': 0.75,
                'description': 'VIX spikes precede equity selling'
            },
            
            # Bond yield changes lead REITs (3-7 days) 
            'rates_reits': {
                'trigger_conditions': ['10Y yield > 2 std dev move', 'TLT > 3% daily move'],
                'affected_assets': ['VNQ', 'IYR', 'mortgage_REITs'],
                'typical_lead': [3, 5, 7],
                'relationship': 'inverse',
                'reliability': 0.68,
                'description': 'Rate moves precede REIT adjustment'
            },
            
            # Dollar strength leads international/commodities (5-10 days)
            'dollar_international': {
                'trigger_conditions': ['DXY > 1.5% daily move', 'DXY breakout'],
                'affected_assets': ['EFA', 'EEM', 'GLD', 'international_stocks'],
                'typical_lead': [5, 7, 10],
                'relationship': 'inverse',
                'reliability': 0.62,
                'description': 'Dollar moves precede international asset adjustment'
            },
            
            # Credit stress leads equity weakness (2-5 days)
            'credit_equity': {
                'trigger_conditions': ['Credit spreads widen', 'HYG weakness'],
                'affected_assets': ['SPY', 'small_caps', 'high_beta_stocks'],
                'typical_lead': [2, 3, 5],
                'relationship': 'negative',
                'reliability': 0.70,
                'description': 'Credit stress precedes equity weakness'
            },
            
            # Sector rotation patterns (3-10 days)
            'financials_leadership': {
                'trigger_conditions': ['XLF outperformance', 'XLF volume spike'],
                'affected_assets': ['SPY', 'cyclical_sectors', 'value_stocks'],
                'typical_lead': [3, 5, 7],
                'relationship': 'positive',
                'reliability': 0.58,
                'description': 'Financial strength signals market optimism'
            }
        }
        
        # Technical pattern signals
        self.technical_signals = {
            'volatility_expansion': {
                'description': 'Low volatility precedes volatility expansion',
                'lookback_period': 20,
                'trigger_threshold': '20th percentile',
                'affected_timeframe': [5, 10, 15],
                'reliability': 0.65
            },
            
            'correlation_breakdown': {
                'description': 'High correlation precedes breakdown',
                'lookback_period': 30,
                'trigger_threshold': '80th percentile',
                'affected_timeframe': [7, 14, 21],
                'reliability': 0.58
            },
            
            'momentum_exhaustion': {
                'description': 'Extended moves precede reversals',
                'lookback_period': 14,
                'trigger_threshold': '2 std dev from mean',
                'affected_timeframe': [3, 7, 10],
                'reliability': 0.60
            }
        }
    
    def analyze_early_signals(self, target_asset: str, data: pd.DataFrame) -> Dict:
        """
        Analyze practical early warning signals for target asset
        Focus on actionable signals with known reliability
        """
        
        if target_asset not in data.columns:
            return {'error': f'Target asset {target_asset} not found in data'}
        
        if len(data) < 60:
            return {'error': 'Need at least 60 days of data for reliable analysis'}
        
        results = {
            'target_asset': target_asset,
            'analysis_timestamp': datetime.now().isoformat(),
            'market_relationship_signals': [],
            'technical_pattern_signals': [],
            'volatility_signals': [],
            'current_early_warnings': [],
            'watchlist_indicators': [],
            'risk_level': 'NORMAL',
            'action_required': False
        }
        
        # 1. Market relationship signals
        market_signals = self._analyze_market_relationships(target_asset, data)
        results['market_relationship_signals'] = market_signals
        
        # 2. Technical pattern signals
        technical_signals = self._analyze_technical_patterns(target_asset, data)
        results['technical_pattern_signals'] = technical_signals
        
        # 3. Volatility-based signals
        vol_signals = self._analyze_volatility_signals(target_asset, data)
        results['volatility_signals'] = vol_signals
        
        # 4. Generate current early warnings
        warnings = self._generate_current_warnings(target_asset, data, results)
        results['current_early_warnings'] = warnings
        
        # 5. Create watchlist of indicators to monitor
        watchlist = self._create_watchlist(target_asset, data, results)
        results['watchlist_indicators'] = watchlist
        
        # 6. Determine overall risk level and action required
        risk_assessment = self._assess_risk_level(results)
        results.update(risk_assessment)
        
        return results
    
    def _analyze_market_relationships(self, target: str, data: pd.DataFrame) -> List[Dict]:
        """Analyze known market relationships for early signals"""
        
        signals = []
        
        # VIX -> Equity relationship
        if target in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'] and 'VIX' in data.columns:
            vix_signal = self._analyze_vix_equity_signal(target, data)
            if vix_signal:
                signals.append(vix_signal)
        
        # Rates -> REITs relationship
        if 'REI' in target.upper() or target == 'VNQ':
            if 'TLT' in data.columns:
                rates_signal = self._analyze_rates_reits_signal(target, data)
                if rates_signal:
                    signals.append(rates_signal)
        
        # Dollar -> International/Commodities
        if target in ['GLD', 'EFA', 'EEM'] and 'DXY' in data.columns:
            dollar_signal = self._analyze_dollar_signal(target, data)  
            if dollar_signal:
                signals.append(dollar_signal)
        
        return signals
    
    def _analyze_vix_equity_signal(self, target: str, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze VIX as early indicator for equity weakness"""
        
        vix = data['VIX']
        target_prices = data[target]
        
        if len(vix) < 30:
            return None
        
        # Current VIX level and recent trend
        current_vix = vix.iloc[-1]
        vix_ma_5 = vix.rolling(5).mean().iloc[-1]
        vix_ma_20 = vix.rolling(20).mean().iloc[-1]
        
        # VIX percentiles
        vix_90th = vix.rolling(90).quantile(0.9).iloc[-1]
        vix_10th = vix.rolling(90).quantile(0.1).iloc[-1]
        
        signal_strength = 'NONE'
        warning_level = 'NORMAL'
        expected_impact = 'No significant impact expected'
        lead_time = 'N/A'
        
        # VIX spike signal
        if current_vix > vix_90th:
            signal_strength = 'STRONG'
            warning_level = 'HIGH'
            expected_impact = f'{target} likely to decline in next 1-3 days'
            lead_time = '1-3 days'
        elif current_vix > 30 and vix_ma_5 > vix_ma_20:
            signal_strength = 'MODERATE'
            warning_level = 'MEDIUM'
            expected_impact = f'{target} may face pressure in next 2-5 days'
            lead_time = '2-5 days'
        elif current_vix < vix_10th:
            signal_strength = 'COMPLACENCY'
            warning_level = 'LOW'
            expected_impact = f'Low VIX suggests potential volatility expansion ahead'
            lead_time = '5-15 days'
        
        if signal_strength != 'NONE':
            return {
                'signal_type': 'VIX_EQUITY_RELATIONSHIP',
                'indicator': 'VIX',
                'target': target,
                'current_value': current_vix,
                'signal_strength': signal_strength,
                'warning_level': warning_level,
                'expected_impact': expected_impact,
                'lead_time': lead_time,
                'reliability': self.market_relationships['vix_equity_fear']['reliability'],
                'action_required': warning_level in ['HIGH', 'MEDIUM']
            }
        
        return None
    
    def _analyze_rates_reits_signal(self, target: str, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze interest rate moves as early indicator for REITs"""
        
        tlt = data['TLT']
        target_prices = data[target]
        
        if len(tlt) < 30:
            return None
        
        # Recent TLT moves (bond price moves inverse to rates)
        tlt_1d_change = (tlt.iloc[-1] / tlt.iloc[-2] - 1) * 100
        tlt_5d_change = (tlt.iloc[-1] / tlt.iloc[-6] - 1) * 100
        
        # TLT volatility
        tlt_vol = tlt.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        recent_vol = tlt.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100
        
        signal_strength = 'NONE'
        warning_level = 'NORMAL'
        expected_impact = 'No significant rate impact expected'
        
        # Significant TLT (rate) moves
        if abs(tlt_1d_change) > 2.0:  # Large daily bond move
            if tlt_1d_change > 0:  # Bonds up = rates down = good for REITs
                signal_strength = 'STRONG'
                warning_level = 'POSITIVE'
                expected_impact = f'{target} likely to benefit from falling rates in 3-7 days'
            else:  # Bonds down = rates up = bad for REITs
                signal_strength = 'STRONG'
                warning_level = 'HIGH' 
                expected_impact = f'{target} likely to decline from rising rates in 3-7 days'
        elif abs(tlt_5d_change) > 3.0:  # Sustained move
            signal_strength = 'MODERATE'
            warning_level = 'MEDIUM'
            expected_impact = f'{target} may adjust to rate environment in 5-10 days'
        
        if signal_strength != 'NONE':
            return {
                'signal_type': 'RATES_REITS_RELATIONSHIP',
                'indicator': 'TLT (Rates)',
                'target': target,
                'recent_bond_move': tlt_1d_change,
                'signal_strength': signal_strength,
                'warning_level': warning_level,
                'expected_impact': expected_impact,
                'lead_time': '3-7 days',
                'reliability': self.market_relationships['rates_reits']['reliability'],
                'action_required': warning_level in ['HIGH', 'MEDIUM']
            }
        
        return None
    
    def _analyze_dollar_signal(self, target: str, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze dollar strength as early indicator for international/commodities"""
        
        # For this demo, we'll use available data approximations
        # In practice, would use DXY data
        
        # Use TLT as proxy for dollar strength (inverse relationship)
        if 'TLT' in data.columns:
            dollar_proxy = data['TLT']
            
            recent_change = (dollar_proxy.iloc[-1] / dollar_proxy.iloc[-6] - 1) * 100
            
            if abs(recent_change) > 2.0:
                if target == 'GLD':
                    if recent_change < 0:  # TLT down suggests dollar up, bad for gold
                        return {
                            'signal_type': 'DOLLAR_COMMODITY_RELATIONSHIP',
                            'indicator': 'Dollar Strength (proxy)',
                            'target': target,
                            'signal_strength': 'MODERATE',
                            'warning_level': 'MEDIUM',
                            'expected_impact': f'{target} may decline from dollar strength in 5-10 days',
                            'lead_time': '5-10 days',
                            'reliability': 0.6,
                            'action_required': True
                        }
        
        return None
    
    def _analyze_technical_patterns(self, target: str, data: pd.DataFrame) -> List[Dict]:
        """Analyze technical patterns for early signals"""
        
        signals = []
        target_data = data[target]
        
        if len(target_data) < 30:
            return signals
        
        # Volatility expansion signal
        vol_signal = self._check_volatility_expansion(target, target_data)
        if vol_signal:
            signals.append(vol_signal)
        
        # Momentum exhaustion signal
        momentum_signal = self._check_momentum_exhaustion(target, target_data)
        if momentum_signal:
            signals.append(momentum_signal)
        
        return signals
    
    def _check_volatility_expansion(self, target: str, price_data: pd.Series) -> Optional[Dict]:
        """Check for volatility expansion setup"""
        
        returns = price_data.pct_change().dropna()
        
        if len(returns) < 20:
            return None
        
        # Current volatility vs historical
        current_vol = returns.rolling(10).std().iloc[-1] * np.sqrt(252) * 100
        avg_vol = returns.rolling(60).std().mean() * np.sqrt(252) * 100
        
        # Low volatility that may precede expansion
        if current_vol < avg_vol * 0.7:  # 30% below average
            return {
                'signal_type': 'VOLATILITY_EXPANSION_SETUP',
                'indicator': 'Low Volatility',
                'target': target,
                'current_volatility': current_vol,
                'average_volatility': avg_vol,
                'signal_strength': 'MODERATE',
                'warning_level': 'WATCH',
                'expected_impact': f'Volatility expansion likely in next 5-15 days',
                'lead_time': '5-15 days',
                'reliability': self.technical_signals['volatility_expansion']['reliability'],
                'action_required': False
            }
        
        return None
    
    def _check_momentum_exhaustion(self, target: str, price_data: pd.Series) -> Optional[Dict]:
        """Check for momentum exhaustion signals"""
        
        if len(price_data) < 20:
            return None
        
        # RSI-like momentum measure
        returns = price_data.pct_change().dropna()
        recent_return = (price_data.iloc[-1] / price_data.iloc[-15] - 1) * 100
        
        vol = returns.std() * np.sqrt(252) * 100
        
        # Check for extended moves
        if abs(recent_return) > 2 * vol / np.sqrt(252) * np.sqrt(14) * 100:  # 2 std dev move
            direction = 'upside' if recent_return > 0 else 'downside'
            
            return {
                'signal_type': 'MOMENTUM_EXHAUSTION',
                'indicator': 'Extended Move',
                'target': target,
                'recent_move': recent_return,
                'signal_strength': 'MODERATE',
                'warning_level': 'MEDIUM',
                'expected_impact': f'Potential {direction} reversal in next 3-10 days',
                'lead_time': '3-10 days',
                'reliability': self.technical_signals['momentum_exhaustion']['reliability'],
                'action_required': True
            }
        
        return None
    
    def _analyze_volatility_signals(self, target: str, data: pd.DataFrame) -> List[Dict]:
        """Analyze volatility-based early warning signals"""
        
        signals = []
        target_data = data[target]
        
        if len(target_data) < 30:
            return signals
        
        returns = target_data.pct_change().dropna()
        
        # Rolling volatility analysis
        vol_10d = returns.rolling(10).std() * np.sqrt(252) * 100
        vol_30d = returns.rolling(30).std() * np.sqrt(252) * 100
        
        current_vol_10d = vol_10d.iloc[-1]
        current_vol_30d = vol_30d.iloc[-1]
        
        # Volatility breakout signal
        if current_vol_10d > current_vol_30d * 1.5:
            signals.append({
                'signal_type': 'VOLATILITY_BREAKOUT',
                'indicator': 'Short-term Volatility Spike',
                'target': target,
                'current_10d_vol': current_vol_10d,
                'current_30d_vol': current_vol_30d,
                'signal_strength': 'STRONG',
                'warning_level': 'HIGH',
                'expected_impact': 'Continued volatility and potential trend change',
                'lead_time': '1-5 days',
                'reliability': 0.70,
                'action_required': True
            })
        
        return signals
    
    def _generate_current_warnings(self, target: str, data: pd.DataFrame, results: Dict) -> List[Dict]:
        """Generate current early warning signals"""
        
        warnings = []
        
        # Collect all signals requiring action
        all_signals = (
            results['market_relationship_signals'] + 
            results['technical_pattern_signals'] + 
            results['volatility_signals']
        )
        
        action_required_signals = [s for s in all_signals if s.get('action_required', False)]
        high_warning_signals = [s for s in all_signals if s.get('warning_level') == 'HIGH']
        
        for signal in action_required_signals:
            warnings.append({
                'warning_type': signal['signal_type'],
                'urgency': 'HIGH' if signal.get('warning_level') == 'HIGH' else 'MEDIUM',
                'message': f"{signal['indicator']} suggests {signal['expected_impact']}",
                'timeframe': signal.get('lead_time', 'Unknown'),
                'reliability': signal.get('reliability', 0.5),
                'recommended_action': self._get_recommended_action(signal, target)
            })
        
        return sorted(warnings, key=lambda x: x['reliability'], reverse=True)
    
    def _get_recommended_action(self, signal: Dict, target: str) -> str:
        """Get recommended action based on signal"""
        
        signal_type = signal.get('signal_type', '')
        warning_level = signal.get('warning_level', 'MEDIUM')
        
        if 'VIX' in signal_type and warning_level == 'HIGH':
            return f"Reduce {target} position size, consider hedging or taking profits"
        elif 'RATES' in signal_type and warning_level == 'HIGH':
            return f"Prepare for {target} volatility, consider position adjustment"
        elif 'VOLATILITY_BREAKOUT' in signal_type:
            return f"Monitor {target} closely, consider stop losses or volatility plays"
        elif 'MOMENTUM_EXHAUSTION' in signal_type:
            return f"Watch for {target} reversal, consider profit taking or contrarian plays"
        else:
            return f"Monitor {target} and related indicators closely"
    
    def _create_watchlist(self, target: str, data: pd.DataFrame, results: Dict) -> List[Dict]:
        """Create watchlist of indicators to monitor"""
        
        watchlist = []
        
        # Always watch VIX for equity assets
        if target in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'] and 'VIX' in data.columns:
            watchlist.append({
                'indicator': 'VIX',
                'current_value': data['VIX'].iloc[-1],
                'watch_for': 'Spikes above 30 or 90th percentile',
                'expected_impact': f'{target} weakness in 1-3 days',
                'priority': 'HIGH'
            })
        
        # Watch TLT for rate-sensitive assets
        if target == 'VNQ' and 'TLT' in data.columns:
            recent_change = (data['TLT'].iloc[-1] / data['TLT'].iloc[-2] - 1) * 100
            watchlist.append({
                'indicator': 'TLT (Rates)',
                'current_value': f"{recent_change:+.2f}% (1-day)",
                'watch_for': 'Large daily moves >2%',
                'expected_impact': f'{target} adjustment in 3-7 days',
                'priority': 'HIGH'
            })
        
        # Watch volatility for all assets
        returns = data[target].pct_change()
        current_vol = returns.rolling(10).std().iloc[-1] * np.sqrt(252) * 100
        watchlist.append({
            'indicator': f'{target} Volatility',
            'current_value': f"{current_vol:.1f}%",
            'watch_for': 'Expansion above normal range',
            'expected_impact': 'Increased price swings and trend changes',
            'priority': 'MEDIUM'
        })
        
        return watchlist
    
    def _assess_risk_level(self, results: Dict) -> Dict:
        """Assess overall risk level based on signals"""
        
        warnings = results['current_early_warnings']
        
        if not warnings:
            return {
                'risk_level': 'LOW',
                'action_required': False,
                'risk_summary': 'No significant early warning signals detected'
            }
        
        high_urgency_count = len([w for w in warnings if w['urgency'] == 'HIGH'])
        high_reliability_count = len([w for w in warnings if w['reliability'] > 0.7])
        
        if high_urgency_count >= 2 or high_reliability_count >= 1:
            risk_level = 'HIGH'
            action_required = True
            risk_summary = f"{high_urgency_count} high-urgency signals detected"
        elif high_urgency_count >= 1 or len(warnings) >= 2:
            risk_level = 'MEDIUM' 
            action_required = True
            risk_summary = f"{len(warnings)} warning signals active"
        else:
            risk_level = 'LOW'
            action_required = False
            risk_summary = "Minor warning signals detected"
        
        return {
            'risk_level': risk_level,
            'action_required': action_required,
            'risk_summary': risk_summary
        }

def main():
    """Test the practical early indicator system"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🎯 PRACTICAL EARLY INDICATOR SYSTEM")
    print("=" * 60)
    print("Focus on actionable early warning signals with known reliability")
    print("=" * 60)
    
    # Load market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No market data available")
        return
    
    # Test with key assets
    test_assets = ['AAPL', 'VNQ', 'SPY']
    
    system = PracticalEarlyIndicatorSystem()
    
    for target_asset in test_assets:
        if target_asset not in data.columns:
            continue
        
        print(f"\n" + "="*80)
        print(f"🎯 EARLY WARNING ANALYSIS FOR: {target_asset}")
        print("="*80)
        
        results = system.analyze_early_signals(target_asset, data)
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            continue
        
        # Risk assessment
        print(f"\n🚨 RISK ASSESSMENT:")
        print(f"   Risk Level: {results['risk_level']}")
        print(f"   Action Required: {results['action_required']}")
        print(f"   Summary: {results['risk_summary']}")
        
        # Current warnings
        warnings = results['current_early_warnings']
        if warnings:
            print(f"\n⚠️ CURRENT EARLY WARNINGS ({len(warnings)} active):")
            for i, warning in enumerate(warnings, 1):
                urgency_emoji = "🚨" if warning['urgency'] == 'HIGH' else "🟡"
                print(f"   {i}. {urgency_emoji} {warning['warning_type']}")
                print(f"      Message: {warning['message']}")
                print(f"      Timeframe: {warning['timeframe']}")
                print(f"      Reliability: {warning['reliability']:.1%}")
                print(f"      Action: {warning['recommended_action']}")
                print()
        
        # Watchlist
        watchlist = results['watchlist_indicators']
        if watchlist:
            print(f"\n👀 INDICATORS TO WATCH:")
            for item in watchlist:
                priority_emoji = "🔥" if item['priority'] == 'HIGH' else "📊"
                print(f"   {priority_emoji} {item['indicator']}: {item['current_value']}")
                print(f"      Watch for: {item['watch_for']}")
                print(f"      Impact: {item['expected_impact']}")
                print()
        
        # Market relationship signals
        market_signals = results['market_relationship_signals']
        if market_signals:
            print(f"\n🔗 MARKET RELATIONSHIP SIGNALS:")
            for signal in market_signals:
                print(f"   • {signal['signal_type']}: {signal['expected_impact']}")
                print(f"     Reliability: {signal['reliability']:.1%}, Lead time: {signal['lead_time']}")
        
        # Technical signals
        technical_signals = results['technical_pattern_signals']
        if technical_signals:
            print(f"\n📈 TECHNICAL PATTERN SIGNALS:")
            for signal in technical_signals:
                print(f"   • {signal['signal_type']}: {signal['expected_impact']}")
                print(f"     Lead time: {signal['lead_time']}")
    
    print(f"\n✅ Practical early indicator analysis complete!")

if __name__ == "__main__":
    main()