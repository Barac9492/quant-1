#!/usr/bin/env python3
"""
Adaptive Early Signal Detection Engine
Built with 20+ years quant experience - handles non-stationary correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.signal import find_peaks
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdaptiveEarlySignalEngine:
    """
    Sophisticated early signal detection accounting for:
    1. Non-stationary correlations
    2. Regime-dependent relationships
    3. Time-varying volatility
    4. Market microstructure effects
    5. Multi-timeframe consistency
    """
    
    def __init__(self):
        self.lookback_periods = {
            'ultra_short': 5,    # Microstructure/momentum
            'short': 15,         # Tactical signals
            'medium': 45,        # Regime identification
            'long': 90,          # Structural relationships
            'cycle': 252         # Full market cycle
        }
        
        # Regime-dependent correlation thresholds
        self.correlation_regimes = {
            'crisis': {'threshold': 0.8, 'decay_factor': 0.95},
            'stress': {'threshold': 0.6, 'decay_factor': 0.98},
            'normal': {'threshold': 0.4, 'decay_factor': 0.99},
            'complacency': {'threshold': 0.2, 'decay_factor': 0.995}
        }
        
        # Asset-specific lead times (learned from historical data)
        self.adaptive_leads = {
            'VIX': {'target_assets': ['SPY', 'QQQ'], 'base_lead': 2, 'max_lead': 7},
            'TLT': {'target_assets': ['VNQ', 'IEF'], 'base_lead': 5, 'max_lead': 12},
            'GLD': {'target_assets': ['TLT', 'DXY'], 'base_lead': 7, 'max_lead': 15}
        }
    
    def detect_adaptive_signals(self, data: pd.DataFrame) -> Dict:
        """Main entry point for adaptive signal detection"""
        
        if len(data) < 60:
            return {'error': 'Insufficient data for adaptive analysis'}
        
        signals = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': self._identify_current_regime(data),
            'correlation_stability': self._assess_correlation_stability(data),
            'adaptive_thresholds': {},
            'regime_adjusted_signals': [],
            'time_varying_leads': {},
            'volatility_adjusted_signals': [],
            'cross_timeframe_confirmation': {},
            'portfolio_impact_signals': [],
            'execution_priority': []
        }
        
        # Step 1: Identify current market regime
        regime = signals['market_regime']
        logger.info(f"Current market regime: {regime['regime_type']}")
        
        # Step 2: Calculate adaptive thresholds based on regime
        adaptive_thresholds = self._calculate_adaptive_thresholds(data, regime)
        signals['adaptive_thresholds'] = adaptive_thresholds
        
        # Step 3: Detect regime-adjusted correlation signals
        regime_signals = self._detect_regime_adjusted_signals(data, regime, adaptive_thresholds)
        signals['regime_adjusted_signals'] = regime_signals
        
        # Step 4: Time-varying lead-lag analysis
        time_varying = self._time_varying_lead_lag_analysis(data)
        signals['time_varying_leads'] = time_varying
        
        # Step 5: Volatility-adjusted signal detection
        vol_adjusted = self._volatility_adjusted_signals(data, regime)
        signals['volatility_adjusted_signals'] = vol_adjusted
        
        # Step 6: Cross-timeframe confirmation
        cross_tf = self._cross_timeframe_confirmation(data)
        signals['cross_timeframe_confirmation'] = cross_tf
        
        # Step 7: Portfolio impact analysis
        portfolio_impact = self._analyze_portfolio_impact(data, regime_signals)
        signals['portfolio_impact_signals'] = portfolio_impact
        
        # Step 8: Generate execution priority list
        execution_priority = self._generate_execution_priority(signals)
        signals['execution_priority'] = execution_priority
        
        return signals
    
    def _identify_current_regime(self, data: pd.DataFrame) -> Dict:
        """Identify current market regime using multiple indicators"""
        
        # Calculate regime indicators
        indicators = {}
        
        # 1. Average correlation regime
        recent_corr = data.tail(15).corr()
        avg_corr = recent_corr.values[np.triu_indices_from(recent_corr.values, k=1)].mean()
        
        # 2. Volatility regime  
        returns = data.pct_change().dropna()
        avg_vol = returns.tail(20).std().mean() * np.sqrt(252)
        vol_percentile = (returns.tail(60).std() < returns.tail(20).std().mean()).mean()
        
        # 3. Momentum regime
        momentum_scores = []
        for col in data.columns:
            if len(data[col]) >= 20:
                momentum = (data[col].iloc[-1] / data[col].iloc[-20] - 1)
                momentum_scores.append(momentum)
        
        avg_momentum = np.mean(momentum_scores) if momentum_scores else 0
        
        # Classify regime
        if avg_corr > 0.7 and avg_vol > 0.25:
            regime_type = 'crisis'
        elif avg_corr > 0.5 and vol_percentile > 0.7:
            regime_type = 'stress'
        elif avg_corr < 0.3 and avg_vol < 0.15:
            regime_type = 'complacency'
        else:
            regime_type = 'normal'
        
        # Regime persistence score
        historical_regimes = []
        for i in range(10, min(len(data), 60), 5):
            hist_data = data.iloc[-i-15:-i]
            if len(hist_data) >= 10:
                hist_corr = hist_data.corr().values[np.triu_indices_from(hist_data.corr().values, k=1)].mean()
                if hist_corr > 0.7:
                    historical_regimes.append('crisis')
                elif hist_corr > 0.5:
                    historical_regimes.append('stress')  
                elif hist_corr < 0.3:
                    historical_regimes.append('complacency')
                else:
                    historical_regimes.append('normal')
        
        regime_persistence = (np.array(historical_regimes) == regime_type).mean() if historical_regimes else 0.5
        
        return {
            'regime_type': regime_type,
            'avg_correlation': avg_corr,
            'avg_volatility': avg_vol,
            'avg_momentum': avg_momentum,
            'regime_persistence': regime_persistence,
            'confidence': 'HIGH' if regime_persistence > 0.7 else 'MEDIUM' if regime_persistence > 0.4 else 'LOW'
        }
    
    def _assess_correlation_stability(self, data: pd.DataFrame) -> Dict:
        """Assess how stable correlations are across different timeframes"""
        
        stability_metrics = {}
        
        # Calculate correlations at different horizons
        horizons = [10, 20, 45, 90]
        correlation_matrices = {}
        
        for horizon in horizons:
            if len(data) >= horizon:
                corr_matrix = data.tail(horizon).corr()
                correlation_matrices[f'{horizon}d'] = corr_matrix
        
        if len(correlation_matrices) < 2:
            return {'stability_score': 0.5, 'regime_shift_probability': 0.5}
        
        # Calculate correlation stability across horizons
        stability_scores = []
        
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns[i+1:], i+1):
                correlations_across_horizons = []
                
                for horizon_key in correlation_matrices:
                    if asset1 in correlation_matrices[horizon_key].index:
                        corr_val = correlation_matrices[horizon_key].loc[asset1, asset2]
                        correlations_across_horizons.append(corr_val)
                
                if len(correlations_across_horizons) >= 2:
                    # Stability = 1 - coefficient of variation
                    stability = 1 - (np.std(correlations_across_horizons) / (abs(np.mean(correlations_across_horizons)) + 1e-6))
                    stability_scores.append(max(0, stability))
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.5
        
        # Regime shift probability (inverse of stability)
        regime_shift_prob = max(0, 1 - avg_stability)
        
        # Detect recent correlation breaks
        correlation_breaks = self._detect_correlation_breaks(data)
        
        return {
            'stability_score': avg_stability,
            'regime_shift_probability': regime_shift_prob,
            'correlation_breaks': correlation_breaks,
            'interpretation': self._interpret_stability(avg_stability, regime_shift_prob)
        }
    
    def _detect_correlation_breaks(self, data: pd.DataFrame) -> List[Dict]:
        """Detect structural breaks in correlation patterns"""
        
        breaks = []
        
        if len(data) < 45:
            return breaks
        
        # Compare short vs medium term correlations
        short_corr = data.tail(15).corr()
        medium_corr = data.tail(45).corr()
        
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns[i+1:], i+1):
                short_val = short_corr.loc[asset1, asset2]
                medium_val = medium_corr.loc[asset1, asset2]
                
                # Detect significant breaks
                break_magnitude = abs(short_val - medium_val)
                
                if break_magnitude > 0.3:  # Significant break
                    break_direction = 'increase' if short_val > medium_val else 'decrease'
                    
                    breaks.append({
                        'assets': f"{asset1}-{asset2}",
                        'break_magnitude': break_magnitude,
                        'direction': break_direction,
                        'short_term_corr': short_val,
                        'medium_term_corr': medium_val,
                        'significance': 'HIGH' if break_magnitude > 0.5 else 'MEDIUM'
                    })
        
        return sorted(breaks, key=lambda x: x['break_magnitude'], reverse=True)[:5]
    
    def _calculate_adaptive_thresholds(self, data: pd.DataFrame, regime: Dict) -> Dict:
        """Calculate regime-dependent correlation thresholds"""
        
        regime_type = regime['regime_type']
        base_threshold = self.correlation_regimes[regime_type]['threshold']
        
        # Adjust thresholds based on recent volatility
        vol_adjustment = min(0.2, regime['avg_volatility'] * 0.5)  # Higher vol = lower threshold
        
        # Adjust based on regime persistence
        persistence_adjustment = (regime['regime_persistence'] - 0.5) * 0.1
        
        adjusted_threshold = base_threshold - vol_adjustment + persistence_adjustment
        adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.9)
        
        return {
            'regime_type': regime_type,
            'base_threshold': base_threshold,
            'volatility_adjustment': -vol_adjustment,
            'persistence_adjustment': persistence_adjustment,
            'final_threshold': adjusted_threshold,
            'signal_decay_factor': self.correlation_regimes[regime_type]['decay_factor']
        }
    
    def _detect_regime_adjusted_signals(self, data: pd.DataFrame, regime: Dict, thresholds: Dict) -> List[Dict]:
        """Detect early signals adjusted for current regime"""
        
        signals = []
        threshold = thresholds['final_threshold']
        
        # Dynamic lookback based on regime
        if regime['regime_type'] == 'crisis':
            short_window, medium_window = 7, 21
        elif regime['regime_type'] == 'stress':
            short_window, medium_window = 10, 30
        else:
            short_window, medium_window = 15, 45
        
        if len(data) < medium_window:
            return signals
        
        short_corr = data.tail(short_window).corr()
        medium_corr = data.tail(medium_window).corr()
        
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns[i+1:], i+1):
                short_val = short_corr.loc[asset1, asset2]
                medium_val = medium_corr.loc[asset1, asset2]
                
                correlation_change = short_val - medium_val
                
                # Regime-adjusted signal detection
                if abs(correlation_change) > threshold:
                    
                    # Calculate signal strength (regime-adjusted)
                    strength_multiplier = {
                        'crisis': 1.5,  # Signals more important in crisis
                        'stress': 1.2,
                        'normal': 1.0,
                        'complacency': 0.8  # Less reliable in complacent markets
                    }[regime['regime_type']]
                    
                    signal_strength = abs(correlation_change) * strength_multiplier
                    
                    # Determine signal type and implications
                    if correlation_change > 0:
                        signal_type = 'CORRELATION_SURGE'
                        implication = 'Diversification breakdown, risk concentration'
                        action = 'Reduce correlated positions, increase hedges'
                    else:
                        signal_type = 'CORRELATION_BREAKDOWN'  
                        implication = 'Diversification opportunity, regime normalization'
                        action = 'Rebalance portfolio, increase position sizes'
                    
                    # Calculate expected lead time (regime-dependent)
                    base_lead = 5
                    regime_lead_adjustment = {
                        'crisis': 2,     # Faster reactions in crisis
                        'stress': 3,
                        'normal': 5,
                        'complacency': 7  # Slower in complacent markets
                    }[regime['regime_type']]
                    
                    expected_lead = regime_lead_adjustment
                    
                    signals.append({
                        'signal_type': signal_type,
                        'assets': f"{asset1}-{asset2}",
                        'correlation_change': correlation_change,
                        'signal_strength': signal_strength,
                        'short_term_corr': short_val,
                        'medium_term_corr': medium_val,
                        'regime_adjusted': True,
                        'regime_type': regime['regime_type'],
                        'expected_lead_days': expected_lead,
                        'implication': implication,
                        'action': action,
                        'confidence': self._calculate_signal_confidence(correlation_change, threshold, regime)
                    })
        
        # Sort by signal strength
        return sorted(signals, key=lambda x: x['signal_strength'], reverse=True)[:10]
    
    def _time_varying_lead_lag_analysis(self, data: pd.DataFrame) -> Dict:
        """Analyze lead-lag relationships that change over time"""
        
        time_varying_leads = {}
        
        # Define rolling windows for lead-lag analysis
        windows = [30, 60, 90] if len(data) >= 90 else [30]
        
        for leader_asset in self.adaptive_leads:
            if leader_asset not in data.columns:
                continue
                
            target_assets = [a for a in self.adaptive_leads[leader_asset]['target_assets'] if a in data.columns]
            
            for target_asset in target_assets:
                lead_analysis = {
                    'asset_pair': f"{leader_asset}_leads_{target_asset}",
                    'optimal_lags': {},
                    'correlation_strength': {},
                    'stability_score': 0,
                    'current_prediction': None
                }
                
                # Analyze lead-lag relationship across different windows
                for window in windows:
                    if len(data) >= window + 10:
                        window_data = data.tail(window)
                        optimal_lag, max_corr = self._find_optimal_lag_adaptive(
                            window_data[leader_asset], 
                            window_data[target_asset],
                            max_lag=self.adaptive_leads[leader_asset]['max_lead']
                        )
                        
                        if abs(max_corr) > 0.2:  # Meaningful correlation
                            lead_analysis['optimal_lags'][f'{window}d'] = optimal_lag
                            lead_analysis['correlation_strength'][f'{window}d'] = max_corr
                
                # Calculate stability of lead-lag relationship
                if len(lead_analysis['optimal_lags']) >= 2:
                    lag_values = list(lead_analysis['optimal_lags'].values())
                    lag_stability = 1 - (np.std(lag_values) / (np.mean(lag_values) + 1e-6))
                    lead_analysis['stability_score'] = max(0, lag_stability)
                    
                    # Generate current prediction if relationship is stable
                    if lag_stability > 0.6:  # Stable relationship
                        current_lag = lag_values[-1]  # Most recent lag
                        recent_leader_move = data[leader_asset].pct_change().tail(current_lag).mean()
                        
                        if abs(recent_leader_move) > data[leader_asset].pct_change().std() * 0.5:
                            direction = 'same' if lead_analysis['correlation_strength'][f'{windows[-1]}d'] > 0 else 'opposite'
                            lead_analysis['current_prediction'] = {
                                'leader_move': recent_leader_move * 100,
                                'expected_lag': current_lag,
                                'expected_direction': direction,
                                'confidence': 'HIGH' if lag_stability > 0.8 else 'MEDIUM'
                            }
                
                if lead_analysis['optimal_lags']:
                    time_varying_leads[lead_analysis['asset_pair']] = lead_analysis
        
        return time_varying_leads
    
    def _find_optimal_lag_adaptive(self, leader: pd.Series, follower: pd.Series, max_lag: int = 10) -> Tuple[int, float]:
        """Find optimal lag with adaptive parameters"""
        
        leader_returns = leader.pct_change().dropna()
        follower_returns = follower.pct_change().dropna()
        
        common_idx = leader_returns.index.intersection(follower_returns.index)
        leader_aligned = leader_returns[common_idx]
        follower_aligned = follower_returns[common_idx]
        
        best_lag = 0
        best_corr = 0
        
        for lag in range(1, min(max_lag + 1, len(leader_aligned) // 4)):
            if len(leader_aligned) - lag > 20:
                leader_lagged = leader_aligned.shift(lag).dropna()
                follower_current = follower_aligned[leader_lagged.index]
                
                if len(leader_lagged) > 10:
                    try:
                        corr, p_value = stats.pearsonr(leader_lagged, follower_current)
                        if p_value < 0.05 and abs(corr) > abs(best_corr):
                            best_lag = lag
                            best_corr = corr
                    except:
                        continue
        
        return best_lag, best_corr
    
    def _volatility_adjusted_signals(self, data: pd.DataFrame, regime: Dict) -> List[Dict]:
        """Generate signals adjusted for current volatility environment"""
        
        signals = []
        
        # Calculate rolling volatilities
        returns = data.pct_change().dropna()
        
        if len(returns) < 30:
            return signals
        
        current_vol = returns.tail(10).std().mean() * np.sqrt(252)
        long_term_vol = returns.tail(60).std().mean() * np.sqrt(252)
        vol_ratio = current_vol / (long_term_vol + 1e-6)
        
        # Volatility regime classification
        if vol_ratio > 1.5:
            vol_regime = 'high_vol'
        elif vol_ratio < 0.7:
            vol_regime = 'low_vol'
        else:
            vol_regime = 'normal_vol'
        
        # Generate volatility-specific signals
        if vol_regime == 'high_vol':
            # In high vol environments, correlations tend to increase
            signals.append({
                'signal_type': 'VOLATILITY_REGIME',
                'regime': 'HIGH_VOLATILITY',
                'vol_ratio': vol_ratio,
                'signal': f'🌪️ HIGH VOLATILITY REGIME: Current vol {current_vol:.1f}% vs long-term {long_term_vol:.1f}%',
                'implication': 'Correlations likely to increase, diversification benefits reduced',
                'expected_duration': '5-15 days',
                'action': 'Reduce position sizes, increase cash allocation',
                'confidence': 'HIGH' if vol_ratio > 2.0 else 'MEDIUM'
            })
        
        elif vol_regime == 'low_vol':
            # Low vol environments often precede volatility spikes
            signals.append({
                'signal_type': 'VOLATILITY_REGIME',
                'regime': 'LOW_VOLATILITY',
                'vol_ratio': vol_ratio,
                'signal': f'😴 LOW VOLATILITY REGIME: Current vol {current_vol:.1f}% vs long-term {long_term_vol:.1f}%',
                'implication': 'Potential for volatility expansion, prepare for regime change',
                'expected_duration': '10-30 days until regime change',
                'action': 'Consider volatility strategies, prepare for correlation changes',
                'confidence': 'MEDIUM'
            })
        
        # Asset-specific volatility signals
        for asset in data.columns:
            asset_returns = returns[asset].dropna()
            if len(asset_returns) >= 30:
                asset_current_vol = asset_returns.tail(10).std() * np.sqrt(252)
                asset_long_vol = asset_returns.tail(60).std() * np.sqrt(252)
                asset_vol_ratio = asset_current_vol / (asset_long_vol + 1e-6)
                
                if asset_vol_ratio > 2.0:  # Significant volatility spike
                    signals.append({
                        'signal_type': 'ASSET_VOLATILITY_SPIKE',
                        'asset': asset,
                        'vol_ratio': asset_vol_ratio,
                        'signal': f'⚡ {asset} VOLATILITY SPIKE: {asset_vol_ratio:.1f}x normal levels',
                        'implication': f'{asset} likely to influence broader market correlations',
                        'expected_lead': '2-7 days',
                        'action': f'Monitor {asset} impact on related assets, consider hedging',
                        'confidence': 'HIGH'
                    })
        
        return signals
    
    def _cross_timeframe_confirmation(self, data: pd.DataFrame) -> Dict:
        """Check for signal confirmation across multiple timeframes"""
        
        confirmations = {}
        
        timeframes = {
            'short': 10,
            'medium': 30,
            'long': 60
        }
        
        # Calculate correlations across timeframes
        correlation_matrices = {}
        for tf_name, window in timeframes.items():
            if len(data) >= window:
                correlation_matrices[tf_name] = data.tail(window).corr()
        
        if len(correlation_matrices) < 2:
            return confirmations
        
        # Look for cross-timeframe confirmation
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns[i+1:], i+1):
                asset_pair = f"{asset1}-{asset2}"
                
                correlations = {}
                for tf_name in correlation_matrices:
                    correlations[tf_name] = correlation_matrices[tf_name].loc[asset1, asset2]
                
                # Check for trend consistency
                corr_values = list(correlations.values())
                trend_direction = None
                
                if all(c > 0.4 for c in corr_values):
                    trend_direction = 'consistently_high'
                elif all(c < -0.4 for c in corr_values):
                    trend_direction = 'consistently_negative'
                elif all(abs(c) < 0.2 for c in corr_values):
                    trend_direction = 'consistently_low'
                
                # Check for trend strengthening/weakening
                if len(corr_values) >= 3:
                    short, medium, long_term = corr_values[-3:]
                    
                    if short > medium > long_term:
                        trend_change = 'strengthening'
                    elif short < medium < long_term:
                        trend_change = 'weakening'
                    else:
                        trend_change = 'mixed'
                    
                    if trend_direction or trend_change != 'mixed':
                        confirmations[asset_pair] = {
                            'correlations': correlations,
                            'trend_direction': trend_direction,
                            'trend_change': trend_change,
                            'confirmation_strength': self._calculate_confirmation_strength(corr_values),
                            'signal_reliability': 'HIGH' if trend_direction and trend_change != 'mixed' else 'MEDIUM'
                        }
        
        return confirmations
    
    def _calculate_confirmation_strength(self, correlation_values: List[float]) -> float:
        """Calculate strength of cross-timeframe confirmation"""
        
        if len(correlation_values) < 2:
            return 0.0
        
        # Consistency across timeframes
        consistency = 1 - (np.std(correlation_values) / (np.mean(np.abs(correlation_values)) + 1e-6))
        
        # Magnitude of correlations
        magnitude = np.mean(np.abs(correlation_values))
        
        return min(1.0, consistency * magnitude)
    
    def _analyze_portfolio_impact(self, data: pd.DataFrame, regime_signals: List[Dict]) -> List[Dict]:
        """Analyze portfolio-level impact of detected signals"""
        
        portfolio_signals = []
        
        if not regime_signals:
            return portfolio_signals
        
        # Calculate portfolio-level correlation concentration
        recent_corr = data.tail(20).corr()
        correlation_values = recent_corr.values[np.triu_indices_from(recent_corr.values, k=1)]
        
        # Portfolio diversification metrics
        avg_correlation = np.mean(correlation_values)
        max_correlation = np.max(correlation_values)
        correlation_concentration = (np.abs(correlation_values) > 0.7).mean()
        
        # Risk concentration analysis
        if correlation_concentration > 0.3:  # More than 30% of pairs highly correlated
            portfolio_signals.append({
                'signal_type': 'PORTFOLIO_RISK_CONCENTRATION',
                'concentration_ratio': correlation_concentration,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'signal': f'🚨 PORTFOLIO CONCENTRATION: {correlation_concentration:.1%} of asset pairs highly correlated',
                'impact': 'Severely reduced diversification benefits',
                'action': 'Urgent portfolio rebalancing required, add uncorrelated assets',
                'priority': 'HIGH',
                'expected_timeframe': '1-5 days'
            })
        
        # Analyze specific signal impacts
        surge_signals = [s for s in regime_signals if s['signal_type'] == 'CORRELATION_SURGE']
        breakdown_signals = [s for s in regime_signals if s['signal_type'] == 'CORRELATION_BREAKDOWN']
        
        if len(surge_signals) >= 3:  # Multiple correlation surges
            affected_assets = set()
            for signal in surge_signals:
                assets = signal['assets'].split('-')
                affected_assets.update(assets)
            
            portfolio_signals.append({
                'signal_type': 'SYSTEMIC_CORRELATION_SURGE',
                'affected_assets': list(affected_assets),
                'num_signals': len(surge_signals),
                'signal': f'🔴 SYSTEMIC RISK: {len(surge_signals)} correlation surges detected',
                'impact': f'{len(affected_assets)} assets showing increased correlation',
                'action': 'Defensive positioning, reduce leverage, increase cash',
                'priority': 'CRITICAL',
                'expected_timeframe': '1-3 days'
            })
        
        elif len(breakdown_signals) >= 3:  # Multiple correlation breakdowns
            portfolio_signals.append({
                'signal_type': 'DIVERSIFICATION_OPPORTUNITY',
                'num_signals': len(breakdown_signals),
                'signal': f'🟢 OPPORTUNITY: {len(breakdown_signals)} correlation breakdowns detected',
                'impact': 'Enhanced diversification opportunities available',
                'action': 'Rebalance portfolio, increase position sizes selectively',
                'priority': 'MEDIUM',
                'expected_timeframe': '5-10 days'
            })
        
        return portfolio_signals
    
    def _generate_execution_priority(self, all_signals: Dict) -> List[Dict]:
        """Generate prioritized execution list with specific actions"""
        
        execution_list = []
        
        # Priority 1: Critical portfolio-level signals
        for signal in all_signals.get('portfolio_impact_signals', []):
            if signal.get('priority') == 'CRITICAL':
                execution_list.append({
                    'priority_rank': 1,
                    'category': 'PORTFOLIO_CRITICAL',
                    'signal': signal['signal'],
                    'action': signal['action'],
                    'timeframe': signal['expected_timeframe'],
                    'confidence': 'HIGH',
                    'execution_urgency': 'IMMEDIATE'
                })
        
        # Priority 2: High-strength regime signals
        for signal in all_signals.get('regime_adjusted_signals', [])[:3]:
            if signal.get('signal_strength', 0) > 0.8:
                execution_list.append({
                    'priority_rank': 2,
                    'category': 'REGIME_CRITICAL',
                    'signal': f"{signal['signal_type']}: {signal['assets']}",
                    'action': signal['action'],
                    'timeframe': f"{signal['expected_lead_days']} days",
                    'confidence': signal['confidence'],
                    'execution_urgency': 'HIGH'
                })
        
        # Priority 3: Confirmed cross-timeframe signals
        for pair, confirmation in all_signals.get('cross_timeframe_confirmation', {}).items():
            if confirmation.get('signal_reliability') == 'HIGH':
                execution_list.append({
                    'priority_rank': 3,
                    'category': 'CROSS_TIMEFRAME_CONFIRMED',
                    'signal': f"Cross-timeframe confirmation: {pair}",
                    'action': f"Act on {confirmation['trend_change']} correlation trend",
                    'timeframe': '3-7 days',
                    'confidence': 'MEDIUM-HIGH',
                    'execution_urgency': 'MEDIUM'
                })
        
        # Priority 4: Time-varying lead predictions
        for pair, analysis in all_signals.get('time_varying_leads', {}).items():
            if analysis.get('current_prediction') and analysis['current_prediction']['confidence'] == 'HIGH':
                pred = analysis['current_prediction']
                execution_list.append({
                    'priority_rank': 4,
                    'category': 'PREDICTIVE_LEAD_LAG',
                    'signal': f"Lead-lag prediction: {pair}",
                    'action': f"Prepare for {pred['expected_direction']} move in follower asset",
                    'timeframe': f"{pred['expected_lag']} days",
                    'confidence': pred['confidence'],
                    'execution_urgency': 'MEDIUM'
                })
        
        # Sort by priority rank and return top 10
        execution_list.sort(key=lambda x: (x['priority_rank'], -len(x['signal'])))
        return execution_list[:10]
    
    def _calculate_signal_confidence(self, correlation_change: float, threshold: float, regime: Dict) -> str:
        """Calculate confidence level for correlation signals"""
        
        # Base confidence on magnitude relative to threshold
        magnitude_ratio = abs(correlation_change) / threshold
        
        # Adjust for regime persistence
        regime_adjustment = regime['regime_persistence']
        
        # Final confidence score
        confidence_score = magnitude_ratio * regime_adjustment
        
        if confidence_score > 1.5:
            return 'HIGH'
        elif confidence_score > 1.0:
            return 'MEDIUM-HIGH'
        elif confidence_score > 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _interpret_stability(self, stability_score: float, regime_shift_prob: float) -> str:
        """Interpret correlation stability metrics"""
        
        if stability_score > 0.8:
            return "HIGH STABILITY: Correlations are stable across timeframes"
        elif stability_score > 0.6:
            return "MODERATE STABILITY: Some variation in correlations"
        elif regime_shift_prob > 0.6:
            return "LOW STABILITY: High probability of regime shift"
        else:
            return "UNSTABLE: Correlations highly variable, regime unclear"

def main():
    """Test adaptive early signal detection"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🧠 ADAPTIVE EARLY SIGNAL DETECTION")
    print("=" * 60)
    print("Built with 20+ years quant experience")
    print("Accounting for non-stationary correlations & regime changes")
    print("=" * 60)
    
    # Load real market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No data available")
        return
    
    # Run adaptive analysis
    engine = AdaptiveEarlySignalEngine()
    signals = engine.detect_adaptive_signals(data)
    
    if 'error' in signals:
        print(f"❌ {signals['error']}")
        return
    
    print(f"\n📊 Analysis Time: {signals['timestamp']}")
    print(f"📈 Data Coverage: {len(data)} days, {len(data.columns)} assets")
    
    # Display market regime
    regime = signals['market_regime']
    print(f"\n🎯 CURRENT MARKET REGIME")
    print("-" * 40)
    print(f"Regime Type: {regime['regime_type'].upper()}")
    print(f"Average Correlation: {regime['avg_correlation']:.3f}")
    print(f"Average Volatility: {regime['avg_volatility']:.1%}")
    print(f"Regime Persistence: {regime['regime_persistence']:.1%}")
    print(f"Confidence: {regime['confidence']}")
    
    # Display correlation stability
    stability = signals['correlation_stability']
    print(f"\n📊 CORRELATION STABILITY ANALYSIS")
    print("-" * 40)
    print(f"Stability Score: {stability['stability_score']:.3f}")
    print(f"Regime Shift Probability: {stability['regime_shift_probability']:.1%}")
    print(f"Interpretation: {stability['interpretation']}")
    
    if stability.get('correlation_breaks'):
        print(f"\n🚨 RECENT CORRELATION BREAKS ({len(stability['correlation_breaks'])} found):")
        for break_info in stability['correlation_breaks'][:3]:
            print(f"   {break_info['assets']}: {break_info['direction']} by {break_info['break_magnitude']:.3f}")
    
    # Display adaptive thresholds
    thresholds = signals['adaptive_thresholds']
    print(f"\n⚖️ ADAPTIVE THRESHOLDS")
    print("-" * 40)
    print(f"Base Threshold: {thresholds['base_threshold']:.3f}")
    print(f"Final Threshold: {thresholds['final_threshold']:.3f}")
    print(f"Volatility Adjustment: {thresholds['volatility_adjustment']:+.3f}")
    print(f"Persistence Adjustment: {thresholds['persistence_adjustment']:+.3f}")
    
    # Display regime-adjusted signals
    regime_signals = signals['regime_adjusted_signals']
    if regime_signals:
        print(f"\n🎯 REGIME-ADJUSTED SIGNALS ({len(regime_signals)} found)")
        print("-" * 50)
        for i, signal in enumerate(regime_signals[:5], 1):
            print(f"\n{i}. {signal['signal_type']}: {signal['assets']}")
            print(f"   Correlation Change: {signal['correlation_change']:+.3f}")
            print(f"   Signal Strength: {signal['signal_strength']:.3f}")
            print(f"   Expected Lead: {signal['expected_lead_days']} days")
            print(f"   Confidence: {signal['confidence']}")
            print(f"   Action: {signal['action']}")
    
    # Display portfolio impact
    portfolio_impact = signals['portfolio_impact_signals']
    if portfolio_impact:
        print(f"\n💼 PORTFOLIO IMPACT ANALYSIS")
        print("-" * 50)
        for impact in portfolio_impact:
            print(f"\n🚨 {impact['signal_type']}: {impact['signal']}")
            print(f"   Impact: {impact['impact']}")
            print(f"   Action: {impact['action']}")
            print(f"   Priority: {impact['priority']}")
    
    # Display execution priorities
    execution = signals['execution_priority']
    if execution:
        print(f"\n🎯 EXECUTION PRIORITY LIST")
        print("=" * 60)
        for i, action in enumerate(execution, 1):
            urgency_emoji = {'IMMEDIATE': '🚨', 'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
            emoji = urgency_emoji.get(action['execution_urgency'], '⚪')
            
            print(f"\n{i}. {emoji} {action['category']} ({action['timeframe']})")
            print(f"   Signal: {action['signal']}")
            print(f"   Action: {action['action']}")
            print(f"   Confidence: {action['confidence']}")
            print(f"   Urgency: {action['execution_urgency']}")
    
    print(f"\n✅ Adaptive early signal analysis complete!")
    print("🧠 Remember: Correlations are non-stationary - adapt your strategy!")

if __name__ == "__main__":
    main()