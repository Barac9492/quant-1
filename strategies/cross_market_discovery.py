#!/usr/bin/env python3
"""
Cross-Market Discovery Engine - Find hidden connections across distant asset classes
The brilliance lies in discovering non-obvious relationships that create alpha

Focus on seemingly unrelated but fundamentally connected markets:
- Commodity futures -> Tech stock volatility  
- FX carry trades -> Emerging market bonds
- Japanese bond auctions -> US small cap momentum
- Baltic Dry Index -> Consumer discretionary earnings
- German Bund yields -> Brazilian bank stocks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class CrossMarketDiscoveryEngine:
    """
    Find non-obvious connections across distant asset classes that provide real alpha
    Built to discover relationships that most market participants miss
    """
    
    def __init__(self):
        # Cross-market connection hypotheses - the distant relationships that matter
        self.cross_market_chains = {
            # Commodity -> Tech volatility chain
            'commodity_tech_volatility': {
                'description': 'Commodity volatility spikes lead to tech risk-off 3-7 days later',
                'chain': (['DJP', 'PDBC', 'GLD'], ['QQQ', 'XLK', 'ARKK']),  # Commodities -> Tech
                'mechanism': 'Inflation fear -> Growth discount rate increase',
                'lead_time_days': [3, 5, 7],
                'volatility_threshold': 2.0,  # 2 std dev move
                'reliability_score': 0.72
            },
            
            # Currency carry -> EM debt chain  
            'carry_trade_unwind': {
                'description': 'USD/JPY carry unwind signals EM bond stress 5-10 days ahead',
                'chain': (['USDJPY=X', 'DXY'], ['EEM', 'EMB', 'EMLC']),  # FX -> EM
                'mechanism': 'Carry trade unwind -> EM capital flight',
                'lead_time_days': [5, 7, 10],
                'move_threshold': 1.5,
                'reliability_score': 0.68
            },
            
            # Asian session -> US market open
            'asian_session_spillover': {
                'description': 'Nikkei overnight moves predict US small cap opening gaps',
                'chain': (['^N225', 'EWJ'], ['IWM', 'VXX']),  # Japan -> US Small Cap
                'mechanism': 'Risk sentiment transmission via overnight futures',
                'lead_time_days': [0, 1],  # Same day and next day
                'correlation_threshold': 0.4,
                'reliability_score': 0.65
            },
            
            # Energy transition signals
            'energy_transition_alpha': {
                'description': 'Oil volatility inversely predicts clean energy momentum',
                'chain': (['USO', 'XLE', '^GSPC'], ['ICLN', 'PBW', 'QCLN']),  # Oil -> Clean Energy
                'mechanism': 'Energy price volatility -> Policy response -> Clean energy investment',
                'lead_time_days': [7, 10, 14],
                'inverse_relationship': True,
                'reliability_score': 0.61
            },
            
            # Credit -> Equity sector rotation
            'credit_sector_rotation': {
                'description': 'HYG spreads predict financials vs utilities rotation',
                'chain': (['HYG', 'LQD', 'TLT'], ['XLF', 'XLU']),  # Credit -> Sector Rotation
                'mechanism': 'Credit conditions -> Banking profitability expectations',
                'lead_time_days': [5, 7, 12],
                'spread_sensitivity': True,
                'reliability_score': 0.75
            },
            
            # Volatility surface -> Individual names
            'vol_surface_alpha': {
                'description': 'VIX term structure predicts high-beta stock performance',
                'chain': (['^VIX', '^VIX3M'], ['TSLA', 'NVDA', 'AMD']),  # Vol Surface -> High Beta
                'mechanism': 'Term structure -> Risk appetite for momentum names',
                'lead_time_days': [2, 4, 6],
                'term_structure_sensitive': True,
                'reliability_score': 0.69
            },
            
            # International monetary policy spillovers
            'central_bank_spillover': {
                'description': 'ECB policy signals affect US growth stocks via dollar',
                'chain': (['EUR=X', 'EFA'], ['QQQ', 'VGT']),  # EUR -> US Growth
                'mechanism': 'EUR strength -> USD weakness -> Growth stock revaluation',
                'lead_time_days': [3, 5, 8],
                'currency_sensitivity': True,
                'reliability_score': 0.58
            },
            
            # Commodity curve signals
            'commodity_curve_divergence': {
                'description': 'Gold/Silver ratio predicts mining stock volatility expansion',
                'chain': (['GLD', 'SLV'], ['GDX', 'GDXJ', 'SIL']),  # Precious Metals Ratio -> Miners
                'mechanism': 'Metal ratio stress -> Mining sector concentration risk',
                'lead_time_days': [7, 12, 18],
                'ratio_based': True,
                'reliability_score': 0.64
            }
        }
        
        # Exotic correlation patterns that most miss
        self.exotic_patterns = {
            'inverse_volatility_momentum': {
                'description': 'Low VIX -> High momentum stock outperformance 10-15 days later',
                'threshold_conditions': ['VIX < 20th percentile for 5+ days'],
                'target_effect': 'Momentum factor (MTUM) outperforms value (VLUE)',
                'lead_days': [10, 12, 15]
            },
            
            'currency_tech_divergence': {
                'description': 'DXY vs EUR divergence predicts US vs EU tech relative performance',  
                'signal': 'DXY/EUR spread expansion',
                'target': 'QQQ vs EXS5 (EU tech) relative returns',
                'lead_days': [5, 8, 12]
            },
            
            'credit_small_cap_spiral': {
                'description': 'IG credit tightening leads small cap underperformance',
                'signal': 'LQD underperforms TLT',
                'target': 'IWM underperforms SPY', 
                'lead_days': [7, 10, 14]
            }
        }
        
    def discover_hidden_connections(self, data: pd.DataFrame, 
                                  min_lead_days: int = 1,
                                  min_alpha_threshold: float = 0.3) -> Dict:
        """
        Main discovery engine for finding non-obvious cross-market connections
        
        Args:
            data: Market data across multiple asset classes
            min_lead_days: Minimum predictive lead time required
            min_alpha_threshold: Minimum alpha generation threshold
            
        Returns:
            Dictionary of discovered hidden connections with alpha potential
        """
        
        if len(data) < 120:  # Need substantial history
            return {'error': 'Need at least 120 days for cross-market discovery'}
        
        results = {
            'discovery_timestamp': datetime.now().isoformat(),
            'cross_market_signals': [],
            'exotic_pattern_alerts': [],
            'alpha_generating_chains': [],
            'regime_dependent_connections': [],
            'volatility_transmission_paths': [],
            'summary_stats': {}
        }
        
        # 1. Test cross-market chains
        cross_market_results = self._test_cross_market_chains(data, min_lead_days, min_alpha_threshold)
        results['cross_market_signals'] = cross_market_results
        
        # 2. Discover exotic patterns
        exotic_results = self._discover_exotic_patterns(data, min_lead_days)
        results['exotic_pattern_alerts'] = exotic_results
        
        # 3. Find alpha-generating chains
        alpha_chains = self._find_alpha_generating_chains(data, min_alpha_threshold)
        results['alpha_generating_chains'] = alpha_chains
        
        # 4. Regime-dependent connections
        regime_connections = self._find_regime_dependent_connections(data)
        results['regime_dependent_connections'] = regime_connections
        
        # 5. Volatility transmission analysis
        vol_transmission = self._analyze_volatility_transmission(data)
        results['volatility_transmission_paths'] = vol_transmission
        
        # 6. Generate summary statistics
        results['summary_stats'] = self._generate_discovery_summary(results)
        
        return results
    
    def _test_cross_market_chains(self, data: pd.DataFrame, 
                                min_lead_days: int, min_alpha: float) -> List[Dict]:
        """Test predefined cross-market connection chains"""
        
        chain_results = []
        
        for chain_name, chain_info in self.cross_market_chains.items():
            try:
                # Parse chain components
                source_assets = self._parse_asset_list(chain_info['chain'][0], data.columns)
                target_assets = self._parse_asset_list(chain_info['chain'][1], data.columns)
                
                if not source_assets or not target_assets:
                    continue
                
                # Test the chain relationship
                chain_result = self._test_specific_chain(
                    data, source_assets, target_assets, 
                    chain_info, min_lead_days, min_alpha
                )
                
                if chain_result and chain_result.get('alpha_potential', 0) >= min_alpha:
                    chain_results.append({
                        'chain_name': chain_name,
                        'description': chain_info['description'],
                        'mechanism': chain_info['mechanism'],
                        'source_assets': source_assets,
                        'target_assets': target_assets,
                        'optimal_lead_days': chain_result['optimal_lead_days'],
                        'alpha_potential': chain_result['alpha_potential'],
                        'reliability_score': chain_result['reliability_score'],
                        'current_signal_strength': chain_result.get('current_signal_strength', 0),
                        'trading_signal': chain_result.get('trading_signal', 'NEUTRAL'),
                        'confidence': chain_result.get('confidence', 'LOW')
                    })
                    
            except Exception as e:
                logger.warning(f"Error testing chain {chain_name}: {str(e)}")
                continue
        
        # Sort by alpha potential
        return sorted(chain_results, key=lambda x: x['alpha_potential'], reverse=True)
    
    def _parse_asset_list(self, asset_spec, available_columns) -> List[str]:
        """Parse asset specification and return available assets"""
        if isinstance(asset_spec, list):
            return [asset for asset in asset_spec if asset in available_columns]
        elif isinstance(asset_spec, str):
            return [asset_spec] if asset_spec in available_columns else []
        else:
            return []
    
    def _test_specific_chain(self, data: pd.DataFrame, source_assets: List[str], 
                           target_assets: List[str], chain_info: Dict,
                           min_lead_days: int, min_alpha: float) -> Optional[Dict]:
        """Test a specific cross-market chain for alpha generation"""
        
        try:
            # Create composite source signal
            source_signal = self._create_composite_signal(data, source_assets, chain_info)
            
            # Test against each target
            best_alpha = 0
            best_result = None
            
            for target in target_assets:
                if target not in data.columns:
                    continue
                
                # Test multiple lead times
                for lead_days in chain_info['lead_time_days']:
                    if lead_days < min_lead_days:
                        continue
                    
                    alpha_score = self._calculate_alpha_score(
                        source_signal, data[target], lead_days, chain_info
                    )
                    
                    if alpha_score > best_alpha:
                        best_alpha = alpha_score
                        
                        # Generate current trading signal
                        current_signal = self._generate_current_signal(
                            source_signal, data[target], lead_days, chain_info
                        )
                        
                        best_result = {
                            'optimal_lead_days': lead_days,
                            'alpha_potential': alpha_score,
                            'reliability_score': chain_info['reliability_score'],
                            'best_target': target,
                            'current_signal_strength': current_signal['strength'],
                            'trading_signal': current_signal['direction'],
                            'confidence': current_signal['confidence']
                        }
            
            return best_result if best_alpha >= min_alpha else None
            
        except Exception as e:
            logger.warning(f"Error in chain testing: {str(e)}")
            return None
    
    def _create_composite_signal(self, data: pd.DataFrame, 
                               assets: List[str], chain_info: Dict) -> pd.Series:
        """Create composite signal from multiple source assets"""
        
        if len(assets) == 1:
            return data[assets[0]].pct_change()
        
        # Equal weight composite for now - could be optimized
        composite_returns = pd.DataFrame()
        
        for asset in assets:
            if asset in data.columns:
                composite_returns[asset] = data[asset].pct_change()
        
        if composite_returns.empty:
            return pd.Series()
        
        # Handle special signal types
        if chain_info.get('ratio_based'):
            # For ratio-based signals (e.g., GLD/SLV)
            if len(assets) >= 2:
                return (data[assets[0]] / data[assets[1]]).pct_change()
        
        if chain_info.get('spread_sensitivity'):
            # For credit spread signals
            if len(assets) >= 2:
                return composite_returns[assets[0]] - composite_returns[assets[1]]
        
        # Default: equal-weighted composite
        return composite_returns.mean(axis=1)
    
    def _calculate_alpha_score(self, source_signal: pd.Series, target_asset: pd.Series,
                             lead_days: int, chain_info: Dict) -> float:
        """Calculate alpha generation potential of the signal"""
        
        try:
            # Shift source signal by lead days
            source_lagged = source_signal.shift(lead_days)
            target_returns = target_asset.pct_change()
            
            # Align data
            common_idx = source_lagged.index.intersection(target_returns.index)
            source_aligned = source_lagged[common_idx].dropna()
            target_aligned = target_returns[source_aligned.index]
            
            if len(source_aligned) < 30:
                return 0.0
            
            # Calculate different alpha metrics
            correlation_alpha = self._correlation_alpha(source_aligned, target_aligned, chain_info)
            directional_alpha = self._directional_alpha(source_aligned, target_aligned, chain_info)
            volatility_alpha = self._volatility_alpha(source_aligned, target_aligned, chain_info)
            
            # Combine alpha measures
            combined_alpha = (correlation_alpha * 0.4 + 
                            directional_alpha * 0.4 + 
                            volatility_alpha * 0.2)
            
            return min(1.0, combined_alpha)
            
        except Exception as e:
            logger.warning(f"Error calculating alpha score: {str(e)}")
            return 0.0
    
    def _correlation_alpha(self, source: pd.Series, target: pd.Series, chain_info: Dict) -> float:
        """Alpha from correlation-based signals"""
        try:
            corr, p_val = stats.pearsonr(source, target)
            
            if chain_info.get('inverse_relationship'):
                corr = -corr
            
            # Strong correlation with statistical significance
            alpha_score = abs(corr) * (1 - p_val) if p_val < 0.05 else 0
            return alpha_score
            
        except:
            return 0.0
    
    def _directional_alpha(self, source: pd.Series, target: pd.Series, chain_info: Dict) -> float:
        """Alpha from directional prediction accuracy"""
        try:
            # Create signal thresholds
            source_threshold = source.std() * chain_info.get('volatility_threshold', 1.0)
            
            # Strong source signals
            strong_up = source > source_threshold
            strong_down = source < -source_threshold
            
            # Target responses
            target_up = target > 0
            target_down = target < 0
            
            # Directional accuracy
            if chain_info.get('inverse_relationship'):
                accuracy_up = (strong_up & target_down).sum() / strong_up.sum() if strong_up.sum() > 0 else 0
                accuracy_down = (strong_down & target_up).sum() / strong_down.sum() if strong_down.sum() > 0 else 0
            else:
                accuracy_up = (strong_up & target_up).sum() / strong_up.sum() if strong_up.sum() > 0 else 0
                accuracy_down = (strong_down & target_down).sum() / strong_down.sum() if strong_down.sum() > 0 else 0
            
            return (accuracy_up + accuracy_down) / 2
            
        except:
            return 0.0
    
    def _volatility_alpha(self, source: pd.Series, target: pd.Series, chain_info: Dict) -> float:
        """Alpha from volatility prediction"""
        try:
            # Source volatility events
            source_vol_events = abs(source) > source.std() * 2
            
            # Target volatility response
            target_vol = target.rolling(5).std()
            target_vol_increase = target_vol > target_vol.rolling(20).mean()
            
            # Volatility transmission accuracy
            vol_transmission_rate = (source_vol_events & target_vol_increase).sum() / source_vol_events.sum()
            return vol_transmission_rate if source_vol_events.sum() > 5 else 0
            
        except:
            return 0.0
    
    def _generate_current_signal(self, source_signal: pd.Series, target_asset: pd.Series,
                               lead_days: int, chain_info: Dict) -> Dict:
        """Generate current trading signal strength and direction"""
        
        try:
            # Recent source signal strength
            recent_source = source_signal.tail(5).mean()
            source_vol = source_signal.std()
            
            signal_strength = abs(recent_source) / source_vol if source_vol > 0 else 0
            
            # Determine direction
            if chain_info.get('inverse_relationship'):
                direction = 'SELL' if recent_source > 0 else 'BUY'
            else:
                direction = 'BUY' if recent_source > 0 else 'SELL'
            
            # Determine confidence
            if signal_strength > 2.0:
                confidence = 'HIGH'
            elif signal_strength > 1.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
                direction = 'NEUTRAL'
            
            return {
                'strength': min(3.0, signal_strength),
                'direction': direction,
                'confidence': confidence
            }
            
        except:
            return {'strength': 0, 'direction': 'NEUTRAL', 'confidence': 'LOW'}
    
    def _discover_exotic_patterns(self, data: pd.DataFrame, min_lead_days: int) -> List[Dict]:
        """Discover exotic patterns that most traders miss"""
        
        exotic_alerts = []
        
        for pattern_name, pattern_info in self.exotic_patterns.items():
            try:
                alert = self._test_exotic_pattern(data, pattern_name, pattern_info, min_lead_days)
                if alert:
                    exotic_alerts.append(alert)
            except Exception as e:
                logger.warning(f"Error testing exotic pattern {pattern_name}: {str(e)}")
        
        return exotic_alerts
    
    def _test_exotic_pattern(self, data: pd.DataFrame, pattern_name: str, 
                           pattern_info: Dict, min_lead_days: int) -> Optional[Dict]:
        """Test a specific exotic pattern"""
        
        if pattern_name == 'inverse_volatility_momentum':
            return self._test_inverse_vol_momentum(data, pattern_info, min_lead_days)
        elif pattern_name == 'currency_tech_divergence':
            return self._test_currency_tech_divergence(data, pattern_info, min_lead_days)
        elif pattern_name == 'credit_small_cap_spiral':
            return self._test_credit_small_cap_spiral(data, pattern_info, min_lead_days)
        
        return None
    
    def _test_inverse_vol_momentum(self, data: pd.DataFrame, pattern_info: Dict, min_lead_days: int) -> Optional[Dict]:
        """Test the inverse volatility-momentum relationship"""
        
        if '^VIX' not in data.columns:
            return None
        
        try:
            vix = data['^VIX']
            
            # Low VIX periods (20th percentile for 5+ days)
            vix_20th = vix.rolling(60).quantile(0.2)
            low_vix_periods = (vix < vix_20th).rolling(5).sum() >= 5
            
            if low_vix_periods.sum() < 10:  # Need enough occurrences
                return None
            
            # Test if momentum stocks outperform after low VIX periods
            momentum_assets = ['QQQ', 'ARKK', 'MTUM']  # Momentum proxies
            value_assets = ['VTV', 'VLUE', 'XLF']       # Value proxies
            
            available_momentum = [a for a in momentum_assets if a in data.columns]
            available_value = [a for a in value_assets if a in data.columns]
            
            if not available_momentum or not available_value:
                return None
            
            # Calculate momentum vs value performance after low VIX
            momentum_performance = []
            value_performance = []
            
            for lead_days in pattern_info['lead_days']:
                if lead_days < min_lead_days:
                    continue
                
                # Future performance after low VIX periods
                for i in low_vix_periods[low_vix_periods].index:
                    future_date = i + pd.Timedelta(days=lead_days)
                    if future_date in data.index:
                        # Calculate forward returns
                        momentum_ret = np.mean([
                            (data.loc[future_date, asset] / data.loc[i, asset] - 1) * 100
                            for asset in available_momentum
                        ])
                        value_ret = np.mean([
                            (data.loc[future_date, asset] / data.loc[i, asset] - 1) * 100
                            for asset in available_value
                        ])
                        
                        momentum_performance.append(momentum_ret)
                        value_performance.append(value_ret)
            
            if len(momentum_performance) < 5:
                return None
            
            # Test if momentum significantly outperforms value
            momentum_avg = np.mean(momentum_performance)
            value_avg = np.mean(value_performance)
            outperformance = momentum_avg - value_avg
            
            # Statistical significance test
            try:
                t_stat, p_val = stats.ttest_ind(momentum_performance, value_performance)
                significant = p_val < 0.05 and outperformance > 0
            except:
                significant = False
            
            if significant and outperformance > 0.5:  # At least 0.5% outperformance
                # Check current VIX condition
                current_vix = vix.iloc[-1]
                recent_vix_20th = vix.rolling(60).quantile(0.2).iloc[-1]
                current_low_vix = current_vix < recent_vix_20th
                
                return {
                    'pattern_name': 'inverse_volatility_momentum',
                    'description': pattern_info['description'],
                    'outperformance': outperformance,
                    'statistical_significance': p_val,
                    'current_signal': 'ACTIVE' if current_low_vix else 'INACTIVE',
                    'expected_lead_days': pattern_info['lead_days'],
                    'confidence': 'HIGH' if p_val < 0.01 else 'MEDIUM'
                }
        
        except Exception as e:
            logger.warning(f"Error in inverse vol momentum test: {str(e)}")
        
        return None
    
    def _test_currency_tech_divergence(self, data: pd.DataFrame, pattern_info: Dict, min_lead_days: int) -> Optional[Dict]:
        """Test USD vs EUR impact on US vs EU tech"""
        
        required_assets = ['DXY', 'EUR=X', 'QQQ']  # Need at least these
        available = [asset for asset in required_assets if asset in data.columns]
        
        if len(available) < 2:
            return None
        
        # Implementation would test DXY vs EUR divergence -> QQQ vs EU tech performance
        # For now, return None as this needs EU tech data
        return None
    
    def _test_credit_small_cap_spiral(self, data: pd.DataFrame, pattern_info: Dict, min_lead_days: int) -> Optional[Dict]:
        """Test credit tightening impact on small caps"""
        
        if not all(asset in data.columns for asset in ['LQD', 'TLT', 'IWM', 'SPY']):
            return None
        
        try:
            # Credit spread proxy: LQD vs TLT performance
            lqd_returns = data['LQD'].pct_change()
            tlt_returns = data['TLT'].pct_change()
            credit_spread_proxy = lqd_returns - tlt_returns  # LQD underperformance = credit tightening
            
            # Small cap vs large cap relative performance
            iwm_returns = data['IWM'].pct_change()
            spy_returns = data['SPY'].pct_change()
            small_vs_large = iwm_returns - spy_returns
            
            # Test if credit tightening predicts small cap underperformance
            alpha_scores = []
            
            for lead_days in pattern_info['lead_days']:
                if lead_days < min_lead_days:
                    continue
                
                credit_lagged = credit_spread_proxy.shift(lead_days)
                
                # Credit tightening events (LQD underperforms TLT significantly)
                credit_tightening = credit_lagged < -credit_lagged.std()
                
                if credit_tightening.sum() < 5:
                    continue
                
                # Small cap performance during credit tightening periods
                small_cap_response = small_vs_large[credit_tightening].mean()
                
                # Negative response indicates small caps underperform (expected)
                alpha_score = -small_cap_response if small_cap_response < 0 else 0
                alpha_scores.append((lead_days, alpha_score))
            
            if not alpha_scores:
                return None
            
            best_lead, best_alpha = max(alpha_scores, key=lambda x: x[1])
            
            if best_alpha > 0.002:  # At least 20 bps alpha
                # Current signal strength
                recent_credit_signal = credit_spread_proxy.tail(5).mean()
                signal_strength = abs(recent_credit_signal) / credit_spread_proxy.std()
                
                return {
                    'pattern_name': 'credit_small_cap_spiral',
                    'description': pattern_info['description'],
                    'optimal_lead_days': best_lead,
                    'alpha_potential': best_alpha,
                    'current_signal_strength': signal_strength,
                    'trading_signal': 'SELL_SMALL_CAP' if recent_credit_signal < -0.001 else 'NEUTRAL',
                    'confidence': 'HIGH' if signal_strength > 1.5 else 'MEDIUM'
                }
        
        except Exception as e:
            logger.warning(f"Error in credit small cap test: {str(e)}")
        
        return None
    
    def _find_alpha_generating_chains(self, data: pd.DataFrame, min_alpha: float) -> List[Dict]:
        """Find unexpected alpha-generating cross-asset chains"""
        
        alpha_chains = []
        
        # Test unexpected asset pairs for alpha generation
        unexpected_pairs = [
            (['GLD', 'SLV'], ['TSLA', 'NVDA'], 'Precious metals volatility -> Tech momentum'),
            (['TLT', '^TNX'], ['XLF', 'BAC'], 'Bond volatility -> Banking sector rotation'),
            (['^VIX'], ['EEM', 'VWO'], 'US fear -> EM weakness transmission'),
            (['DXY'], ['QQQ', 'XLK'], 'Dollar strength -> Growth stock discount'),
            (['USO', 'XLE'], ['ICLN', 'TAN'], 'Energy volatility -> Clean energy inverse momentum')
        ]
        
        for sources, targets, description in unexpected_pairs:
            available_sources = [s for s in sources if s in data.columns]
            available_targets = [t for t in targets if t in data.columns]
            
            if not available_sources or not available_targets:
                continue
            
            try:
                chain_alpha = self._test_alpha_chain(data, available_sources, available_targets, description)
                if chain_alpha and chain_alpha['alpha_score'] >= min_alpha:
                    alpha_chains.append(chain_alpha)
            except Exception as e:
                logger.warning(f"Error testing alpha chain: {str(e)}")
        
        return sorted(alpha_chains, key=lambda x: x['alpha_score'], reverse=True)
    
    def _test_alpha_chain(self, data: pd.DataFrame, sources: List[str], 
                         targets: List[str], description: str) -> Optional[Dict]:
        """Test a specific alpha-generating chain"""
        
        try:
            # Create source composite signal
            source_composite = pd.DataFrame()
            for source in sources:
                source_composite[source] = data[source].pct_change()
            source_signal = source_composite.mean(axis=1)
            
            # Test against targets
            best_alpha = 0
            best_config = None
            
            for target in targets:
                target_returns = data[target].pct_change()
                
                for lag in [3, 5, 7, 10, 14]:
                    source_lagged = source_signal.shift(lag)
                    
                    # Calculate alpha metric
                    common_idx = source_lagged.index.intersection(target_returns.index)
                    if len(common_idx) < 50:
                        continue
                    
                    source_aligned = source_lagged[common_idx].dropna()
                    target_aligned = target_returns[source_aligned.index]
                    
                    if len(source_aligned) < 30:
                        continue
                    
                    # Alpha calculation: Sharpe ratio of signal-based strategy
                    alpha_score = self._calculate_strategy_alpha(source_aligned, target_aligned)
                    
                    if alpha_score > best_alpha:
                        best_alpha = alpha_score
                        best_config = {
                            'target': target,
                            'optimal_lag': lag,
                            'source_assets': sources,
                            'description': description
                        }
            
            if best_alpha > 0.2:  # Minimum alpha threshold
                return {
                    'alpha_score': best_alpha,
                    'optimal_lag': best_config['optimal_lag'],
                    'target_asset': best_config['target'],
                    'source_assets': sources,
                    'description': description,
                    'sharpe_equivalent': best_alpha * 2  # Rough Sharpe approximation
                }
        
        except Exception as e:
            logger.warning(f"Error in alpha chain test: {str(e)}")
        
        return None
    
    def _calculate_strategy_alpha(self, signal: pd.Series, target_returns: pd.Series) -> float:
        """Calculate alpha of a signal-based trading strategy"""
        
        try:
            # Simple strategy: go long when signal > 0, short when < 0
            strategy_returns = np.sign(signal) * target_returns
            
            # Remove extreme outliers
            strategy_returns = strategy_returns[abs(strategy_returns) < strategy_returns.std() * 3]
            
            if len(strategy_returns) < 20:
                return 0.0
            
            # Alpha metrics
            mean_return = strategy_returns.mean()
            volatility = strategy_returns.std()
            
            if volatility == 0:
                return 0.0
            
            # Sharpe-like ratio (annualized)
            alpha_score = (mean_return / volatility) * np.sqrt(252)
            
            # Information ratio consideration
            hit_rate = (strategy_returns > 0).mean()
            
            # Combine return alpha with hit rate
            combined_alpha = alpha_score * (0.7 + 0.3 * (hit_rate - 0.5) * 2)
            
            return max(0, combined_alpha)
            
        except:
            return 0.0
    
    def _find_regime_dependent_connections(self, data: pd.DataFrame) -> List[Dict]:
        """Find connections that only work in specific market regimes"""
        
        # Implementation for regime-dependent analysis
        return []
    
    def _analyze_volatility_transmission(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze how volatility transmits across distant markets"""
        
        # Implementation for volatility transmission analysis
        return []
    
    def _generate_discovery_summary(self, results: Dict) -> Dict:
        """Generate summary statistics of discoveries"""
        
        total_signals = len(results['cross_market_signals'])
        high_alpha_signals = len([s for s in results['cross_market_signals'] if s.get('alpha_potential', 0) > 0.5])
        
        exotic_alerts = len(results['exotic_pattern_alerts'])
        alpha_chains = len(results['alpha_generating_chains'])
        
        return {
            'total_cross_market_signals': total_signals,
            'high_alpha_signals': high_alpha_signals,
            'exotic_pattern_alerts': exotic_alerts,
            'alpha_generating_chains': alpha_chains,
            'discovery_quality_score': (high_alpha_signals * 2 + exotic_alerts + alpha_chains) / 10
        }

def main():
    """Test the cross-market discovery engine"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("🔍 CROSS-MARKET DISCOVERY ENGINE")
    print("=" * 60)
    print("Finding hidden connections across distant asset classes")
    print("=" * 60)
    
    # Load market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No market data available")
        return
    
    print(f"\n📊 Available assets: {', '.join(data.columns[:10])}{'...' if len(data.columns) > 10 else ''}")
    
    engine = CrossMarketDiscoveryEngine()
    results = engine.discover_hidden_connections(data, min_lead_days=2, min_alpha_threshold=0.25)
    
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    # Display results
    print(f"\n🎯 DISCOVERY SUMMARY:")
    summary = results['summary_stats']
    print(f"   Cross-Market Signals: {summary['total_cross_market_signals']}")
    print(f"   High Alpha Signals: {summary['high_alpha_signals']}")
    print(f"   Exotic Pattern Alerts: {summary['exotic_pattern_alerts']}")
    print(f"   Alpha Generating Chains: {summary['alpha_generating_chains']}")
    print(f"   Discovery Quality Score: {summary['discovery_quality_score']:.2f}/1.0")
    
    # Cross-market signals
    cross_signals = results['cross_market_signals']
    if cross_signals:
        print(f"\n🔗 CROSS-MARKET CONNECTIONS ({len(cross_signals)} found):")
        for i, signal in enumerate(cross_signals[:3], 1):
            print(f"\n   {i}. {signal['chain_name'].upper()}")
            print(f"      Description: {signal['description']}")
            print(f"      Mechanism: {signal['mechanism']}")
            print(f"      Sources: {' + '.join(signal['source_assets'])}")
            print(f"      Targets: {' + '.join(signal['target_assets'])}")
            print(f"      Optimal Lead: {signal['optimal_lead_days']} days")
            print(f"      Alpha Potential: {signal['alpha_potential']:.3f}")
            print(f"      Current Signal: {signal['trading_signal']} ({signal['confidence']})")
    
    # Exotic patterns
    exotic_patterns = results['exotic_pattern_alerts']
    if exotic_patterns:
        print(f"\n🎨 EXOTIC PATTERN ALERTS ({len(exotic_patterns)} found):")
        for i, pattern in enumerate(exotic_patterns, 1):
            print(f"\n   {i}. {pattern['pattern_name'].upper()}")
            print(f"      Description: {pattern['description']}")
            print(f"      Current Status: {pattern.get('current_signal', 'Unknown')}")
            print(f"      Confidence: {pattern.get('confidence', 'Unknown')}")
    
    # Alpha generating chains
    alpha_chains = results['alpha_generating_chains']
    if alpha_chains:
        print(f"\n💰 ALPHA GENERATING CHAINS ({len(alpha_chains)} found):")
        for i, chain in enumerate(alpha_chains, 1):
            print(f"\n   {i}. {chain['description']}")
            print(f"      Sources: {' + '.join(chain['source_assets'])}")
            print(f"      Target: {chain['target_asset']}")
            print(f"      Optimal Lead: {chain['optimal_lag']} days")
            print(f"      Alpha Score: {chain['alpha_score']:.3f}")
            print(f"      Sharpe Equivalent: {chain['sharpe_equivalent']:.2f}")
    
    print(f"\n✨ Cross-market discovery analysis complete!")

if __name__ == "__main__":
    main()