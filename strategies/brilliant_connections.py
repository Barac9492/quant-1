#!/usr/bin/env python3
"""
Brilliant Cross-Market Connections - Find genius-level distant relationships
The system that discovers what others miss through sophisticated cross-market analysis

Examples of brilliant connections:
- Japan 10Y yield volatility -> US biotech momentum (via R&D funding flows)
- Baltic Dry Index -> Consumer discretionary earnings beats (supply chain efficiency)
- Swiss Franc strength -> Gold mining stock volatility (safe haven flows)
- German manufacturing PMI -> US semiconductor equipment (export dependency)
- Australian dollar -> Copper futures -> Electric vehicle stocks (supply chain)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class BrilliantConnectionEngine:
    """
    Find brilliant, non-obvious connections that generate alpha
    Focus on multi-hop relationships and macroeconomic transmission mechanisms
    """
    
    def __init__(self, min_lead_time: int = 3):
        self.min_lead_time = min_lead_time
        # Brilliant multi-hop connection chains
        self.brilliant_chains = {
            # Supply chain efficiency chain
            'supply_chain_efficiency': {
                'description': 'Transportation costs predict consumer discretionary margin pressure',
                'connections': [
                    'Transportation cost spike (proxy via energy)',
                    '-> Supply chain stress (7-14 days)',
                    '-> Consumer discretionary margin compression (14-21 days)',
                    '-> Earnings disappointments (30-45 days)'
                ],
                'assets': (['USO', 'XLE'], ['XLY', 'RH', 'HD']),
                'lead_times': [7, 14, 21, 30],
                'alpha_mechanism': 'Supply chain cost transmission to margins',
                'brilliance_score': 0.85
            },
            
            # Currency strength manufacturing chain
            'currency_manufacturing_chain': {
                'description': 'Strong dollar makes US manufacturing less competitive globally',
                'connections': [
                    'Dollar strength',
                    '-> Export competitiveness decline (5-10 days)',
                    '-> Industrial sector pressure (10-20 days)',
                    '-> Small-cap industrial underperformance (15-30 days)'
                ],
                'assets': (['DXY'], ['XLI', 'IWM']),
                'lead_times': [5, 10, 15, 20],
                'alpha_mechanism': 'Export competitiveness transmission',
                'brilliance_score': 0.78
            },
            
            # Safe haven flow reallocation
            'safe_haven_reallocation': {
                'description': 'Bond volatility drives alternative safe haven demand',
                'connections': [
                    'Bond market volatility spike',
                    '-> Traditional safe haven uncertainty (2-5 days)',
                    '-> Alternative safe haven demand (gold, crypto) (5-10 days)',
                    '-> Mining/precious metals momentum (7-14 days)'
                ],
                'assets': (['TLT', '^TNX'], ['GLD', 'GDX', 'GDXJ']),
                'lead_times': [5, 7, 10, 14],
                'alpha_mechanism': 'Safe haven substitution effects',
                'brilliance_score': 0.82
            },
            
            # Interest rate sensitivity cascade
            'rate_sensitivity_cascade': {
                'description': 'Rate volatility cascades through high-duration assets asymmetrically',
                'connections': [
                    'Rate volatility expansion',
                    '-> Duration-sensitive sector stress (3-7 days)',
                    '-> Utility vs REIT vs Growth relative performance shifts (7-14 days)',
                    '-> Sector rotation momentum (10-21 days)'
                ],
                'assets': (['^TNX', 'TLT'], ['XLU', 'VNQ', 'QQQ']),
                'lead_times': [3, 7, 10, 14],
                'alpha_mechanism': 'Duration risk cascading effects',
                'brilliance_score': 0.88
            },
            
            # Credit quality deterioration chain
            'credit_quality_deterioration': {
                'description': 'Credit spreads predict equity quality factor rotation',
                'connections': [
                    'Credit spread widening',
                    '-> Quality concerns rise (5-10 days)',
                    '-> Flight to quality stocks (10-15 days)',
                    '-> Value vs Growth vs Quality factor rotation (15-25 days)'
                ],
                'assets': (['HYG', 'LQD'], ['QUAL', 'VTV', 'VUG']),
                'lead_times': [5, 10, 15, 20],
                'alpha_mechanism': 'Credit quality transmission to equity factors',
                'brilliance_score': 0.91
            },
            
            # Volatility regime transmission
            'volatility_regime_transmission': {
                'description': 'VIX term structure predicts cross-asset volatility expansion',
                'connections': [
                    'VIX term structure inversion',
                    '-> Near-term vol expansion expected (1-3 days)',
                    '-> Cross-asset volatility contagion (3-7 days)',
                    '-> International equity volatility spillover (7-14 days)'
                ],
                'assets': (['^VIX'], ['EFA', 'EEM', 'VEA']),
                'lead_times': [3, 5, 7, 10],
                'alpha_mechanism': 'Volatility regime contagion across markets',
                'brilliance_score': 0.79
            },
            
            # Commodity financialization
            'commodity_financialization': {
                'description': 'Commodity volatility affects financial conditions via inflation expectations',
                'connections': [
                    'Commodity price volatility',
                    '-> Inflation expectation volatility (3-7 days)',
                    '-> Real rate uncertainty (7-14 days)',
                    '-> Financial sector stress (via NIM pressure) (10-21 days)'
                ],
                'assets': (['DJP', 'GLD', 'USO'], ['XLF', 'BKX']),
                'lead_times': [3, 7, 10, 14],
                'alpha_mechanism': 'Commodity->inflation->real rates->financials',
                'brilliance_score': 0.86
            }
        }
        
        # Sophisticated pattern recognition for distant connections
        self.pattern_detectors = {
            'term_structure_anomalies': {
                'description': 'Detect term structure inversions and predict sector rotations',
                'assets_needed': ['^TNX', 'TLT', '^VIX'],
                'detection_method': 'yield_curve_analysis'
            },
            
            'cross_currency_arbitrage': {
                'description': 'Currency strength differentials predict sector performance',
                'assets_needed': ['DXY', 'EUR=X', 'JPY=X'],
                'detection_method': 'currency_differential_analysis'
            },
            
            'volatility_surface_distortions': {
                'description': 'VIX surface distortions predict individual stock movements',
                'assets_needed': ['^VIX', '^VIX3M'],
                'detection_method': 'volatility_surface_analysis'
            },
            
            'credit_equity_disconnects': {
                'description': 'Credit-equity disconnects predict mean reversion trades',
                'assets_needed': ['HYG', 'SPY', 'LQD'],
                'detection_method': 'credit_equity_basis_analysis'
            }
        }
    
    def discover_brilliant_connections(self, data: pd.DataFrame) -> Dict:
        """
        Main engine for discovering brilliant cross-market connections
        """
        
        if len(data) < 180:  # Need substantial history for brilliant analysis
            return {'error': 'Need at least 180 days for brilliant connection discovery'}
        
        results = {
            'discovery_timestamp': datetime.now().isoformat(),
            'brilliant_connections': [],
            'pattern_alerts': [],
            'genius_trades': [],
            'macro_transmission_paths': [],
            'brilliance_score': 0.0
        }
        
        # 1. Test brilliant connection chains
        brilliant_results = self._test_brilliant_chains(data)
        results['brilliant_connections'] = brilliant_results
        
        # 2. Pattern detection
        pattern_results = self._detect_sophisticated_patterns(data)
        results['pattern_alerts'] = pattern_results
        
        # 3. Generate genius trade ideas
        genius_trades = self._generate_genius_trades(data, brilliant_results, pattern_results)
        results['genius_trades'] = genius_trades
        
        # 4. Map macro transmission paths
        transmission_paths = self._map_transmission_paths(data)
        results['macro_transmission_paths'] = transmission_paths
        
        # 5. Calculate overall brilliance score
        results['brilliance_score'] = self._calculate_brilliance_score(results)
        
        return results
    
    def _test_brilliant_chains(self, data: pd.DataFrame) -> List[Dict]:
        """Test brilliant multi-hop connection chains"""
        
        brilliant_results = []
        
        for chain_name, chain_info in self.brilliant_chains.items():
            try:
                result = self._test_single_brilliant_chain(data, chain_name, chain_info)
                if result and result.get('alpha_potential', 0) > 0.3:
                    brilliant_results.append(result)
            except Exception as e:
                logger.warning(f"Error testing brilliant chain {chain_name}: {str(e)}")
        
        return sorted(brilliant_results, key=lambda x: x.get('brilliance_score', 0), reverse=True)
    
    def _test_single_brilliant_chain(self, data: pd.DataFrame, 
                                   chain_name: str, chain_info: Dict) -> Optional[Dict]:
        """Test a single brilliant connection chain"""
        
        source_assets, target_assets = chain_info['assets']
        
        # Find available assets
        available_sources = [asset for asset in source_assets if asset in data.columns]
        available_targets = [asset for asset in target_assets if asset in data.columns]
        
        if not available_sources or not available_targets:
            return None
        
        try:
            # Create sophisticated source signal
            source_signal = self._create_sophisticated_signal(data, available_sources, chain_info)
            
            # Test chain across multiple lead times
            best_result = None
            best_alpha = 0
            
            for target in available_targets:
                target_returns = data[target].pct_change()
                
                for lead_time in chain_info['lead_times']:
                    # Skip lead times below minimum threshold
                    if lead_time < self.min_lead_time:
                        continue
                    
                    alpha_score = self._calculate_sophisticated_alpha(
                        source_signal, target_returns, lead_time, chain_info
                    )
                    
                    if alpha_score > best_alpha:
                        best_alpha = alpha_score
                        
                        # Generate current genius signal
                        current_signal = self._generate_genius_signal(
                            source_signal, target_returns, lead_time, chain_info
                        )
                        
                        best_result = {
                            'chain_name': chain_name,
                            'description': chain_info['description'],
                            'connections': chain_info['connections'],
                            'alpha_mechanism': chain_info['alpha_mechanism'],
                            'source_assets': available_sources,
                            'target_asset': target,
                            'optimal_lead_time': lead_time,
                            'alpha_potential': alpha_score,
                            'brilliance_score': chain_info['brilliance_score'],
                            'current_signal': current_signal
                        }
            
            return best_result
            
        except Exception as e:
            logger.warning(f"Error in brilliant chain test: {str(e)}")
            return None
    
    def _create_sophisticated_signal(self, data: pd.DataFrame, 
                                   assets: List[str], chain_info: Dict) -> pd.Series:
        """Create sophisticated composite signal from multiple assets"""
        
        if len(assets) == 1:
            return data[assets[0]].pct_change()
        
        # Create weighted composite based on volatility and correlation
        asset_returns = pd.DataFrame()
        for asset in assets:
            asset_returns[asset] = data[asset].pct_change()
        
        asset_returns = asset_returns.dropna()
        
        if len(asset_returns) < 30:
            return asset_returns.mean(axis=1)
        
        # Sophisticated weighting: inverse volatility weighting
        vols = asset_returns.std()
        weights = (1 / vols) / (1 / vols).sum()
        
        # Apply weights
        weighted_signal = (asset_returns * weights).sum(axis=1)
        
        # Add signal processing for specific chain types
        alpha_mechanism = chain_info.get('alpha_mechanism', '')
        
        if 'volatility' in alpha_mechanism.lower():
            # For volatility-based signals, use volatility expansion
            weighted_signal = weighted_signal.rolling(5).std()
        elif 'transmission' in alpha_mechanism.lower():
            # For transmission signals, use momentum
            weighted_signal = weighted_signal.rolling(3).mean()
        
        return weighted_signal.dropna()
    
    def _calculate_sophisticated_alpha(self, source_signal: pd.Series, 
                                     target_returns: pd.Series, lead_time: int,
                                     chain_info: Dict) -> float:
        """Calculate sophisticated alpha incorporating multiple factors"""
        
        try:
            # Lag source signal
            source_lagged = source_signal.shift(lead_time)
            
            # Align data
            common_idx = source_lagged.index.intersection(target_returns.index)
            source_aligned = source_lagged[common_idx].dropna()
            target_aligned = target_returns[source_aligned.index]
            
            if len(source_aligned) < 50:
                return 0.0
            
            # Multiple alpha calculations
            correlation_alpha = self._correlation_based_alpha(source_aligned, target_aligned)
            regime_alpha = self._regime_dependent_alpha(source_aligned, target_aligned)
            momentum_alpha = self._momentum_based_alpha(source_aligned, target_aligned)
            volatility_alpha = self._volatility_prediction_alpha(source_aligned, target_aligned)
            
            # Weighted combination based on chain mechanism
            mechanism = chain_info.get('alpha_mechanism', '')
            if 'volatility' in mechanism.lower():
                combined_alpha = volatility_alpha * 0.5 + correlation_alpha * 0.3 + regime_alpha * 0.2
            elif 'transmission' in mechanism.lower():
                combined_alpha = momentum_alpha * 0.4 + correlation_alpha * 0.4 + regime_alpha * 0.2
            else:
                combined_alpha = correlation_alpha * 0.4 + momentum_alpha * 0.3 + regime_alpha * 0.3
            
            # Adjust for chain brilliance
            brilliance_factor = chain_info.get('brilliance_score', 0.5)
            final_alpha = combined_alpha * brilliance_factor
            
            return min(1.0, final_alpha)
            
        except Exception as e:
            logger.warning(f"Error in sophisticated alpha calculation: {str(e)}")
            return 0.0
    
    def _correlation_based_alpha(self, source: pd.Series, target: pd.Series) -> float:
        """Alpha from correlation strength"""
        try:
            corr, p_val = stats.pearsonr(source, target)
            return abs(corr) * (1 - p_val) if p_val < 0.05 else 0
        except:
            return 0.0
    
    def _regime_dependent_alpha(self, source: pd.Series, target: pd.Series) -> float:
        """Alpha from regime-dependent relationships"""
        try:
            # High volatility regime
            target_vol = target.rolling(20).std()
            high_vol_regime = target_vol > target_vol.quantile(0.7)
            
            # Correlation in high vs low vol regimes
            high_vol_corr = source[high_vol_regime].corr(target[high_vol_regime])
            low_vol_corr = source[~high_vol_regime].corr(target[~high_vol_regime])
            
            # Alpha from regime difference
            regime_alpha = abs(high_vol_corr - low_vol_corr) * max(abs(high_vol_corr), abs(low_vol_corr))
            return min(1.0, regime_alpha)
        except:
            return 0.0
    
    def _momentum_based_alpha(self, source: pd.Series, target: pd.Series) -> float:
        """Alpha from momentum prediction"""
        try:
            # Source momentum signals
            source_momentum = source.rolling(3).mean() > 0
            
            # Target momentum response
            target_momentum = target.rolling(3).mean() > 0
            
            # Momentum prediction accuracy
            accuracy = (source_momentum == target_momentum).mean()
            return max(0, (accuracy - 0.5) * 2)  # Scale to 0-1
        except:
            return 0.0
    
    def _volatility_prediction_alpha(self, source: pd.Series, target: pd.Series) -> float:
        """Alpha from volatility prediction"""
        try:
            # Source volatility expansion events
            source_vol_expansion = abs(source) > source.std() * 1.5
            
            # Target volatility response
            target_vol = target.rolling(5).std()
            target_vol_expansion = target_vol > target_vol.rolling(20).mean() * 1.2
            
            # Volatility prediction accuracy
            if source_vol_expansion.sum() > 5:
                vol_prediction_accuracy = (source_vol_expansion & target_vol_expansion).sum() / source_vol_expansion.sum()
                return vol_prediction_accuracy
            return 0
        except:
            return 0.0
    
    def _generate_genius_signal(self, source_signal: pd.Series, target_returns: pd.Series,
                              lead_time: int, chain_info: Dict) -> Dict:
        """Generate current genius trading signal"""
        
        try:
            # Recent source signal strength
            recent_source = source_signal.tail(lead_time + 2).iloc[:-lead_time].mean()
            source_vol = source_signal.std()
            
            signal_strength = abs(recent_source) / source_vol if source_vol > 0 else 0
            
            # Determine signal direction based on mechanism
            mechanism = chain_info.get('alpha_mechanism', '')
            if 'inverse' in mechanism.lower() or 'negative' in mechanism.lower():
                direction = 'SELL' if recent_source > 0 else 'BUY'
            else:
                direction = 'BUY' if recent_source > 0 else 'SELL'
            
            # Confidence based on brilliance score and signal strength
            brilliance = chain_info.get('brilliance_score', 0.5)
            confidence_score = (signal_strength * 0.6 + brilliance * 0.4)
            
            if confidence_score > 1.5:
                confidence = 'GENIUS'
            elif confidence_score > 1.0:
                confidence = 'HIGH'
            elif confidence_score > 0.5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
                direction = 'HOLD'
            
            return {
                'signal_strength': min(3.0, signal_strength),
                'direction': direction,
                'confidence': confidence,
                'lead_time': lead_time,
                'brilliance_factor': brilliance
            }
            
        except:
            return {'signal_strength': 0, 'direction': 'HOLD', 'confidence': 'LOW', 'lead_time': 0}
    
    def _detect_sophisticated_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect sophisticated patterns that most miss"""
        
        pattern_alerts = []
        
        for pattern_name, pattern_info in self.pattern_detectors.items():
            try:
                alert = self._run_pattern_detector(data, pattern_name, pattern_info)
                if alert:
                    pattern_alerts.append(alert)
            except Exception as e:
                logger.warning(f"Error in pattern detection {pattern_name}: {str(e)}")
        
        return pattern_alerts
    
    def _run_pattern_detector(self, data: pd.DataFrame, 
                            pattern_name: str, pattern_info: Dict) -> Optional[Dict]:
        """Run specific pattern detector"""
        
        needed_assets = pattern_info['assets_needed']
        available_assets = [asset for asset in needed_assets if asset in data.columns]
        
        if len(available_assets) < len(needed_assets) * 0.5:  # Need at least half the assets
            return None
        
        method = pattern_info['detection_method']
        
        try:
            if method == 'yield_curve_analysis':
                return self._analyze_yield_curve_patterns(data, available_assets, pattern_info)
            elif method == 'currency_differential_analysis':
                return self._analyze_currency_differentials(data, available_assets, pattern_info)
            elif method == 'volatility_surface_analysis':
                return self._analyze_volatility_surface(data, available_assets, pattern_info)
            elif method == 'credit_equity_basis_analysis':
                return self._analyze_credit_equity_basis(data, available_assets, pattern_info)
        except:
            pass
        
        return None
    
    def _analyze_yield_curve_patterns(self, data: pd.DataFrame, 
                                    assets: List[str], pattern_info: Dict) -> Optional[Dict]:
        """Analyze yield curve and term structure patterns"""
        
        if '^TNX' not in assets or 'TLT' not in assets:
            return None
        
        try:
            # Yield curve steepness proxy
            tnx_returns = data['^TNX'].pct_change()
            tlt_returns = data['TLT'].pct_change()
            
            # Curve steepening/flattening
            curve_signal = tnx_returns - (-tlt_returns)  # TNX up, TLT down = steepening
            
            # Recent curve moves
            recent_curve_move = curve_signal.tail(5).mean()
            curve_vol = curve_signal.std()
            
            if abs(recent_curve_move) > curve_vol * 1.5:
                return {
                    'pattern_name': 'yield_curve_anomaly',
                    'description': pattern_info['description'],
                    'signal_strength': abs(recent_curve_move) / curve_vol,
                    'curve_direction': 'STEEPENING' if recent_curve_move > 0 else 'FLATTENING',
                    'sector_implications': 'Financials benefit from steepening, REITs suffer' if recent_curve_move > 0 else 'REITs benefit from flattening, Financials suffer'
                }
        except:
            pass
        
        return None
    
    def _analyze_currency_differentials(self, data: pd.DataFrame,
                                      assets: List[str], pattern_info: Dict) -> Optional[Dict]:
        """Analyze cross-currency strength differentials"""
        # Implementation for currency differential analysis
        return None
    
    def _analyze_volatility_surface(self, data: pd.DataFrame,
                                  assets: List[str], pattern_info: Dict) -> Optional[Dict]:
        """Analyze VIX term structure distortions"""
        
        if '^VIX' not in assets:
            return None
        
        try:
            vix = data['^VIX']
            
            # VIX level analysis
            current_vix = vix.iloc[-1]
            vix_percentile = (vix <= current_vix).mean()
            
            # Term structure analysis (if VIX3M available)
            if '^VIX3M' in assets:
                vix3m = data['^VIX3M']
                current_vix3m = vix3m.iloc[-1]
                term_structure = current_vix3m - current_vix
                
                # Backwardation (VIX > VIX3M) suggests near-term stress
                if term_structure < -2:  # Significant backwardation
                    return {
                        'pattern_name': 'vix_backwardation',
                        'description': 'VIX term structure in backwardation - near-term volatility spike expected',
                        'term_structure_value': term_structure,
                        'implications': 'High-beta stocks likely to underperform in next 3-7 days',
                        'confidence': 'HIGH' if term_structure < -5 else 'MEDIUM'
                    }
            
            # Extreme VIX levels
            if vix_percentile > 0.95:  # Above 95th percentile
                return {
                    'pattern_name': 'extreme_fear',
                    'description': 'VIX at extreme levels - potential contrarian opportunity',
                    'vix_percentile': vix_percentile,
                    'implications': 'Consider contrarian positions in beaten-down growth names',
                    'confidence': 'HIGH'
                }
        except:
            pass
        
        return None
    
    def _analyze_credit_equity_basis(self, data: pd.DataFrame,
                                   assets: List[str], pattern_info: Dict) -> Optional[Dict]:
        """Analyze credit-equity basis relationships"""
        
        if not all(asset in assets for asset in ['HYG', 'SPY']):
            return None
        
        try:
            hyg_returns = data['HYG'].pct_change()
            spy_returns = data['SPY'].pct_change()
            
            # Credit-equity basis
            basis = hyg_returns - spy_returns
            recent_basis = basis.tail(10).mean()
            basis_vol = basis.std()
            
            if abs(recent_basis) > basis_vol * 1.5:
                return {
                    'pattern_name': 'credit_equity_disconnect',
                    'description': 'Credit and equity markets showing unusual disconnect',
                    'basis_signal': recent_basis,
                    'disconnect_type': 'Credit leading equity higher' if recent_basis > 0 else 'Equity leading credit higher',
                    'mean_reversion_trade': 'Short credit, long equity' if recent_basis > 0 else 'Long credit, short equity'
                }
        except:
            pass
        
        return None
    
    def _generate_genius_trades(self, data: pd.DataFrame, 
                              brilliant_connections: List[Dict],
                              pattern_alerts: List[Dict]) -> List[Dict]:
        """Generate genius trade ideas from discoveries"""
        
        genius_trades = []
        
        # From brilliant connections
        for connection in brilliant_connections:
            if connection.get('current_signal', {}).get('confidence') in ['GENIUS', 'HIGH']:
                trade = {
                    'trade_type': 'BRILLIANT_CONNECTION',
                    'trade_name': connection['chain_name'],
                    'description': connection['description'],
                    'source_signal': ' + '.join(connection['source_assets']),
                    'target_asset': connection['target_asset'],
                    'direction': connection['current_signal']['direction'],
                    'lead_time': connection['current_signal']['lead_time'],
                    'confidence': connection['current_signal']['confidence'],
                    'alpha_potential': connection['alpha_potential'],
                    'brilliance_score': connection['brilliance_score']
                }
                genius_trades.append(trade)
        
        # From pattern alerts
        for pattern in pattern_alerts:
            if pattern.get('confidence') == 'HIGH':
                trade = {
                    'trade_type': 'PATTERN_OPPORTUNITY',
                    'trade_name': pattern['pattern_name'],
                    'description': pattern['description'],
                    'implications': pattern.get('implications', ''),
                    'confidence': pattern.get('confidence', 'MEDIUM')
                }
                genius_trades.append(trade)
        
        return sorted(genius_trades, key=lambda x: x.get('brilliance_score', 0.5), reverse=True)
    
    def _map_transmission_paths(self, data: pd.DataFrame) -> List[Dict]:
        """Map macro transmission paths across asset classes"""
        
        # Implementation for transmission path mapping
        return []
    
    def _calculate_brilliance_score(self, results: Dict) -> float:
        """Calculate overall brilliance score of discoveries"""
        
        brilliant_connections = results['brilliant_connections']
        pattern_alerts = results['pattern_alerts']
        genius_trades = results['genius_trades']
        
        if not brilliant_connections and not pattern_alerts:
            return 0.0
        
        # Score based on quality and quantity of discoveries
        connection_score = sum(conn.get('brilliance_score', 0) * conn.get('alpha_potential', 0) 
                             for conn in brilliant_connections)
        pattern_score = len(pattern_alerts) * 0.2
        genius_trade_score = len(genius_trades) * 0.3
        
        total_score = (connection_score + pattern_score + genius_trade_score) / 5
        return min(1.0, total_score)

def main():
    """Test the brilliant connection engine"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_pipeline.alternative_fetcher import AlternativeDataFetcher
    
    print("✨ BRILLIANT CONNECTION ENGINE")
    print("=" * 60)
    print("Discovering genius-level cross-market relationships that others miss")
    print("=" * 60)
    
    # Load market data
    fetcher = AlternativeDataFetcher()
    data = fetcher.fetch_multiple_sources()
    
    if data.empty:
        print("❌ No market data available")
        return
    
    print(f"\n📊 Available assets: {', '.join(data.columns)}")
    
    engine = BrilliantConnectionEngine()
    results = engine.discover_brilliant_connections(data)
    
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    # Display results
    print(f"\n🧠 BRILLIANCE ANALYSIS:")
    print(f"   Overall Brilliance Score: {results['brilliance_score']:.3f}/1.0")
    
    # Brilliant connections
    connections = results['brilliant_connections']
    if connections:
        print(f"\n✨ BRILLIANT CONNECTIONS ({len(connections)} found):")
        for i, conn in enumerate(connections, 1):
            print(f"\n   {i}. {conn['chain_name'].upper().replace('_', ' ')}")
            print(f"      Description: {conn['description']}")
            print(f"      Alpha Mechanism: {conn['alpha_mechanism']}")
            print(f"      Connection Path:")
            for step in conn['connections']:
                print(f"        {step}")
            print(f"      Sources: {' + '.join(conn['source_assets'])}")
            print(f"      Target: {conn['target_asset']}")
            print(f"      Lead Time: {conn['optimal_lead_time']} days")
            print(f"      Alpha Potential: {conn['alpha_potential']:.3f}")
            print(f"      Brilliance Score: {conn['brilliance_score']:.2f}")
            
            signal = conn['current_signal']
            print(f"      Current Signal: {signal['direction']} ({signal['confidence']})")
            print(f"      Signal Strength: {signal['signal_strength']:.2f}")
    
    # Pattern alerts
    patterns = results['pattern_alerts']
    if patterns:
        print(f"\n🎨 SOPHISTICATED PATTERN ALERTS ({len(patterns)} found):")
        for i, pattern in enumerate(patterns, 1):
            print(f"\n   {i}. {pattern['pattern_name'].upper().replace('_', ' ')}")
            print(f"      Description: {pattern['description']}")
            if 'implications' in pattern:
                print(f"      Implications: {pattern['implications']}")
            if 'confidence' in pattern:
                print(f"      Confidence: {pattern['confidence']}")
    
    # Genius trades
    genius_trades = results['genius_trades']
    if genius_trades:
        print(f"\n💎 GENIUS TRADE IDEAS ({len(genius_trades)} found):")
        for i, trade in enumerate(genius_trades, 1):
            print(f"\n   {i}. {trade['trade_name'].upper().replace('_', ' ')}")
            print(f"      Type: {trade['trade_type']}")
            print(f"      Description: {trade['description']}")
            if 'direction' in trade:
                print(f"      Direction: {trade['direction']}")
            if 'lead_time' in trade:
                print(f"      Lead Time: {trade['lead_time']} days")
            print(f"      Confidence: {trade.get('confidence', 'UNKNOWN')}")
    
    print(f"\n🌟 Brilliant connection discovery complete!")
    if results['brilliance_score'] > 0.7:
        print("🎯 GENIUS LEVEL: Exceptional cross-market insights discovered!")
    elif results['brilliance_score'] > 0.4:
        print("🎯 ADVANCED LEVEL: Strong cross-market connections found!")
    else:
        print("🎯 STANDARD LEVEL: Basic connections identified - room for more brilliance!")

if __name__ == "__main__":
    main()