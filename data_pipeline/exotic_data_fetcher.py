#!/usr/bin/env python3
"""
Exotic Data Fetcher - Collect distant asset classes for brilliant cross-market analysis
Built to find the data sources that most traders ignore but create alpha
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ExoticDataFetcher:
    """
    Fetch exotic asset classes to enable brilliant cross-market discovery
    Focus on distant relationships that create alpha
    """
    
    def __init__(self):
        # Exotic asset universe for brilliant analysis
        self.exotic_universe = {
            # Core markets for cross-asset analysis
            'core_equity': ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'EEM', 'EFA'],
            'core_fixed_income': ['TLT', 'IEF', 'SHY', 'TIP', 'HYG', 'LQD', 'EMB'],
            'core_commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DJP', 'PDBC'],
            
            # Exotic sectors for distant connections
            'exotic_sectors': ['XBI', 'ICLN', 'TAN', 'ARKK', 'FINX', 'BOTZ', 'HACK'],
            'international_exotic': ['EWJ', 'EWG', 'EWU', 'EWZ', 'INDA', 'FXI', 'RSX'],
            'currency_proxies': ['UUP', 'FXE', 'FXY', 'EUO', 'YCS'],
            
            # Volatility and alternatives
            'volatility': ['VXX', 'SVXY', 'VIX9D'],
            'alternatives': ['REIT', 'VNQ', 'VNQI', 'MORT', 'PFF'],
            
            # Industry-specific exotic connections
            'semiconductors': ['SMH', 'SOXX', 'NVDA', 'AMD', 'TSM'],
            'biotech': ['XBI', 'IBB', 'ARKG', 'PBE'],
            'energy_transition': ['ICLN', 'PBW', 'QCLN', 'TAN', 'LIT'],
            'financials_deep': ['XLF', 'KBE', 'KRE', 'IAT'],
            
            # Economic proxies
            'transportation': ['IYT', 'JETS', 'SHIP'],
            'materials': ['XLB', 'REMX', 'PICK', 'SIL'],
            'consumer_cyclical': ['XLY', 'XRT', 'RH', 'HD'],
            
            # Macro indicators (attempt to get)
            'macro_indicators': ['^VIX', '^VIX3M', '^TNX', '^TYX', 'DXY', '^GSPC'],
            
            # Interest rate sensitivity spectrum
            'duration_spectrum': ['SHY', 'IEI', 'IEF', 'TLH', 'TLT', 'EDV'],
            
            # Credit spectrum
            'credit_spectrum': ['AGG', 'LQD', 'HYG', 'JNK', 'EMB', 'PCY'],
            
            # International developed
            'developed_markets': ['EFA', 'VEA', 'IEFA', 'EWJ', 'EWG', 'EWU'],
            
            # Emerging markets deep
            'emerging_markets': ['EEM', 'VWO', 'IEMG', 'EWZ', 'INDA', 'FXI']
        }
        
        # Flattened universe for easy access
        self.all_tickers = []
        for category, tickers in self.exotic_universe.items():
            self.all_tickers.extend(tickers)
        
        # Remove duplicates while preserving order
        seen = set()
        self.all_tickers = [x for x in self.all_tickers if not (x in seen or seen.add(x))]
        
        print(f"Exotic Data Fetcher initialized with {len(self.all_tickers)} unique tickers across {len(self.exotic_universe)} categories")
    
    def fetch_exotic_data(self, period: str = "1y", max_attempts: int = 3) -> pd.DataFrame:
        """
        Fetch exotic asset data for brilliant cross-market analysis
        
        Args:
            period: Time period for data ('1y', '6mo', '2y', etc.)
            max_attempts: Maximum retry attempts for failed fetches
            
        Returns:
            DataFrame with exotic asset price data
        """
        
        print(f"🌍 Fetching exotic cross-market data for {len(self.all_tickers)} assets...")
        
        # Split into batches to avoid API limits
        batch_size = 20
        batches = [self.all_tickers[i:i+batch_size] for i in range(0, len(self.all_tickers), batch_size)]
        
        all_data = pd.DataFrame()
        successful_fetches = 0
        failed_tickers = []
        
        for batch_num, batch in enumerate(batches, 1):
            print(f"  Fetching batch {batch_num}/{len(batches)}: {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}")
            
            try:
                # Fetch batch data
                batch_data = yf.download(batch, period=period, interval='1d', 
                                       group_by='ticker', auto_adjust=True, 
                                       prepost=False, threads=True, progress=False)
                
                if batch_data.empty:
                    failed_tickers.extend(batch)
                    continue
                
                # Process batch data
                processed_batch = self._process_batch_data(batch_data, batch)
                
                if not processed_batch.empty:
                    if all_data.empty:
                        all_data = processed_batch
                    else:
                        all_data = pd.concat([all_data, processed_batch], axis=1)
                    
                    successful_fetches += len(processed_batch.columns)
                else:
                    failed_tickers.extend(batch)
            
            except Exception as e:
                logger.warning(f"Error fetching batch {batch_num}: {str(e)}")
                failed_tickers.extend(batch)
        
        # Clean and validate data
        if not all_data.empty:
            all_data = self._clean_exotic_data(all_data)
        
        print(f"✅ Successfully fetched {successful_fetches} exotic assets")
        if failed_tickers:
            print(f"⚠️ Failed to fetch {len(failed_tickers)} assets: {', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''}")
        
        return all_data
    
    def _process_batch_data(self, batch_data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Process batch data and extract closing prices"""
        
        processed = pd.DataFrame()
        
        try:
            if len(tickers) == 1:
                # Single ticker case
                ticker = tickers[0]
                if not batch_data.empty and 'Close' in batch_data.columns:
                    processed[ticker] = batch_data['Close']
            else:
                # Multiple tickers case
                for ticker in tickers:
                    try:
                        if ticker in batch_data.columns:
                            # Multi-level columns case
                            if 'Close' in batch_data[ticker].columns:
                                processed[ticker] = batch_data[ticker]['Close']
                        elif (ticker, 'Close') in batch_data.columns:
                            # Flat columns case
                            processed[ticker] = batch_data[(ticker, 'Close')]
                    except Exception as e:
                        logger.debug(f"Error processing {ticker}: {str(e)}")
                        continue
            
            # Drop empty columns
            processed = processed.dropna(axis=1, how='all')
            
        except Exception as e:
            logger.warning(f"Error processing batch data: {str(e)}")
        
        return processed
    
    def _clean_exotic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate exotic data"""
        
        try:
            # Remove columns with too much missing data (>50%)
            min_data_threshold = len(data) * 0.5
            data = data.dropna(axis=1, thresh=min_data_threshold)
            
            # Forward fill small gaps (max 5 days)
            data = data.fillna(method='ffill', limit=5)
            
            # Remove any remaining rows with too much missing data
            min_assets_threshold = len(data.columns) * 0.7
            data = data.dropna(axis=0, thresh=min_assets_threshold)
            
            # Remove extreme outliers (more than 5 std devs)
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    col_data = data[col].dropna()
                    if len(col_data) > 20:  # Need enough data for stats
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        if std_val > 0:
                            outlier_mask = abs(col_data - mean_val) > 5 * std_val
                            data.loc[outlier_mask, col] = np.nan
            
            # Final forward fill for remaining gaps
            data = data.fillna(method='ffill')
            
            # Sort by date
            data = data.sort_index()
            
            print(f"🧹 Cleaned data: {len(data)} days, {len(data.columns)} assets")
            
        except Exception as e:
            logger.warning(f"Error cleaning data: {str(e)}")
        
        return data
    
    def get_category_assets(self, category: str) -> List[str]:
        """Get assets for a specific category"""
        return self.exotic_universe.get(category, [])
    
    def get_available_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.exotic_universe.keys())
    
    def create_brilliant_dataset(self) -> pd.DataFrame:
        """Create a dataset optimized for brilliant cross-market discovery"""
        
        print("🎯 Creating brilliant cross-market dataset...")
        
        # Fetch exotic data
        exotic_data = self.fetch_exotic_data(period="18mo")  # More history for better analysis
        
        if exotic_data.empty:
            print("❌ Failed to create brilliant dataset - no data available")
            return pd.DataFrame()
        
        # Enhance with calculated fields for better cross-market analysis
        enhanced_data = self._enhance_for_brilliant_analysis(exotic_data)
        
        return enhanced_data
    
    def _enhance_for_brilliant_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with calculated fields for brilliant analysis"""
        
        enhanced = data.copy()
        
        try:
            # Add ratio-based signals for brilliant connections
            if 'GLD' in data.columns and 'SLV' in data.columns:
                enhanced['GLD_SLV_RATIO'] = data['GLD'] / data['SLV']
            
            if 'SPY' in data.columns and 'IWM' in data.columns:
                enhanced['LARGE_SMALL_RATIO'] = data['SPY'] / data['IWM']
            
            if 'QQQ' in data.columns and 'XLF' in data.columns:
                enhanced['GROWTH_FINANCIAL_RATIO'] = data['QQQ'] / data['XLF']
            
            if 'HYG' in data.columns and 'LQD' in data.columns:
                enhanced['CREDIT_QUALITY_SPREAD'] = data['HYG'] / data['LQD']
            
            if 'TLT' in data.columns and 'SHY' in data.columns:
                enhanced['YIELD_CURVE_PROXY'] = data['TLT'] / data['SHY']
            
            # Add volatility proxies
            for asset in ['SPY', 'QQQ', 'TLT', 'GLD']:
                if asset in data.columns:
                    vol_name = f'{asset}_VOLATILITY'
                    enhanced[vol_name] = data[asset].pct_change().rolling(10).std() * np.sqrt(252)
            
            # Add momentum signals
            for asset in ['SPY', 'QQQ', 'IWM']:
                if asset in data.columns:
                    mom_name = f'{asset}_MOMENTUM'
                    enhanced[mom_name] = data[asset] / data[asset].rolling(20).mean() - 1
            
            print(f"✨ Enhanced dataset: {len(enhanced.columns)} total fields ({len(enhanced.columns) - len(data.columns)} calculated)")
        
        except Exception as e:
            logger.warning(f"Error enhancing data: {str(e)}")
            return data
        
        return enhanced

def main():
    """Test the exotic data fetcher"""
    
    print("🌍 EXOTIC DATA FETCHER TEST")
    print("=" * 60)
    print("Fetching distant asset classes for brilliant cross-market analysis")
    print("=" * 60)
    
    fetcher = ExoticDataFetcher()
    
    # Show available categories
    print(f"\n📋 Available Categories ({len(fetcher.get_available_categories())}):")
    for category in fetcher.get_available_categories()[:8]:  # Show first 8
        assets = fetcher.get_category_assets(category)
        print(f"   • {category}: {', '.join(assets[:5])}{'...' if len(assets) > 5 else ''}")
    
    # Create brilliant dataset
    brilliant_data = fetcher.create_brilliant_dataset()
    
    if not brilliant_data.empty:
        print(f"\n✨ BRILLIANT DATASET CREATED:")
        print(f"   Time Range: {brilliant_data.index[0].strftime('%Y-%m-%d')} to {brilliant_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Assets & Signals: {len(brilliant_data.columns)}")
        print(f"   Data Points: {len(brilliant_data)}")
        
        # Show data quality
        data_quality = (brilliant_data.notna().sum() / len(brilliant_data)).mean()
        print(f"   Data Quality: {data_quality:.1%} complete")
        
        # Show sample of exotic assets
        sample_assets = [col for col in brilliant_data.columns if not any(suffix in col for suffix in ['_RATIO', '_VOLATILITY', '_MOMENTUM'])][:15]
        print(f"   Sample Assets: {', '.join(sample_assets)}")
        
        # Show calculated signals
        calculated_signals = [col for col in brilliant_data.columns if any(suffix in col for suffix in ['_RATIO', '_VOLATILITY', '_MOMENTUM'])]
        if calculated_signals:
            print(f"   Calculated Signals: {', '.join(calculated_signals[:10])}")
        
        # Test with brilliant connection engine
        print(f"\n🧠 Testing with Brilliant Connection Engine...")
        
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from strategies.brilliant_connections import BrilliantConnectionEngine
            
            engine = BrilliantConnectionEngine()
            results = engine.discover_brilliant_connections(brilliant_data)
            
            print(f"   Brilliance Score: {results['brilliance_score']:.3f}/1.0")
            print(f"   Brilliant Connections: {len(results['brilliant_connections'])}")
            print(f"   Pattern Alerts: {len(results['pattern_alerts'])}")
            print(f"   Genius Trades: {len(results['genius_trades'])}")
            
        except Exception as e:
            print(f"   ⚠️ Brilliant engine test failed: {str(e)}")
    
    else:
        print("❌ Failed to create brilliant dataset")
    
    print(f"\n🌟 Exotic data fetcher test complete!")

if __name__ == "__main__":
    main()