import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class HypothesisEngine:
    def __init__(self):
        self.hypotheses = {
            "reits_rates": self._reits_interest_rates_hypothesis,
            "dollar_yen_bitcoin": self._dollar_yen_bitcoin_hypothesis, 
            "vix_tech": self._vix_tech_hypothesis
        }
    
    def test_hypothesis(self, hypothesis_name: str, data: pd.DataFrame) -> Dict:
        """특정 가설 테스트"""
        if hypothesis_name in self.hypotheses:
            return self.hypotheses[hypothesis_name](data)
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis_name}")
    
    def test_all_hypotheses(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """모든 가설 테스트"""
        results = {}
        for name, test_func in self.hypotheses.items():
            try:
                results[name] = test_func(data)
                logger.info(f"✓ Tested hypothesis: {name}")
            except Exception as e:
                logger.error(f"✗ Failed to test {name}: {e}")
                results[name] = {"error": str(e)}
        return results
    
    def _reits_interest_rates_hypothesis(self, data: pd.DataFrame) -> Dict:
        """
        가설 1: REITs와 금리는 역상관관계를 가진다
        - VNQ (REITs ETF)와 ^TNX (10년 국채 수익률) 분석
        - 금리 상승 시 REITs 가격 하락 예상
        """
        required_assets = ["VNQ", "^TNX"]
        available_assets = [asset for asset in required_assets if asset in data.columns]
        
        if len(available_assets) < 2:
            return {"error": f"Required assets not available. Need: {required_assets}, Have: {available_assets}"}
        
        reits_data = data["VNQ"].dropna()
        rates_data = data["^TNX"].dropna()
        
        # 공통 날짜로 정렬
        common_dates = reits_data.index.intersection(rates_data.index)
        reits_aligned = reits_data[common_dates]
        rates_aligned = rates_data[common_dates]
        
        # 30일 롤링 상관관계
        combined_data = pd.DataFrame({
            'REITs': reits_aligned,
            'Rates': rates_aligned
        })
        rolling_corr = combined_data['REITs'].rolling(30).corr(combined_data['Rates'])
        
        # 통계 계산
        current_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else None
        avg_corr = rolling_corr.mean()
        strong_negative_days = (rolling_corr < -0.5).sum()
        
        # 최근 7일 추세 분석
        recent_reits_change = (reits_aligned.iloc[-1] / reits_aligned.iloc[-8] - 1) * 100
        recent_rates_change = rates_aligned.iloc[-1] - rates_aligned.iloc[-8]
        
        # 신호 생성
        signal = self._generate_reits_signal(current_corr, recent_reits_change, recent_rates_change)
        
        return {
            "hypothesis": "REITs vs Interest Rates (Inverse Correlation)",
            "current_correlation": round(current_corr, 4) if current_corr else None,
            "average_correlation": round(avg_corr, 4),
            "strong_inverse_days": int(strong_negative_days),
            "recent_reits_change_pct": round(recent_reits_change, 2),
            "recent_rates_change_pts": round(recent_rates_change, 4),
            "signal": signal,
            "confidence": self._calculate_confidence(rolling_corr, "inverse"),
            "data_points": len(rolling_corr)
        }
    
    def _dollar_yen_bitcoin_hypothesis(self, data: pd.DataFrame) -> Dict:
        """
        가설 2: 달러 강세 → 엔화 약세 → 비트코인 변동성 증가
        - DX-Y.NYB (달러 지수), USDJPY=X (달러/엔), BTC-USD 분석
        - 지연 효과(lag effect) 고려한 분석
        """
        required_assets = ["DX-Y.NYB", "USDJPY=X", "BTC-USD"]
        available_assets = [asset for asset in required_assets if asset in data.columns]
        
        if len(available_assets) < 3:
            return {"error": f"Required assets not available. Need: {required_assets}, Have: {available_assets}"}
        
        # 데이터 준비
        dxy_data = data["DX-Y.NYB"].dropna()
        usdjpy_data = data["USDJPY=X"].dropna()
        btc_data = data["BTC-USD"].dropna()
        
        # 변동성 계산 (20일 롤링 표준편차)
        btc_volatility = btc_data.pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        # 지연 효과 분석 (0, 1, 3, 7일)
        lag_correlations = {}
        for lag in [0, 1, 3, 7]:
            if lag == 0:
                dxy_lagged = dxy_data
            else:
                dxy_lagged = dxy_data.shift(lag)
            
            common_dates = dxy_lagged.index.intersection(btc_volatility.index)
            if len(common_dates) > 30:
                corr = dxy_lagged[common_dates].corr(btc_volatility[common_dates])
                lag_correlations[f"lag_{lag}"] = corr
        
        # 최근 추세 분석
        recent_dxy_change = (dxy_data.iloc[-1] / dxy_data.iloc[-8] - 1) * 100
        recent_usdjpy_change = (usdjpy_data.iloc[-1] / usdjpy_data.iloc[-8] - 1) * 100
        current_btc_vol = btc_volatility.iloc[-1] if not btc_volatility.empty else None
        avg_btc_vol = btc_volatility.mean()
        
        # 달러-엔 상관관계
        common_dates = dxy_data.index.intersection(usdjpy_data.index)
        dxy_usdjpy_corr = dxy_data[common_dates].corr(usdjpy_data[common_dates])
        
        # 신호 생성
        signal = self._generate_crypto_signal(recent_dxy_change, recent_usdjpy_change, current_btc_vol, avg_btc_vol)
        
        return {
            "hypothesis": "Dollar Strength → Yen Weakness → Bitcoin Volatility",
            "lag_correlations": {k: round(v, 4) for k, v in lag_correlations.items()},
            "dxy_usdjpy_correlation": round(dxy_usdjpy_corr, 4),
            "recent_dxy_change_pct": round(recent_dxy_change, 2),
            "recent_usdjpy_change_pct": round(recent_usdjpy_change, 2),
            "current_btc_volatility": round(current_btc_vol, 2) if current_btc_vol else None,
            "average_btc_volatility": round(avg_btc_vol, 2),
            "signal": signal,
            "confidence": self._calculate_lag_confidence(lag_correlations),
            "data_points": len(common_dates)
        }
    
    def _vix_tech_hypothesis(self, data: pd.DataFrame) -> Dict:
        """
        가설 3: VIX 30 이상일 때 기술주 ETF들의 상관관계 증가
        - ^VIX, QQQ, ARKK 분석
        - 공포 지수 높을 때 기술주들 동조화 현상
        """
        required_assets = ["^VIX", "QQQ", "ARKK"]
        available_assets = [asset for asset in required_assets if asset in data.columns]
        
        if len(available_assets) < 3:
            return {"error": f"Required assets not available. Need: {required_assets}, Have: {available_assets}"}
        
        vix_data = data["^VIX"].dropna()
        qqq_data = data["QQQ"].dropna()
        arkk_data = data["ARKK"].dropna()
        
        # 공통 날짜
        common_dates = vix_data.index.intersection(qqq_data.index).intersection(arkk_data.index)
        
        vix_aligned = vix_data[common_dates]
        qqq_aligned = qqq_data[common_dates]
        arkk_aligned = arkk_data[common_dates]
        
        # VIX > 30인 기간과 그렇지 않은 기간 구분
        high_vix_mask = vix_aligned > 30
        normal_vix_mask = ~high_vix_mask
        
        # 각 상황에서의 QQQ-ARKK 상관관계
        if high_vix_mask.sum() > 10:  # 충분한 데이터 포인트 확인
            high_vix_corr = qqq_aligned[high_vix_mask].corr(arkk_aligned[high_vix_mask])
        else:
            high_vix_corr = None
            
        if normal_vix_mask.sum() > 10:
            normal_vix_corr = qqq_aligned[normal_vix_mask].corr(arkk_aligned[normal_vix_mask])
        else:
            normal_vix_corr = None
        
        # 전체 상관관계
        overall_corr = qqq_aligned.corr(arkk_aligned)
        
        # 현재 상황
        current_vix = vix_aligned.iloc[-1]
        current_condition = "High Fear" if current_vix > 30 else "Normal"
        
        # 최근 30일 QQQ-ARKK 상관관계
        recent_corr = qqq_aligned.iloc[-30:].corr(arkk_aligned.iloc[-30:])
        
        # VIX와 상관관계의 관계
        rolling_corr_qqq_arkk = qqq_aligned.rolling(20).corr(arkk_aligned)
        vix_corr_relationship = vix_aligned.corr(rolling_corr_qqq_arkk)
        
        # 신호 생성
        signal = self._generate_vix_signal(current_vix, high_vix_corr, normal_vix_corr, recent_corr)
        
        return {
            "hypothesis": "VIX > 30 increases Tech ETF correlation",
            "current_vix": round(current_vix, 2),
            "current_condition": current_condition,
            "high_vix_correlation": round(high_vix_corr, 4) if high_vix_corr else None,
            "normal_vix_correlation": round(normal_vix_corr, 4) if normal_vix_corr else None,
            "overall_correlation": round(overall_corr, 4),
            "recent_30d_correlation": round(recent_corr, 4),
            "vix_corr_relationship": round(vix_corr_relationship, 4),
            "high_vix_days": int(high_vix_mask.sum()),
            "signal": signal,
            "confidence": self._calculate_vix_confidence(high_vix_corr, normal_vix_corr),
            "data_points": len(common_dates)
        }
    
    def _generate_reits_signal(self, corr: float, reits_change: float, rates_change: float) -> str:
        """REITs 가설 기반 신호 생성"""
        if corr is None:
            return "Insufficient Data"
        
        if corr < -0.6 and rates_change > 0.1:
            return "🔴 SELL REITS - Strong inverse correlation + Rising rates"
        elif corr < -0.4 and reits_change > 5:
            return "🟡 CAUTION - Negative correlation but REITs rallying"
        elif corr > -0.2:
            return "🟢 NEUTRAL - Weak correlation, fundamentals matter more"
        else:
            return "🟡 MONITOR - Moderate inverse correlation"
    
    def _generate_crypto_signal(self, dxy_change: float, usdjpy_change: float, 
                               current_vol: float, avg_vol: float) -> str:
        """암호화폐 가설 기반 신호 생성"""
        if current_vol is None:
            return "Insufficient Data"
        
        if dxy_change > 2 and current_vol > avg_vol * 1.5:
            return "🔴 HIGH VOLATILITY - Strong dollar + Elevated BTC vol"
        elif dxy_change > 1 and usdjpy_change > 1:
            return "🟡 INCREASED RISK - Dollar strength affecting JPY"
        elif current_vol < avg_vol * 0.8:
            return "🟢 LOW VOLATILITY - Stable conditions"
        else:
            return "🟡 MONITOR - Normal volatility range"
    
    def _generate_vix_signal(self, vix: float, high_vix_corr: float, 
                            normal_vix_corr: float, recent_corr: float) -> str:
        """VIX 가설 기반 신호 생성"""
        if vix > 35:
            return "🔴 EXTREME FEAR - Tech stocks highly correlated"
        elif vix > 30:
            if high_vix_corr and high_vix_corr > 0.8:
                return "🟡 HIGH FEAR - Strong tech correlation confirmed"
            else:
                return "🟡 ELEVATED FEAR - Monitor tech divergence"
        elif vix < 20:
            return "🟢 COMPLACENCY - Tech stocks may diverge"
        else:
            return "🟡 NEUTRAL - Normal fear levels"
    
    def _calculate_confidence(self, rolling_corr: pd.Series, expected_direction: str) -> str:
        """신뢰도 계산"""
        if rolling_corr.empty:
            return "No Data"
        
        if expected_direction == "inverse":
            negative_ratio = (rolling_corr < -0.3).sum() / len(rolling_corr)
            if negative_ratio > 0.7:
                return "High"
            elif negative_ratio > 0.5:
                return "Medium"
            else:
                return "Low"
        
        return "Medium"
    
    def _calculate_lag_confidence(self, lag_correlations: Dict) -> str:
        """지연 효과 신뢰도 계산"""
        if not lag_correlations:
            return "No Data"
        
        max_corr = max(abs(v) for v in lag_correlations.values())
        if max_corr > 0.4:
            return "High"
        elif max_corr > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_vix_confidence(self, high_vix_corr: Optional[float], 
                                 normal_vix_corr: Optional[float]) -> str:
        """VIX 가설 신뢰도 계산"""
        if high_vix_corr is None or normal_vix_corr is None:
            return "Insufficient Data"
        
        diff = high_vix_corr - normal_vix_corr
        if diff > 0.2:
            return "High"
        elif diff > 0.1:
            return "Medium"
        else:
            return "Low"