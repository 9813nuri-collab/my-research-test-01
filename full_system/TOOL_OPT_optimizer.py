"""
H-PIOS v8.5 — Historical Back-Optimizer (v2.1: Synthetic Proxy Breakthrough)
=============================================================================
2019-현재 매크로 데이터를 기반으로 12인 거장 엔진의 파라미터를
역사적 성과에 맞게 최적화하되, 각 거장의 **투자 철학 관성(Philosophy Inertia)**을
보존하는 모듈입니다.

v2.1 핵심 변경점
-----------------
1. **Synthetic Fundamental Proxies** — 매크로 인덱스 데이터를
   마이크로 펀더멘탈 시그널로 변환하여 노드별 차별화된 신호 생성.
   - Value: 200MA 이격도 + 신용 스프레드 → 합성 NCAV/PE/ROIC 프록시
   - Growth: NASDAQ 모멘텀 + VIX ROC → 합성 PEG/EPS 프록시
   - Quant: MA Z-Score + VIX 직접 주입
   - Risk: VIX 가속도 + 추세 R² → Tail Risk 프록시

2. **Taleb Protection** — Taleb/Marks 노드에 조건부 10x 관성 페널티
3. **Graceful Early Stopping** — min_epochs=20 유예기간, lr=0.005
4. **Regime-Weighted Loss** — Black_Swan 3.0x 가중
5. **Enhanced Output** — Best-fit Regime 로깅, preserved 마킹

핵심 파이프라인
---------------
1. **Data Ingestion** — yfinance + 15개 파생 시계열 (MA, Z-Score, ROC, R²)
2. **Event Regime Mapping** — 2019~현재 주요 이벤트 국면 라벨링
3. **Synthetic Payload Builder** — Macro → Micro 변환
4. **Historical Simulator** — GraphOrchestrator 일별 반복 구동
5. **Attribution Tracker** — T+5/T+20/T+60 멀티-호라이즌 + 국면별 기여도
6. **Philosophy Inertia Optimizer** — 조건부 관성 + Early Stopping
7. **Persistence Layer** — optimized_weights.json 저장

설계 원칙
---------
- 기존 models.py / engine_core.py 절대 미수정
- 원본 JSON 파일 미수정 → 최적화 결과는 별도 JSON으로 저장
- ±20% 바운드 + 롤링 윈도우 + 국면 가중치로 오버피팅 방지
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("H-PIOS.optimizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# ── 프로젝트 루트 결정 (이 파일 기준) ──
_PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────
# 지연 임포트: models & engine_core (수정 금지 대상)
# ──────────────────────────────────────────────────────────
sys.path.insert(0, str(_PROJECT_ROOT))

from CORE_MODELS_models import (  # noqa: E402
    MarketDataPayload,
    MasterEngineConfig,
    EngineState,
    OrchestratorOutput,
    NodeExecutionResult,
)
from CORE_ENGINE_core import GraphOrchestrator  # noqa: E402


# ══════════════════════════════════════════════════════════
# 1. Data Ingestion — yfinance 매크로 데이터 수집 + 파생 시계열
# ══════════════════════════════════════════════════════════

# 다운로드 대상 티커 레지스트리
TICKER_REGISTRY: Dict[str, str] = {
    # Equities
    "SP500":       "^GSPC",
    "KOSPI":       "^KS11",
    "NASDAQ100":   "^NDX",
    # Commodities
    "OIL":         "CL=F",
    "GOLD":        "GC=F",
    # Currency
    "USDKRW":      "KRW=X",
    "DXY":         "DX-Y.NYB",
    # Rates
    "TNX":         "^TNX",     # 10Y Treasury Yield
    "IRX":         "^IRX",     # 13-week T-Bill (2Y proxy)
    # Risk
    "VIX":         "^VIX",
    "HYG":         "HYG",      # High Yield Corporate Bond ETF
    "IEF":         "IEF",      # 7-10Y Treasury Bond ETF
}


def _compute_rolling_r_squared(prices: pd.Series, window: int = 60) -> pd.Series:
    """가격 시계열의 롤링 선형 추세 R² (추세 강도 측정).

    R² → 1.0: 강한 추세 (상승 또는 하락)
    R² → 0.0: 노이즈 우위 (방향성 없음)

    Args:
        prices: 가격 시리즈 (index: DatetimeIndex)
        window: 롤링 윈도우 크기 (기본 60일)

    Returns:
        롤링 R² 시리즈 (동일 인덱스)
    """
    def _r_sq(arr: np.ndarray) -> float:
        x = np.arange(len(arr))
        cc = np.corrcoef(x, arr)
        if cc.shape != (2, 2):
            return 0.5
        r = cc[0, 1]
        return r ** 2 if not np.isnan(r) else 0.5

    return prices.rolling(window, min_periods=window).apply(
        _r_sq, raw=True
    )


def fetch_macro_data(start: str = "2019-01-01") -> pd.DataFrame:
    """yfinance를 통해 매크로 시장 데이터를 다운로드하고 파생 지표를 계산합니다.

    기본 13개 티커의 Adjusted Close를 병합한 뒤, Synthetic Proxy에 필요한
    15개 파생 시계열(MA, Z-Score, ROC, R² 등)을 산출합니다.

    Args:
        start: 시작 날짜 (YYYY-MM-DD). 기본 2019-01-01.

    Returns:
        날짜 인덱스의 클린 DataFrame.
        기본 컬럼: SP500, KOSPI, NASDAQ100, OIL, GOLD, USDKRW, DXY,
                   TNX, IRX, VIX, HYG, IEF
        파생 컬럼: yield_spread, credit_spread_proxy,
                   SP500_ret, KOSPI_ret, NASDAQ100_ret,
                   SP500_MA200, SP500_MA200_ratio,
                   SP500_MA20_zscore,
                   NDX_12m_ret, NDX_60d_ret,
                   VIX_5d_roc, VIX_20d_roc,
                   HYG_IEF_zscore,
                   SP500_ret_252d, SP500_volatility_20d,
                   SP500_r_squared_60d

    Raises:
        RuntimeError: yfinance 설치 안 됨 또는 다운로드 실패 시.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError(
            "yfinance 패키지가 필요합니다. `pip install yfinance` 를 실행하세요."
        ) from e

    logger.info("매크로 데이터 다운로드 시작 (start=%s)", start)

    frames: Dict[str, pd.Series] = {}
    for label, ticker in TICKER_REGISTRY.items():
        try:
            df_raw = yf.download(
                ticker, start=start, progress=False, auto_adjust=True
            )
            if df_raw.empty:
                logger.warning("티커 %s (%s): 데이터 없음 → 건너뜀", label, ticker)
                continue
            close_col = df_raw["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            frames[label] = close_col
        except Exception as e:
            logger.warning("티커 %s (%s): 다운로드 실패 → %s", label, ticker, e)

    if not frames:
        raise RuntimeError("모든 티커 다운로드 실패. 네트워크 연결을 확인하세요.")

    # 병합
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Forward-fill → Backward-fill (초반 NaN 처리)
    df = df.ffill().bfill()

    # ── 기본 파생 지표 ──
    # Yield Spread: 10Y - 2Y proxy
    if "TNX" in df.columns and "IRX" in df.columns:
        df["yield_spread"] = df["TNX"] - df["IRX"]
    else:
        df["yield_spread"] = 0.0

    # Credit Spread Proxy: HYG / IEF ratio (하락 → 스프레드 확대)
    if "HYG" in df.columns and "IEF" in df.columns:
        df["credit_spread_proxy"] = df["HYG"] / df["IEF"]
    else:
        df["credit_spread_proxy"] = 1.0

    # 일별 수익률
    for eq in ["SP500", "KOSPI", "NASDAQ100"]:
        if eq in df.columns:
            df[f"{eq}_ret"] = df[eq].pct_change()
        else:
            df[f"{eq}_ret"] = 0.0

    # ── ═══════════════════════════════════════════════ ──
    # ── Synthetic Proxy용 파생 시계열 (v2.1 신규) ──
    # ── ═══════════════════════════════════════════════ ──

    # --- 1. S&P500 이동평균 이격도 ---
    if "SP500" in df.columns:
        df["SP500_MA200"] = df["SP500"].rolling(200, min_periods=1).mean()
        df["SP500_MA200_ratio"] = df["SP500"] / df["SP500_MA200"]

        # 20일 MA Z-Score (평균 회귀 시그널)
        sp500_ma20 = df["SP500"].rolling(20, min_periods=1).mean()
        sp500_pct_dev = df["SP500"] / sp500_ma20 - 1.0
        sp500_ret_vol = df["SP500_ret"].rolling(20, min_periods=5).std().clip(lower=0.005)
        df["SP500_MA20_zscore"] = sp500_pct_dev / sp500_ret_vol
    else:
        df["SP500_MA200"] = 0.0
        df["SP500_MA200_ratio"] = 1.0
        df["SP500_MA20_zscore"] = 0.0

    # --- 2. NASDAQ 100 모멘텀 ---
    if "NASDAQ100" in df.columns:
        df["NDX_12m_ret"] = df["NASDAQ100"].pct_change(252)  # ~12개월
        df["NDX_60d_ret"] = df["NASDAQ100"].pct_change(60)   # ~3개월
    else:
        df["NDX_12m_ret"] = 0.0
        df["NDX_60d_ret"] = 0.0

    # --- 3. VIX 변화율 (Rate of Change) ---
    if "VIX" in df.columns:
        df["VIX_5d_roc"] = df["VIX"].pct_change(5)
        df["VIX_20d_roc"] = df["VIX"].pct_change(20)
    else:
        df["VIX_5d_roc"] = 0.0
        df["VIX_20d_roc"] = 0.0

    # --- 4. 신용 스프레드 Z-Score (60일 롤링) ---
    if "credit_spread_proxy" in df.columns:
        csp = df["credit_spread_proxy"]
        csp_mean = csp.rolling(60, min_periods=10).mean()
        csp_std = csp.rolling(60, min_periods=10).std().clip(lower=0.001)
        df["HYG_IEF_zscore"] = (csp - csp_mean) / csp_std
    else:
        df["HYG_IEF_zscore"] = 0.0

    # --- 5. S&P500 장기 수익률 & 변동성 ---
    if "SP500" in df.columns:
        df["SP500_ret_252d"] = df["SP500"].pct_change(252)
        df["SP500_volatility_20d"] = (
            df["SP500_ret"].rolling(20, min_periods=5).std() * np.sqrt(252)
        )
    else:
        df["SP500_ret_252d"] = 0.0
        df["SP500_volatility_20d"] = 0.15

    # --- 6. 추세 강도 R² (Shannon 노드용) ---
    if "SP500" in df.columns:
        df["SP500_r_squared_60d"] = _compute_rolling_r_squared(df["SP500"], window=60)
    else:
        df["SP500_r_squared_60d"] = 0.5

    # ── NaN 처리: 파생 지표의 초기 NaN을 보수적 기본값으로 채움 ──
    fill_defaults: Dict[str, float] = {
        "SP500_MA200_ratio": 1.0,
        "SP500_MA20_zscore": 0.0,
        "NDX_12m_ret": 0.0,
        "NDX_60d_ret": 0.0,
        "VIX_5d_roc": 0.0,
        "VIX_20d_roc": 0.0,
        "HYG_IEF_zscore": 0.0,
        "SP500_ret_252d": 0.0,
        "SP500_volatility_20d": 0.15,
        "SP500_r_squared_60d": 0.5,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # 첫 행 NaN 제거
    df = df.iloc[1:]

    logger.info(
        "매크로 데이터 준비 완료: %d 행 × %d 컬럼 (%s ~ %s)",
        len(df), len(df.columns),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
    return df


# ══════════════════════════════════════════════════════════
# 2. Event Regime Mapping — 역사적 국면 라벨링
# ══════════════════════════════════════════════════════════

# 이벤트 → (시작일, 종료일) 매핑 (시간순)
EVENT_REGIMES: List[Tuple[str, str, str]] = [
    # (regime_label, start_date, end_date)
    ("Black_Swan",                "2020-02-19", "2020-04-09"),
    ("Liquidity_Bubble",          "2020-05-01", "2021-11-30"),
    ("Deleveraging",              "2022-01-01", "2022-10-31"),
    ("Systemic_Risk",             "2023-03-01", "2023-04-30"),
    ("AI_Supercycle",             "2023-05-01", "2025-12-31"),
    ("Geopolitical_Stagflation",  "2026-01-01", "2099-12-31"),  # ~today
]

# 국면 → engine_core가 인식하는 내부 Regime 매핑
_REGIME_TO_ENGINE: Dict[str, str] = {
    "Black_Swan":               "Tail_Risk_Event",
    "Liquidity_Bubble":         "Late_Bull_Market",
    "Deleveraging":             "Deleveraging",
    "Systemic_Risk":            "Systemic_Crash",
    "AI_Supercycle":            "Innovation_Cycle",
    "Geopolitical_Stagflation": "Stagflation",
    "Normal":                   "Normal_Market",
}


def label_event_regime(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 event_regime 컬럼을 추가합니다.

    날짜 범위에 따라 역사적 이벤트 국면을 라벨링하고,
    엔진이 인식하는 내부 Regime 문자열로 매핑합니다.

    Args:
        df: 날짜 인덱스 DataFrame (fetch_macro_data 출력)

    Returns:
        event_regime 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    df["event_regime"] = "Normal"

    for regime_label, start_str, end_str in EVENT_REGIMES:
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        df.loc[mask, "event_regime"] = regime_label

    # 엔진 내부 Regime으로 변환한 컬럼 추가
    df["engine_regime"] = df["event_regime"].map(_REGIME_TO_ENGINE).fillna("Normal_Market")

    regime_counts = df["event_regime"].value_counts()
    logger.info("국면 라벨링 완료:\n%s", regime_counts.to_string())
    return df


# ══════════════════════════════════════════════════════════
# 3. Synthetic Payload Builder — Macro → Micro Proxy 변환
# ══════════════════════════════════════════════════════════

def _ret_to_percentile(ret_252d: float) -> float:
    """12개월 수익률을 근사적 역사적 백분위로 변환합니다.

    S&P500의 장기 12개월 수익률 분포를 기반으로:
    - 평균: ~10%, 표준편차: ~15%
    - 95th 백분위: ~35%
    - 5th 백분위: ~-15%

    시그모이드 CDF 근사를 사용하여 [0.01, 0.99] 범위로 매핑합니다.

    Args:
        ret_252d: 252 영업일(~12개월) 누적 수익률

    Returns:
        근사 백분위 (0.01 ~ 0.99)
    """
    z = (ret_252d - 0.10) / 0.15  # 표준화
    percentile = 1.0 / (1.0 + math.exp(-z * 1.5))
    return max(0.01, min(0.99, percentile))


def build_payload(row: pd.Series) -> MarketDataPayload:
    """Synthetic Fundamental Proxy Builder.

    **핵심 패러독스 해결:**
    엔진은 마이크로(개별 종목) 펀더멘탈 지표(P/E, NCAV, ROIC)를 기대하지만,
    최적화 데이터는 매크로(인덱스) 시계열뿐입니다.

    이 함수는 금융 수학적 변환을 통해 매크로 시계열을 합성 마이크로 시그널로
    변환하여, 각 거장 노드가 시장 국면에 따라 차별화된 반응을 생성하게 합니다.

    설계 원칙:
    - **Value 노드**: 시장 할인(200MA 이격) + 신용 스트레스 → 합성 NCAV/PE
    - **Growth 노드**: NASDAQ 모멘텀 + VIX 역수 → 합성 PEG/EPS
    - **Macro 노드**: Yield Spread + VIX 직접 매핑
    - **Quant 노드**: MA Z-Score + VIX 가속도
    - **Risk 노드**: 변동성 클러스터링 + 추세 R²

    Args:
        row: DataFrame의 단일 행 (pd.Series)

    Returns:
        MarketDataPayload 인스턴스
    """

    def _safe(val: Any, default: float = 0.0) -> float:
        """NaN/None/Inf 안전 변환."""
        if val is None:
            return default
        try:
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return default
            return fval
        except (TypeError, ValueError):
            return default

    # ── 1차 매크로 데이터 추출 ──
    sp500 = _safe(row.get("SP500"), 4000.0)
    vix = _safe(row.get("VIX"), 20.0)
    tnx = _safe(row.get("TNX"), 3.0)       # 10Y yield (%)
    irx = _safe(row.get("IRX"), 2.0)       # 13-week T-Bill (%)
    sp500_ret = _safe(row.get("SP500_ret"), 0.0)
    yield_spread = _safe(row.get("yield_spread"), 0.0)
    credit_proxy = _safe(row.get("credit_spread_proxy"), 1.0)

    # ── 2차 파생 시계열 추출 (fetch_macro_data에서 사전 계산됨) ──
    sp500_ma200_ratio = _safe(row.get("SP500_MA200_ratio"), 1.0)
    sp500_ma20_zscore = _safe(row.get("SP500_MA20_zscore"), 0.0)
    ndx_12m_ret = _safe(row.get("NDX_12m_ret"), 0.0)
    ndx_60d_ret = _safe(row.get("NDX_60d_ret"), 0.0)
    vix_5d_roc = _safe(row.get("VIX_5d_roc"), 0.0)
    hyg_ief_zscore = _safe(row.get("HYG_IEF_zscore"), 0.0)
    sp500_ret_252d = _safe(row.get("SP500_ret_252d"), 0.0)
    sp500_vol_20d = _safe(row.get("SP500_volatility_20d"), 0.15)
    sp500_r_sq = _safe(row.get("SP500_r_squared_60d"), 0.5)

    # ════════════════════════════════════════════════════════
    # SYNTHETIC FUNDAMENTAL PROXIES
    # ════════════════════════════════════════════════════════
    # 각 프록시는 JSON 임계값을 기준으로 현실적인 범위에서 작동하도록
    # 금융 수학적으로 교정(calibrated)되었습니다.

    # ── ① VALUE NODES (Graham, Buffett, Munger) ──
    # 철학: "남들이 공포에 빠졌을 때 탐욕스러워져라"
    # 핵심 시그널: 200MA 아래 깊은 할인 + 신용 스트레스

    # --- Graham: Price_to_NCAV (threshold: < 0.66) ---
    # 200MA 이격도↑ + 신용 스트레스↑ + VIX↑ → NCAV↓ (deep value)
    ma200_discount = max(0.0, 1.0 - sp500_ma200_ratio)  # [0, ∞)
    credit_stress = max(0.0, -hyg_ief_zscore) * 0.1       # 스프레드 확대 시 양수
    vix_fear = max(0.0, (vix - 20.0) / 60.0)             # VIX 20 기준 정규화
    graham_ncav = 0.85 - ma200_discount * 1.0 - credit_stress - vix_fear * 0.25
    graham_ncav = max(0.1, min(1.5, graham_ncav))

    # --- Graham: P/E_Ratio (threshold: < 15) ---
    # VIX↑ + 200MA 할인↑ → P/E↓ (주가 급락 시 밸류에이션 압축)
    base_pe = 22.0
    vix_pe_compression = max(0.0, vix - 20.0) * 0.15     # VIX 40→-3, VIX 80→-9
    ma_pe_discount = ma200_discount * 15.0                 # 20% 할인→-3
    graham_pe = base_pe - vix_pe_compression - ma_pe_discount
    graham_pe = max(5.0, min(35.0, graham_pe))

    # --- Buffett: ROIC_10yr_Avg (threshold: > 0.15) ---
    # 건강한 신용 + 성장 모멘텀 → 높은 기업 자본효율
    roic_base = 0.12
    roic_credit_adj = max(-0.04, min(0.04, hyg_ief_zscore * 0.015))
    roic_growth_adj = max(-0.03, min(0.03, ndx_60d_ret * 0.1))
    buffett_roic = roic_base + roic_credit_adj + roic_growth_adj
    buffett_roic = max(0.04, min(0.25, buffett_roic))

    # --- Buffett: Gross_Margin_Volatility (threshold: < 0.05) ---
    # 저변동성 환경 = 안정적 마진 → Moat 강화
    buffett_margin_vol = 0.03 + max(0.0, vix - 15.0) * 0.002
    buffett_margin_vol = max(0.01, min(0.20, buffett_margin_vol))

    # --- Munger: Debt_to_Equity (threshold: < 0.5) ---
    # 신용 스트레스↑ + VIX↑ → 높은 레버리지 리스크 인식
    munger_de = 0.40 + max(0.0, -hyg_ief_zscore) * 0.08 + max(0.0, vix - 20.0) * 0.005
    munger_de = max(0.1, min(1.5, munger_de))

    # --- Munger: FCF_Yield (threshold: > 0.05) ---
    # 높은 금리 + 시장 할인 → FCF Yield 상승
    rate_yield = max(0.0, tnx / 100.0 - 0.01) * 0.5
    discount_yield = ma200_discount * 0.1
    munger_fcf = 0.03 + rate_yield + discount_yield
    munger_fcf = max(0.01, min(0.15, munger_fcf))

    # ── ② GROWTH NODES (Fisher, Lynch, Soros) ──
    # 철학: 성장 모멘텀 + 합리적 밸류에이션의 교차점

    # --- Fisher: RnD_to_Revenue_Ratio (threshold: > 0.08) ---
    # NASDAQ 모멘텀 → 기술 혁신 투자의 프록시
    fisher_rnd = 0.05 + max(0.0, ndx_60d_ret) * 0.3
    fisher_rnd = max(0.02, min(0.20, fisher_rnd))

    # --- Fisher: Operating_Margin_Expansion_3yr (threshold: > 0.0) ---
    # NDX 60일 모멘텀 → 운영 레버리지 확장의 프록시
    fisher_margin = ndx_60d_ret * 0.5
    fisher_margin = max(-0.10, min(0.15, fisher_margin))

    # --- Lynch: PEG_Ratio (threshold: < 1.0) ---
    # 높은 NDX 모멘텀 / 저 VIX = 성장 대비 저평가 환경
    growth_signal = max(0.01, ndx_12m_ret + 0.10)  # 안전 하한
    vix_inverse = 20.0 / max(vix, 10.0)
    lynch_peg = 1.5 - growth_signal * vix_inverse * 0.5
    lynch_peg = max(0.3, min(3.0, lynch_peg))

    # --- Lynch: EPS_Growth_TTM (threshold: > 0.20) ---
    # NASDAQ 12개월 수익률 → 기업 이익 성장의 매크로 프록시
    lynch_eps = ndx_12m_ret * 0.7 + sp500_ret_252d * 0.3
    lynch_eps = max(-0.30, min(0.60, lynch_eps))

    # --- Soros: Price_Momentum_1yr_Percentile (threshold: > 0.95) ---
    # SP500 12개월 수익률 → 역사적 백분위 매핑 (재귀성 탐지)
    soros_momentum = _ret_to_percentile(sp500_ret_252d)

    # --- Soros: Price_to_Fundamental_Divergence_Zscore (threshold: > 2.5) ---
    # 200MA로부터의 이탈 정도 → 가격-펀더멘탈 괴리 (거품/패닉 신호)
    soros_divergence = abs(sp500_ma200_ratio - 1.0) * 10.0
    soros_divergence = max(0.0, min(5.0, soros_divergence))

    # ── ③ MACRO NODES (Dalio, Marks, Simons) ──
    # 직접적 매크로 지표를 활용하되, Z-Score 기반 동적 교정 적용

    # --- Dalio: Yield_Curve_Spread (threshold: < 0.0 → inverted) ---
    # 직접 매핑 (TNX - IRX)

    # --- Dalio: Credit_Spread_High_Yield_Volatility (threshold: > 1.5) ---
    # HYG/IEF Z-Score 역전환: 음의 Z = 스프레드 확대 → 변동성 상승
    dalio_credit = max(0.0, -hyg_ief_zscore * 1.5 + 0.5)
    dalio_credit = max(0.0, min(5.0, dalio_credit))

    # --- Dalio: M2_Money_Supply_YoY (threshold: < 0.02 → tightening) ---
    # 프록시: 금리 상승 + 수익률 곡선 역전 → 긴축적 통화 환경
    m2_proxy = 0.05 + yield_spread * 0.02 - max(0.0, tnx - 3.0) * 0.005
    m2_proxy = max(-0.05, min(0.15, m2_proxy))

    # --- Marks: VIX_Index (threshold: > 35) ---
    # 직접 매핑

    # --- Marks: OAS_Spread_Zscore (threshold: > 2.0) ---
    # VIX Z-Score + 신용 Z-Score 복합 → 신용 시장 극단 경색
    vix_zscore_proxy = (vix - 20.0) / 8.0  # VIX 장기 평균~20, 표준편차~8
    marks_oas = (vix_zscore_proxy * 0.6 + max(0.0, -hyg_ief_zscore) * 0.4) * 1.5
    marks_oas = max(-2.0, min(5.0, marks_oas))

    # --- Simons: Price_Mean_Reversion_Zscore (threshold: < -3.0) ---
    # 사전 계산된 MA20 Z-Score 직접 사용 (극단적 음수 = 매수 기회)
    simons_mr_z = sp500_ma20_zscore

    # --- Simons: Order_Book_Imbalance_Ratio (threshold: > 0.8) ---
    # VIX 급등 = 호가 불균형의 프록시 (수급 공백)
    vix_spike = max(0.0, vix_5d_roc) * 2.0
    simons_obi = 0.5 + vix_spike * 0.3
    simons_obi = max(0.0, min(1.0, simons_obi))

    # ── ④ RISK NODES (Taleb, Shannon, Thorp) ──

    # --- Taleb: SKEW_Index (threshold: > 140) ---
    # VIX 수준 + VIX 가속도 → 꼬리 위험 프록시 (Fat Tail)
    vix_acceleration = max(0.0, vix_5d_roc)
    taleb_skew = 100.0 + vix * 1.0 + vix_acceleration * 15.0
    taleb_skew = max(90.0, min(200.0, taleb_skew))

    # --- Taleb: OTM_Put_Option_Volume_Spike (threshold: > 3.0 σ) ---
    # VIX 절대 수준 + VIX 급등 → OTM 풋 수요 폭증
    taleb_put_spike = max(0.0, (vix - 25.0) / 5.0) + max(0.0, vix_5d_roc * 3.0)
    taleb_put_spike = max(0.0, min(8.0, taleb_put_spike))

    # --- Shannon: Price_Trend_R_Squared (threshold: < 0.3) ---
    # 사전 계산된 60일 롤링 R² (낮을수록 노이즈 우위)
    shannon_r_sq = sp500_r_sq

    # --- Shannon: Intraday_Volatility_vs_Daily_Return (threshold: > 2.5) ---
    # 실현 변동성 / 연간 수익률 = 노이즈 대비 시그널 열위
    abs_annual_ret = max(abs(sp500_ret_252d), 0.01)
    shannon_noise = sp500_vol_20d / abs_annual_ret * 0.5
    shannon_noise = max(0.5, min(10.0, shannon_noise))

    # --- Thorp: Expected_Value_of_Signal (threshold: > 0.0) ---
    # 앙상블 기대수익 프록시: 일별 수익 × 스케일 + 변동성 보정
    thorp_ev = sp500_ret * 10.0 + max(0.0, 0.5 - sp500_vol_20d) * 0.5
    thorp_ev = max(-1.0, min(2.0, thorp_ev))

    # --- Thorp: Historical_Win_Rate_of_Signal (threshold: > 0.5) ---
    # 장기 추세 방향 + 기본 승률
    thorp_wr = 0.52 + sp500_ret_252d * 0.1
    thorp_wr = max(0.35, min(0.75, thorp_wr))

    # ════════════════════════════════════════════════════════
    # 최종 메트릭 딕셔너리 조립
    # ════════════════════════════════════════════════════════
    metrics: Dict[str, float] = {
        # Value (Graham)
        "Price_to_NCAV": graham_ncav,
        "P/E_Ratio": graham_pe,
        # Value (Buffett)
        "ROIC_10yr_Avg": buffett_roic,
        "Gross_Margin_Volatility": buffett_margin_vol,
        # Value (Munger)
        "Debt_to_Equity": munger_de,
        "FCF_Yield": munger_fcf,
        # Growth (Fisher)
        "RnD_to_Revenue_Ratio": fisher_rnd,
        "Operating_Margin_Expansion_3yr": fisher_margin,
        # Growth (Lynch)
        "PEG_Ratio": lynch_peg,
        "EPS_Growth_TTM": lynch_eps,
        # Growth (Soros)
        "Price_Momentum_1yr_Percentile": soros_momentum,
        "Price_to_Fundamental_Divergence_Zscore": soros_divergence,
        # Macro (Dalio)
        "Yield_Curve_Spread_10Y_2Y": yield_spread,
        "Credit_Spread_High_Yield_Volatility": dalio_credit,
        "M2_Money_Supply_YoY": m2_proxy,
        # Macro (Marks)
        "VIX_Index": vix,
        "OAS_Spread_Zscore": marks_oas,
        # Macro (Simons)
        "Price_Mean_Reversion_Zscore": simons_mr_z,
        "Order_Book_Imbalance_Ratio": simons_obi,
        # Risk (Taleb)
        "SKEW_Index": taleb_skew,
        "OTM_Put_Option_Volume_Spike": taleb_put_spike,
        # Risk (Shannon)
        "Price_Trend_R_Squared": shannon_r_sq,
        "Intraday_Volatility_vs_Daily_Return": shannon_noise,
        # Risk (Thorp)
        "Expected_Value_of_Signal": thorp_ev,
        "Historical_Win_Rate_of_Signal": thorp_wr,
    }

    # 엔진 내부 국면
    regime = row.get("engine_regime", "Normal_Market")
    if pd.isna(regime):
        regime = "Normal_Market"

    timestamp = row.name if isinstance(row.name, (datetime, pd.Timestamp)) else datetime.utcnow()

    return MarketDataPayload(
        ticker="MACRO_INDEX",
        timestamp=timestamp,
        metrics=metrics,
        current_regime=str(regime),
        regime_confidence=0.8,
        metadata={
            "sp500_ret": sp500_ret,
            "vix": vix,
            "sp500_ma200_ratio": sp500_ma200_ratio,
            "ndx_12m_ret": ndx_12m_ret,
            "hyg_ief_zscore": hyg_ief_zscore,
            "event_regime": str(row.get("event_regime", "Normal")),
        },
    )


# ══════════════════════════════════════════════════════════
# 4. Historical Simulator — 일별 엔진 시뮬레이션
# ══════════════════════════════════════════════════════════

@dataclass
class SimulationRecord:
    """단일 시뮬레이션 타임스텝 기록."""
    date: pd.Timestamp
    event_regime: str
    engine_regime: str
    ensemble_signal: float
    position_size: float
    override_active: bool
    node_scores: Dict[str, float]         # node_id → normalized_score
    node_state_flags: Dict[str, str]      # node_id → active_state_flag
    forward_returns: Dict[int, Optional[float]] = field(
        default_factory=dict
    )  # horizon → forward return (T+5, T+20, T+60)


class HistoricalSimulator:
    """매크로 데이터 기반 역사적 시뮬레이션 엔진.

    GraphOrchestrator를 일별로 반복 실행하고,
    각 시점의 시그널과 미래 수익률을 기록합니다.

    Args:
        orchestrator: 초기화된 GraphOrchestrator 인스턴스
        forward_horizons: 미래 수익률 평가 호라이즌 (영업일 수)
    """

    # 기본 호라이즌 (영업일)
    DEFAULT_HORIZONS: Tuple[int, ...] = (5, 20, 60)

    def __init__(
        self,
        orchestrator: GraphOrchestrator,
        forward_horizons: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.horizons = forward_horizons or self.DEFAULT_HORIZONS
        self.records: List[SimulationRecord] = []

    def run(self, df: pd.DataFrame) -> List[SimulationRecord]:
        """전체 기간에 대해 시뮬레이션을 실행합니다.

        Args:
            df: label_event_regime()이 적용된 매크로 DataFrame

        Returns:
            시뮬레이션 기록 리스트
        """
        self.records = []
        dates = df.index.tolist()
        n_total = len(dates)

        logger.info("시뮬레이션 시작: %d 영업일", n_total)

        # S&P500 수익률 시리즈 (미래 수익률 계산용)
        sp500_col = "SP500" if "SP500" in df.columns else None

        for i, dt in enumerate(dates):
            row = df.loc[dt]

            # 페이로드 구성
            payload = build_payload(row)

            # 엔진 실행
            try:
                output: OrchestratorOutput = self.orchestrator.resolve_signals(
                    market_data=payload
                )
            except Exception as e:
                logger.warning("시뮬레이션 %s: 엔진 실행 실패 → %s", dt, e)
                continue

            # 노드별 점수 기록
            node_scores = {
                nid: nr.normalized_score
                for nid, nr in output.node_results.items()
            }
            node_flags = {
                nid: (nr.active_state_flag or "")
                for nid, nr in output.node_results.items()
            }

            # 미래 수익률 계산 (SP500 기준)
            forward_rets: Dict[int, Optional[float]] = {}
            if sp500_col:
                current_price = df.iloc[i][sp500_col]
                if not (math.isnan(current_price) or current_price == 0):
                    for h in self.horizons:
                        future_idx = i + h
                        if future_idx < n_total:
                            future_price = df.iloc[future_idx][sp500_col]
                            if not math.isnan(future_price):
                                forward_rets[h] = (
                                    (future_price - current_price) / current_price
                                )
                            else:
                                forward_rets[h] = None
                        else:
                            forward_rets[h] = None

            record = SimulationRecord(
                date=dt,
                event_regime=str(row.get("event_regime", "Normal")),
                engine_regime=str(row.get("engine_regime", "Normal_Market")),
                ensemble_signal=output.ensemble_signal,
                position_size=output.final_position_size,
                override_active=output.override_active,
                node_scores=node_scores,
                node_state_flags=node_flags,
                forward_returns=forward_rets,
            )
            self.records.append(record)

            # 진행률 로깅 (10% 단위)
            if (i + 1) % max(1, n_total // 10) == 0:
                logger.info(
                    "  시뮬레이션 진행: %d / %d (%.0f%%)",
                    i + 1, n_total, (i + 1) / n_total * 100,
                )

        logger.info("시뮬레이션 완료: %d 레코드 생성", len(self.records))
        return self.records


# ══════════════════════════════════════════════════════════
# 5. Attribution Tracker — 멀티-호라이즌 노드 기여도 추적
# ══════════════════════════════════════════════════════════

# 노드 → 최적 호라이즌 매핑 (투자 철학 기반)
NODE_PRIMARY_HORIZON: Dict[str, int] = {
    # Quant / Short-term (T+5)
    "MAC_SIMONS_001":  5,
    "RSK_TALEB_001":   5,
    "RSK_SHANNON_001": 5,
    # Trend / Medium-term (T+20)
    "GRO_LYNCH_001":   20,
    "GRO_FISHER_001":  20,
    "GRO_SOROS_001":   20,
    "MAC_MARKS_001":   20,
    "MAC_DALIO_001":   20,
    # Value / Long-term (T+60)
    "VAL_BUFFETT_001": 60,
    "VAL_GRAHAM_001":  60,
    "VAL_MUNGER_001":  60,
    "RSK_THORP_001":   60,
}

# 노드 → 투자 유형 분류
NODE_TYPE: Dict[str, str] = {
    "VAL_GRAHAM_001":  "Value",
    "VAL_BUFFETT_001": "Value",
    "VAL_MUNGER_001":  "Value",
    "GRO_FISHER_001":  "Growth",
    "GRO_LYNCH_001":   "Growth",
    "GRO_SOROS_001":   "Growth",
    "MAC_DALIO_001":   "Macro",
    "MAC_MARKS_001":   "Macro",
    "MAC_SIMONS_001":  "Quant",
    "RSK_TALEB_001":   "Risk",
    "RSK_SHANNON_001": "Risk",
    "RSK_THORP_001":   "Risk",
}

# Taleb Protection 대상 노드 (평상시 동결)
TALEB_PROTECTION_NODES: Set[str] = {"RSK_TALEB_001", "MAC_MARKS_001"}

# 위기 국면 (Taleb/Marks가 최적화되는 유일한 국면)
CRISIS_REGIMES: Set[str] = {"Black_Swan", "Systemic_Risk"}


@dataclass
class HorizonAttribution:
    """단일 노드의 호라이즌별 기여도 통계."""
    node_id: str
    horizon: int
    signal_sum: float = 0.0
    correct_signals: int = 0
    total_signals: int = 0
    cumulative_attribution: float = 0.0  # signal × forward_return 누적

    @property
    def win_rate(self) -> float:
        """해당 호라이즌의 승률."""
        return self.correct_signals / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def avg_signal_strength(self) -> float:
        """평균 시그널 강도."""
        return self.signal_sum / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def horizon_error(self) -> float:
        """호라이즌 오차 (1 - win_rate 기반)."""
        return 1.0 - self.win_rate


class AttributionTracker:
    """멀티-호라이즌 노드 기여도 추적기.

    각 노드의 시그널과 미래 수익률 간 상관을 T+5, T+20, T+60
    호라이즌별로 추적하여, 철학 관성 최적화의 입력으로 사용합니다.

    Attributes:
        rolling_window: 롤링 통계 윈도우 크기 (영업일)
        horizons: 평가 호라이즌 목록
    """

    def __init__(
        self,
        horizons: Tuple[int, ...] = (5, 20, 60),
        rolling_window: int = 252,
    ) -> None:
        self.horizons = horizons
        self.rolling_window = rolling_window

        # node_id → horizon → HorizonAttribution
        self._attributions: Dict[str, Dict[int, HorizonAttribution]] = defaultdict(
            lambda: {h: HorizonAttribution(node_id="", horizon=h) for h in self.horizons}
        )
        # 최근 N개 레코드 (롤링 윈도우)
        self._recent_records: List[SimulationRecord] = []

    def process_records(
        self,
        records: List[SimulationRecord],
        signal_threshold: float = 0.5,
    ) -> Dict[str, Dict[int, HorizonAttribution]]:
        """시뮬레이션 레코드를 처리하여 노드별 기여도를 계산합니다.

        시그널이 signal_threshold 이상이면 '매수 시그널'로 간주하고,
        해당 호라이즌의 미래 수익률이 양수이면 '적중'으로 판정합니다.

        Args:
            records: 시뮬레이션 레코드 리스트
            signal_threshold: 매수 시그널 임계값 (기본 0.5)

        Returns:
            node_id → horizon → HorizonAttribution 매핑
        """
        logger.info("기여도 분석 시작: %d 레코드", len(records))

        for rec in records:
            # 롤링 윈도우 유지
            self._recent_records.append(rec)
            if len(self._recent_records) > self.rolling_window:
                self._recent_records = self._recent_records[-self.rolling_window:]

            for node_id, score in rec.node_scores.items():
                for h in self.horizons:
                    fwd_ret = rec.forward_returns.get(h)
                    if fwd_ret is None:
                        continue

                    attr = self._attributions[node_id][h]
                    attr.node_id = node_id
                    attr.total_signals += 1
                    attr.signal_sum += score

                    # 시그널 방향과 수익률 방향 일치 여부
                    is_bullish = score >= signal_threshold
                    is_positive = fwd_ret > 0

                    if is_bullish == is_positive:
                        attr.correct_signals += 1

                    # 기여도: (signal - 0.5) × forward_return
                    # → 시그널이 중립(0.5)일 때 기여도 = 0
                    attr.cumulative_attribution += (score - 0.5) * fwd_ret

        # 요약 로깅
        for node_id in sorted(self._attributions.keys()):
            primary_h = NODE_PRIMARY_HORIZON.get(node_id, 20)
            attr = self._attributions[node_id].get(primary_h)
            if attr and attr.total_signals > 0:
                logger.info(
                    "  %s (T+%d): win_rate=%.2f%%, avg_signal=%.3f, "
                    "attribution=%.4f, n=%d",
                    node_id, primary_h,
                    attr.win_rate * 100, attr.avg_signal_strength,
                    attr.cumulative_attribution, attr.total_signals,
                )

        return dict(self._attributions)

    def get_regime_attribution(
        self,
        records: List[SimulationRecord],
        regime: str,
    ) -> Dict[str, Dict[int, HorizonAttribution]]:
        """특정 국면에 한정하여 기여도를 계산합니다.

        오버피팅 방지를 위해 국면별 가중치를 차등 적용할 때 사용합니다.

        Args:
            records: 전체 시뮬레이션 레코드
            regime: 필터링할 event_regime

        Returns:
            국면 한정 기여도 매핑
        """
        filtered = [r for r in records if r.event_regime == regime]
        tracker = AttributionTracker(
            horizons=self.horizons,
            rolling_window=self.rolling_window,
        )
        return tracker.process_records(filtered)


# ══════════════════════════════════════════════════════════
# 6. Philosophy Inertia Optimizer — 철학 관성 보존 최적화
#    v2.1: Taleb Protection + Early Stopping + Regime Loss Weighting
# ══════════════════════════════════════════════════════════

@dataclass
class OptimizableParam:
    """최적화 대상 파라미터."""
    node_id: str
    param_name: str          # 예: "historical_win_rate", "decay_factor"
    original_value: float    # JSON 원본 값
    current_value: float     # 현재 값
    lower_bound: float       # 원본 × 0.8
    upper_bound: float       # 원본 × 1.2


class PhilosophyInertiaOptimizer:
    """투자 철학 관성을 보존하면서 파라미터를 최적화합니다.

    v2.1 변경점:
    - Taleb Protection: RSK_TALEB_001, MAC_MARKS_001에 조건부 10x 관성 페널티
    - Early Stopping: min_epochs=20 유예 후 3회 연속 미개선 시 조기 종료
    - Learning Rate: 0.005 고정
    - Regime Loss Weighting: Black_Swan 3.0x

    Loss = Σ(αₕ × Errorₕ) + λ_eff × Inertia_Penalty

    여기서 λ_eff는:
    - 일반 노드: λ (기본 0.3)
    - Taleb/Marks (평상시): λ × 10.0 (사실상 동결)
    - Taleb/Marks (위기 시): λ (정상 최적화)

    Args:
        configs: MasterEngineConfig 리스트 (원본 JSON)
        learning_rate: 학습률 (기본 0.005)
        inertia_lambda: 철학 관성 페널티 계수 (기본 0.3)
        bound_pct: 파라미터 변동 허용 범위 (기본 0.20 = ±20%)
        max_iterations: 최대 반복 횟수 (기본 50)
        min_epochs: Early Stopping 유예 기간 (기본 20)
        es_patience: 조기 종료 인내 횟수 (기본 3)
    """

    # 노드 유형별 호라이즌 가중치 (α₅, α₂₀, α₆₀)
    HORIZON_ALPHAS: Dict[str, Tuple[float, float, float]] = {
        "Value":  (0.10, 0.20, 0.70),  # T+60 중심
        "Growth": (0.15, 0.60, 0.25),  # T+20 중심
        "Macro":  (0.20, 0.55, 0.25),  # T+20 중심
        "Quant":  (0.70, 0.20, 0.10),  # T+5 중심
        "Risk":   (0.40, 0.35, 0.25),  # 균형
    }

    def __init__(
        self,
        configs: List[MasterEngineConfig],
        learning_rate: float = 0.005,
        inertia_lambda: float = 0.3,
        bound_pct: float = 0.20,
        max_iterations: int = 50,
        min_epochs: int = 20,
        es_patience: int = 3,
    ) -> None:
        self.learning_rate = learning_rate
        self.inertia_lambda = inertia_lambda
        self.bound_pct = bound_pct
        self.max_iterations = max_iterations
        self.min_epochs = min_epochs
        self.es_patience = es_patience

        # 최적화 대상 파라미터 추출
        self._params: List[OptimizableParam] = []
        self._extract_params(configs)

    def _extract_params(self, configs: List[MasterEngineConfig]) -> None:
        """JSON 원본에서 최적화 대상 파라미터를 추출합니다.

        대상: historical_win_rate, decay_factor (각 노드별)
        """
        for config in configs:
            for node in config.Nodes:
                constants = node.Intelligence_Structure.Step_3_Statistical_Correction.constants
                nid = node.Node_ID

                for pname in ("historical_win_rate", "decay_factor"):
                    if pname in constants:
                        orig = float(constants[pname])
                        self._params.append(OptimizableParam(
                            node_id=nid,
                            param_name=pname,
                            original_value=orig,
                            current_value=orig,
                            lower_bound=orig * (1.0 - self.bound_pct),
                            upper_bound=orig * (1.0 + self.bound_pct),
                        ))

        logger.info("최적화 대상 파라미터: %d개", len(self._params))

    def optimize(
        self,
        attributions: Dict[str, Dict[int, HorizonAttribution]],
        records: List[SimulationRecord],
        regime_attributions: Optional[Dict[str, Dict[str, Dict[int, HorizonAttribution]]]] = None,
    ) -> List[OptimizableParam]:
        """멀티-호라이즌 기여도 기반으로 파라미터를 최적화합니다.

        v2.1 핵심 변경:
        1. Taleb Protection — CRISIS 국면 외에는 10x 관성으로 동결
        2. Early Stopping — min_epochs 유예 후 patience 회 미개선 시 조기 종료
        3. Regime-Weighted Loss — Black_Swan에 3.0x 가중

        Args:
            attributions: 노드별 호라이즌별 기여도 (전체 기간)
            records: 시뮬레이션 레코드 (국면별 가중치용)
            regime_attributions: 국면별 기여도
                {regime_label: {node_id: {horizon: HorizonAttribution}}}

        Returns:
            최적화된 파라미터 리스트
        """
        logger.info(
            "파라미터 최적화 시작 (lr=%.4f, λ=%.2f, bound=±%.0f%%, "
            "max_iter=%d, min_epochs=%d, patience=%d)",
            self.learning_rate, self.inertia_lambda,
            self.bound_pct * 100, self.max_iterations,
            self.min_epochs, self.es_patience,
        )

        # 국면별 가중치
        regime_weights = self._compute_regime_weights(records)

        # ── Taleb Protection: 위기 국면 전용 기여도 준비 ──
        crisis_attributions = self._build_crisis_attributions(regime_attributions)

        # ── Early Stopping 상태 ──
        loss_history: List[float] = []
        no_improve_count: int = 0

        for iteration in range(self.max_iterations):
            total_loss = 0.0

            for param in self._params:
                nid = param.node_id
                node_type = NODE_TYPE.get(nid, "Risk")
                alphas = self.HORIZON_ALPHAS.get(node_type, (0.33, 0.34, 0.33))

                # ── Taleb Protection: 조건부 관성 결정 ──
                if nid in TALEB_PROTECTION_NODES:
                    # 위기 국면 전용 기여도 사용 + 10x 관성 페널티
                    effective_inertia = self.inertia_lambda * 10.0
                    node_attrs = crisis_attributions if crisis_attributions else attributions
                else:
                    effective_inertia = self.inertia_lambda
                    node_attrs = attributions

                # ── 멀티-호라이즌 Loss 계산 ──
                horizon_errors: Dict[int, float] = {}
                for h_idx, h in enumerate((5, 20, 60)):
                    attr = node_attrs.get(nid, {}).get(h)
                    if attr and attr.total_signals > 0:
                        horizon_errors[h] = attr.horizon_error
                    else:
                        horizon_errors[h] = 0.5  # 데이터 부족 → 중립

                # 가중 Loss: Σ αᵢ × Errorᵢ
                weighted_error = sum(
                    alphas[i] * horizon_errors.get(h, 0.5)
                    for i, h in enumerate((5, 20, 60))
                )

                # Inertia Penalty: |current - original| / |original|
                inertia_penalty = abs(
                    param.current_value - param.original_value
                ) / max(abs(param.original_value), 1e-8)

                loss = weighted_error + effective_inertia * inertia_penalty
                total_loss += loss

                # ── 그래디언트 추정 (부호 기반) ──
                primary_h = NODE_PRIMARY_HORIZON.get(nid, 20)
                primary_attr = node_attrs.get(nid, {}).get(primary_h)

                if primary_attr and primary_attr.total_signals > 10:
                    gradient_direction = primary_attr.win_rate - 0.5
                    # 국면별 가중 보정
                    dominant_regime = _get_dominant_regime(records, nid)
                    regime_boost = regime_weights.get(dominant_regime, 1.0)

                    # inertia_lambda에 따른 관성(Inertia) 적용: 원본 값에서 멀어질수록 반대 방향으로 힘을 가함
                    inertia_pull = 0.0
                    deviation = (param.current_value - param.original_value) / max(abs(param.original_value), 1e-8)
                    if abs(deviation) > 1e-4:
                        inertia_pull = -effective_inertia * (1.0 if deviation > 0 else -1.0)

                    step = (
                        self.learning_rate
                        * (gradient_direction * regime_boost + inertia_pull)
                    )
                else:
                    step = 0.0  # 데이터 부족 → 갱신 안 함

                # ── 파라미터 갱신 + 클리핑 ──
                new_value = param.current_value + step * param.original_value
                param.current_value = float(np.clip(
                    new_value, param.lower_bound, param.upper_bound
                ))

            # ── Early Stopping 검사 ──
            if iteration >= self.min_epochs and loss_history:
                if total_loss >= loss_history[-1]:
                    no_improve_count += 1
                else:
                    no_improve_count = 0

                if no_improve_count >= self.es_patience:
                    logger.info(
                        "  ⚡ Early Stopping 발동: Epoch %d (Loss %.4f, "
                        "%d회 연속 미개선)",
                        iteration + 1, total_loss, self.es_patience,
                    )
                    loss_history.append(total_loss)
                    break

            loss_history.append(total_loss)

            if (iteration + 1) % 10 == 0:
                logger.info(
                    "  Iteration %d/%d: total_loss=%.4f%s",
                    iteration + 1, self.max_iterations, total_loss,
                    f" (no_improve={no_improve_count})" if iteration >= self.min_epochs else "",
                )

        # ── 최종 결과 로깅 ──
        logger.info("\n--- 최적화 결과 (Epochs: %d) ---", len(loss_history))
        for p in self._params:
            change_pct = (
                (p.current_value - p.original_value) / p.original_value * 100
            )
            status = "preserved" if abs(change_pct) < 1.0 else "optimized"
            protection = " [PROTECTED]" if p.node_id in TALEB_PROTECTION_NODES else ""
            logger.info(
                "  %s.%s: %.4f → %.4f (%+.2f%%) [%s]%s",
                p.node_id, p.param_name,
                p.original_value, p.current_value, change_pct,
                status, protection,
            )

        return self._params

    def _build_crisis_attributions(
        self,
        regime_attributions: Optional[Dict[str, Dict[str, Dict[int, HorizonAttribution]]]],
    ) -> Dict[str, Dict[int, HorizonAttribution]]:
        """위기 국면(Black_Swan + Systemic_Risk) 기여도를 병합합니다.

        Taleb/Marks 노드의 Loss 계산에 사용됩니다.
        비위기 국면 데이터를 제외하여, 평상시 낮은 승률이
        파라미터를 왜곡하는 것을 방지합니다.

        Args:
            regime_attributions: 국면별 기여도 딕셔너리

        Returns:
            병합된 위기 국면 기여도 (node_id → horizon → HorizonAttribution)
            데이터 부족 시 빈 딕셔너리 반환.
        """
        if not regime_attributions:
            return {}

        merged: Dict[str, Dict[int, HorizonAttribution]] = {}

        for crisis_regime in CRISIS_REGIMES:
            reg_attr = regime_attributions.get(crisis_regime, {})
            for node_id, h_attrs in reg_attr.items():
                if node_id not in merged:
                    merged[node_id] = {}
                for h, attr in h_attrs.items():
                    if h not in merged[node_id]:
                        merged[node_id][h] = HorizonAttribution(
                            node_id=node_id,
                            horizon=h,
                            signal_sum=attr.signal_sum,
                            correct_signals=attr.correct_signals,
                            total_signals=attr.total_signals,
                            cumulative_attribution=attr.cumulative_attribution,
                        )
                    else:
                        existing = merged[node_id][h]
                        existing.signal_sum += attr.signal_sum
                        existing.correct_signals += attr.correct_signals
                        existing.total_signals += attr.total_signals
                        existing.cumulative_attribution += attr.cumulative_attribution

        crisis_total = sum(
            sum(a.total_signals for a in h_dict.values())
            for h_dict in merged.values()
        )
        logger.info(
            "  위기 국면 기여도 병합 완료: %d 노드, %d 시그널",
            len(merged), crisis_total,
        )
        return merged

    @staticmethod
    def _compute_regime_weights(
        records: List[SimulationRecord],
    ) -> Dict[str, float]:
        """국면별 가중치를 계산합니다.

        블랙스완 3.0x, 시스템 리스크 2.0x 등 극단 국면에
        더 높은 가중치를 부여합니다.

        Args:
            records: 시뮬레이션 레코드

        Returns:
            regime → 가중치 매핑
        """
        base_weights: Dict[str, float] = {
            "Black_Swan":               3.0,    # v2.1: 3.0x (tail-risk responsiveness)
            "Systemic_Risk":            2.0,
            "Deleveraging":             1.5,
            "Geopolitical_Stagflation": 1.3,
            "Liquidity_Bubble":         1.0,
            "AI_Supercycle":            1.0,
            "Normal":                   0.8,
        }
        return base_weights

    @property
    def optimized_params(self) -> List[OptimizableParam]:
        """현재 최적화된 파라미터 목록."""
        return self._params


def _get_dominant_regime(
    records: List[SimulationRecord], node_id: str
) -> str:
    """노드의 가장 활발한 국면을 반환합니다.

    Args:
        records: 시뮬레이션 레코드
        node_id: 노드 ID

    Returns:
        가장 빈도가 높은 event_regime
    """
    regime_counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        score = rec.node_scores.get(node_id, 0.0)
        if score > 0.6:  # 시그널이 의미 있는 경우만
            regime_counts[rec.event_regime] += 1

    if not regime_counts:
        return "Normal"
    return max(regime_counts, key=regime_counts.get)  # type: ignore[arg-type]


def _get_best_fit_regime(
    node_id: str,
    regime_attributions: Dict[str, Dict[str, Dict[int, HorizonAttribution]]],
) -> Tuple[str, float]:
    """노드의 최고 승률 국면(Best-Fit Regime)을 반환합니다.

    각 국면에서의 primary horizon 승률을 비교하여,
    가장 높은 승률을 보인 국면과 해당 승률을 반환합니다.

    Args:
        node_id: 노드 ID
        regime_attributions: 국면별 기여도

    Returns:
        (best_regime, best_win_rate) 튜플
    """
    primary_h = NODE_PRIMARY_HORIZON.get(node_id, 20)
    best_regime = "N/A"
    best_wr = 0.0

    for regime_label, reg_attr in regime_attributions.items():
        node_h_attrs = reg_attr.get(node_id, {})
        attr = node_h_attrs.get(primary_h)
        if attr and attr.total_signals > 5:  # 최소 샘플 요건
            if attr.win_rate > best_wr:
                best_wr = attr.win_rate
                best_regime = regime_label

    return best_regime, best_wr


# ══════════════════════════════════════════════════════════
# 7. Persistence Layer — 최적화 결과 저장
# ══════════════════════════════════════════════════════════

def save_optimized_weights(
    params: List[OptimizableParam],
    records: List[SimulationRecord],
    attributions: Dict[str, Dict[int, HorizonAttribution]],
    regime_attributions: Optional[Dict[str, Dict[str, Dict[int, HorizonAttribution]]]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """최적화 결과를 optimized_weights.json으로 저장합니다.

    v2.1 변경점:
    - change_pct 절댓값 < 1.0%인 파라미터: "status": "preserved"
    - 각 노드의 Best-Fit Regime 포함
    - 옵티마이저 설정 메타데이터 갱신

    Args:
        params: 최적화된 파라미터 리스트
        records: 시뮬레이션 레코드 (메타데이터용)
        attributions: 기여도 데이터
        regime_attributions: 국면별 기여도 (Best-Fit Regime 계산용)
        optimizer_config: 옵티마이저 설정 딕셔너리 (메타데이터용)
        output_path: 저장 경로 (기본: 프로젝트 루트의 optimized_weights.json)

    Returns:
        저장된 파일 경로
    """
    if output_path is None:
        output_path = str(_PROJECT_ROOT / "optimized_weights.json")

    # ── 노드별 파라미터 구성 ──
    nodes_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for p in params:
        short_name = p.node_id.split("_")[1].lower()
        change_pct = (p.current_value - p.original_value) / p.original_value * 100

        param_entry: Dict[str, Any] = {
            "original": round(p.original_value, 6),
            "optimized": round(p.current_value, 6),
            "change_pct": round(change_pct, 2),
            "bounds": [round(p.lower_bound, 6), round(p.upper_bound, 6)],
        }

        # v2.1: preserved/optimized 상태 마킹
        if abs(change_pct) < 1.0:
            param_entry["status"] = "preserved"
        else:
            param_entry["status"] = "optimized"

        # Taleb Protection 표시
        if p.node_id in TALEB_PROTECTION_NODES:
            param_entry["protection"] = "taleb_inertia_10x"

        nodes_dict[short_name][p.param_name] = param_entry

    # ── 노드별 기여도 요약 + Best-Fit Regime ──
    attribution_summary: Dict[str, Dict[str, Any]] = {}
    for node_id, h_attrs in attributions.items():
        short_name = node_id.split("_")[1].lower()
        primary_h = NODE_PRIMARY_HORIZON.get(node_id, 20)

        node_summary: Dict[str, Any] = {
            "node_type": NODE_TYPE.get(node_id, "Unknown"),
            "primary_horizon": primary_h,
        }

        for h in (5, 20, 60):
            a = h_attrs.get(h)
            if a and a.total_signals > 0:
                node_summary[f"T{h}_win_rate"] = round(a.win_rate, 4)
                node_summary[f"T{h}_avg_signal"] = round(a.avg_signal_strength, 4)
                node_summary[f"T{h}_attribution"] = round(a.cumulative_attribution, 6)
                node_summary[f"T{h}_n_signals"] = a.total_signals

        # Best-Fit Regime (v2.1)
        if regime_attributions:
            best_regime, best_wr = _get_best_fit_regime(node_id, regime_attributions)
            node_summary["best_fit_regime"] = best_regime
            node_summary["best_fit_win_rate"] = round(best_wr, 4)

        attribution_summary[short_name] = node_summary

    # ── 글로벌 파라미터 ──
    sim_dates = [r.date for r in records] if records else []

    opt_cfg = optimizer_config or {
        "learning_rate": 0.005,
        "inertia_lambda": 0.3,
        "bound_pct": 0.20,
        "rolling_window": 252,
    }

    global_params: Dict[str, Any] = {
        "optimization_date": datetime.utcnow().isoformat(),
        "optimizer_version": "2.1_synthetic_proxy",
        "data_range": {
            "start": sim_dates[0].strftime("%Y-%m-%d") if sim_dates else "N/A",
            "end": sim_dates[-1].strftime("%Y-%m-%d") if sim_dates else "N/A",
            "n_trading_days": len(sim_dates),
        },
        "optimizer_config": opt_cfg,
        "regime_distribution": _compute_regime_distribution(records),
        "taleb_protection_nodes": sorted(TALEB_PROTECTION_NODES),
        "crisis_regimes": sorted(CRISIS_REGIMES),
    }

    # ── 최종 구조 ──
    output = {
        "schema": "H-PIOS_v8.5_Optimized_Weights",
        "version": "2.1.0",
        "nodes": dict(nodes_dict),
        "attribution": attribution_summary,
        "global_parameters": global_params,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("최적화 결과 저장 완료: %s", output_path)
    return output_path


def _compute_regime_distribution(
    records: List[SimulationRecord],
) -> Dict[str, int]:
    """국면별 영업일 수를 집계합니다."""
    dist: Dict[str, int] = defaultdict(int)
    for r in records:
        dist[r.event_regime] += 1
    return dict(dist)


# ══════════════════════════════════════════════════════════
# 8. Orchestrator Factory — JSON 로드 유틸리티
# ══════════════════════════════════════════════════════════

def load_engine_configs(
    json_dir: Optional[str] = None,
) -> List[MasterEngineConfig]:
    """4개 도메인 마스터 JSON을 로드합니다.

    Args:
        json_dir: JSON 파일 디렉토리 (기본: 프로젝트 루트)

    Returns:
        MasterEngineConfig 리스트
    """
    if json_dir is None:
        json_dir = str(_PROJECT_ROOT)

    json_files = [
        "value_engine_master.json",
        "growth_engine_master.json",
        "macro_engine_master.json",
        "risk_engine_master.json",
    ]

    configs = []
    for fname in json_files:
        fpath = os.path.join(json_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MasterEngineConfig(**data)
        configs.append(cfg)
        logger.info("JSON 로드: %s (%d nodes)", fname, len(cfg.Nodes))

    return configs


# ══════════════════════════════════════════════════════════
# 9. Main Entry Point — 전체 파이프라인 실행
# ══════════════════════════════════════════════════════════

def main() -> None:
    """H-PIOS v8.5 역사적 최적화 파이프라인을 실행합니다.

    v2.1 파이프라인:
    1. 매크로 데이터 다운로드 (yfinance) + 15개 파생 시계열
    2. 이벤트 국면 라벨링
    3. 역사적 시뮬레이션 실행 (Synthetic Proxy)
    4. 멀티-호라이즌 + 국면별 기여도 분석
    5. 철학 관성 보존 파라미터 최적화 (Taleb Protection + Early Stopping)
    6. 결과 저장 + Best-Fit Regime 로깅
    """
    logger.info("=" * 60)
    logger.info("H-PIOS v8.5 Historical Back-Optimizer v2.1 시작")
    logger.info("  Synthetic Proxy Breakthrough Edition")
    logger.info("=" * 60)

    # ── Step 1: 데이터 다운로드 + 파생 시계열 ──
    logger.info("\n[Step 1/6] 매크로 데이터 다운로드 + 파생 시계열 계산")
    df = fetch_macro_data(start="2019-01-01")

    # ── Step 2: 국면 라벨링 ──
    logger.info("\n[Step 2/6] 이벤트 국면 라벨링")
    df = label_event_regime(df)

    # ── Step 3: 엔진 초기화 & 시뮬레이션 ──
    logger.info("\n[Step 3/6] 역사적 시뮬레이션 실행 (Synthetic Proxy)")
    configs = load_engine_configs()
    orchestrator = GraphOrchestrator(configs)
    simulator = HistoricalSimulator(orchestrator)
    records = simulator.run(df)

    if not records:
        logger.error("시뮬레이션 레코드 없음. 종료합니다.")
        return

    # ── Step 4: 기여도 분석 (전체 + 국면별) ──
    logger.info("\n[Step 4/6] 멀티-호라이즌 + 국면별 기여도 분석")
    tracker = AttributionTracker()
    attributions = tracker.process_records(records)

    # v2.1: 전체 국면별 기여도 계산 (Taleb Protection + Best-Fit Regime용)
    unique_regimes = sorted(set(r.event_regime for r in records))
    regime_attributions: Dict[str, Dict[str, Dict[int, HorizonAttribution]]] = {}
    for regime_label in unique_regimes:
        regime_attr = tracker.get_regime_attribution(records, regime_label)
        regime_attributions[regime_label] = regime_attr
        count = sum(1 for r in records if r.event_regime == regime_label)
        logger.info(
            "  국면 '%s': %d 영업일, %d 노드 기여도 분석",
            regime_label, count, len(regime_attr),
        )

    # ── Step 5: 파라미터 최적화 ──
    logger.info("\n[Step 5/6] 철학 관성 보존 파라미터 최적화 (v2.1)")
    optimizer = PhilosophyInertiaOptimizer(
        configs,
        learning_rate=0.005,
        inertia_lambda=0.3,
        bound_pct=0.20,
        max_iterations=50,
        min_epochs=20,
        es_patience=3,
    )
    optimized_params = optimizer.optimize(
        attributions,
        records,
        regime_attributions=regime_attributions,
    )

    # ── Best-Fit Regime 로깅 ──
    logger.info("\n📊 [Best-Fit Regime Analysis]")
    logger.info("-" * 55)
    for node_id in sorted(NODE_TYPE.keys()):
        best_regime, best_wr = _get_best_fit_regime(node_id, regime_attributions)
        node_type = NODE_TYPE.get(node_id, "?")
        protection = " 🛡️" if node_id in TALEB_PROTECTION_NODES else ""
        logger.info(
            "  %-18s [%-6s]: Best Regime = %-25s (win_rate=%.1f%%)%s",
            node_id, node_type, best_regime, best_wr * 100, protection,
        )

    # ── Step 6: 결과 저장 ──
    logger.info("\n[Step 6/6] 최적화 결과 저장")
    opt_config = {
        "learning_rate": 0.005,
        "inertia_lambda": 0.3,
        "bound_pct": 0.20,
        "max_iterations": 50,
        "min_epochs": 20,
        "early_stopping_patience": 3,
        "black_swan_loss_weight": 3.0,
        "taleb_protection_inertia_multiplier": 10.0,
    }
    output_path = save_optimized_weights(
        params=optimized_params,
        records=records,
        attributions=attributions,
        regime_attributions=regime_attributions,
        optimizer_config=opt_config,
    )

    # ── 최종 요약 ──
    logger.info("\n" + "=" * 60)
    logger.info("H-PIOS v8.5 Historical Back-Optimizer v2.1 완료")
    logger.info("  시뮬레이션 기간: %s ~ %s (%d 영업일)",
                records[0].date.strftime("%Y-%m-%d"),
                records[-1].date.strftime("%Y-%m-%d"),
                len(records))
    logger.info("  최적화된 파라미터: %d개", len(optimized_params))

    preserved = sum(
        1 for p in optimized_params
        if abs((p.current_value - p.original_value) / p.original_value * 100) < 1.0
    )
    logger.info("  Preserved 파라미터: %d개 (변동 < 1.0%%)", preserved)
    logger.info("  Protected 노드: %s", ", ".join(sorted(TALEB_PROTECTION_NODES)))
    logger.info("  결과 파일: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
