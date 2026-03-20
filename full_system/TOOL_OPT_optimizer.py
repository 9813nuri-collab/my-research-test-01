"""
H-PIOS v8.5 — Historical Back-Optimizer (v2.1: Synthetic Proxy Breakthrough)
=============================================================================
Optimizes the parameters of the 12-master engine based on macro data from
2019 to the present, fitting them to historical performance while preserving
each master's **Philosophy Inertia**.

v2.1 Key Changes
-----------------
1. **Synthetic Fundamental Proxies** — Converts macro index data into
   micro fundamental signals, generating differentiated signals per node.
   - Value: 200MA deviation + credit spread → synthetic NCAV/PE/ROIC proxy
   - Growth: NASDAQ momentum + VIX ROC → synthetic PEG/EPS proxy
   - Quant: MA Z-Score + direct VIX injection
   - Risk: VIX acceleration + trend R² → tail risk proxy

2. **Taleb Protection** — Conditional 10x inertia penalty for Taleb/Marks nodes
3. **Graceful Early Stopping** — min_epochs=20 grace period, lr=0.005
4. **Regime-Weighted Loss** — Black_Swan 3.0x weighting
5. **Enhanced Output** — Best-fit regime logging, preserved marking

Core Pipeline
---------------
1. **Data Ingestion** — yfinance + 15 derived time series (MA, Z-Score, ROC, R²)
2. **Event Regime Mapping** — 2019–present major event regime labeling
3. **Synthetic Payload Builder** — Macro → Micro conversion
4. **Historical Simulator** — Daily iteration via GraphOrchestrator
5. **Attribution Tracker** — T+5/T+20/T+60 multi-horizon + regime-level attribution
6. **Philosophy Inertia Optimizer** — Conditional inertia + Early Stopping
7. **Persistence Layer** — Save to optimized_weights.json

Design Principles
---------
- Never modify existing models.py / engine_core.py
- Never modify original JSON files → optimization results saved as separate JSON
- Overfitting prevention via ±20% bounds + rolling window + regime weighting

Runtime messages (logs, RuntimeError text) may be Korean and are left unchanged.
English gloss: 매크로 데이터 다운로드 시작 — starting macro download;
티커 … 데이터 없음 / 다운로드 실패 — no data / download failed;
모든 티커 다운로드 실패 — all tickers failed; 국면 라벨링 — regime labeling;
시뮬레이션 — simulation; 기여도 — attribution; Early Stopping 발동 — early stop;
최적화 — optimization; 위기 국면 — crisis regime; 결과 저장 — save results;
Step 1/6 … Step 6/6 — numbered pipeline stages.
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

# ── Determine project root (relative to this file) ──
_PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────
# Lazy imports: models & engine_core (must not be modified)
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
# 1. Data Ingestion — Macro data collection via yfinance + derived time series
# ══════════════════════════════════════════════════════════

# Ticker registry for download targets
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
    """Compute rolling linear trend R² of a price series (trend strength measure).

    R² → 1.0: Strong trend (either up or down)
    R² → 0.0: Noise-dominant (no directional bias)

    Args:
        prices: Price series (index: DatetimeIndex)
        window: Rolling window size (default 60 days)

    Returns:
        Rolling R² series (same index)
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
    """Download macro market data via yfinance and compute derived indicators.

    Merges Adjusted Close from 13 tickers, then computes 15 derived time
    series (MA, Z-Score, ROC, R², etc.) required for Synthetic Proxies.

    Args:
        start: Start date (YYYY-MM-DD). Default 2019-01-01.

    Returns:
        Clean DataFrame with DatetimeIndex.
        Base columns: SP500, KOSPI, NASDAQ100, OIL, GOLD, USDKRW, DXY,
                      TNX, IRX, VIX, HYG, IEF
        Derived columns: yield_spread, credit_spread_proxy,
                         SP500_ret, KOSPI_ret, NASDAQ100_ret,
                         SP500_MA200, SP500_MA200_ratio,
                         SP500_MA20_zscore,
                         NDX_12m_ret, NDX_60d_ret,
                         VIX_5d_roc, VIX_20d_roc,
                         HYG_IEF_zscore,
                         SP500_ret_252d, SP500_volatility_20d,
                         SP500_r_squared_60d

    Raises:
        RuntimeError: If yfinance is not installed or download fails.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        # KO: "yfinance required; run pip install yfinance"
        raise RuntimeError(
            "yfinance 패키지가 필요합니다. `pip install yfinance` 를 실행하세요."
        ) from e

    # KO: "Starting macro data download"
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

    # Merge
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Forward-fill → Backward-fill (handle leading NaNs)
    df = df.ffill().bfill()

    # ── Basic derived indicators ──
    # Yield Spread: 10Y - 2Y proxy
    if "TNX" in df.columns and "IRX" in df.columns:
        df["yield_spread"] = df["TNX"] - df["IRX"]
    else:
        df["yield_spread"] = 0.0

    # Credit Spread Proxy: HYG / IEF ratio (decline → spread widening)
    if "HYG" in df.columns and "IEF" in df.columns:
        df["credit_spread_proxy"] = df["HYG"] / df["IEF"]
    else:
        df["credit_spread_proxy"] = 1.0

    # Daily returns
    for eq in ["SP500", "KOSPI", "NASDAQ100"]:
        if eq in df.columns:
            df[f"{eq}_ret"] = df[eq].pct_change()
        else:
            df[f"{eq}_ret"] = 0.0

    # ── ═══════════════════════════════════════════════ ──
    # ── Derived time series for Synthetic Proxies (v2.1 new) ──
    # ── ═══════════════════════════════════════════════ ──

    # --- 1. S&P500 moving average deviation ---
    if "SP500" in df.columns:
        df["SP500_MA200"] = df["SP500"].rolling(200, min_periods=1).mean()
        df["SP500_MA200_ratio"] = df["SP500"] / df["SP500_MA200"]

        # 20-day MA Z-Score (mean-reversion signal)
        sp500_ma20 = df["SP500"].rolling(20, min_periods=1).mean()
        sp500_pct_dev = df["SP500"] / sp500_ma20 - 1.0
        sp500_ret_vol = df["SP500_ret"].rolling(20, min_periods=5).std().clip(lower=0.005)
        df["SP500_MA20_zscore"] = sp500_pct_dev / sp500_ret_vol
    else:
        df["SP500_MA200"] = 0.0
        df["SP500_MA200_ratio"] = 1.0
        df["SP500_MA20_zscore"] = 0.0

    # --- 2. NASDAQ 100 momentum ---
    if "NASDAQ100" in df.columns:
        df["NDX_12m_ret"] = df["NASDAQ100"].pct_change(252)  # ~12 months
        df["NDX_60d_ret"] = df["NASDAQ100"].pct_change(60)   # ~3 months
    else:
        df["NDX_12m_ret"] = 0.0
        df["NDX_60d_ret"] = 0.0

    # --- 3. VIX Rate of Change ---
    if "VIX" in df.columns:
        df["VIX_5d_roc"] = df["VIX"].pct_change(5)
        df["VIX_20d_roc"] = df["VIX"].pct_change(20)
    else:
        df["VIX_5d_roc"] = 0.0
        df["VIX_20d_roc"] = 0.0

    # --- 4. Credit spread Z-Score (60-day rolling) ---
    if "credit_spread_proxy" in df.columns:
        csp = df["credit_spread_proxy"]
        csp_mean = csp.rolling(60, min_periods=10).mean()
        csp_std = csp.rolling(60, min_periods=10).std().clip(lower=0.001)
        df["HYG_IEF_zscore"] = (csp - csp_mean) / csp_std
    else:
        df["HYG_IEF_zscore"] = 0.0

    # --- 5. S&P500 long-term return & volatility ---
    if "SP500" in df.columns:
        df["SP500_ret_252d"] = df["SP500"].pct_change(252)
        df["SP500_volatility_20d"] = (
            df["SP500_ret"].rolling(20, min_periods=5).std() * np.sqrt(252)
        )
    else:
        df["SP500_ret_252d"] = 0.0
        df["SP500_volatility_20d"] = 0.15

    # --- 6. Trend strength R² (for Shannon node) ---
    if "SP500" in df.columns:
        df["SP500_r_squared_60d"] = _compute_rolling_r_squared(df["SP500"], window=60)
    else:
        df["SP500_r_squared_60d"] = 0.5

    # ── NaN handling: fill initial NaNs in derived indicators with conservative defaults ──
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

    # Remove first row NaN
    df = df.iloc[1:]

    logger.info(
        "매크로 데이터 준비 완료: %d 행 × %d 컬럼 (%s ~ %s)",
        len(df), len(df.columns),
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
    return df


# ══════════════════════════════════════════════════════════
# 2. Event Regime Mapping — Historical regime labeling
# ══════════════════════════════════════════════════════════

# Event → (start_date, end_date) mapping (chronological)
EVENT_REGIMES: List[Tuple[str, str, str]] = [
    # (regime_label, start_date, end_date)
    ("Black_Swan",                "2020-02-19", "2020-04-09"),
    ("Liquidity_Bubble",          "2020-05-01", "2021-11-30"),
    ("Deleveraging",              "2022-01-01", "2022-10-31"),
    ("Systemic_Risk",             "2023-03-01", "2023-04-30"),
    ("AI_Supercycle",             "2023-05-01", "2025-12-31"),
    ("Geopolitical_Stagflation",  "2026-01-01", "2099-12-31"),  # ~today
]

# Regime → engine_core internal regime mapping
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
    """Add an event_regime column to the DataFrame.

    Labels historical event regimes based on date ranges and maps them
    to the engine's internal regime strings.

    Args:
        df: DataFrame with DatetimeIndex (output of fetch_macro_data)

    Returns:
        DataFrame with event_regime column appended
    """
    df = df.copy()
    df["event_regime"] = "Normal"

    for regime_label, start_str, end_str in EVENT_REGIMES:
        start_dt = pd.Timestamp(start_str)
        end_dt = pd.Timestamp(end_str)
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        df.loc[mask, "event_regime"] = regime_label

    # Add column mapped to engine internal regime
    df["engine_regime"] = df["event_regime"].map(_REGIME_TO_ENGINE).fillna("Normal_Market")

    regime_counts = df["event_regime"].value_counts()
    logger.info("국면 라벨링 완료:\n%s", regime_counts.to_string())
    return df


# ══════════════════════════════════════════════════════════
# 3. Synthetic Payload Builder — Macro → Micro proxy conversion
# ══════════════════════════════════════════════════════════

def _ret_to_percentile(ret_252d: float) -> float:
    """Convert 12-month return to an approximate historical percentile.

    Based on the long-term S&P500 12-month return distribution:
    - Mean: ~10%, Std: ~15%
    - 95th percentile: ~35%
    - 5th percentile: ~-15%

    Uses a sigmoid CDF approximation to map to the [0.01, 0.99] range.

    Args:
        ret_252d: Cumulative return over 252 trading days (~12 months)

    Returns:
        Approximate percentile (0.01 – 0.99)
    """
    z = (ret_252d - 0.10) / 0.15  # Standardize
    percentile = 1.0 / (1.0 + math.exp(-z * 1.5))
    return max(0.01, min(0.99, percentile))


def build_payload(row: pd.Series) -> MarketDataPayload:
    """Synthetic Fundamental Proxy Builder.

    **Core paradox resolution:**
    The engine expects micro (individual stock) fundamental metrics
    (P/E, NCAV, ROIC), but the optimization data consists only of
    macro (index-level) time series.

    This function applies financial-mathematical transformations to convert
    macro time series into synthetic micro signals, enabling each master node
    to generate differentiated responses across market regimes.

    Design principles:
    - **Value nodes**: Market discount (200MA deviation) + credit stress → synthetic NCAV/PE
    - **Growth nodes**: NASDAQ momentum + inverse VIX → synthetic PEG/EPS
    - **Macro nodes**: Yield spread + direct VIX mapping
    - **Quant nodes**: MA Z-Score + VIX acceleration
    - **Risk nodes**: Volatility clustering + trend R²

    Args:
        row: Single row from the DataFrame (pd.Series)

    Returns:
        MarketDataPayload instance
    """

    def _safe(val: Any, default: float = 0.0) -> float:
        """Safe conversion handling NaN/None/Inf."""
        if val is None:
            return default
        try:
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return default
            return fval
        except (TypeError, ValueError):
            return default

    # ── Primary macro data extraction ──
    sp500 = _safe(row.get("SP500"), 4000.0)
    vix = _safe(row.get("VIX"), 20.0)
    tnx = _safe(row.get("TNX"), 3.0)       # 10Y yield (%)
    irx = _safe(row.get("IRX"), 2.0)       # 13-week T-Bill (%)
    sp500_ret = _safe(row.get("SP500_ret"), 0.0)
    yield_spread = _safe(row.get("yield_spread"), 0.0)
    credit_proxy = _safe(row.get("credit_spread_proxy"), 1.0)

    # ── Secondary derived time series (pre-computed in fetch_macro_data) ──
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
    # Each proxy is financially calibrated to operate within realistic ranges
    # relative to the JSON threshold values.

    # ── ① VALUE NODES (Graham, Buffett, Munger) ──
    # Philosophy: "Be greedy when others are fearful"
    # Key signals: Deep discount below 200MA + credit stress

    # --- Graham: Price_to_NCAV (threshold: < 0.66) ---
    # 200MA deviation↑ + credit stress↑ + VIX↑ → NCAV↓ (deep value)
    ma200_discount = max(0.0, 1.0 - sp500_ma200_ratio)  # [0, ∞)
    credit_stress = max(0.0, -hyg_ief_zscore) * 0.1       # Positive when spread widens
    vix_fear = max(0.0, (vix - 20.0) / 60.0)             # Normalized relative to VIX 20
    graham_ncav = 0.85 - ma200_discount * 1.0 - credit_stress - vix_fear * 0.25
    graham_ncav = max(0.1, min(1.5, graham_ncav))

    # --- Graham: P/E_Ratio (threshold: < 15) ---
    # VIX↑ + 200MA discount↑ → P/E↓ (valuation compression during selloffs)
    base_pe = 22.0
    vix_pe_compression = max(0.0, vix - 20.0) * 0.15     # VIX 40→-3, VIX 80→-9
    ma_pe_discount = ma200_discount * 15.0                 # 20% discount→-3
    graham_pe = base_pe - vix_pe_compression - ma_pe_discount
    graham_pe = max(5.0, min(35.0, graham_pe))

    # --- Buffett: ROIC_10yr_Avg (threshold: > 0.15) ---
    # Healthy credit + growth momentum → high corporate capital efficiency
    roic_base = 0.12
    roic_credit_adj = max(-0.04, min(0.04, hyg_ief_zscore * 0.015))
    roic_growth_adj = max(-0.03, min(0.03, ndx_60d_ret * 0.1))
    buffett_roic = roic_base + roic_credit_adj + roic_growth_adj
    buffett_roic = max(0.04, min(0.25, buffett_roic))

    # --- Buffett: Gross_Margin_Volatility (threshold: < 0.05) ---
    # Low-volatility environment = stable margins → stronger moat
    buffett_margin_vol = 0.03 + max(0.0, vix - 15.0) * 0.002
    buffett_margin_vol = max(0.01, min(0.20, buffett_margin_vol))

    # --- Munger: Debt_to_Equity (threshold: < 0.5) ---
    # Credit stress↑ + VIX↑ → higher perceived leverage risk
    munger_de = 0.40 + max(0.0, -hyg_ief_zscore) * 0.08 + max(0.0, vix - 20.0) * 0.005
    munger_de = max(0.1, min(1.5, munger_de))

    # --- Munger: FCF_Yield (threshold: > 0.05) ---
    # Higher rates + market discount → rising FCF yield
    rate_yield = max(0.0, tnx / 100.0 - 0.01) * 0.5
    discount_yield = ma200_discount * 0.1
    munger_fcf = 0.03 + rate_yield + discount_yield
    munger_fcf = max(0.01, min(0.15, munger_fcf))

    # ── ② GROWTH NODES (Fisher, Lynch, Soros) ──
    # Philosophy: Intersection of growth momentum + reasonable valuation

    # --- Fisher: RnD_to_Revenue_Ratio (threshold: > 0.08) ---
    # NASDAQ momentum → proxy for technology innovation investment
    fisher_rnd = 0.05 + max(0.0, ndx_60d_ret) * 0.3
    fisher_rnd = max(0.02, min(0.20, fisher_rnd))

    # --- Fisher: Operating_Margin_Expansion_3yr (threshold: > 0.0) ---
    # NDX 60-day momentum → proxy for operating leverage expansion
    fisher_margin = ndx_60d_ret * 0.5
    fisher_margin = max(-0.10, min(0.15, fisher_margin))

    # --- Lynch: PEG_Ratio (threshold: < 1.0) ---
    # High NDX momentum / low VIX = undervalued relative to growth
    growth_signal = max(0.01, ndx_12m_ret + 0.10)  # Safe lower bound
    vix_inverse = 20.0 / max(vix, 10.0)
    lynch_peg = 1.5 - growth_signal * vix_inverse * 0.5
    lynch_peg = max(0.3, min(3.0, lynch_peg))

    # --- Lynch: EPS_Growth_TTM (threshold: > 0.20) ---
    # NASDAQ 12-month return → macro proxy for corporate earnings growth
    lynch_eps = ndx_12m_ret * 0.7 + sp500_ret_252d * 0.3
    lynch_eps = max(-0.30, min(0.60, lynch_eps))

    # --- Soros: Price_Momentum_1yr_Percentile (threshold: > 0.95) ---
    # SP500 12-month return → historical percentile mapping (reflexivity detection)
    soros_momentum = _ret_to_percentile(sp500_ret_252d)

    # --- Soros: Price_to_Fundamental_Divergence_Zscore (threshold: > 2.5) ---
    # Deviation from 200MA → price-fundamental divergence (bubble/panic signal)
    soros_divergence = abs(sp500_ma200_ratio - 1.0) * 10.0
    soros_divergence = max(0.0, min(5.0, soros_divergence))

    # ── ③ MACRO NODES (Dalio, Marks, Simons) ──
    # Directly leverages macro indicators with Z-Score-based dynamic calibration

    # --- Dalio: Yield_Curve_Spread (threshold: < 0.0 → inverted) ---
    # Direct mapping (TNX - IRX)

    # --- Dalio: Credit_Spread_High_Yield_Volatility (threshold: > 1.5) ---
    # HYG/IEF Z-Score inversion: negative Z = spread widening → higher volatility
    dalio_credit = max(0.0, -hyg_ief_zscore * 1.5 + 0.5)
    dalio_credit = max(0.0, min(5.0, dalio_credit))

    # --- Dalio: M2_Money_Supply_YoY (threshold: < 0.02 → tightening) ---
    # Proxy: Rising rates + yield curve inversion → tightening monetary environment
    m2_proxy = 0.05 + yield_spread * 0.02 - max(0.0, tnx - 3.0) * 0.005
    m2_proxy = max(-0.05, min(0.15, m2_proxy))

    # --- Marks: VIX_Index (threshold: > 35) ---
    # Direct mapping

    # --- Marks: OAS_Spread_Zscore (threshold: > 2.0) ---
    # VIX Z-Score + credit Z-Score composite → extreme credit market tightening
    vix_zscore_proxy = (vix - 20.0) / 8.0  # VIX long-term mean ~20, std ~8
    marks_oas = (vix_zscore_proxy * 0.6 + max(0.0, -hyg_ief_zscore) * 0.4) * 1.5
    marks_oas = max(-2.0, min(5.0, marks_oas))

    # --- Simons: Price_Mean_Reversion_Zscore (threshold: < -3.0) ---
    # Directly uses pre-computed MA20 Z-Score (extreme negative = buying opportunity)
    simons_mr_z = sp500_ma20_zscore

    # --- Simons: Order_Book_Imbalance_Ratio (threshold: > 0.8) ---
    # VIX spike = proxy for order book imbalance (supply-demand gap)
    vix_spike = max(0.0, vix_5d_roc) * 2.0
    simons_obi = 0.5 + vix_spike * 0.3
    simons_obi = max(0.0, min(1.0, simons_obi))

    # ── ④ RISK NODES (Taleb, Shannon, Thorp) ──

    # --- Taleb: SKEW_Index (threshold: > 140) ---
    # VIX level + VIX acceleration → tail risk proxy (fat tail)
    vix_acceleration = max(0.0, vix_5d_roc)
    taleb_skew = 100.0 + vix * 1.0 + vix_acceleration * 15.0
    taleb_skew = max(90.0, min(200.0, taleb_skew))

    # --- Taleb: OTM_Put_Option_Volume_Spike (threshold: > 3.0 σ) ---
    # VIX absolute level + VIX spike → OTM put demand surge
    taleb_put_spike = max(0.0, (vix - 25.0) / 5.0) + max(0.0, vix_5d_roc * 3.0)
    taleb_put_spike = max(0.0, min(8.0, taleb_put_spike))

    # --- Shannon: Price_Trend_R_Squared (threshold: < 0.3) ---
    # Pre-computed 60-day rolling R² (lower = noise-dominant)
    shannon_r_sq = sp500_r_sq

    # --- Shannon: Intraday_Volatility_vs_Daily_Return (threshold: > 2.5) ---
    # Realized volatility / annualized return = signal weakness relative to noise
    abs_annual_ret = max(abs(sp500_ret_252d), 0.01)
    shannon_noise = sp500_vol_20d / abs_annual_ret * 0.5
    shannon_noise = max(0.5, min(10.0, shannon_noise))

    # --- Thorp: Expected_Value_of_Signal (threshold: > 0.0) ---
    # Ensemble expected return proxy: daily return × scale + volatility adjustment
    thorp_ev = sp500_ret * 10.0 + max(0.0, 0.5 - sp500_vol_20d) * 0.5
    thorp_ev = max(-1.0, min(2.0, thorp_ev))

    # --- Thorp: Historical_Win_Rate_of_Signal (threshold: > 0.5) ---
    # Long-term trend direction + base win rate
    thorp_wr = 0.52 + sp500_ret_252d * 0.1
    thorp_wr = max(0.35, min(0.75, thorp_wr))

    # ════════════════════════════════════════════════════════
    # Final metrics dictionary assembly
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

    # Engine internal regime
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
# 4. Historical Simulator — Daily engine simulation
# ══════════════════════════════════════════════════════════

@dataclass
class SimulationRecord:
    """Record for a single simulation timestep."""
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
    """Historical simulation engine based on macro data.

    Runs the GraphOrchestrator daily and records the signal
    and forward returns at each timestep.

    Args:
        orchestrator: Initialized GraphOrchestrator instance
        forward_horizons: Forward return evaluation horizons (in trading days)
    """

    # Default horizons (trading days)
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
        """Run simulation over the entire period.

        Args:
            df: Macro DataFrame with label_event_regime() applied

        Returns:
            List of simulation records
        """
        self.records = []
        dates = df.index.tolist()
        n_total = len(dates)

        logger.info("시뮬레이션 시작: %d 영업일", n_total)

        # S&P500 return series (for forward return computation)
        sp500_col = "SP500" if "SP500" in df.columns else None

        for i, dt in enumerate(dates):
            row = df.loc[dt]

            # Build payload
            payload = build_payload(row)

            # Run engine
            try:
                output: OrchestratorOutput = self.orchestrator.resolve_signals(
                    market_data=payload
                )
            except Exception as e:
                logger.warning("시뮬레이션 %s: 엔진 실행 실패 → %s", dt, e)
                continue

            # Record per-node scores
            node_scores = {
                nid: nr.normalized_score
                for nid, nr in output.node_results.items()
            }
            node_flags = {
                nid: (nr.active_state_flag or "")
                for nid, nr in output.node_results.items()
            }

            # Compute forward returns (based on SP500)
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

            # Progress logging (every 10%)
            if (i + 1) % max(1, n_total // 10) == 0:
                logger.info(
                    "  시뮬레이션 진행: %d / %d (%.0f%%)",
                    i + 1, n_total, (i + 1) / n_total * 100,
                )

        logger.info("시뮬레이션 완료: %d 레코드 생성", len(self.records))
        return self.records


# ══════════════════════════════════════════════════════════
# 5. Attribution Tracker — Multi-horizon node attribution tracking
# ══════════════════════════════════════════════════════════

# Node → optimal horizon mapping (based on investment philosophy)
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

# Node → investment type classification
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

# Nodes subject to Taleb Protection (frozen during normal times)
TALEB_PROTECTION_NODES: Set[str] = {"RSK_TALEB_001", "MAC_MARKS_001"}

# Crisis regimes (the only regimes where Taleb/Marks are optimized)
CRISIS_REGIMES: Set[str] = {"Black_Swan", "Systemic_Risk"}


@dataclass
class HorizonAttribution:
    """Per-horizon attribution statistics for a single node."""
    node_id: str
    horizon: int
    signal_sum: float = 0.0
    correct_signals: int = 0
    total_signals: int = 0
    cumulative_attribution: float = 0.0  # cumulative signal × forward_return

    @property
    def win_rate(self) -> float:
        """Win rate for this horizon."""
        return self.correct_signals / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def avg_signal_strength(self) -> float:
        """Average signal strength."""
        return self.signal_sum / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def horizon_error(self) -> float:
        """Horizon error (1 - win_rate based)."""
        return 1.0 - self.win_rate


class AttributionTracker:
    """Multi-horizon node attribution tracker.

    Tracks the correlation between each node's signal and forward returns
    across T+5, T+20, T+60 horizons, serving as input for the philosophy
    inertia optimizer.

    Attributes:
        rolling_window: Rolling statistics window size (trading days)
        horizons: List of evaluation horizons
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
        # Recent N records (rolling window)
        self._recent_records: List[SimulationRecord] = []

    def process_records(
        self,
        records: List[SimulationRecord],
        signal_threshold: float = 0.5,
    ) -> Dict[str, Dict[int, HorizonAttribution]]:
        """Process simulation records and compute per-node attribution.

        Signals at or above signal_threshold are treated as 'buy signals';
        a positive forward return at the corresponding horizon counts as a 'hit'.

        Args:
            records: List of simulation records
            signal_threshold: Buy signal threshold (default 0.5)

        Returns:
            node_id → horizon → HorizonAttribution mapping
        """
        logger.info("기여도 분석 시작: %d 레코드", len(records))

        for rec in records:
            # Maintain rolling window
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

                    # Check agreement between signal direction and return direction
                    is_bullish = score >= signal_threshold
                    is_positive = fwd_ret > 0

                    if is_bullish == is_positive:
                        attr.correct_signals += 1

                    # Attribution: (signal - 0.5) × forward_return
                    # → attribution = 0 when signal is neutral (0.5)
                    attr.cumulative_attribution += (score - 0.5) * fwd_ret

        # Summary logging
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
        """Compute attribution restricted to a specific regime.

        Used when applying differential per-regime weighting to prevent
        overfitting.

        Args:
            records: Full simulation records
            regime: event_regime to filter by

        Returns:
            Regime-restricted attribution mapping
        """
        filtered = [r for r in records if r.event_regime == regime]
        tracker = AttributionTracker(
            horizons=self.horizons,
            rolling_window=self.rolling_window,
        )
        return tracker.process_records(filtered)


# ══════════════════════════════════════════════════════════
# 6. Philosophy Inertia Optimizer — Philosophy-preserving optimization
#    v2.1: Taleb Protection + Early Stopping + Regime Loss Weighting
# ══════════════════════════════════════════════════════════

@dataclass
class OptimizableParam:
    """Optimizable parameter."""
    node_id: str
    param_name: str          # e.g., "historical_win_rate", "decay_factor"
    original_value: float    # Original JSON value
    current_value: float     # Current value
    lower_bound: float       # original × 0.8
    upper_bound: float       # original × 1.2


class PhilosophyInertiaOptimizer:
    """Optimizes parameters while preserving investment philosophy inertia.

    v2.1 changes:
    - Taleb Protection: Conditional 10x inertia penalty for RSK_TALEB_001, MAC_MARKS_001
    - Early Stopping: Early termination after min_epochs grace when no improvement for patience epochs
    - Learning Rate: Fixed at 0.005
    - Regime Loss Weighting: Black_Swan 3.0x

    Loss = Σ(αₕ × Errorₕ) + λ_eff × Inertia_Penalty

    where λ_eff is:
    - Normal nodes: λ (default 0.3)
    - Taleb/Marks (normal times): λ × 10.0 (effectively frozen)
    - Taleb/Marks (during crisis): λ (normal optimization)

    Args:
        configs: List of MasterEngineConfig (original JSON)
        learning_rate: Learning rate (default 0.005)
        inertia_lambda: Philosophy inertia penalty coefficient (default 0.3)
        bound_pct: Allowed parameter variation range (default 0.20 = ±20%)
        max_iterations: Maximum iteration count (default 50)
        min_epochs: Early Stopping grace period (default 20)
        es_patience: Early stopping patience count (default 3)
    """

    # Per-node-type horizon weights (α₅, α₂₀, α₆₀)
    HORIZON_ALPHAS: Dict[str, Tuple[float, float, float]] = {
        "Value":  (0.10, 0.20, 0.70),  # T+60 focused
        "Growth": (0.15, 0.60, 0.25),  # T+20 focused
        "Macro":  (0.20, 0.55, 0.25),  # T+20 focused
        "Quant":  (0.70, 0.20, 0.10),  # T+5 focused
        "Risk":   (0.40, 0.35, 0.25),  # Balanced
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

        # Extract optimizable parameters
        self._params: List[OptimizableParam] = []
        self._extract_params(configs)

    def _extract_params(self, configs: List[MasterEngineConfig]) -> None:
        """Extract optimizable parameters from original JSON configs.

        Targets: historical_win_rate, decay_factor (per node)
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
        """Optimize parameters based on multi-horizon attribution.

        v2.1 key changes:
        1. Taleb Protection — Frozen with 10x inertia outside CRISIS regimes
        2. Early Stopping — Early termination after min_epochs grace with patience-based check
        3. Regime-Weighted Loss — Black_Swan weighted at 3.0x

        Args:
            attributions: Per-node per-horizon attribution (full period)
            records: Simulation records (for regime weighting)
            regime_attributions: Per-regime attribution
                {regime_label: {node_id: {horizon: HorizonAttribution}}}

        Returns:
            List of optimized parameters
        """
        logger.info(
            "파라미터 최적화 시작 (lr=%.4f, λ=%.2f, bound=±%.0f%%, "
            "max_iter=%d, min_epochs=%d, patience=%d)",
            self.learning_rate, self.inertia_lambda,
            self.bound_pct * 100, self.max_iterations,
            self.min_epochs, self.es_patience,
        )

        # Regime weights
        regime_weights = self._compute_regime_weights(records)

        # ── Taleb Protection: prepare crisis-regime-only attribution ──
        crisis_attributions = self._build_crisis_attributions(regime_attributions)

        # ── Early Stopping state ──
        loss_history: List[float] = []
        no_improve_count: int = 0

        for iteration in range(self.max_iterations):
            total_loss = 0.0

            for param in self._params:
                nid = param.node_id
                node_type = NODE_TYPE.get(nid, "Risk")
                alphas = self.HORIZON_ALPHAS.get(node_type, (0.33, 0.34, 0.33))

                # ── Taleb Protection: conditional inertia determination ──
                if nid in TALEB_PROTECTION_NODES:
                    # Use crisis-only attribution + 10x inertia penalty
                    effective_inertia = self.inertia_lambda * 10.0
                    node_attrs = crisis_attributions if crisis_attributions else attributions
                else:
                    effective_inertia = self.inertia_lambda
                    node_attrs = attributions

                # ── Multi-horizon loss computation ──
                horizon_errors: Dict[int, float] = {}
                for h_idx, h in enumerate((5, 20, 60)):
                    attr = node_attrs.get(nid, {}).get(h)
                    if attr and attr.total_signals > 0:
                        horizon_errors[h] = attr.horizon_error
                    else:
                        horizon_errors[h] = 0.5  # Insufficient data → neutral

                # Weighted Loss: Σ αᵢ × Errorᵢ
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

                # ── Gradient estimation (sign-based) ──
                primary_h = NODE_PRIMARY_HORIZON.get(nid, 20)
                primary_attr = node_attrs.get(nid, {}).get(primary_h)

                if primary_attr and primary_attr.total_signals > 10:
                    gradient_direction = primary_attr.win_rate - 0.5
                    # Regime-weighted adjustment
                    dominant_regime = _get_dominant_regime(records, nid)
                    regime_boost = regime_weights.get(dominant_regime, 1.0)

                    # Inertia pull: exerts opposing force as parameter deviates from original
                    inertia_pull = 0.0
                    deviation = (param.current_value - param.original_value) / max(abs(param.original_value), 1e-8)
                    if abs(deviation) > 1e-4:
                        inertia_pull = -effective_inertia * (1.0 if deviation > 0 else -1.0)

                    step = (
                        self.learning_rate
                        * (gradient_direction * regime_boost + inertia_pull)
                    )
                else:
                    step = 0.0  # Insufficient data → no update

                # ── Parameter update + clipping ──
                new_value = param.current_value + step * param.original_value
                param.current_value = float(np.clip(
                    new_value, param.lower_bound, param.upper_bound
                ))

            # ── Early Stopping check ──
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

        # ── Final result logging ──
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
        """Merge attribution from crisis regimes (Black_Swan + Systemic_Risk).

        Used for loss computation of Taleb/Marks nodes. By excluding
        non-crisis regime data, this prevents low win rates during normal
        times from distorting parameters.

        Args:
            regime_attributions: Per-regime attribution dictionary

        Returns:
            Merged crisis regime attribution (node_id → horizon → HorizonAttribution).
            Returns empty dict if insufficient data.
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
        """Compute per-regime weights.

        Assigns higher weights to extreme regimes such as Black Swan (3.0x)
        and Systemic Risk (2.0x).

        Args:
            records: Simulation records

        Returns:
            regime → weight mapping
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
        """Current list of optimized parameters."""
        return self._params


def _get_dominant_regime(
    records: List[SimulationRecord], node_id: str
) -> str:
    """Return the regime in which the node is most active.

    Args:
        records: Simulation records
        node_id: Node ID

    Returns:
        event_regime with the highest frequency
    """
    regime_counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        score = rec.node_scores.get(node_id, 0.0)
        if score > 0.6:  # Only count meaningful signals
            regime_counts[rec.event_regime] += 1

    if not regime_counts:
        return "Normal"
    return max(regime_counts, key=regime_counts.get)  # type: ignore[arg-type]


def _get_best_fit_regime(
    node_id: str,
    regime_attributions: Dict[str, Dict[str, Dict[int, HorizonAttribution]]],
) -> Tuple[str, float]:
    """Return the node's best-fit regime (highest win rate regime).

    Compares the primary horizon win rate across all regimes and returns
    the regime with the highest win rate along with that rate.

    Args:
        node_id: Node ID
        regime_attributions: Per-regime attribution

    Returns:
        (best_regime, best_win_rate) tuple
    """
    primary_h = NODE_PRIMARY_HORIZON.get(node_id, 20)
    best_regime = "N/A"
    best_wr = 0.0

    for regime_label, reg_attr in regime_attributions.items():
        node_h_attrs = reg_attr.get(node_id, {})
        attr = node_h_attrs.get(primary_h)
        if attr and attr.total_signals > 5:  # Minimum sample requirement
            if attr.win_rate > best_wr:
                best_wr = attr.win_rate
                best_regime = regime_label

    return best_regime, best_wr


# ══════════════════════════════════════════════════════════
# 7. Persistence Layer — Save optimization results
# ══════════════════════════════════════════════════════════

def save_optimized_weights(
    params: List[OptimizableParam],
    records: List[SimulationRecord],
    attributions: Dict[str, Dict[int, HorizonAttribution]],
    regime_attributions: Optional[Dict[str, Dict[str, Dict[int, HorizonAttribution]]]] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Save optimization results to optimized_weights.json.

    v2.1 changes:
    - Parameters with |change_pct| < 1.0%: "status": "preserved"
    - Includes Best-Fit Regime per node
    - Updated optimizer configuration metadata

    Args:
        params: List of optimized parameters
        records: Simulation records (for metadata)
        attributions: Attribution data
        regime_attributions: Per-regime attribution (for Best-Fit Regime computation)
        optimizer_config: Optimizer configuration dictionary (for metadata)
        output_path: Save path (default: optimized_weights.json in project root)

    Returns:
        Path of the saved file
    """
    if output_path is None:
        output_path = str(_PROJECT_ROOT / "optimized_weights.json")

    # ── Per-node parameter construction ──
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

        # v2.1: preserved/optimized status marking
        if abs(change_pct) < 1.0:
            param_entry["status"] = "preserved"
        else:
            param_entry["status"] = "optimized"

        # Taleb Protection indicator
        if p.node_id in TALEB_PROTECTION_NODES:
            param_entry["protection"] = "taleb_inertia_10x"

        nodes_dict[short_name][p.param_name] = param_entry

    # ── Per-node attribution summary + Best-Fit Regime ──
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

    # ── Global parameters ──
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

    # ── Final structure ──
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
    """Aggregate the number of trading days per regime."""
    dist: Dict[str, int] = defaultdict(int)
    for r in records:
        dist[r.event_regime] += 1
    return dict(dist)


# ══════════════════════════════════════════════════════════
# 8. Orchestrator Factory — JSON loading utility
# ══════════════════════════════════════════════════════════

def load_engine_configs(
    json_dir: Optional[str] = None,
) -> List[MasterEngineConfig]:
    """Load the four domain master JSON files.

    Args:
        json_dir: JSON file directory (default: project root)

    Returns:
        List of MasterEngineConfig
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
# 9. Main Entry Point — Full pipeline execution
# ══════════════════════════════════════════════════════════

def main() -> None:
    """Execute the H-PIOS v8.5 historical optimization pipeline.

    v2.1 pipeline:
    1. Download macro data (yfinance) + 15 derived time series
    2. Event regime labeling
    3. Historical simulation (Synthetic Proxy)
    4. Multi-horizon + per-regime attribution analysis
    5. Philosophy inertia-preserving parameter optimization (Taleb Protection + Early Stopping)
    6. Save results + Best-Fit Regime logging
    """
    logger.info("=" * 60)
    logger.info("H-PIOS v8.5 Historical Back-Optimizer v2.1 시작")
    logger.info("  Synthetic Proxy Breakthrough Edition")
    logger.info("=" * 60)

    # ── Step 1: Data download + derived time series ──
    logger.info("\n[Step 1/6] 매크로 데이터 다운로드 + 파생 시계열 계산")
    df = fetch_macro_data(start="2019-01-01")

    # ── Step 2: Regime labeling ──
    logger.info("\n[Step 2/6] 이벤트 국면 라벨링")
    df = label_event_regime(df)

    # ── Step 3: Engine initialization & simulation ──
    logger.info("\n[Step 3/6] 역사적 시뮬레이션 실행 (Synthetic Proxy)")
    configs = load_engine_configs()
    orchestrator = GraphOrchestrator(configs)
    simulator = HistoricalSimulator(orchestrator)
    records = simulator.run(df)

    if not records:
        logger.error("시뮬레이션 레코드 없음. 종료합니다.")
        return

    # ── Step 4: Attribution analysis (full + per-regime) ──
    logger.info("\n[Step 4/6] 멀티-호라이즌 + 국면별 기여도 분석")
    tracker = AttributionTracker()
    attributions = tracker.process_records(records)

    # v2.1: Compute per-regime attribution (for Taleb Protection + Best-Fit Regime)
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

    # ── Step 5: Parameter optimization ──
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

    # ── Best-Fit Regime logging ──
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

    # ── Step 6: Save results ──
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

    # ── Final summary ──
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
