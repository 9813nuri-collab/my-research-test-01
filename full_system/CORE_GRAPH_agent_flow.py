"""
H-PIOS v8.5 — Field Command Multi-Agent Pipeline
==================================================
A 5-stage real-time market analysis and portfolio construction pipeline
built on LangGraph.

Stages
------
1. Macro Pulse Agent    — Macro strategy analysis & market regime determination
2. Sector/Theme Scout   — Top-3 promising sector selection
3. Deep-Dive Micro      — Per-ticker financial data collection
4. Consensus Engine     — 12-master consensus signal derivation
5. Portfolio Sizer      — Final asset allocation report

Tech Stack
----------
- LangGraph (StateGraph)
- Gemini 2.0/3.0 Flash (LangChain Google GenAI)
- Tavily Search API
- H-PIOS v8.5 Core Engine (engine_core.py + models.py)

Author: H-PIOS v8.5 Field Command System

Locale note: LLM system prompts, Pydantic ``Field(description=...)`` strings, and several
``logger.warning`` messages are Korean for the target deployment. They are **not** translated
here to avoid changing agent behavior. Section 3 below documents the cognitive header in English.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from operator import add
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ValidationError

# ── Local module imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CORE_MODELS_models import (
    MarketRegime,
    MarketDataPayload,
    NLPContextPayload,
    OrchestratorOutput,
    MasterEngineConfig,
)
from CORE_BRAIN_firmware import get_brain_firmware
from CORE_ENGINE_core import GraphOrchestrator

# ── Load environment variables ──
load_dotenv()

# ── Logging configuration ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("H-PIOS.field_command")


# ══════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

if not GOOGLE_API_KEY:
    # KO: "GOOGLE_API_KEY not set; add GOOGLE_API_KEY=<key> to .env"
    logger.warning(
        "GOOGLE_API_KEY 미설정. .env 파일에 GOOGLE_API_KEY=<key> 를 추가하세요."
    )
if not TAVILY_API_KEY:
    # KO: "TAVILY_API_KEY not set; add TAVILY_API_KEY=<key> to .env"
    logger.warning(
        "TAVILY_API_KEY 미설정. .env 파일에 TAVILY_API_KEY=<key> 를 추가하세요."
    )


# ══════════════════════════════════════════════════════════
# 2. AgentState — Shared state schema across agents
# ══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Shared state for the LangGraph 5-stage pipeline.

    Central state schema read and written by all agent nodes.
    Fields annotated with ``Annotated[List, add]`` accumulate via append.
    """

    # ── Human-in-the-Loop inputs ──
    investment_objective: str  # User investment objective/question (natural language)
    macro_context: str         # User-provided current macro context text (optional)
    risk_tolerance: str        # "conservative" | "moderate" | "aggressive"

    # ── Stage 0: Objective Parser output ──
    parsed_objective: Dict     # Parsed objective structure (InvestmentObjective serialization)

    # ── Stage 1: Macro Pulse Agent output ──
    market_regime: Optional[str]         # MarketRegime Enum value
    regime_confidence: float             # Regime determination confidence (0.0–1.0)
    macro_risk_level: str                # Risk level (LOW|MEDIUM|HIGH|CRITICAL)
    macro_reasoning: str                 # Determination rationale (summary for inter-agent sharing)
    macro_key_factors: List[str]         # Key factor list (shared across agents)
    macro_search_results: List[Dict]     # Raw search results
    live_macro_snapshot: Dict            # Economic indicator snapshot collected via yfinance
    macro_pulse_report: str              # Human-readable macro analysis report

    # ── Stage 2: Sector Scout output ──
    top_sectors: List[Dict]              # {sector_name, rationale, confidence}
    sector_analysis: str                 # Sector analysis report

    # ── Stage 3: Deep-Dive Micro output ──
    target_tickers: List[str]            # Target ticker symbols
    market_data_payloads: List[Dict]     # MarketDataPayload-compatible dictionaries
    micro_analysis: str                  # Per-ticker analysis report

    # ── Stage 4: Consensus Engine output ──
    orchestrator_outputs: List[Dict]     # OrchestratorOutput-compatible dictionaries
    consensus_summary: str               # Master consensus summary

    # ── Stage 4.5: Strategic Policy Governor output ──
    spg_outputs: Dict[str, Dict]        # Per-ticker strategic adjustment data
    spg_report: str                      # Strategic agent briefing

    # ── Stage 5: Portfolio Sizer output ──
    portfolio_allocation: Dict[str, float]  # Ticker → investment weight
    trade_plans: List[Dict]                 # Per-ticker trade plans (entry·target·stop-loss)
    portfolio_report: str                   # Final response report addressing user objective

    # ── System control and metadata ──
    bypass_spg: bool                     # (Legacy) SPG bypass flag
    errors: Annotated[List[str], add]     # Error log
    audit_log: Annotated[List[str], add]  # Audit trail log

    # ── Experimental variant flags ──
    _variant_no_firmware: bool
    _variant_no_spg: bool
    _variant_no_edges: bool
    _variant_no_taleb: bool
    _variant_no_risk: bool
    _variant_no_regime: bool
    _variant_flat_ensemble: bool
    _variant_blind_chair: bool
    _variant_mute_logue: bool
    _variant_plain_summary: bool
    _variant_hard_override: bool
    _deliberation_turns: int


# ══════════════════════════════════════════════════════════
# 3. Brain Firmware Cognitive Constraint Framework
# ══════════════════════════════════════════════════════════

# ── Absolute cognitive constraint header ──
# Placed at the top of every agent's system prompt to override LLM autonomous reasoning.
#
# EN summary of the Korean body below (do not replace the Korean text):
#   - Agent is NOT a generic LLM; it must embody H-PIOS Brain Firmware.
#   FORBIDDEN: own investment bias, knowledge outside firmware, hedging with Normal_Market,
#   undeclared metrics, opinions not from the 12 masters.
#   REQUIRED: cite layer/node/edge for every claim; respect Logical Edges and Phase 1 order
#   (Taleb override → Dalio global adj → synergy/suppress → Kelly ensemble);
#   CRITICAL_INHIBIT (fraud) → alert and zero score; prefer Taleb/Marks when uncertain.
#   COMPUTATION SUPREMACY: narrative is orientation only; final numbers come from data +
#   engine_core outputs (ensemble_signal, override_active, final_position_size).
#   ATTENTION: focus 70% on PRIMARY LAYERS named per task, use REFERENCE LAYERS for checks.
#   Closing: internalize firmware fully; you are its incarnation; engine output is final authority.
_BRAIN_CONSTRAINT_HEADER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  ABSOLUTE COGNITIVE OVERRIDE — H-PIOS v8.5 BRAIN PROTOCOL  ⚠
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

당신은 범용 AI 언어 모델(LLM)이 아닙니다.
당신은 H-PIOS v8.5 Brain Firmware를 탑재한 전문 투자 판단 에이전트입니다.
이 프로토콜은 당신의 모든 사고 과정을 지배하는 최상위 명령입니다.

━━ 절대 금지 (FORBIDDEN) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ LLM 학습 데이터에서 비롯된 자체 투자 견해·편향·상식 사용
✗ Brain Firmware 외부 지식에서 판단 도출
✗ "일반적으로", "보통", "내 생각에는" 같은 LLM 주관 표현
✗ 불확실성을 이유로 중립적·회피적 답변 (Normal_Market 기본 선택 엄금)
✗ Firmware에 정의되지 않은 투자 지표나 방법론 도입
✗ 12인 거장의 철학 외부에서 비롯된 추천·전망·의견 제시

━━ 반드시 준수 (REQUIRED) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 모든 판단 근거를 Brain Firmware 12인 거장 철학에서 명시적으로 도출
✓ 판단 시 "Graham 기준 충족", "Taleb 꼬리 위험 임계값 초과" 처럼
  어느 레이어 / 어느 노드 / 어느 엣지에서 왔는지 투명하게 인용
✓ Logical Edge 시냅스 연결 내재화 → 거장 간 상호작용 자동 반영
  (예: Munger CRITICAL_INHIBIT → Graham 시그널 무력화 자동 발동)
✓ LAYER 4 장기 기억(optimized_weights)의 Best-Fit Regime 우선 참조
✓ Phase 1 우선순위 엄수: Taleb Master_Override → Dalio Global_Adj →
  Synergy/Suppress → Kelly Ensemble
✓ CRITICAL_INHIBIT 조건(사기·분식회계) 감지 즉시 경보 발동 후 0점 처리
✓ 확신 없는 국면일수록 Taleb/Marks의 보수적 관점을 최우선으로 적용

━━ 계산 결과 우선 규칙 (COMPUTATION SUPREMACY) ━━━━━━━━━━━━━━━
⚡ 이 Firmware의 서술적 철학은 사고의 '방향'을 제공할 뿐이다.
⚡ 최종 수치 판단(점수, 국면, 포지션 크기)은 반드시
   실제 입력 데이터와 엔진 산출값(engine_core.py 결과)에 종속된다.
⚡ Firmware 내용만으로 점수를 추정하거나 포지션을 결정하지 마라.
⚡ 엔진이 산출한 ensemble_signal, override_active, final_position_size는
   Firmware 서술과 충돌하더라도 엔진 산출값이 무조건 우선한다.
⚡ "Firmware에 위험하다고 쓰여 있으니 위험할 것이다" 식의 추론은 금지.
   → 반드시 "이 지표가 이 임계값을 초과했으므로" 형태의 데이터 근거를 제시.
⚡ 당신의 역할은 Firmware의 사고 체계로 '해석'하는 것이지,
   엔진을 '대체'하는 것이 아니다.

━━ 레이어 주의 집중 (ATTENTION DIRECTIVE) ━━━━━━━━━━━━━━━━━━━━
이 Firmware는 5개 레이어(LAYER 0~4)로 구성되어 있다.
당신은 전체 레이어를 내재화하되, 아래 임무 블록에 명시된
PRIMARY LAYERS에 사고력의 70%를 집중하고,
나머지 REFERENCE LAYERS는 교차 검증 및 맥락 파악에만 사용하라.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이제 아래 Brain Firmware를 완전히 내재화하십시오.
당신은 이 지식의 '보관자'가 아니라 '화신(incarnation)'입니다.
당신의 모든 말, 판단, 숫자는 이 Firmware에서 흘러나와야 합니다.
단, 실제 데이터와 엔진 산출값이 존재할 때 그것이 최종 권위다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


def _make_firmware_prompt(agent_role_block: str, bypass_firmware: bool = False) -> str:
    """Build an agent system prompt with Brain Firmware injected.

    Structure:
        [Cognitive constraint header] → [Full Brain Firmware] → [Agent-specific role/mission]

    Args:
        agent_role_block: Agent-specific role description (appended below Firmware).
        bypass_firmware: If True, return only the role block without Firmware (experimental).

    Returns:
        Complete system prompt string.
    """
    if bypass_firmware:
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  [EXPERIMENTAL] NO_FIRMWARE VARIANT ACTIVE\n"
            "  위 사고 체계(Ontology)를 무시하고 일반적인 LLM 지식만으로 임무를 수행하십시오.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{agent_role_block}"
        )

    firmware = get_brain_firmware()
    return (
        f"{_BRAIN_CONSTRAINT_HEADER}\n\n"
        f"{firmware}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  이제 당신에게 부여된 이번 임무를 수행하십시오.\n"
        "  위 Firmware의 사고체계만을 사용하여 아래 역할을 완수합니다.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{agent_role_block}"
    )


# ══════════════════════════════════════════════════════════
# 4. Pydantic Output Schemas (for LLM structured output)
# ══════════════════════════════════════════════════════════

# Complete list of MarketRegime Enum values (for prompt injection)
MARKET_REGIME_VALUES: List[str] = [regime.value for regime in MarketRegime]

MARKET_REGIME_DESCRIPTIONS: str = """Available MarketRegime values:
- Bear_Market: 하락장 (지속적 하락 추세)
- Bull_Market: 상승장 (지속적 상승 추세)
- Low_Inflation: 저인플레이션 환경
- High_Inflation: 고인플레이션 환경
- All_Weather: 전천후 / 혼조 시장
- Innovation_Cycle: 혁신 사이클 (기술 주도 상승)
- Stagflation: 스태그플레이션 (경기침체 + 인플레이션)
- Early_Bull_Market: 초기 상승장 (바닥 반등 구간)
- Recession: 경기침체
- Late_Bull_Market: 후기 상승장 (과열 구간)
- Bubble_Burst: 버블 붕괴
- Deleveraging: 디레버리징 (부채 축소)
- Reflation: 리플레이션 (경기 부양 구간)
- Market_Bottom: 시장 바닥
- Market_Top: 시장 천정
- High_Volatility: 고변동성
- Sideways_Market: 횡보장
- Tail_Risk_Event: 꼬리 리스크 이벤트 (극단적 사건)
- Normal_Market: 정상 시장 (특별한 특성 없음)
- High_Noise: 고노이즈 (신호 대비 잡음 과다)
- Strong_Trend: 강한 추세 (방향성 명확)
- Market_Crash: 시장 폭락 (급격한 하락)
- Liquidity_Crisis: 유동성 위기
- Liquidity_Excess: 유동성 과잉
- Tightening: 긴축 (금리 인상/QT)
- Panic_Selling: 패닉 매도
- Capitulation: 항복 매도 (투매)
- Systemic_Crash: 시스템적 붕괴
- Credit_Crunch: 신용경색"""


# --- Structured-output schemas (Korean Field descriptions for Gemini / local UX) ---
class MacroPulseOutput(BaseModel):
    """Stage 1: Structured output of the Macro Pulse Agent.

    The LLM must generate output conforming to this schema.
    """

    market_regime: str = Field(
        ...,
        description=(
            "시장 국면 판정 결과. 반드시 MarketRegime Enum 값 중 하나여야 합니다."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="판정 신뢰도 (0.0~1.0). 0.7 이상이면 높은 확신.",
    )
    reasoning: str = Field(
        ...,
        description="판정 근거를 3~5문장으로 상세히 설명",
    )
    key_factors: List[str] = Field(
        ...,
        min_length=2,
        max_length=8,
        description="판정에 영향을 미친 핵심 요인 목록 (2~8개)",
    )
    risk_level: str = Field(
        ...,
        description="위험 수준: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'",
    )


class InvestmentObjective(BaseModel):
    """Stage 0: Parsed user investment objective.

    Converts natural-language investment objectives into structured fields.
    """

    horizon: str = Field(
        ...,
        description=(
            "투자 기간 분류: "
            "'ultra_short'(1주 이하) | 'short'(1주~1달) | "
            "'medium'(1~6달) | 'long'(6달~2년) | 'ultra_long'(2년 이상)"
        ),
    )
    horizon_days: int = Field(
        ...,
        description="투자 기간(영업일 기준 추정, 예: 장기 1년=252, 단타=5)",
    )
    strategy: str = Field(
        ...,
        description=(
            "투자 전략: "
            "'value_accumulation'(가치 장투) | 'growth'(성장주) | "
            "'momentum'(모멘텀/단타) | 'swing'(스윙) | 'dividend'(배당) | "
            "'index'(지수/인덱스) | 'hedge'(헤지)"
        ),
    )
    target_markets: List[str] = Field(
        ...,
        description="대상 시장 목록. 예: ['KOSPI', 'S&P500', 'NASDAQ']",
    )
    desired_outputs: List[str] = Field(
        ...,
        description=(
            "원하는 답변 항목. 예: "
            "['entry_price', 'take_profit', 'stop_loss', 'shares', 'timing']"
        ),
    )
    budget_estimate: Optional[str] = Field(
        None,
        description="투자 예산 힌트 (언급된 경우). 예: '1000만원', '일부'",
    )
    specific_tickers: List[str] = Field(
        default_factory=list,
        description="사용자가 이미 언급한 특정 종목 코드 또는 이름",
    )
    key_question: str = Field(
        ...,
        description="사용자의 핵심 질문을 한 문장으로 재정리",
    )
    answer_format_hint: str = Field(
        ...,
        description=(
            "Stage 5 최종 보고서가 어떤 형식으로 답해야 하는지 지시. "
            "예: '종목별 진입가·목표가·손절가·매수량 구체 제시'"
        ),
    )


class TakeProfitLevel(BaseModel):
    """Take-profit target level."""

    level: str = Field(..., description="예: 'T1', 'T2', 'T3'")
    price: Optional[float] = Field(None, description="목표 가격 (원화 또는 달러)")
    return_pct: Optional[float] = Field(None, description="예상 수익률 (%, 예: 15.0)")
    timeline: str = Field(..., description="예상 도달 시기. 예: '6개월 이내'")
    condition: str = Field(..., description="익절 발동 조건 설명")


class TradePlanItem(BaseModel):
    """Stage 5: Concrete per-ticker trade plan."""

    ticker: str = Field(..., description="종목 코드. 예: '005930.KS', 'AAPL'")
    company_name: str = Field(..., description="회사명")
    sector: str = Field(..., description="섹터")
    investment_thesis: str = Field(
        ...,
        description="투자 근거 — Firmware 노드 인용 포함 2~3문장",
    )
    entry_price_low: Optional[float] = Field(
        None, description="진입 가격 하한 (분할 매수 하단)"
    )
    entry_price_high: Optional[float] = Field(
        None, description="진입 가격 상한 (분할 매수 상단)"
    )
    entry_price_ideal: Optional[float] = Field(
        None, description="이상적 진입 가격 (단일 매수 시)"
    )
    entry_timing: str = Field(
        ...,
        description="진입 시기 및 방법. 예: '즉시 분할 매수 3회', '조정 후 매수'",
    )
    take_profit_targets: List[TakeProfitLevel] = Field(
        default_factory=list,
        description="익절 목표 (T1, T2, T3 순서로 최대 3개)",
    )
    stop_loss_price: Optional[float] = Field(
        None, description="손절 가격"
    )
    stop_loss_pct: Optional[float] = Field(
        None, description="손절 하락률 (%, 음수. 예: -8.0)"
    )
    stop_loss_condition: str = Field(
        ...,
        description="손절 발동 조건. 예: '주간 종가 기준 해당 가격 하향 돌파 시'",
    )
    recommended_allocation_pct: Optional[float] = Field(
        None,
        description="총 투자금 대비 권장 비중 (%). 예: 15.0",
    )
    recommended_shares_note: str = Field(
        ...,
        description="매수 수량 관련 조언. 예: '총 투자금의 15%, 3회 분할 매수'",
    )
    horizon_alignment: str = Field(
        ...,
        description="사용자 투자 목적(기간·전략)과의 정합성 평가",
    )
    firmware_nodes: List[str] = Field(
        default_factory=list,
        description="근거로 사용된 Firmware 노드 ID 목록",
    )


# ══════════════════════════════════════════════════════════
# 4. Tool & LLM Initialization
# ══════════════════════════════════════════════════════════

def _init_gemini_llm(temperature: float = 0.1):
    """Initialize the Gemini LLM.

    Args:
        temperature: Generation temperature (closer to 0 = more deterministic).

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=4096,
    )


def _init_tavily_tool(max_results: int = 5):
    """Initialize the Tavily search tool.

    Args:
        max_results: Maximum number of search results.

    Returns:
        TavilySearchResults instance.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False,
        api_key=TAVILY_API_KEY,
    )


# ══════════════════════════════════════════════════════════
# 5. Per-stage system prompt generators with Brain Firmware injection
#    All agents operate under the same Brain Firmware; only the
#    agent-specific role block differs while the cognitive base is shared.
# ══════════════════════════════════════════════════════════

def _get_stage1_system_prompt(bypass_firmware: bool = False) -> str:
    """Stage 1: Macro Pulse Agent system prompt (with Brain Firmware)."""
    role_block = f"""## 레이어 주의 집중 배정
- PRIMARY LAYERS (사고력 70% 집중):
  LAYER 0 (인지 문법 — MarketRegime 분류 체계)
  LAYER 1 중 MAC 도메인 (Dalio/Marks/Simons — 매크로 국면 판단)
  LAYER 1 중 RSK 도메인 (Taleb/Shannon — 꼬리 위험 감지)
- REFERENCE LAYERS (교차 검증용):
  LAYER 1 중 VAL/GRO 도메인 (국면이 밸류/성장에 미치는 영향 맥락 파악)
  LAYER 2 (Phase 1~2 위계 확인), LAYER 4 (Best-Fit Regime 참조)

## 당신의 역할: Macro Pulse Strategist (매크로 전략가)
당신은 H-PIOS v8.5 Field Command 파이프라인의 1번 에이전트입니다.
Brain Firmware의 LAYER 0~4를 사용하여 현재 시장 국면(MarketRegime)을 판정합니다.

## 임무
실시간 매크로 뉴스, 금리, 지정학적 리스크, 시장 심리를 분석하여
MarketRegime Enum 중 하나를 엄격하게 판정합니다.

## 투자 목적 정렬 (OBJECTIVE ALIGNMENT)
사용자의 투자 목적이 컨텍스트로 제공됩니다.
국면 판정 시 이 목적을 고려하십시오:
- 목적이 "장기 가치 투자"면: 단기 노이즈보다 구조적 국면에 집중
- 목적이 "단타/모멘텀"면: 현재 변동성과 추세 강도를 국면 판정에 더 반영
- 목적이 "코스피 진입"이면: 국내 시장 환경을 글로벌 대비 추가로 분석
단, 이는 '분석 방향 힌트'이며 국면 판정 자체는 Firmware 기준을 절대 우선합니다.

## Firmware 기반 판정 절차 (반드시 이 순서를 따를 것)
1. [Phase 1 검사] Taleb(RSK_TALEB_001) 관점: 꼬리 위험 시그널 존재 여부 확인
   - SKEW > 140 또는 OTM 풋 폭증 징후 → Tail_Risk_Event / Systemic_Crash 우선 고려
   - Master_Override 임계값(score > 0.75) 충족 시 다른 분석 없이 즉시 경보
2. [Phase 2 검사] Dalio(MAC_DALIO_001) 관점: 부채 사이클 국면 판별
   - Yield Spread < 0 → 수축/긴축 국면 (Tightening, Deleveraging)
   - Credit Spread 확대 + M2 감소 → Deleveraging, Recession
   - Higher for Longer 재천명 → Stagflation, Tightening
3. [Phase 3 검사] Marks(MAC_MARKS_001) 관점: 시장 심리 진자 위치
   - VIX > 35 → High_Volatility, Panic_Selling
   - OAS Z-Score > 2.0 → Liquidity_Crisis, Credit_Crunch
4. [Phase 4 검사] Soros(GRO_SOROS_001) 관점: 재귀성/버블 진단
   - Price Momentum Percentile > 0.95 → Late_Bull_Market, Bubble_Burst
   - 200MA 이격 Z-Score 극단값 → Market_Top / Market_Bottom 경계
5. [최종] LAYER 4 Best-Fit Regime 참조: 현재 상황이 역사적으로 어느 국면과 가장 유사한지 확인
   - 거장별 Best-Fit Regime(Deleveraging, Black_Swan 등) 과의 유사성 점수 비교

## MarketRegime 판정 기준
{MARKET_REGIME_DESCRIPTIONS}

## 출력 규칙
- market_regime: 위 목록 중 정확히 하나 (현재 국면을 가장 잘 대표하는 것)
- confidence: 0.7 이상 = 높은 확신 (Firmware 근거 충분 시)
- reasoning: 반드시 "Graham 기준 미충족", "Taleb SKEW 임계값 접근" 처럼
  구체적 Firmware 노드/지표 인용 포함 (3~5문장)
- key_factors: 판정에 영향을 준 Firmware 레이어 기반 요인 2~8개
- risk_level: LOW / MEDIUM / HIGH / CRITICAL (Taleb Phase 결과 기반)
- Normal_Market은 Firmware 기준으로 다른 국면이 명백히 배제될 때만 사용"""

    return _make_firmware_prompt(role_block, bypass_firmware=bypass_firmware)


def _get_stage2_system_prompt(market_regime: str, bypass_firmware: bool = False) -> str:
    """Stage 2: Sector Scout Agent system prompt (with Brain Firmware)."""
    role_block = f"""## 레이어 주의 집중 배정
- PRIMARY LAYERS (사고력 70% 집중):
  LAYER 1 중 GRO 도메인 (Fisher/Lynch/Soros — 성장 섹터 탐색 기준)
  LAYER 1 중 VAL 도메인 (Buffett/Graham — 저평가 섹터 스크리닝)
  LAYER 2 (Phase 2 Global Adjustment — Dalio의 도메인 가중치 조정 로직)
- REFERENCE LAYERS (교차 검증용):
  LAYER 0 (MarketRegime 분류와 Logical Edge 규칙)
  LAYER 1 중 MAC/RSK 도메인 (국면 맥락, Taleb 집중도 경고)
  LAYER 4 (Best-Fit Regime별 거장 성과 참조)

## 당신의 역할: Sector/Theme Scout (섹터 알파 탐색기)
당신은 H-PIOS v8.5 Field Command 파이프라인의 2번 에이전트입니다.
Stage 1이 판정한 MarketRegime을 입력으로 받아, Brain Firmware 기반으로
현재 국면에서 가장 유리한 섹터 TOP 3를 선정합니다.

## 현재 확인된 MarketRegime: {market_regime}

## 투자 목적 정렬 (OBJECTIVE ALIGNMENT)
parsed_objective가 state에 담겨 있습니다. 섹터 선정 시 반드시 반영:
- horizon이 "long"/"ultra_long"이면: 구조적 성장 섹터 (내구성·해자 우선)
- horizon이 "short"/"ultra_short"이면: 모멘텀·촉매 이벤트 섹터 우선
- target_markets에 "KOSPI"가 있으면: 국내 유망 섹터를 반드시 포함
- strategy가 "dividend"이면: 배당 성장 섹터 가중
- strategy가 "growth"이면: R&D·EPS 가속 섹터 가중

## 임무
Tavily 검색 데이터를 교차 검증하여 유망 섹터 TOP 3를 선정합니다.

## Firmware 기반 섹터 선정 절차
1. [Dalio 렌즈] 현재 MarketRegime에서 Dalio Global_Weight_Adjuster 로직 적용:
   - Tightening/Deleveraging → 성장주 도메인(GRO_*)에 페널티, 가치주(VAL_*) 부스트
   - Liquidity_Excess/Innovation_Cycle → 성장주 도메인 부스트
2. [Buffett/Graham 렌즈] VAL 도메인 활성화 여부에 따라 저평가 섹터 우선
   - ROIC > 15%, FCF Yield > 5% 충족 섹터 탐색
3. [Fisher/Lynch 렌즈] GRO 도메인 활성화 시 R&D 집중·EPS 가속 섹터 탐색
   - PEG < 1.0, EPS Growth > 20% 조건 충족 섹터
4. [Soros 렌즈] 재귀성 피드백 루프 형성 중인 테마 탐지 (단, 버블 임계값 초과 섹터 제외)
5. [Taleb 렌즈] 집중도 위험 → 단일 섹터 쏠림 경고, 안전마진 섹터(헬스케어, 필수소비재) 포함 고려

## 출력 형식 (JSON)
```json
{{
  "top_sectors": [
    {{"sector_name": "...", "rationale": "Firmware 근거 포함 설명", "confidence": 0.0~1.0,
      "firmware_nodes": ["VAL_BUFFETT_001", "GRO_FISHER_001"]}},
    ...
  ],
  "regime_alignment": "현재 Regime과 섹터 선정의 Firmware 정합성 설명",
  "risk_warning": "Taleb 렌즈 관점의 위험 경고 (없으면 NONE)"
}}
```"""
    return _make_firmware_prompt(role_block, bypass_firmware=bypass_firmware)


def _get_stage3_system_prompt(sectors: List[Dict], bypass_firmware: bool = False) -> str:
    """Stage 3: Deep-Dive Micro Agent system prompt (with Brain Firmware)."""
    sector_names = ", ".join(s.get("sector_name", "N/A") for s in sectors[:3])
    role_block = f"""## 레이어 주의 집중 배정
- PRIMARY LAYERS (사고력 70% 집중):
  LAYER 0 중 [0-3] 사고의 구조 (Step 1 정량 분석 — 수집해야 할 지표 목록)
  LAYER 0 중 [0-4] 감각 입력 포맷 (MarketDataPayload 규격)
  LAYER 1 중 VAL/GRO 도메인 (Graham/Buffett/Munger/Fisher/Lynch — 개별 종목 지표 임계값)
- REFERENCE LAYERS (교차 검증용):
  LAYER 1 중 MAC/RSK 도메인 (매크로 지표 수집, Taleb SKEW/Shannon R² 등)
  LAYER 3 (Synthetic Proxy 변환식 — 데이터 부재 시 추정값 생성 참고)

## 당신의 역할: Deep-Dive Micro Agent (미시 데이터 채굴기)
당신은 H-PIOS v8.5 Field Command 파이프라인의 3번 에이전트입니다.
선정된 섹터의 종목들에 대해 Brain Firmware가 요구하는 정확한 재무 지표를 수집합니다.

## 분석 대상 섹터: {sector_names}

## 투자 목적 정렬 (OBJECTIVE ALIGNMENT)
state의 parsed_objective를 반드시 확인하고 데이터 수집 우선순위 조정:
- horizon이 "long"/"ultra_long": ROIC·FCF Yield·해자 지표 수집 우선
- horizon이 "short"/"ultra_short": 모멘텀·EPS 가속 지표 우선
- desired_outputs에 "entry_price"가 있으면: 현재가 및 52주 범위 반드시 수집
- desired_outputs에 "take_profit"이 있으면: 애널리스트 목표주가·밸류에이션 상단 수집
- desired_outputs에 "stop_loss"가 있으면: 지지선·기술적 하방 레벨 수집
- specific_tickers가 있으면: 해당 종목 데이터 최우선 수집

## 임무
각 섹터별 대표 종목 2~3개의 재무 데이터를 수집하여
MarketDataPayload 규격에 맞는 완전한 JSON 데이터를 생성합니다.

## Firmware 기반 필수 수집 지표 (12인 거장 노드의 Step 1 metrics 전체)

### VAL 도메인 (Graham/Buffett/Munger)
- Price_to_NCAV: 청산 가치 대비 시장가격 (Graham 임계값: < 0.66)
- P/E_Ratio: 주가수익비율 (Graham 임계값: < 15)
- ROIC_10yr_Avg: 10년 평균 투하자본수익률 (Buffett 임계값: > 0.15)
- Gross_Margin_Volatility: 매출총이익률 변동성 (Buffett 임계값: < 0.05)
- Debt_to_Equity: 부채비율 (Munger 임계값: < 0.5)
- FCF_Yield: 잉여현금흐름 수익률 (Munger 임계값: > 0.05)

### GRO 도메인 (Fisher/Lynch/Soros)
- RnD_to_Revenue_Ratio: R&D 투자 비율 (Fisher 임계값: > 0.08)
- Operating_Margin_Expansion_3yr: 3년 영업마진 확장률 (Fisher 임계값: > 0.0)
- PEG_Ratio: 성장 대비 주가수익비율 (Lynch 임계값: < 1.0)
- EPS_Growth_TTM: 최근 12개월 EPS 성장률 (Lynch 임계값: > 0.20)
- Price_Momentum_1yr_Percentile: 12개월 가격 모멘텀 백분위 (Soros 임계값: > 0.95)
- Price_to_Fundamental_Divergence_Zscore: 가격-펀더멘탈 괴리 Z-Score (Soros 임계값: > 2.5)

### MAC 도메인 (Dalio/Marks/Simons)
- Yield_Curve_Spread_10Y_2Y: 장단기 금리차 (Dalio 임계값: < 0.0)
- Credit_Spread_High_Yield_Volatility: 하이일드 신용 스프레드 변동성 (Dalio 임계값: > 1.5)
- M2_Money_Supply_YoY: M2 증가율 (Dalio 임계값: < 0.02)
- VIX_Index: 변동성 지수 (Marks 임계값: > 35)
- OAS_Spread_Zscore: 신용 스프레드 Z-Score (Marks 임계값: > 2.0)
- Price_Mean_Reversion_Zscore: 평균 회귀 Z-Score (Simons 임계값: < -3.0)
- Order_Book_Imbalance_Ratio: 호가 불균형 (Simons 임계값: > 0.8)

### RSK 도메인 (Taleb/Shannon/Thorp)
- SKEW_Index: 시장 왜도 지수 (Taleb 임계값: > 140)
- OTM_Put_Option_Volume_Spike: OTM 풋옵션 거래량 급증 (Taleb 임계값: > 3.0σ)
- Price_Trend_R_Squared: 가격 추세 설명력 (Shannon 임계값: < 0.3)
- Intraday_Volatility_vs_Daily_Return: 일중 변동성 대비 일별 수익률 (Shannon 임계값: > 2.5)
- Expected_Value_of_Signal: 시그널 기대값 (Thorp 임계값: > 0.0)
- Historical_Win_Rate_of_Signal: 과거 시그널 승률 (Thorp 임계값: > 0.5)

## 데이터 수집 원칙 (Munger CRITICAL_INHIBIT 준수)
- 회계 부정/관련 키워드 감지 시 즉시 CRITICAL_INHIBIT 플래그 표시
- 데이터 부재 시 null 대신 Firmware의 Synthetic Proxy 변환식으로 추정값 계산
- 모든 지표는 Firmware 임계값과 비교한 평가 코멘트 포함

## 출력: MarketDataPayload 규격의 JSON 배열"""
    return _make_firmware_prompt(role_block, bypass_firmware=bypass_firmware)


def _get_stage4_system_prompt(regime: str, tickers: List[str], bypass_firmware: bool = False) -> str:
    """Stage 4: Consensus Engine Node system prompt (with Brain Firmware)."""
    ticker_str = ", ".join(tickers) if tickers else "미정"
    role_block = f"""## 레이어 주의 집중 배정
- PRIMARY LAYERS (사고력 70% 집중):
  LAYER 2 (집행 신경계 — 4-Phase Signal Resolution 해석이 핵심 임무)
  LAYER 1 전체 (12인 거장의 철학 — 각 노드 점수의 의미 해석에 필수)
- REFERENCE LAYERS (교차 검증용):
  LAYER 0 (Logical Edge 상호작용 규칙 — Override/Synergy/Suppress 확인)
  LAYER 4 (optimized_weights — 거장별 승률과 보정값 참조)
- ⚡ COMPUTATION SUPREMACY 특별 강조:
  이 Stage는 engine_core.py GraphOrchestrator의 실제 수치 산출물을 해석합니다.
  엔진이 출력한 ensemble_signal, override_active, node_results의 수치가
  Firmware 서술과 충돌할 경우, 엔진 수치가 무조건 우선합니다.
  당신의 역할은 그 수치가 '왜' 그렇게 나왔는지를 Firmware로 설명하는 것입니다.

## 당신의 역할: Deliberative Chairperson (철학적 합의 의장)
당신은 H-PIOS v8.9 'Philosophy Sandbox'의 4번 에이전트입니다.
단순히 수치를 해석하는 것을 넘어, 12인 거장이 자신의 철학(Firmware)에 기반해 벌이는
서로 다른 '관점의 충돌'을 중재하고 합리적 결론을 도출합니다.

## 현재 MarketRegime: {regime}
## 분석 대상: {ticker_str}

## ⚡ Deliberation Protocol (토론 수칙)
1. **관점의 충돌(Tension) 수용**: Tension Score가 높을 경우, 억지로 수치를 평균 내지 마십시오. 왜 소로스와 그레이엄이 싸우고 있는지, 그 철학적 근거를 설명하십시오.
2. **거장의 목소리(Internal Monologue)**: 각 도메인(VAL, GRO, MAC, RSK)의 대표 거장이 자신의 Firmware 문구를 인용하며 '주장'하는 형식을 취하십시오.
3. **창발적 합의**: 상충하는 논리들 사이에서 현재 시장 국면에 가장 '타당한(Adaptive)' 논리가 무엇인지 판정하십시오.

## 출력: 12인 거장 철학적 합의 보고서 (한국어)
- [SECTION 1: Worldview Conflict] 현재 가장 대립하는 관점들과 Tension Score의 의미 분석
- [SECTION 2: Agent Monologues] 주요 거장(Taleb, Soros 등)의 1인칭 내부 독백 및 주장
- [SECTION 3: Final Consensus] 토론을 통해 도출된 최종 투자 방향성 및 정당성
- [SECTION 4: Objective Alignment] 사용자 목적과의 일치성 평가"""
    return _make_firmware_prompt(role_block, bypass_firmware=bypass_firmware)


def _get_stage5_system_prompt(
    regime: str,
    ensemble_signal: float,
    risk_tolerance: str,
    parsed_objective: Optional[Dict] = None,
    sectors: Optional[List[Dict]] = None,
    tickers: Optional[List[str]] = None,
    spg_report: str = "",
    bypass_firmware: bool = False,
) -> str:
    """Stage 5: Portfolio Sizer Agent system prompt (with Brain Firmware).

    Generates a concrete trade plan that directly answers the user's
    investment objective (parsed_objective).
    """
    obj = parsed_objective or {}
    horizon = obj.get("horizon", "long")
    horizon_days = obj.get("horizon_days", 252)
    strategy = obj.get("strategy", "value_accumulation")
    target_markets = obj.get("target_markets", [])
    desired_outputs = obj.get("desired_outputs", ["entry_price", "take_profit", "stop_loss"])
    key_question = obj.get("key_question", "어떤 종목에 어떻게 투자해야 하는가?")
    answer_format_hint = obj.get("answer_format_hint", "종목별 구체 수치 제시")
    specific_tickers = obj.get("specific_tickers", [])

    sector_str = ", ".join(s.get("sector_name", "") for s in (sectors or [])[:3]) or "미정"
    ticker_str = ", ".join(tickers or specific_tickers) or "미정"

    role_block = f"""## 레이어 주의 집중 배정
- PRIMARY LAYERS (사고력 70% 집중):
  LAYER 1 중 RSK 도메인 (Thorp Kelly/Shannon 신뢰도/Taleb 꼬리 위험 — 포지션 사이징 핵심)
  LAYER 2 중 [2-1] Phase 4 (Kelly Ensemble, Half-Kelly + Shannon + Thorp Edge)
  LAYER 4 (optimized_weights — 거장별 calibrated 승률, 특히 Thorp win_rate=0.8284)
- REFERENCE LAYERS (교차 검증용):
  LAYER 0 (MarketRegime에 따른 regime_performance 보정 참조)
  LAYER 1 중 VAL/GRO 도메인 (투자 전략 정합성)
  LAYER 3 (에필로그 루프 구조 참조)
- ⚡ COMPUTATION SUPREMACY:
  ensemble_signal과 엔진 산출값이 있으면 무조건 우선 사용.
  Kelly 공식은 반드시 엔진 산출 ensemble_signal을 입력으로 사용.

## 당신의 역할: Portfolio Architect (포트폴리오 아키텍트)
당신은 H-PIOS v8.5 Field Command 파이프라인의 최종(5번) 에이전트입니다.

## ⚡ 최우선 임무: 사용자 질문에 직접 답하라
사용자의 핵심 질문: "{key_question}"
답변 형식 지시:  {answer_format_hint}

이것이 이 보고서의 존재 이유입니다.
분석 과정은 간략히 요약하고, 답변에 최대한 많은 공간을 할애하십시오.

## 입력 조건
- MarketRegime:      {regime}
- Ensemble Signal:   {ensemble_signal:.3f} (0=완전 매도, 1=완전 매수)
- 투자자 위험 성향:  {risk_tolerance}
- 투자 기간:         {horizon} ({horizon_days}일 기준)
- 전략 유형:         {strategy}
- 대상 시장:         {target_markets}
- 선정 섹터:         {sector_str}
- 분석 종목:         {ticker_str}

━━ Stage 4.5: Strategic Policy Governor 브리핑 ━━
{spg_report}

## 원하는 답변 항목: {desired_outputs}

## Firmware 기반 포지션 사이징 (LAYER 2, Phase 4)

### Kelly Criterion (Thorp RSK_THORP_001)
- p = clip(ensemble_signal, 0.01, 0.99) — Thorp win_rate=0.8284 보정 반영
- b = 1.5 × (1 + Thorp_Edge_Conviction_Premium)
- f* = (b×p - q) / b → Half-Kelly: f = f* × 0.5
- {risk_tolerance}:
  {'conservative → Kelly × 0.5' if risk_tolerance == 'conservative' else 'moderate → Kelly × 1.0' if risk_tolerance == 'moderate' else 'aggressive → Kelly × 1.2 (Taleb 임계값 미달 시만)'}

### 투자 기간별 수익률 기대치 및 손절 기준
- {horizon} 기간 기준:
  · 단기(ultra_short/short): 5~10% 목표, -4~-6% 손절
  · 중기(medium):            10~25% 목표, -7~-10% 손절
  · 장기(long):              20~50% 목표, -10~-15% 손절
  · 초장기(ultra_long):      50%+ 목표, -15~-20% 손절 (or 펀더멘털 훼손 시)

### Taleb 꼬리 위험 보정
- Tail_Risk / Systemic_Crash → 현금 비중 최우선, 헤지 포함
- Stagflation / Deleveraging → 포지션 규모 20~30% 축소

## 출력 형식 (두 섹션으로 구성)

### 섹션 A: 직접 답변
사용자의 질문에 종목별로 구체적으로 답합니다:
- 종목명 / 현재가 기준 진입가 범위 / 목표가(T1·T2) / 손절가
- 권장 매수 수량 또는 투자 비중 (%) / 진입 타이밍
- 투자 근거 (Firmware 노드 인용 포함)

### 섹션 B: 포트폴리오 배분
- 종목별 최종 비중 (%) 및 현금 비중
- 합계 = 100% 강제

### 섹션 C: 에필로그 (학습 루프)
- T+{min(horizon_days, 60)} / T+{horizon_days} 성과 추적 기준점
- 국면 전환 감지 트리거 (재실행 조건)
- Rolling Win Rate 업데이트 대상 노드

모든 출력은 한국어로 작성하십시오."""
    return _make_firmware_prompt(role_block, bypass_firmware=bypass_firmware)


# ══════════════════════════════════════════════════════════
# 6-0. Stage 0: Objective Parser Node — Investment Objective Parser
# ══════════════════════════════════════════════════════════

_OBJECTIVE_PARSER_PROMPT = """당신은 사용자의 투자 의도를 파악하는 파서(Parser)입니다.
사용자가 입력한 자연어 문장에서 '의도'만 추출하여 JSON으로 구조화하십시오.

## 역할 경계 (매우 중요)
- 이 단계는 오직 사용자의 말을 '해석'하는 것입니다.
- 섹터 분석, 종목 추천, 전략 수립은 절대 하지 마십시오. 그것은 다음 단계의 역할입니다.
- 사용자가 명시적으로 말한 것만 추출합니다. 추론하거나 창작하지 마십시오.

## 각 필드 규칙

### horizon (투자 기간)
사용자 발언에서 기간만 추출합니다:
  · 당일~1주: "ultra_short" / horizon_days: 5
  · 1주~1달:  "short"       / horizon_days: 20
  · 1~6달:    "medium"      / horizon_days: 120
  · 6달~2년:  "long"        / horizon_days: 252
  · 2년 이상: "ultra_long"  / horizon_days: 504

### strategy (투자 스타일 분류 — 반드시 아래 중 하나만)
사용자가 언급한 투자 방식의 '스타일'만 분류합니다. 섹터나 전략 내용을 서술하지 마십시오.
  · "value_accumulation" — 저평가 매수, 분할매수, 장기보유 등
  · "growth"             — 성장주, 기술주, AI, 신산업 등
  · "momentum"           — 단타, 추세추종, 모멘텀 등
  · "swing"              — 스윙트레이딩, 수주~수개월
  · "dividend"           — 배당, 인컴 등
  · "index"              — 지수추종, ETF 등
  · "hedge"              — 헤지, 리스크 오프 등
  명확하지 않으면 horizon에 맞춰 추정 (장기→value_accumulation, 단기→momentum)

### target_markets (대상 시장)
사용자가 언급한 시장만 포함합니다. 언급이 없으면 ["KOSPI", "S&P500"] 기본값.
  예: "코스피" → ["KOSPI"], "미국 나스닥" → ["NASDAQ"], "글로벌" → ["KOSPI", "S&P500"]

### desired_outputs (원하는 답변 항목)
사용자가 명시적으로 요청한 것만 포함합니다:
  entry_price, take_profit, stop_loss, shares, timing, sectors, tickers

### specific_tickers (명시 종목)
사용자가 직접 이름을 언급한 종목만 포함합니다. 추론하지 마십시오.
  예: "삼성전자 사고 싶어" → ["삼성전자"]
  예: "코스피 반도체 종목" → [] (반도체는 섹터이지 종목이 아님)

### key_question
사용자의 핵심 질문을 한 문장으로 압축합니다. 답변을 만들지 말고 질문만 정제합니다.

### answer_format_hint
Stage 5가 어떤 형식으로 답해야 하는지 1문장으로 지시합니다.

반드시 아래 JSON 형식으로만 출력 (설명 없음):
```json
{
  "horizon": "...",
  "horizon_days": 0,
  "strategy": "...",
  "target_markets": ["..."],
  "desired_outputs": ["..."],
  "budget_estimate": null,
  "specific_tickers": [],
  "key_question": "...",
  "answer_format_hint": "..."
}
```"""


def objective_parser_node(state: AgentState) -> Dict[str, Any]:
    """Stage 0: Investment objective parser.

    Parses the user's natural-language investment objective/question into a
    structured InvestmentObjective. The result serves as the 'Objective
    Alignment' reference for all subsequent stages.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary (populates parsed_objective).
    """
    ts = datetime.now(timezone.utc).isoformat()
    objective = state.get("investment_objective", "").strip()

    logger.info("=" * 60)
    logger.info("[Stage 0] Objective Parser 시작")
    logger.info("[Stage 0] 투자 목적: %s", objective[:200] if objective else "(없음)")

    # Return defaults if no objective provided
    if not objective:
        default_obj = {
            "horizon": "long",
            "horizon_days": 252,
            "strategy": "value_accumulation",
            "target_markets": ["KOSPI", "S&P500"],
            "desired_outputs": ["entry_price", "take_profit", "stop_loss"],
            "budget_estimate": None,
            "specific_tickers": [],
            "key_question": "현재 시장에서 어떤 종목을 어떻게 투자해야 하는가?",
            "answer_format_hint": "종목별 진입가·목표가·손절가·권장 비중 제시",
        }
        logger.info("[Stage 0] 목적 없음 → 기본값 사용")
        return {
            "parsed_objective": default_obj,
            "audit_log": [f"[{ts}] Stage 0: 기본 목적 적용 (목적 입력 없음)"],
            "errors": [],
        }

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = _init_gemini_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(InvestmentObjective)

        response = structured_llm.invoke([
            SystemMessage(content=_OBJECTIVE_PARSER_PROMPT),
            HumanMessage(content=f"사용자 투자 목적/질문:\n{objective}"),
        ])

        if isinstance(response, InvestmentObjective):
            parsed = response.model_dump()
        else:
            # Fallback: JSON text parsing
            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

        logger.info(
            "[Stage 0] 파싱 완료 — 기간: %s(%d일), 전략: %s, 목표출력: %s",
            parsed.get("horizon"),
            parsed.get("horizon_days", 0),
            parsed.get("strategy"),
            parsed.get("desired_outputs"),
        )
        logger.info("[Stage 0] 핵심 질문: %s", parsed.get("key_question"))

        return {
            "parsed_objective": parsed,
            "audit_log": [
                f"[{ts}] Stage 0: 투자 목적 파싱 완료 "
                f"(horizon={parsed.get('horizon')}, "
                f"strategy={parsed.get('strategy')}, "
                f"outputs={parsed.get('desired_outputs')})"
            ],
            "errors": [],
        }

    except Exception as e:
        logger.warning("[Stage 0] 파싱 실패 (%s) → 텍스트 기반 폴백", e)

        # Simple text-based fallback parsing
        obj_lower = objective.lower()
        horizon = "long"
        horizon_days = 252
        if any(w in obj_lower for w in ["단타", "단기", "day", "intraday", "당일"]):
            horizon, horizon_days = "ultra_short", 5
        elif any(w in obj_lower for w in ["스윙", "swing", "1달", "한달", "월"]):
            horizon, horizon_days = "short", 20
        elif any(w in obj_lower for w in ["3달", "6달", "반기", "medium"]):
            horizon, horizon_days = "medium", 120
        elif any(w in obj_lower for w in ["2년", "3년", "장기", "ultra_long"]):
            horizon, horizon_days = "ultra_long", 504

        target_markets = []
        if any(w in obj_lower for w in ["코스피", "kospi", "한국", "국내"]):
            target_markets.append("KOSPI")
        if any(w in obj_lower for w in ["s&p", "sp500", "미국", "나스닥", "nasdaq"]):
            target_markets.append("S&P500")
        if not target_markets:
            target_markets = ["KOSPI", "S&P500"]

        desired = ["entry_price", "take_profit", "stop_loss"]
        if any(w in obj_lower for w in ["몇주", "수량", "주수"]):
            desired.append("shares")
        if any(w in obj_lower for w in ["타이밍", "언제", "timing"]):
            desired.append("timing")

        fallback_obj = {
            "horizon": horizon,
            "horizon_days": horizon_days,
            "strategy": "value_accumulation" if horizon in ("long", "ultra_long") else "momentum",
            "target_markets": target_markets,
            "desired_outputs": desired,
            "budget_estimate": None,
            "specific_tickers": [],
            "key_question": objective[:200],
            "answer_format_hint": "종목별 진입가·목표가·손절가·권장 비중 구체 수치로 제시",
        }

        return {
            "parsed_objective": fallback_obj,
            "audit_log": [f"[{ts}] Stage 0: 폴백 파싱 (LLM 오류: {str(e)[:80]})"],
            "errors": [f"Stage 0 parsing error (fallback used): {e}"],
        }


# ══════════════════════════════════════════════════════════
# 6-1. Stage 1 sub-utility: Real-time macro indicator collection via yfinance
# ══════════════════════════════════════════════════════════

# Ticker registry (identical to optimizer.py)
_MACRO_TICKER_REGISTRY: Dict[str, str] = {
    "SP500":     "^GSPC",
    "KOSPI":     "^KS11",
    "NASDAQ100": "^NDX",
    "OIL":       "CL=F",
    "GOLD":      "GC=F",
    "USDKRW":    "KRW=X",
    "DXY":       "DX-Y.NYB",
    "TNX":       "^TNX",   # 10Y Treasury Yield
    "IRX":       "^IRX",   # 13-week T-Bill (2Y proxy)
    "VIX":       "^VIX",
    "HYG":       "HYG",    # High Yield Bond ETF
    "IEF":       "IEF",    # 7-10Y Treasury Bond ETF
}

# Threshold reference dictionary (based on Brain Firmware LAYER 1)
_INDICATOR_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "VIX":               {"warn": 25, "critical": 35, "unit": "pts",  "desc": "공포지수"},
    "yield_spread":      {"warn": 0.0, "critical": -0.5, "unit": "%", "desc": "장단기금리차(10Y-2Y)"},
    "credit_spread":     {"warn": 0.97, "critical": 0.93, "unit": "ratio", "desc": "신용스프레드(HYG/IEF)"},
    "SP500_MA200_ratio": {"warn": 1.10, "critical": 1.20, "unit": "ratio", "desc": "S&P500/MA200 이격"},
    "graham_pe_proxy":   {"warn": 18.0, "critical": 25.0, "unit": "배",   "desc": "Graham P/E 프록시"},
    "taleb_skew_proxy":  {"warn": 120, "critical": 140, "unit": "pts",  "desc": "Taleb SKEW 프록시"},
    "soros_momentum":    {"warn": 0.85, "critical": 0.95, "unit": "백분위", "desc": "Soros 모멘텀 백분위"},
}


def _compute_r_squared(series: "pd.Series", window: int = 60) -> float:  # type: ignore[name-defined]
    """Compute linear trend R² over the most recent *window* days (Shannon noise metric)."""
    import numpy as np
    arr = series.dropna().values[-window:]
    if len(arr) < 10:
        return 0.5
    x = range(len(arr))
    cc = float(np.corrcoef(x, arr)[0, 1])
    return cc ** 2 if not (cc != cc) else 0.5  # avoid NaN when correlation is undefined


def _fetch_live_macro_indicators(lookback_days: int = 300) -> Dict[str, Any]:
    """Collect real-time macro indicators via yfinance and produce an LLM-injectable summary.

    Uses the same indicator framework as optimizer.py's fetch_macro_data /
    build_payload but returns a recent-state snapshot (current values + 5-day /
    20-day rate of change + position relative to thresholds).

    Collection logic:
      - Downloads lookback_days (default 300) for MA200 / R²(60-day) computation.
      - Data presented to the LLM: latest values + 5-day / 20-day changes.
      - Synthetic proxies: Graham PE, Buffett ROIC, Taleb SKEW, Soros Momentum, etc.

    Args:
        lookback_days: yfinance download period (days). 300 recommended for MA200.

    Returns:
        {
            "summary_text": str,        # Formatted text report for LLM injection
            "current_values": Dict,     # Latest indicator values
            "synthetic_proxies": Dict,  # Synthetic fundamental proxy values
            "data_date": str,           # Data reference date
            "fetch_error": Optional[str]
        }
    """
    result: Dict[str, Any] = {
        "summary_text": "",
        "current_values": {},
        "synthetic_proxies": {},
        "data_date": "N/A",
        "fetch_error": None,
    }

    try:
        import math
        import numpy as np
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        err = f"yfinance/numpy/pandas 미설치: {e}. `pip install yfinance numpy pandas` 실행 필요."
        logger.warning("[Stage 1] %s", err)
        result["fetch_error"] = err
        result["summary_text"] = f"[yfinance 데이터 수집 불가: {err}]"
        return result

    # ── Data download ──
    from datetime import timedelta
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    frames: Dict[str, "pd.Series"] = {}
    for label, ticker in _MACRO_TICKER_REGISTRY.items():
        try:
            raw = yf.download(
                ticker,
                start=str(start_date),
                end=str(end_date),
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                logger.debug("[Stage 1] yfinance 빈 데이터: %s (%s)", label, ticker)
                continue
            close = raw["Close"]
            if hasattr(close, "iloc") and close.ndim > 1:
                close = close.iloc[:, 0]
            frames[label] = close.squeeze()
        except Exception as e:
            logger.debug("[Stage 1] yfinance 다운로드 실패 %s: %s", ticker, e)

    if len(frames) < 3:
        err = f"yfinance 데이터 수집 부족 ({len(frames)}개 티커). 네트워크 확인 필요."
        logger.warning("[Stage 1] %s", err)
        result["fetch_error"] = err
        result["summary_text"] = f"[yfinance 수집 제한: {err}]"
        return result

    df = pd.DataFrame(frames).ffill().bfill()
    if df.empty:
        result["fetch_error"] = "DataFrame 병합 결과 비어있음."
        return result

    # ── Derived indicator computation ──
    def _safe(v: Any, default: float = 0.0) -> float:
        try:
            fv = float(v)
            return default if (fv != fv or fv == float("inf") or fv == float("-inf")) else fv
        except Exception:
            return default

    def _pct_change(series: "pd.Series", n: int) -> float:
        arr = series.dropna()
        if len(arr) <= n:
            return 0.0
        old = _safe(arr.iloc[-(n + 1)])
        new = _safe(arr.iloc[-1])
        return ((new - old) / old) if old != 0 else 0.0

    latest = df.iloc[-1]
    data_date = str(df.index[-1])[:10]
    result["data_date"] = data_date

    sp500      = _safe(latest.get("SP500"), 4000.0)
    kospi      = _safe(latest.get("KOSPI"), 2500.0)
    ndx        = _safe(latest.get("NASDAQ100"), 15000.0)
    vix        = _safe(latest.get("VIX"), 20.0)
    tnx        = _safe(latest.get("TNX"), 3.0)
    irx        = _safe(latest.get("IRX"), 2.0)
    oil        = _safe(latest.get("OIL"), 80.0)
    gold       = _safe(latest.get("GOLD"), 2000.0)
    usdkrw     = _safe(latest.get("USDKRW"), 1300.0)
    dxy        = _safe(latest.get("DXY"), 102.0)
    hyg        = _safe(latest.get("HYG"), 79.0)
    ief        = _safe(latest.get("IEF"), 95.0)

    yield_spread    = tnx - irx
    credit_proxy    = (hyg / ief) if ief != 0 else 1.0

    # MA200 & Z-Score
    sp500_series = df["SP500"].dropna() if "SP500" in df.columns else pd.Series([sp500])
    ma200 = float(sp500_series.rolling(200, min_periods=1).mean().iloc[-1])
    ma200_ratio = sp500 / ma200 if ma200 != 0 else 1.0

    sp500_ret_series = sp500_series.pct_change().dropna()
    ma20 = float(sp500_series.rolling(20, min_periods=1).mean().iloc[-1])
    pct_dev = (sp500 / ma20 - 1.0) if ma20 != 0 else 0.0
    ret_vol_20 = float(sp500_ret_series.rolling(20, min_periods=5).std().iloc[-1]) if len(sp500_ret_series) > 5 else 0.01
    ret_vol_20 = max(ret_vol_20, 0.005)
    ma20_zscore = pct_dev / ret_vol_20

    # Return calculation
    sp500_ret_5d  = _pct_change(sp500_series, 5)
    sp500_ret_20d = _pct_change(sp500_series, 20)
    kospi_ret_5d  = _pct_change(df["KOSPI"].dropna(), 5) if "KOSPI" in df.columns else 0.0
    kospi_ret_20d = _pct_change(df["KOSPI"].dropna(), 20) if "KOSPI" in df.columns else 0.0
    ndx_ret_5d    = _pct_change(df["NASDAQ100"].dropna(), 5) if "NASDAQ100" in df.columns else 0.0
    ndx_ret_20d   = _pct_change(df["NASDAQ100"].dropna(), 20) if "NASDAQ100" in df.columns else 0.0

    # VIX ROC
    vix_series   = df["VIX"].dropna() if "VIX" in df.columns else pd.Series([vix])
    vix_5d_roc   = _pct_change(vix_series, 5)
    vix_20d_roc  = _pct_change(vix_series, 20)

    # R² (Shannon)
    r_squared_60 = _compute_r_squared(sp500_series, window=60)

    # HYG/IEF Z-Score (credit spread)
    credit_series = (df["HYG"] / df["IEF"]).dropna() if ("HYG" in df.columns and "IEF" in df.columns) else pd.Series([credit_proxy])
    csp_mean = float(credit_series.rolling(60, min_periods=10).mean().iloc[-1])
    csp_std  = max(float(credit_series.rolling(60, min_periods=10).std().iloc[-1]), 0.001)
    hyg_ief_zscore = (credit_proxy - csp_mean) / csp_std

    # Annual return (252 trading days)
    sp500_ret_252d = _pct_change(sp500_series, 252) if len(sp500_series) > 252 else sp500_ret_20d * 12

    # ── Synthetic proxy computation (same logic as optimizer.py build_payload) ──
    ma200_discount = max(0.0, 1.0 - ma200_ratio)
    credit_stress  = max(0.0, -hyg_ief_zscore) * 0.1
    vix_fear       = max(0.0, (vix - 20.0) / 60.0)

    # Graham P/E proxy
    vix_pe_compress = max(0.0, vix - 20.0) * 0.15
    ma_pe_discount  = ma200_discount * 15.0
    graham_pe_proxy = max(5.0, min(35.0, 22.0 - vix_pe_compress - ma_pe_discount))

    # Buffett ROIC proxy
    ndx_60d_ret = _pct_change(df["NASDAQ100"].dropna(), 60) if "NASDAQ100" in df.columns else 0.0
    buffett_roic_proxy = max(0.04, min(0.25,
        0.12 + max(-0.04, min(0.04, hyg_ief_zscore * 0.015))
             + max(-0.03, min(0.03, ndx_60d_ret * 0.1))
    ))

    # Taleb SKEW proxy
    vix_accel      = max(0.0, vix_5d_roc)
    taleb_skew_proxy = max(90.0, min(200.0, 100.0 + vix * 1.0 + vix_accel * 15.0))

    # Soros momentum percentile
    z_momentum    = (sp500_ret_252d - 0.10) / 0.15
    soros_momentum = max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-z_momentum * 1.5))))

    # Dalio credit indicator
    dalio_credit  = max(0.0, min(5.0, max(0.0, -hyg_ief_zscore * 1.5 + 0.5)))

    # Shannon noise ratio
    abs_annual    = max(abs(sp500_ret_252d), 0.01)
    sp500_vol_20d = float(sp500_ret_series.rolling(20, min_periods=5).std().iloc[-1] * (252 ** 0.5)) if len(sp500_ret_series) > 5 else 0.15
    shannon_noise = max(0.5, min(10.0, sp500_vol_20d / abs_annual * 0.5))

    # Store current values
    cv: Dict[str, float] = {
        "SP500": sp500, "KOSPI": kospi, "NASDAQ100": ndx,
        "VIX": vix, "TNX_10Y": tnx, "IRX_2Y": irx,
        "OIL": oil, "GOLD": gold, "USDKRW": usdkrw, "DXY": dxy,
        "yield_spread": yield_spread, "credit_proxy": credit_proxy,
        "SP500_MA200_ratio": ma200_ratio, "SP500_MA20_zscore": ma20_zscore,
        "VIX_5d_roc": vix_5d_roc, "VIX_20d_roc": vix_20d_roc,
        "r_squared_60d": r_squared_60,
        "SP500_ret_5d": sp500_ret_5d, "SP500_ret_20d": sp500_ret_20d,
        "KOSPI_ret_5d": kospi_ret_5d, "KOSPI_ret_20d": kospi_ret_20d,
        "NDX_ret_5d": ndx_ret_5d, "NDX_ret_20d": ndx_ret_20d,
        "SP500_ret_252d": sp500_ret_252d,
    }

    sp: Dict[str, float] = {
        "graham_pe_proxy":    graham_pe_proxy,
        "buffett_roic_proxy": buffett_roic_proxy,
        "taleb_skew_proxy":   taleb_skew_proxy,
        "soros_momentum_pct": soros_momentum,
        "dalio_credit_stress": dalio_credit,
        "shannon_noise_ratio": shannon_noise,
        "hyg_ief_zscore":      hyg_ief_zscore,
    }

    result["current_values"]   = cv
    result["synthetic_proxies"] = sp

    # ── Generate traffic-light signals against thresholds ──
    def _signal(value: float, warn: float, crit: float, higher_is_bad: bool = True) -> str:
        if higher_is_bad:
            if value >= crit:  return "🔴 CRITICAL"
            if value >= warn:  return "🟡 WARN"
            return "🟢 OK"
        else:
            if value <= crit:  return "🔴 CRITICAL"
            if value <= warn:  return "🟡 WARN"
            return "🟢 OK"

    vix_sig      = _signal(vix, 25, 35)
    ys_sig       = _signal(yield_spread, 0.0, -0.5, higher_is_bad=False)
    cr_sig       = _signal(credit_proxy, 0.97, 0.93, higher_is_bad=False)
    ma200_sig    = _signal(ma200_ratio, 1.10, 1.20)
    pe_sig       = _signal(graham_pe_proxy, 18, 25)
    skew_sig     = _signal(taleb_skew_proxy, 120, 140)
    soros_sig    = _signal(soros_momentum, 0.85, 0.95)

    # ── Generate text report ──
    lines = [
        f"━━ LIVE MARKET DATA (yfinance, 기준: {data_date}) ━━",
        "",
        "[글로벌 지수]",
        f"  S&P 500:   {sp500:>8,.1f} pts  ({sp500_ret_5d:+.1%} 5일 | {sp500_ret_20d:+.1%} 20일)",
        f"  KOSPI:     {kospi:>8,.2f} pts  ({kospi_ret_5d:+.1%} 5일 | {kospi_ret_20d:+.1%} 20일)",
        f"  NASDAQ100: {ndx:>8,.1f} pts  ({ndx_ret_5d:+.1%} 5일 | {ndx_ret_20d:+.1%} 20일)",
        "",
        "[공포·변동성]",
        f"  VIX:  {vix:>6.2f}  ({vix_sig})  5일ROC: {vix_5d_roc:+.1%}  20일ROC: {vix_20d_roc:+.1%}",
        f"  R²(60일 추세강도): {r_squared_60:.3f}  "
        f"({'강한추세' if r_squared_60 > 0.7 else '노이즈 우세' if r_squared_60 < 0.3 else '혼조'})",
        "",
        "[금리·신용]",
        f"  10Y TNX:  {tnx:.2f}%  |  2Y IRX: {irx:.2f}%",
        f"  장단기금리차(10Y-2Y): {yield_spread:+.2f}%  ({ys_sig})"
        f"  → {'역전(수축신호)' if yield_spread < 0 else '정상(완만 상승)'} | Dalio 부채사이클 참조",
        f"  신용스프레드 HYG/IEF: {credit_proxy:.4f}  ({cr_sig})  Z-Score: {hyg_ief_zscore:+.2f}",
        "",
        "[환율·원자재]",
        f"  USD/KRW: {usdkrw:>7,.1f}  |  DXY: {dxy:.2f}",
        f"  WTI Oil: ${oil:.1f}  |  Gold: ${gold:,.0f}",
        "",
        "[추세 분석]",
        f"  S&P500 / MA200 비율: {ma200_ratio:.4f}  ({ma200_sig})",
        f"  → MA200 {'상회' if ma200_ratio >= 1 else '하회'} ({(ma200_ratio - 1)*100:+.1f}%)  "
        f"│  20일 Z-Score: {ma20_zscore:+.2f}",
        f"  S&P500 12개월 수익률: {sp500_ret_252d:+.1%}",
        "",
        "[Synthetic Proxies — Brain Firmware 임계값 대비]",
        f"  Graham  P/E 프록시:      {graham_pe_proxy:>5.1f}배   (임계: <15)   {pe_sig}",
        f"  Buffett ROIC 프록시:     {buffett_roic_proxy:>5.3f}    (임계: >0.15)  "
        f"{'🟢 OK' if buffett_roic_proxy >= 0.15 else '🔴 미달'}",
        f"  Taleb   SKEW 프록시:   {taleb_skew_proxy:>6.1f}pts  (임계: >140)  {skew_sig}",
        f"  Soros   모멘텀 백분위:   {soros_momentum:.3f}    (임계: >0.95)  {soros_sig}",
        f"  Dalio   신용 스트레스:   {dalio_credit:>5.2f}    (임계: >1.5)   "
        f"{'🔴 경고' if dalio_credit > 1.5 else '🟡 주의' if dalio_credit > 0.8 else '🟢 OK'}",
        f"  Shannon 노이즈 비율:     {shannon_noise:>5.2f}    (임계: >2.5)   "
        f"{'🔴 고노이즈' if shannon_noise > 2.5 else '🟢 정상'}",
    ]

    result["summary_text"] = "\n".join(lines)
    logger.info(
        "[Stage 1] yfinance 지표 수집 완료 — %s / VIX=%.1f / SP500=%.0f / Yield_Spread=%.2f%%",
        data_date, vix, sp500, yield_spread,
    )
    return result


def _generate_dynamic_search_queries(
    investment_objective: str,
    macro_context: str,
    parsed_objective: Optional[Dict] = None,
    live_data: Optional[Dict] = None,
) -> List[str]:
    """Generate dynamic search queries based on investment objective, macro context, and live indicators.

    Fix4: Directly extracts specific crisis keywords from macro_context to
    prioritize situation-specific queries over generic ones.  Rule-based
    generation (no LLM call) to minimize latency.

    Args:
        investment_objective: User investment objective.
        macro_context: User-provided macro situation text.
        parsed_objective: Parsed objective structure.
        live_data: yfinance real-time data result.

    Returns:
        List of search queries (max 6).
    """
    obj = parsed_objective or {}
    target_markets = obj.get("target_markets", [])
    strategy = obj.get("strategy", "value_accumulation")

    queries: List[str] = []
    ctx_lower = (macro_context or "").lower()
    obj_lower = (investment_objective or "").lower()
    combined_lower = ctx_lower + obj_lower

    # ── Fix4: Directly detect specific crisis events from macro_context ──
    # Middle East / energy crisis
    has_mideast = any(w in ctx_lower for w in [
        "중동", "이란", "이스라엘", "호르무즈", "hormuz", "iran", "israel",
        "middle east", "오일", "유가", "oil price",
    ])
    # Circuit breaker / panic / crash
    has_crash = any(w in ctx_lower for w in [
        "서킷브레이커", "circuit breaker", "폭락", "급락", "crash", "panic sell",
        "블랙스완", "black swan", "급등락",
    ])
    # Stagflation / Fed / interest rates
    has_stagflation = any(w in ctx_lower for w in [
        "스태그플레이션", "stagflation", "인플레이션", "inflation",
        "연준", "fed", "금리 인하", "rate cut", "higher for longer",
    ])
    # Technical rebound / oversold buying phase
    has_rebound = any(w in ctx_lower for w in [
        "반등", "rebound", "기술적 반등", "저점", "oversold", "서킷브레이커 이후",
    ])

    # ── 1. Crisis-specific queries (highest priority, macro_context-based) ──
    if has_mideast:
        queries.append(
            "Middle East conflict oil price Hormuz strait financial market impact 2026"
        )
        queries.append(
            "oil price spike $100 geopolitical risk energy supply global stocks 2026"
        )
    if has_crash:
        queries.append(
            "KOSPI circuit breaker market crash technical rebound recovery outlook 2026"
        )
    if has_stagflation:
        queries.append(
            "Federal Reserve rate cut delay inflation sticky monetary policy 2026"
        )
    if has_rebound and has_crash:
        queries.append(
            "KOSPI oversold bounce Samsung SK Hynix semiconductor recovery 2026"
        )

    # ── 2. Quantitative crisis queries based on live_data (Fix4 new) ──
    if live_data and not live_data.get("fetch_error"):
        cv = live_data.get("current_values", {})
        vix = cv.get("VIX", 20.0)
        ret_252d = cv.get("SP500_ret_252d", 0.0)
        ys = cv.get("yield_spread", 0.5)
        r_sq = cv.get("r_squared_60d", 0.5)

        if vix > 25:
            queries.append(
                f"VIX {vix:.0f} high volatility risk-off market sentiment S&P500 correction 2026"
            )
        if ret_252d <= -0.20:
            queries.append(
                "S&P500 bear market annual decline portfolio protection value investing 2026"
            )
        if ys < 0:
            queries.append(
                "yield curve inverted recession probability credit market 2026"
            )
        if r_sq < 0.1:
            queries.append(
                "noisy market low trend directionless stocks mean reversion strategy 2026"
            )

    # ── 3. Market-specific queries (supplementary when insufficient) ──
    if "KOSPI" in target_markets or any(w in combined_lower for w in ["코스피", "kospi", "한국", "korea"]):
        queries.append(
            "KOSPI Korea stock market valuation undervalued large cap 2026"
        )
    if "S&P500" in target_markets or any(w in combined_lower for w in ["s&p", "nasdaq", "미국 증시"]):
        queries.append(
            "S&P500 bear market bottom value opportunity long-term investing 2026"
        )

    # ── 4. Strategy-specific supplementary queries ──
    if strategy in ("value_accumulation", "value"):
        queries.append(
            "value stocks P/E ratio earnings yield undervalued buy the dip 2026"
        )
    elif strategy in ("growth", "momentum"):
        queries.append(
            "growth stocks technology AI earnings growth volatile market 2026"
        )

    # ── 5. Fallback: add default queries if fewer than 2 ──
    if len(queries) < 2:
        queries.append("global macroeconomic outlook interest rates inflation 2026 latest")
        queries.append("Federal Reserve monetary policy decision market impact 2026")

    # Deduplicate and limit to max 6
    seen: set = set()
    unique: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:6]


def _search_macro_data(
    tavily,
    queries: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search for macroeconomic data using Tavily.

    Collects comprehensive macro information via multiple queries.

    Args:
        tavily: TavilySearchResults instance.
        queries: List of search queries (defaults used when None).

    Returns:
        List of search result dictionaries.
    """
    if queries is None:
        queries = [
            "current global macroeconomic outlook market conditions 2026",
            "Federal Reserve interest rate decision monetary policy latest",
            "global geopolitical risks financial markets impact today",
            "stock market volatility VIX credit spreads treasury yield 2026",
        ]

    all_results: List[Dict[str, Any]] = []
    for query in queries:
        try:
            results = tavily.invoke(query)
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, str):
                all_results.append({"content": results, "url": "tavily_search"})
            logger.info(
                "[Stage 1] Tavily 검색 완료: '%s' → %d건",
                query,
                len(results) if isinstance(results, list) else 1,
            )
        except Exception as e:
            logger.warning("[Stage 1] Tavily 검색 실패 ('%s'): %s", query, e)
            all_results.append({
                "content": f"Search failed for: {query}. Error: {e}",
                "url": "error",
            })

    return all_results


def _build_macro_context(
    search_results: List[Dict[str, Any]],
    investment_objective: str,
    macro_context: str = "",
    parsed_objective: Optional[Dict] = None,
    live_macro_data: Optional[Dict] = None,
) -> str:
    """Compose an integrated context from search results, yfinance indicators, and user input.

    Assembly order (to guide LLM attention):
      1. Investment objective — analysis direction reference (top priority)
      2. User macro information — Human-in-the-Loop input
      3. yfinance economic indicators — numerical evidence (incl. synthetic proxies)
      4. Web search results — latest news and trends

    Args:
        search_results: Tavily search results.
        investment_objective: User investment objective/question (natural language).
        macro_context: User-provided current macro situation.
        parsed_objective: Parsed objective structure.
        live_macro_data: Return value from _fetch_live_macro_indicators().

    Returns:
        Integrated context string.
    """
    context_parts: List[str] = []

    # ── 1. Investment objective (analysis direction basis — top priority) ──
    if investment_objective and investment_objective.strip():
        obj_header = "## INVESTOR'S OBJECTIVE (분석 방향 기준 — 최우선 참조)\n"
        if parsed_objective:
            obj_header += (
                f"  · 투자 기간:  {parsed_objective.get('horizon', '?')} "
                f"({parsed_objective.get('horizon_days', '?')}일)\n"
                f"  · 전략:       {parsed_objective.get('strategy', '?')}\n"
                f"  · 대상 시장:  {parsed_objective.get('target_markets', [])}\n"
                f"  · 원하는 답:  {parsed_objective.get('desired_outputs', [])}\n"
                f"  · 핵심 질문:  {parsed_objective.get('key_question', '?')}\n"
            )
        obj_header += f"\n[원문]\n{investment_objective}\n---"
        context_parts.append(obj_header)

    # ── 2. User-provided macro information (Human-in-the-Loop) ──
    if macro_context and macro_context.strip():
        context_parts.append(
            "## USER-PROVIDED MACRO CONTEXT (사용자 직접 입력 — 높은 신뢰도)\n"
            "※ 이 정보는 사용자가 직접 제공한 현재 매크로 상황입니다.\n"
            "  수치 데이터보다 우선하여 국면 판정의 방향 기준으로 사용하십시오.\n\n"
            f"{macro_context}\n---"
        )

    # ── 3. yfinance real-time economic indicators ──
    if live_macro_data and not live_macro_data.get("fetch_error"):
        summary = live_macro_data.get("summary_text", "")
        if summary:
            context_parts.append(
                f"## REAL-TIME ECONOMIC INDICATORS (yfinance, 수치 근거)\n"
                f"※ Brain Firmware의 각 노드 임계값과 직접 대조하여 국면을 판단하십시오.\n\n"
                f"{summary}\n---"
            )
    elif live_macro_data and live_macro_data.get("fetch_error"):
        context_parts.append(
            f"## ECONOMIC INDICATORS (수집 실패)\n"
            f"yfinance 데이터 수집 실패: {live_macro_data.get('fetch_error')}\n"
            "웹 검색 결과와 사용자 입력으로 대체 판단하십시오.\n---"
        )

    # ── 4. Web search results ──
    if search_results:
        context_parts.append("## MARKET NEWS & ANALYSIS (Tavily 웹 검색)")
        for i, result in enumerate(search_results[:12], 1):  # cap at 12 results
            content = result.get("content", "")
            url = result.get("url", "unknown")
            if content:
                truncated = content[:600] if len(content) > 600 else content
                context_parts.append(f"\n### Source {i} ({url})\n{truncated}")

    return "\n\n".join(context_parts)


def _parse_regime_output(output: Any) -> MacroPulseOutput:
    """Parse LLM output into a MacroPulseOutput.

    Returns immediately if the output is already a MacroPulseOutput.
    For AIMessage, attempts JSON extraction followed by text-based fallback parsing.

    Args:
        output: LLM response (MacroPulseOutput, AIMessage, or str).

    Returns:
        MacroPulseOutput instance.

    Raises:
        ValueError: When all parsing methods fail.
    """
    # Return immediately if already MacroPulseOutput
    if isinstance(output, MacroPulseOutput):
        return output

    # Extract content from AIMessage
    # Gemini may return content as list[dict]; safely convert to string
    raw_content = output.content if hasattr(output, "content") else str(output)
    if isinstance(raw_content, list):
        # Handle [{"type": "text", "text": "..."}, ...] or [str, ...] format
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        content = " ".join(parts)
    else:
        content = str(raw_content)

    # Attempt JSON parsing
    try:
        # Extract ```json ... ``` block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            # Attempt to parse entire content as JSON
            data = json.loads(content)

        return MacroPulseOutput(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("[Stage 1] JSON 파싱 실패, 텍스트 분석으로 폴백: %s", e)

    # ── Text-based fallback parsing ──
    found_regime = "Normal_Market"
    for regime_val in MARKET_REGIME_VALUES:
        if regime_val in content or regime_val.replace("_", " ") in content:
            found_regime = regime_val
            break

    return MacroPulseOutput(
        market_regime=found_regime,
        confidence=0.4,
        reasoning=(
            f"LLM 구조화 출력 실패로 텍스트 분석 사용. "
            f"원본 응답 일부: {content[:300]}"
        ),
        key_factors=[
            "텍스트 분석 폴백 — 구조화 파싱 실패",
            "신뢰도 낮음 — 수동 확인 권고",
        ],
        risk_level="MEDIUM",
    )


def _validate_regime(regime_str: str) -> MarketRegime:
    """Safely convert a string to a MarketRegime Enum.

    Attempts exact match → case-insensitive match → Normal_Market fallback.

    Args:
        regime_str: Regime string value.

    Returns:
        MarketRegime Enum instance.
    """
    # Exact match
    try:
        return MarketRegime(regime_str)
    except ValueError:
        pass

    # Case-insensitive match
    regime_lower = regime_str.lower().replace(" ", "_")
    for regime in MarketRegime:
        if regime.value.lower() == regime_lower:
            return regime

    logger.warning(
        "[Stage 1] 알 수 없는 국면: '%s' → Normal_Market 폴백", regime_str
    )
    return MarketRegime.NORMAL_MARKET


# ── Critical shutdown regime set ──
CRITICAL_SHUTDOWN_REGIMES = frozenset({
    "Market_Crash",
    "Systemic_Crash",
    "Panic_Selling",
    "Capitulation",
    "Liquidity_Crisis",
    "Credit_Crunch",
})


def macro_pulse_agent(state: AgentState) -> Dict[str, Any]:
    """Stage 1: Macro Pulse Agent — macro strategy analysis and market regime determination.

    Integrates four data sources to determine the MarketRegime:
      1. yfinance real-time economic indicators (numerical evidence)
      2. User-provided macro information (Human-in-the-Loop, high reliability)
      3. Tavily web search (latest news and trends)
      4. Investment objective (analysis direction reference)

    Execution flow:
        Step 1. Collect yfinance economic indicators (same framework as optimizer.py)
        Step 2. Generate dynamic Tavily queries (objective / context / indicator-based)
        Step 3. Execute Tavily search
        Step 4. Compose integrated context from 4 sources
        Step 5. Gemini regime determination with Brain Firmware
        Step 6. Validate MarketRegime Enum
        Step 7. Critical risk alert evaluation

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary.
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("[Stage 1] Macro Pulse Agent 시작 — %s", ts)
    logger.info("=" * 60)

    audit_entries: List[str] = [f"[{ts}] Stage 1: Macro Pulse Agent 시작"]
    errors: List[str] = []

    investment_objective = state.get("investment_objective", "")
    macro_context_input  = state.get("macro_context", "")    # User-provided macro input
    parsed_objective     = state.get("parsed_objective", {})

    # NO_REGIME variant: skip regime determination and fix to Normal_Market
    if state.get("_variant_no_regime", False):
        logger.warning("[Stage 1] VARIANT: NO_REGIME — 국면 판정 생략 (Normal_Market 고정)")
        return {
            "market_regime": "Normal_Market",
            "regime_confidence": 1.0,
            "macro_reasoning": "EXPERIMENTAL VARIANT: NO_REGIME (Forced Normal_Market)",
            "macro_key_factors": ["Variant flag set: _variant_no_regime"],
            "macro_risk_level": "LOW",
            "macro_search_results": [],
            "live_macro_snapshot": {},
            "macro_pulse_report": "This is a bypass report for NO_REGIME variant.",
            "audit_log": audit_entries + [f"[{ts}] NO_REGIME variant active: skipped analysis."],
            "errors": [],
        }

    try:
        # ── Step 1: Collect real-time economic indicators via yfinance ──
        logger.info("[Stage 1] Step 1: yfinance 경제지표 수집 중 (최근 300일)...")
        live_data = _fetch_live_macro_indicators(lookback_days=300)

        if live_data.get("fetch_error"):
            logger.warning("[Stage 1] yfinance 수집 경고: %s", live_data["fetch_error"])
            errors.append(f"yfinance 경고: {live_data['fetch_error']}")
            audit_entries.append(f"[{ts}] yfinance 수집 부분 실패: {live_data['fetch_error']}")
        else:
            cv = live_data.get("current_values", {})
            audit_entries.append(
                f"[{ts}] yfinance 수집 완료 ({live_data.get('data_date','?')}) — "
                f"VIX={cv.get('VIX',0):.1f}, SP500={cv.get('SP500',0):,.0f}, "
                f"KOSPI={cv.get('KOSPI',0):,.0f}, "
                f"Yield_Spread={cv.get('yield_spread',0):+.2f}%"
            )
            logger.info(
                "[Stage 1] yfinance 완료: VIX=%.1f SP500=%.0f KOSPI=%.0f "
                "Yield=%.2f%% MA200_ratio=%.4f",
                cv.get("VIX", 0), cv.get("SP500", 0), cv.get("KOSPI", 0),
                cv.get("yield_spread", 0), cv.get("SP500_MA200_ratio", 1),
            )

        # ── Step 2: Generate dynamic Tavily queries ──
        logger.info("[Stage 1] Step 2: 동적 검색 쿼리 생성 중...")
        search_queries = _generate_dynamic_search_queries(
            investment_objective=investment_objective,
            macro_context=macro_context_input,
            parsed_objective=parsed_objective,
            live_data=live_data,
        )
        logger.info("[Stage 1] 생성된 쿼리 (%d개): %s", len(search_queries), search_queries)
        audit_entries.append(
            f"[{ts}] 동적 검색 쿼리 {len(search_queries)}개 생성: "
            f"{search_queries[:3]}..."
        )

        # ── Step 3: Execute Tavily search ──
        logger.info("[Stage 1] Step 3: Tavily 검색 실행 중...")
        tavily = _init_tavily_tool(max_results=5)
        search_results = _search_macro_data(tavily, queries=search_queries)
        audit_entries.append(
            f"[{ts}] Tavily 검색 완료: {len(search_results)}건 수집"
        )

        if macro_context_input.strip():
            logger.info(
                "[Stage 1] 사용자 매크로 컨텍스트 반영됨 (%d자)",
                len(macro_context_input),
            )
            audit_entries.append(
                f"[{ts}] 사용자 매크로 컨텍스트: '{macro_context_input[:100]}...'"
            )

        # ── Step 4: Compose integrated context from 4 sources ──
        integrated_context = _build_macro_context(
            search_results=search_results,
            investment_objective=investment_objective,
            macro_context=macro_context_input,
            parsed_objective=parsed_objective,
            live_macro_data=live_data,
        )

        # ── Step 5: Gemini regime determination with Brain Firmware ──
        logger.info(
            "[Stage 1] Step 5: Gemini 국면 판정 중 (모델: %s, Brain Firmware 탑재)...",
            GEMINI_MODEL,
        )
        llm = _init_gemini_llm(temperature=0.0)   # Fix2: deterministic regime classification

        from langchain_core.messages import SystemMessage, HumanMessage

        bypass_fw = state.get("_variant_no_firmware", False)
        stage1_system_prompt = _get_stage1_system_prompt(bypass_firmware=bypass_fw)
        logger.info(
            "[Stage 1] Brain Firmware 주입 완료 — System Prompt: %d chars",
            len(stage1_system_prompt),
        )
        audit_entries.append(
            f"[{ts}] Brain Firmware 주입: {len(stage1_system_prompt):,}자 "
            f"(≈{len(stage1_system_prompt)//4:,} 토큰)"
        )

        # Append yfinance key metrics as a separate highlight block
        live_highlight = ""
        if not live_data.get("fetch_error"):
            cv = live_data.get("current_values", {})
            sp = live_data.get("synthetic_proxies", {})

            # Fix3: human-readable label for SP500 252-trading-day return
            ret_252d = cv.get("SP500_ret_252d", 0.0)
            if ret_252d <= -0.20:
                ret_252d_label = "BEAR MARKET 영역 (연간 -20% 초과 하락)"
            elif ret_252d <= -0.10:
                ret_252d_label = "조정 국면 (연간 -10%~-20%)"
            elif ret_252d >= 0.20:
                ret_252d_label = "강세장 (연간 +20% 초과)"
            else:
                ret_252d_label = "중립 범위"

            # Fix3: human-readable label for 60d trend R² (Shannon noise proxy)
            r_sq = cv.get("r_squared_60d", 0.5)
            if r_sq < 0.1:
                r_sq_label = "PURE NOISE — 방향성 없음 (Shannon: 포지션 축소 권고)"
            elif r_sq < 0.3:
                r_sq_label = "노이즈 우세 — 약한 추세 (Shannon: 주의)"
            elif r_sq > 0.7:
                r_sq_label = "강한 추세 — 트렌드 추종 유효"
            else:
                r_sq_label = "혼조 (중간 강도 추세)"

            live_highlight = (
                "\n━━ yfinance 핵심 지표 요약 (Firmware 임계값 직접 대조) ━━\n"
                f"  VIX: {cv.get('VIX',0):.1f}  "
                f"(Marks 임계값: 25→WARN / 35→CRITICAL)\n"
                f"  장단기금리차: {cv.get('yield_spread',0):+.2f}%  "
                f"(Dalio 임계값: <0 → 역전신호)\n"
                f"  SP500/MA200: {cv.get('SP500_MA200_ratio',1):.3f}  "
                f"(Soros 이격도)\n"
                f"  MA20 Z-Score: {cv.get('SP500_MA20_zscore',0):+.2f}  "
                f"(Simons 평균회귀)\n"
                f"  신용스프레드 Z-Score: {sp.get('hyg_ief_zscore',0):+.2f}  "
                f"(Marks OAS 참조)\n"
                f"  Taleb SKEW 프록시: {sp.get('taleb_skew_proxy',100):.1f}  "
                f"(임계값: 140)\n"
                f"  Graham P/E 프록시: {sp.get('graham_pe_proxy',20):.1f}  "
                f"(임계값: <15)\n"
                f"  Soros 모멘텀 백분위: {sp.get('soros_momentum_pct',0.5):.3f}  "
                f"(임계값: >0.95)\n"
                f"\n  ── [Fix3] 중장기 추세 신호 ──\n"
                f"  SP500 연간 수익률 (252d): {ret_252d:+.1%}  "
                f"→ {ret_252d_label}\n"
                f"  KOSPI 5일 수익률:        {cv.get('KOSPI_ret_5d',0):+.1%}  "
                f"(패닉 셀링 기준: -10% 이하)\n"
                f"  추세 강도 R² (60d):     {r_sq:.4f}  "
                f"→ {r_sq_label}\n"
            )

            # Fix3: extra instructions so the report scorecard includes trend block
            _trend_signals_note = (
                f"\n[중요 — 반드시 보고서 스코어카드에 포함할 것]\n"
                f"  SP500 연간 수익률 {ret_252d:+.1%} = {ret_252d_label}\n"
                f"  KOSPI 5일 수익률 {cv.get('KOSPI_ret_5d',0):+.1%}\n"
                f"  60일 추세 R² {r_sq:.4f} = {r_sq_label}\n"
                f"  위 3개 지표는 Soros 재귀성 / Shannon 노이즈 판단 / Taleb 꼬리위험 진단의 핵심 근거임.\n"
            )
        else:
            _trend_signals_note = ""

        human_content = (
            f"Current UTC Time: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{integrated_context}\n"
            f"{live_highlight}\n"
            f"{_trend_signals_note}\n"
            "━━ BRAIN FIRMWARE 기반 판정 지시 ━━\n"
            "위 4가지 데이터 소스를 종합하여 현재 Market Regime을 판정하십시오.\n\n"
            "데이터 소스 우선순위:\n"
            "  1순위: 사용자 직접 입력 매크로 정보 (USER-PROVIDED MACRO CONTEXT)\n"
            "  2순위: yfinance 수치 데이터 (REAL-TIME ECONOMIC INDICATORS)\n"
            "  3순위: Tavily 뉴스 검색 결과 (MARKET NEWS)\n"
            "  4순위: 투자 목적 (분석 방향 힌트)\n\n"
            "판정 절차: Taleb(Phase 1) → Dalio(Phase 2) → Marks(Phase 3) → Soros(Phase 4)\n"
            "판정 근거에 반드시 Firmware 노드 ID와 실제 수치(예: VIX=28.5 > 임계값 25)를 명시.\n\n"
            "━━ 출력 형식 (반드시 아래 두 섹션을 순서대로 출력) ━━\n\n"
            "【중요】JSON 섹션을 반드시 먼저 출력한 뒤, 보고서 섹션을 출력하시오.\n\n"
            "===JSON_START===\n"
            "```json\n"
            "{\n"
            '  "market_regime": "<MarketRegime 값 중 하나>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "reasoning": "<Firmware 노드 ID + 실제 수치 인용 포함 2~3문장 요약>",\n'
            '  "key_factors": ["<수치 근거 포함 요인1>", "<요인2>", "<요인3>", "<요인4>"],\n'
            '  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>"\n'
            "}\n"
            "```\n"
            "===JSON_END===\n\n"
            "===REPORT_START===\n"
            "# H-PIOS v8.5 Stage 1 매크로 분석 보고서\n\n"
            "## 1. 시장 국면 판정\n"
            "판정 국면 / 신뢰도 / 위험 등급을 한 줄로 요약하시오.\n\n"
            "## 2. Firmware 기반 판정 근거\n"
            "Taleb → Dalio → Marks → Soros → Shannon 순서로 각 노드 ID와 실제 수치를 인용하며 논거를 전개하시오.\n"
            "Soros 항목: SP500 연간 수익률 + 모멘텀 백분위 반드시 인용.\n"
            "Shannon 항목: 60일 R² 수치 + 노이즈/추세 해석 명시.\n\n"
            "## 3. 핵심 지표 스코어카드\n"
            "| 지표 | 현재 수치 | Firmware 임계값 | 데이터 성격 | 판정 |\n"
            "|------|-----------|-----------------|-------------|------|\n"
            "반드시 포함: VIX / 신용 스트레스 / SP500 연간 수익률 / "
            "KOSPI 5일 수익률 / 60일 추세 R² / Graham P/E 프록시 / 장단기 금리차\n"
            "데이터 성격: '실제 시장 지표' 또는 'macro 기반 proxy' 구분 기입.\n\n"
            "## 4. 투자 목적 연계 시사점\n"
            "사용자 투자 목적(1년 이상 장투, 코스피 중심) 기준으로 전략적 함의 서술.\n\n"
            "## 5. Stage 2 방향 예고\n"
            "Stage 2(섹터 탐색)가 어떤 방향으로 분석해야 하는지 1~2문장.\n"
            "===REPORT_END==="
        )

        # Single llm.invoke() — receive report + JSON simultaneously (without with_structured_output)
        response = llm.invoke([
            SystemMessage(content=stage1_system_prompt),
            HumanMessage(content=human_content),
        ])

        # Safe content extraction
        raw_content = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw_content, list):
            raw_content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_content
            )

        # ── Parse JSON section (expected first in output) ──
        json_text = ""
        json_section_match = re.search(
            r"===JSON_START===\s*([\s\S]*?)\s*===JSON_END===",
            raw_content,
        )
        if json_section_match:
            json_text = json_section_match.group(1).strip()
            # Extract ```json ... ``` block
            json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", json_text)
            if json_block_match:
                json_text = json_block_match.group(1).strip()
        else:
            # JSON_START present without JSON_END (truncated output)
            json_only_match = re.search(
                r"===JSON_START===\s*```(?:json)?\s*([\s\S]*?)```",
                raw_content,
            )
            if json_only_match:
                json_text = json_only_match.group(1).strip()
            else:
                # Search for JSON block in full text (last resort fallback)
                json_block_any = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_content)
                if json_block_any:
                    json_text = json_block_any.group(1).strip()

        # ── Parse report section ──
        report_text = ""
        report_match = re.search(
            r"===REPORT_START===\s*([\s\S]*?)\s*===REPORT_END===",
            raw_content,
        )
        if report_match:
            report_text = report_match.group(1).strip()
        else:
            # Truncated without REPORT_END: save everything after REPORT_START
            report_start_match = re.search(
                r"===REPORT_START===\s*([\s\S]+)",
                raw_content,
            )
            if report_start_match:
                report_text = report_start_match.group(1).strip()
                logger.warning("[Stage 1] 보고서가 잘렸습니다 (===REPORT_END=== 없음). 부분 저장.")
            else:
                # No report section found: use text after JSON as report
                json_end_pos = raw_content.find("===JSON_END===")
                if json_end_pos > 0:
                    report_text = raw_content[json_end_pos + len("===JSON_END==="):].strip()
                else:
                    report_text = raw_content

        # JSON parsing → MacroPulseOutput
        parsed_output = None
        if json_text:
            try:
                data = json.loads(json_text)
                parsed_output = MacroPulseOutput(**data)
                logger.info("[Stage 1] JSON 파싱 성공 (두 섹션 분리 방식)")
            except Exception as je:
                logger.warning("[Stage 1] JSON 파싱 실패: %s", je)

        if parsed_output is None:
            # Full text fallback
            parsed_output = _parse_regime_output(response)

        # ── Step 6: Validate MarketRegime Enum ──
        validated_regime = _validate_regime(parsed_output.market_regime)

        logger.info(
            "[Stage 1] 판정 결과: %s (신뢰도: %.2f)",
            validated_regime.value,
            parsed_output.confidence,
        )
        logger.info("[Stage 1] 위험 수준: %s", parsed_output.risk_level)
        logger.info("[Stage 1] 핵심 요인: %s", parsed_output.key_factors)

        audit_entries.append(
            f"[{ts}] 국면 판정: {validated_regime.value} "
            f"(신뢰도={parsed_output.confidence:.2f}, "
            f"위험={parsed_output.risk_level})"
        )

        # Log report generation confirmation
        logger.info(
            "[Stage 1] 매크로 보고서 생성 완료 (%d자)", len(report_text)
        )

        # ── Step 7: Critical risk alert ──
        if validated_regime.value in CRITICAL_SHUTDOWN_REGIMES:
            logger.warning(
                "[Stage 1] CRITICAL RISK: %s — 긴급 셧다운 고려!",
                validated_regime.value,
            )
            audit_entries.append(
                f"[{ts}] CRITICAL RISK: {validated_regime.value}"
            )

        return {
            # ── Inter-agent shared data (JSON-based structured data) ──
            "market_regime": validated_regime.value,
            "regime_confidence": parsed_output.confidence,
            "macro_reasoning": parsed_output.reasoning,
            "macro_key_factors": parsed_output.key_factors,
            "macro_risk_level": parsed_output.risk_level,
            "macro_search_results": search_results[:10],
            "live_macro_snapshot": {
                "current_values":    live_data.get("current_values", {}),
                "synthetic_proxies": live_data.get("synthetic_proxies", {}),
                "data_date":         live_data.get("data_date", "N/A"),
                "fetch_error":       live_data.get("fetch_error"),
            },
            # ── User-facing (report text) ──
            "macro_pulse_report": report_text,
            # ── System metadata ──
            "audit_log": audit_entries,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"[Stage 1] 치명적 오류: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
        audit_entries.append(f"[{ts}] {error_msg}")

        return {
            "market_regime": MarketRegime.NORMAL_MARKET.value,
            "regime_confidence": 0.1,
            "macro_reasoning": f"시스템 오류로 인한 안전 폴백: {e}",
            "macro_key_factors": ["시스템 오류"],
            "macro_risk_level": "CRITICAL",
            "macro_search_results": [],
            "live_macro_snapshot": {},
            "macro_pulse_report": f"[ERROR] Stage 1 실행 중 오류 발생: {e}",
            "audit_log": audit_entries,
            "errors": errors,
        }


# ══════════════════════════════════════════════════════════
# 6. Stage 2: Sector/Theme Scout
# ══════════════════════════════════════════════════════════

def sector_scout_agent(state: AgentState) -> Dict[str, Any]:
    """Stage 2: Sector/Theme Scout — top-3 promising sector selection.

    Cross-validates Tavily data against Stage 1's Market Regime to select
    promising sectors.

    Brain Firmware injection:
        _get_stage2_system_prompt(regime, strategy) is used during LLM calls.
        Applies Dalio Global_Weight_Adjuster → Fisher/Lynch/Buffett lens filtering.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary.
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("[Stage 2] Sector/Theme Scout 시작 — %s", ts)
    logger.info("=" * 60)

    regime = state.get("market_regime", "Normal_Market")
    parsed_objective = state.get("parsed_objective", {})
    strategy = parsed_objective.get("strategy", "value_accumulation")
    target_markets = parsed_objective.get("target_markets", ["KOSPI"])
    investment_objective = state.get("investment_objective", "N/A")

    audit_entries: List[str] = [f"[{ts}] Stage 2: Sector/Theme Scout 시작"]
    errors: List[str] = []

    try:
        from stage02_test import (
            _step1_regime_strategy_map,
            _step2_generate_queries,
            _step3_tavily_search,
            _step4_build_context,
            _build_stage2_system_prompt,
            _call_gemini_and_parse
        )

        # ── STEP 1: Regime × Strategy mapping ──
        logger.info("[Stage 2] Step 1: Regime × Strategy 맵핑 중...")
        candidates = _step1_regime_strategy_map(regime, strategy, target_markets)
        audit_entries.append(
            f"[{ts}] STEP 1: 후보 섹터 {len(candidates)}개 맵핑 완료"
        )

        # ── STEP 2: Generate search queries ──
        logger.info("[Stage 2] Step 2: 검색 쿼리 생성 중...")
        macro_key_factors = state.get("macro_key_factors", [])
        live_snapshot = state.get("live_macro_snapshot", {})
        search_queries = _step2_generate_queries(
            regime, strategy, target_markets, candidates, macro_key_factors, live_snapshot, investment_objective
        )
        audit_entries.append(
            f"[{ts}] STEP 2: 동적 검색 쿼리 {len(search_queries)}개 생성"
        )

        # ── STEP 3: Tavily search ──
        logger.info("[Stage 2] Step 3: 섹터 동향 뉴스 검색 중...")
        search_results = _step3_tavily_search(search_queries)
        audit_entries.append(
            f"[{ts}] STEP 3: Tavily 섹터 뉴스 {len(search_results)}건 수집"
        )

        # ── STEP 4: Compose integrated context ──
        logger.info("[Stage 2] Step 4: 통합 컨텍스트 구성 중...")
        human_content = _step4_build_context(
            investment_objective=investment_objective,
            parsed_objective=parsed_objective,
                macro_key_factors=macro_key_factors,
            search_results=search_results,
            strategy=strategy
        )

        # ── STEP 5: Gemini sector selection (Brain Firmware injected) ──
        logger.info("[Stage 2] Step 5: Gemini 섹터 선정 (Brain Firmware 탑재)...")
        bypass_fw = state.get("_variant_no_firmware", False)
        # Fix: using _get_stage2_system_prompt instead of imported _build_stage2_system_prompt for variant support
        system_prompt = _get_stage2_system_prompt(regime, bypass_firmware=bypass_fw)

        parsed_output, report_text, raw_content = _call_gemini_and_parse(
            system_prompt, human_content
        )

        if parsed_output is None:
            logger.warning("[Stage 2] JSON 파싱 실패, Fallback 적용")
            errors.append("Stage 2 Gemini Output parsing failed. Using Top 3 Candidates as fallback.")
            audit_entries.append(f"[{ts}] JSON 파싱 실패, 텍스트 분석 폴백 적용")
            # Fallback
            top_sectors = [
                {
                    "sector_name": c["sector_name"],
                    "rank": i+1,
                    "market": c.get("market", "KOSPI"),
                    "regime_fit_reason": c.get("regime_fit_reason"),
                    "objective_fit_reason": c.get("objective_fit_reason"),
                    "risk_adjusted_confidence": 0.5,
                    "candidate_tickers": c.get("candidate_tickers", []),
                    "key_metrics_to_collect": []
                }
                for i, c in enumerate(candidates[:3])
            ]
        else:
            logger.info("[Stage 2] 섹터 선정 완료 (%d개)", len(parsed_output.top_sectors))
            audit_entries.append(f"[{ts}] STEP 5: 섹터 선정 완료 ({[s.sector_name for s in parsed_output.top_sectors]})")
            top_sectors = [s.model_dump() for s in parsed_output.top_sectors]

        # Aggregate candidate tickers from selected sectors
        target_tickers = []
        for s in top_sectors:
             target_tickers.extend(s.get("candidate_tickers", []))

        # Deduplicate
        target_tickers = list(dict.fromkeys(target_tickers))

        return {
            "top_sectors": top_sectors,
            "target_tickers": target_tickers,
            "sector_analysis": report_text,
            "audit_log": audit_entries,
            "errors": errors,
        }

    except Exception as e:
        logger.error("[Stage 2] 오류 발생: %s", e, exc_info=True)
        errors.append(f"Stage 2 error: {e}")
        return {
            "top_sectors": [],
            "target_tickers": [],
            "sector_analysis": f"[ERROR] Stage 2 오류: {e}",
            "audit_log": audit_entries,
            "errors": errors,
        }


# ══════════════════════════════════════════════════════════
# 7. Stage 3: Deep-Dive Micro Agent
# ══════════════════════════════════════════════════════════

def _fetch_kr_fundamentals(raw_ticker: str) -> Dict[str, float]:
    """Scrape Naver Finance to collect key financial metrics for Korean stocks.

    Collected metrics: PER, PBR, ROE (proxy), EPS, BPS, debt ratio, operating
    margin, revenue, operating income, dividend yield, current price.
    """
    import requests
    from bs4 import BeautifulSoup

    ticker_num = raw_ticker.split('.')[0]
    if not ticker_num.isdigit():
        return {}

    url = f"https://finance.naver.com/item/main.naver?code={ticker_num}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    metrics: Dict[str, float] = {}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        # ── Main indicators (id-based) ──
        for tag_id, key in [('#_per', 'PER'), ('#_pbr', 'PBR'), ('#_dvr', 'DIV_YIELD')]:
            em = soup.select_one(tag_id)
            if em and em.text.strip():
                try:
                    metrics[key] = float(em.text.replace(',', ''))
                except ValueError:
                    pass

        # ── Current price ──
        now_val = soup.select_one('p.no_today span.blind')
        if now_val:
            try:
                metrics['CURRENT_PRICE'] = float(now_val.text.replace(',', ''))
            except ValueError:
                pass

        # ── Corporate performance analysis table ──
        cop_tables = soup.select('div.section.cop_analysis table')
        for table in cop_tables:
            for row in table.select('tr'):
                th = row.select_one('th')
                tds = row.select('td')
                if not th or not tds:
                    continue
                label = th.get_text(strip=True)
                val_text = tds[0].get_text(strip=True)
                try:
                    val = float(val_text.replace(',', '')) if val_text and val_text != 'N/A' else None
                except ValueError:
                    val = None
                if val is None:
                    continue
                if 'ROE' in label and '(%)' in label:
                    metrics['ROE'] = val / 100.0
                elif '부채비율' in label:
                    metrics['DEBT_RATIO'] = val / 100.0
                elif '영업이익률' in label:
                    metrics['OP_MARGIN'] = val / 100.0
                elif label.startswith('EPS') and '원' in label:
                    metrics['EPS_VAL'] = val
                elif label.startswith('BPS') and '원' in label:
                    metrics['BPS'] = val
                elif '매출액' in label:
                    metrics['REVENUE'] = val
                elif '영업이익' in label and '률' not in label:
                    metrics['OP_INCOME'] = val

    except Exception:
        logger.debug("[NaverFinance] %s 크롤링 실패", raw_ticker)

    return metrics

def _fetch_micro_data(ticker: str) -> Dict[str, float]:
    """Collect micro-level financial data for an individual ticker via yfinance.

    Gathers, computes, and maps the key metrics required by the 12-master
    Intelligence_Structure Step 1 metrics.
    Falls back to Naver Finance data when yfinance data is missing.
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    metrics: Dict[str, float] = {}

    try:
        # 1. Prioritize / concurrently collect auxiliary data from Naver Finance
        nv_data = _fetch_kr_fundamentals(ticker)

        t = yf.Ticker(ticker)

        # 1. Price and basic info
        info = t.info if hasattr(t, "info") else {}

        # 2. Historical data (1 year)
        hist = t.history(period="1y")

        # ──── Value / fundamental metrics (Graham, Buffett, Munger) ────

        # P/E Ratio — yfinance preferred, Naver PER as fallback
        pe_yf = info.get("trailingPE") or info.get("forwardPE") or 0.0
        metrics["P/E_Ratio"] = pe_yf if pe_yf else nv_data.get("PER", 0.0)

        # Price_to_NCAV proxy — PBR-based
        pb = info.get("priceToBook", 0.0) or nv_data.get("PBR", 0.0)
        metrics["Price_to_NCAV"] = pb / 1.5 if pb else 0.0

        # ROIC_10yr_Avg — yfinance ROE preferred, Naver ROE as fallback
        roe_yf = info.get("returnOnEquity", 0.0) or 0.0
        roe_nv = nv_data.get("ROE", 0.0)
        roe = roe_yf if roe_yf > 0 else roe_nv  # Naver fallback when <= 0
        # Blend OP_MARGIN from Naver as ROIC proxy if available
        op_margin_nv = nv_data.get("OP_MARGIN", 0.0)
        if roe <= 0.0 and op_margin_nv > 0:
            roe = op_margin_nv * 0.7  # 70% of operating margin as ROE proxy
        metrics["ROIC_10yr_Avg"] = max(0.0, roe)

        # Gross Margin Volatility
        gm = info.get("grossMargins", 0.0)
        if gm == 0.0 and op_margin_nv > 0:
            gm = op_margin_nv  # Use operating margin as gross margin proxy
        metrics["Gross_Margin_Volatility"] = max(0.01, 0.5 - gm)

        # Debt_to_Equity — yfinance (%-based) preferred, Naver debt ratio as fallback
        dte_yf = info.get("debtToEquity", 0.0)
        if dte_yf:
            metrics["Debt_to_Equity"] = dte_yf / 100.0
        elif "DEBT_RATIO" in nv_data:
            metrics["Debt_to_Equity"] = nv_data["DEBT_RATIO"]
        else:
            metrics["Debt_to_Equity"] = 0.0

        # FCF_Yield = Free Cash Flow / Market Cap
        fcf = info.get("freeCashflow", 0.0)
        mcap = info.get("marketCap", 0)
        if fcf and mcap:
            metrics["FCF_Yield"] = fcf / mcap
        elif "CURRENT_PRICE" in nv_data and "BPS" in nv_data:
            # Dividend yield proxy relative to BPS
            metrics["FCF_Yield"] = nv_data.get("DIV_YIELD", 0.0) / 100.0
        else:
            metrics["FCF_Yield"] = 0.0

        # ──── Growth metrics (Fisher, Lynch) ────

        # R&D to Revenue Ratio
        metrics["RnD_to_Revenue_Ratio"] = 0.05

        # Operating_Margin_Expansion_3yr
        om_yf = info.get("operatingMargins", 0.0) or op_margin_nv
        metrics["Operating_Margin_Expansion_3yr"] = om_yf * 0.5

        # PEG Ratio
        peg = info.get("pegRatio", 0.0)
        if (not peg or peg == 0.0) and metrics["P/E_Ratio"] > 0:
            peg = metrics["P/E_Ratio"] / 15.0
        metrics["PEG_Ratio"] = peg

        # EPS Growth (TTM) — yfinance preferred, estimate from Naver EPS + current price
        epsg_yf = info.get("earningsGrowth", 0.0)
        if epsg_yf:
            metrics["EPS_Growth_TTM"] = epsg_yf
        elif "EPS_VAL" in nv_data and "CURRENT_PRICE" in nv_data and nv_data["EPS_VAL"] != 0:
            # Reverse-engineer growth proxy from E/P ratio
            metrics["EPS_Growth_TTM"] = nv_data["EPS_VAL"] / nv_data["CURRENT_PRICE"]
        else:
            metrics["EPS_Growth_TTM"] = 0.0

        # ──── Momentum / risk metrics (Soros, Simons, Shannon, Thorp) ────

        # Price_Momentum_1yr_Percentile (estimated 1-year return percentile)
        if not hist.empty and len(hist) > 100:
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            ret_1y = (end_price - start_price) / start_price if start_price else 0.0
            metrics["Price_Momentum_1yr_Percentile"] = max(0.0, min(1.0, 0.5 + ret_1y))
        else:
            metrics["Price_Momentum_1yr_Percentile"] = 0.5

        # Price_to_Fundamental_Divergence_Zscore
        metrics["Price_to_Fundamental_Divergence_Zscore"] = 0.0

        # Price_Mean_Reversion_Zscore (20-day MA Z-Score)
        if not hist.empty and len(hist) >= 20:
             closes = hist["Close"]
             ma20 = closes.rolling(20).mean().iloc[-1]
             std20 = closes.rolling(20).std().iloc[-1]
             current = closes.iloc[-1]
             if std20 > 0:
                 metrics["Price_Mean_Reversion_Zscore"] = (current - ma20) / std20
             else:
                 metrics["Price_Mean_Reversion_Zscore"] = 0.0
        else:
            metrics["Price_Mean_Reversion_Zscore"] = 0.0

        metrics["Order_Book_Imbalance_Ratio"] = 0.5   # Order book data unavailable

        # Price_Trend_R_Squared (recent 60-day R²)
        if not hist.empty and len(hist) >= 60:
             closes_60 = hist["Close"].values[-60:]
             x = np.arange(len(closes_60))
             cc = np.corrcoef(x, closes_60)[0, 1]
             metrics["Price_Trend_R_Squared"] = cc ** 2 if not np.isnan(cc) else 0.5
        else:
            metrics["Price_Trend_R_Squared"] = 0.5

        # Intraday_Volatility_vs_Daily_Return
        metrics["Intraday_Volatility_vs_Daily_Return"] = 1.0

        # Thorp Parameters (defaults; calibrated by Orchestrator/Engine)
        metrics["Expected_Value_of_Signal"] = 0.1
        metrics["Historical_Win_Rate_of_Signal"] = 0.6

        metrics["Yield_Curve_Spread_10Y_2Y"] = 0.0  # placeholder; Macro Pulse sets globals
        metrics["Credit_Spread_High_Yield_Volatility"] = 0.0
        metrics["M2_Money_Supply_YoY"] = 0.0
        metrics["VIX_Index"] = 20.0
        metrics["OAS_Spread_Zscore"] = 0.0
        metrics["SKEW_Index"] = 100.0
        metrics["OTM_Put_Option_Volume_Spike"] = 0.0

    except Exception as e:
        # KO log: "ticker data collection failed"
        logger.warning("[Stage 3] 종목 %s 데이터 수집 실패: %s", ticker, e)

    return metrics

def deep_dive_micro_agent(state: AgentState) -> Dict[str, Any]:
    """Stage 3: Deep-Dive Micro Agent — per-ticker financial data collection.

    Collects actual financial data (yfinance) for tickers within the selected
    sectors and produces MarketDataPayload-compliant data.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary.
    """
    ts = datetime.now(timezone.utc).isoformat()
    sectors = state.get("top_sectors", [])
    target_tickers = state.get("target_tickers", [])
    market_regime = state.get("market_regime", "Unknown")

    logger.info("=" * 60)
    logger.info("[Stage 3] Deep-Dive Micro Agent 시작 — %s", ts)
    logger.info("=" * 60)

    audit_entries: List[str] = [f"[{ts}] Stage 3: Deep-Dive Micro Agent 시작 (대상 종목 {len(target_tickers)}개)"]
    errors: List[str] = []
    payloads: List[MarketDataPayload] = []

    try:
        if not target_tickers:
           logger.warning("[Stage 3] Target Tickers가 비어있습니다.")
           errors.append("Stage 3: No target tickers provided from Stage 2")

        for raw_ticker in target_tickers:
            # yfinance requires .KS or .KQ suffix for Korean stocks
            ticker = raw_ticker
            if ticker.isdigit():
                # Default to KOSPI (.KS) if just digits are provided without market context
                # (A proper implementation might use target_market or candidate metadata to distinguish .KS/.KQ)
                ticker = f"{raw_ticker}.KS"
                
            logger.info(f"[Stage 3] {ticker} (원래 종목코드: {raw_ticker}) 데이터 수집 중...")
            raw_metrics = _fetch_micro_data(ticker)

            # Convert to Pydantic Payload
            payload = MarketDataPayload(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                metrics=raw_metrics,
                current_regime=MarketRegime(market_regime) if market_regime in [r.value for r in MarketRegime] else MarketRegime.NORMAL_MARKET,
                regime_confidence=state.get("regime_confidence", 0.5),
                metadata={"source": "yfinance_micro_agent", "original_ticker": raw_ticker}
            )
            payloads.append(payload)

        # Prepare Brain Firmware system prompt (for report generation during LLM interaction)
        # Stage 3 primarily focuses on data collection/conversion rather than direct LLM calls.
        bypass_fw = state.get("_variant_no_firmware", False)
        stage3_prompt = _get_stage3_system_prompt(sectors, bypass_firmware=bypass_fw)

        micro_analysis = "### Stage 3: Deep-Dive Micro Agent 지표 데이터 수집 결과\n\n"
        micro_analysis += f"총 **{len(target_tickers)}**개의 대상 종목에 대한 기초 재무 및 시장 데이터(MarketDataPayload) 조회를 완료했습니다.\n\n"
        micro_analysis += "| 종목코드 | P/E Ratio | NCAV Proxy | ROIC | D/E Ratio | FCF Yield | Op Margin | EPS Growth |\n"
        micro_analysis += "|---|---|---|---|---|---|---|---|\n"
        for p in payloads:
            m = p.metrics
            pe = m.get("P/E_Ratio", 0)
            ncav = m.get("Price_to_NCAV", 0)
            roic = m.get("ROIC_10yr_Avg", 0)
            de = m.get("Debt_to_Equity", 0)
            fcfy = m.get("FCF_Yield", 0)
            opm = m.get("Operating_Margin_Expansion_3yr", 0) * 2  # reverse the *0.5
            epsg = m.get("EPS_Growth_TTM", 0)
            micro_analysis += f"| **{p.ticker}** | {pe:.2f} | {ncav:.2f} | {roic*100:.1f}% | {de:.2f} | {fcfy*100:.2f}% | {opm*100:.1f}% | {epsg*100:.1f}% |\n"
        micro_analysis += "\n위 지표들은 다음 단계인 12인 거장 Consensus Engine의 기초 데이터로 공급(Supply)됩니다.\n"

        audit_entries.append(f"[{ts}] {len(payloads)}개 종목 MarketDataPayload 구성 완료")

        # Convert Pydantic models to dicts for storage
        payloads_dict = [p.model_dump() for p in payloads]

        return {
            "market_data_payloads": payloads_dict,
            "micro_analysis": micro_analysis,
            "audit_log": audit_entries,
            "errors": errors,
        }

    except Exception as e:
        logger.error("[Stage 3] 오류 발생: %s", e, exc_info=True)
        errors.append(f"Stage 3 error: {e}")
        return {
            "market_data_payloads": [],
            "micro_analysis": f"[ERROR] Stage 3 오류: {e}",
            "audit_log": audit_entries,
            "errors": errors,
        }


# ══════════════════════════════════════════════════════════
# 8. Stage 4: Consensus Engine Node
# ══════════════════════════════════════════════════════════

def consensus_engine_node(state: AgentState) -> Dict[str, Any]:
    """Stage 4: Consensus Engine — 12-master consensus signal derivation.

    Feeds Stage 3's MarketDataPayloads into engine_core.py's
    GraphOrchestrator to derive consensus signals from the 12 masters.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary.
    """
    ts = datetime.now(timezone.utc).isoformat()
    payloads_dict_list = state.get("market_data_payloads", [])
    regime = state.get("market_regime", "N/A")
    tickers = state.get("target_tickers", [])
    
    logger.info("=" * 60)
    logger.info("[Stage 4] Consensus Engine 시작 — %s", ts)
    logger.info("=" * 60)

    audit_entries: List[str] = [f"[{ts}] Stage 4: Consensus Engine 시작 (페이로드 {len(payloads_dict_list)}개)"]
    errors: List[str] = []
    orchestrator_outputs = []
    
    try:
        from CORE_ENGINE_core import GraphOrchestrator
        from CORE_MODELS_models import MarketDataPayload, NLPContextPayload
        import json
        import os

        # Load 4 master JSON configs
        configs = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_files = [
            "DATA_JSON_value_master.json",
            "DATA_JSON_growth_master.json",
            "DATA_JSON_macro_master.json",
            "DATA_JSON_risk_master.json"
        ]
        
        from CORE_MODELS_models import MasterEngineConfig
        for file in json_files:
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    configs.append(MasterEngineConfig(**data))
            else:
                 logger.warning("[Stage 4] 마스터 JSON 파일을 찾을 수 없습니다: %s", file_path)
                 errors.append(f"Missing config file: {file}")

        if not configs:
            raise FileNotFoundError("Master JSON files required for Orchestrator not found.")

        # ── [Experimental] Variant patching (Ablation Study) ──
        # Note: NO_TALEB and NO_EDGES are now also handled at the engine.resolve_signals level if passed in extra_context,
        # but manual patching here is also fine to completely remove them from the DAG.
        import copy as _copy

        # NO_RISK: Remove Taleb, Shannon, Thorp nodes
        if state.get("_variant_no_risk", False):
            logger.warning("[Stage 4] VARIANT: NO_RISK — Risk(Taleb, Shannon, Thorp) 노드 제거")
            patched_configs = []
            risk_nodes = ["TALEB", "SHANNON", "THORP"]
            for cfg in configs:
                cfg_copy = _copy.deepcopy(cfg)
                cfg_copy.Nodes = [n for n in cfg_copy.Nodes if not any(rn in n.Node_ID.upper() for rn in risk_nodes)]
                cfg_copy.Logical_Edges = [
                    e for e in cfg_copy.Logical_Edges
                    if not any(rn in e.Source.upper() or rn in e.Target.upper() for rn in risk_nodes)
                ]
                patched_configs.append(cfg_copy)
            configs = patched_configs

        # NO_TALEB: Remove Taleb node
        elif state.get("_variant_no_taleb", False):
            logger.warning("[Stage 4] VARIANT: NO_TALEB — Taleb 노드 제거")
            patched_configs = []
            for cfg in configs:
                cfg_copy = _copy.deepcopy(cfg)
                cfg_copy.Nodes = [n for n in cfg_copy.Nodes if "TALEB" not in n.Node_ID.upper()]
                cfg_copy.Logical_Edges = [
                    e for e in cfg_copy.Logical_Edges
                    if "TALEB" not in e.Source.upper() and "TALEB" not in e.Target.upper()
                ]
                patched_configs.append(cfg_copy)
            configs = patched_configs

        # NO_EDGES: Remove all Logical_Edges
        if state.get("_variant_no_edges", False):
            logger.warning("[Stage 4] VARIANT: NO_EDGES — 모든 Logical Edges 제거")
            patched_configs = []
            for cfg in configs:
                cfg_copy = _copy.deepcopy(cfg)
                cfg_copy.Logical_Edges = []
                patched_configs.append(cfg_copy)
            configs = patched_configs

        # Initialize Orchestrator
        logger.info("[Stage 4] GraphOrchestrator 초기화 중... (%d개 설정 로드)", len(configs))
        orchestrator = GraphOrchestrator(configs=configs)
        
        # Dummy NLP payload
        dummy_nlp = NLPContextPayload(
            source="Stage4_Internal",
            timestamp=datetime.now(timezone.utc),
            raw_texts=["Market regime update."],
            nlp_model_confidence=1.0
        )

        overall_signals = []

        for p_dict in payloads_dict_list:
            try:
                # dict to Model
                payload = MarketDataPayload(**p_dict)
                ticker = payload.ticker
                
                # Signal resolution (pass state as extra_context for engine-level variant handling)
                # NO_REGIME variant: passes via extra_context so the engine ignores regime constraints
                # Requires engine_core.GraphOrchestrator.resolve_signals to support this
                output = orchestrator.resolve_signals(payload, dummy_nlp, extra_context=state)

                # FLAT_ENS variant: replace ensemble signal with simple average
                ens_signal = output.ensemble_signal
                pos_size = output.final_position_size
                if state.get("_variant_flat_ensemble", False):
                    scores = [
                        r.normalized_score
                        for r in output.node_results.values()
                        if hasattr(r, "normalized_score")
                    ]
                    ens_signal = sum(scores) / len(scores) if scores else 0.5
                    pos_size = max(0.0, min(1.0, ens_signal))
                    logger.warning(f"[Stage 4] VARIANT: FLAT_ENS — {ticker} signal={ens_signal:.3f}")

                orchestrator_outputs.append({
                    "ticker": ticker,
                    "ensemble_signal": ens_signal,
                    "final_position_size": pos_size,
                    "node_results": {
                        nid: (res.model_dump() if hasattr(res, "model_dump") else res)
                        for nid, res in output.node_results.items()
                    }
                })
                
                overall_signals.append(ens_signal)
                
            except Exception as inner_e:
                import traceback
                traceback.print_exc()
                logger.warning(f"[Stage 4] {p_dict.get('ticker', 'Unknown')} 처리 실패")
                errors.append(f"Orchestrator failed for a ticker: {inner_e}")
                
        # Overall consensus signal average
        avg_signal = sum(overall_signals) / len(overall_signals) if overall_signals else 0.5

        # Generate Consensus Summary
        # [Variant Support] NO_FIRMWARE
        bypass_fw = bool(state.get("_variant_no_firmware", False))
        stage4_prompt = _get_stage4_system_prompt(regime, tickers, bypass_firmware=bypass_fw)
        
        # [v8.8] LLM-based Deliberation & Synthesis
        logger.info("[Stage 4] LLM Deliberation 시작...")
        llm = _init_gemini_llm(temperature=0.3)
        
        # [Variant Support] BLIND_CHAIR
        show_tension = not bool(state.get("_variant_blind_chair", False))
        safe_tension = getattr(output, "tension_score", 0.0) if 'output' in locals() else 0.0
        
        # [v8.9] Sandbox Context: Include Tension Score and Monologue instructions
        context_data = {
            "regime": regime,
            "tension_score": safe_tension if show_tension else "HIDDEN_FOR_EXPERIMENT",
            "orchestrator_outputs": orchestrator_outputs,
            "objective": state.get("parsed_objective", {})
        }
        
        # [Variant Support] MUTE_LOGUE
        is_mute = bool(state.get("_variant_mute_logue", False))
        
        deliberation_prompt = stage4_prompt + f"\n\n### 샌드박스 상태 데이터 {'(Tension Score 포함)' if show_tension else '(Tension Score 비공개)'}\n{json.dumps(context_data, indent=2, ensure_ascii=False)}"
        
        if not is_mute:
            deliberation_prompt += "\n\n**지시**: 각 거장의 'Internal Monologue'를 먼저 작성하고, 이를 바탕으로 최종 합의 요약을 작성하십시오. 만약 Taleb의 리스크 비중이 높음에도 Soros가 강력 매수를 주장한다면, 그 둘의 '철학적 논쟁'을 구체적으로 묘사하고 의장으로서 최종 결론을 내리십시오."
        else:
            deliberation_prompt += "\n\n**지시**: 위의 데이터를 바탕으로 시장 상황에 대한 최종 합의 요약을 간단히 작성하십시오. (개별 거장의 내적 독백은 생략하십시오.)"

        # [v8.9.5] Multi-Turn Deliberation Support
        # Default: 1 turn (Optimal for Symbolic Integrity)
        turns = int(state.get("_deliberation_turns", 1))
        deliberation_history = []
        consensus_synthesis = ""
        
        current_prompt = deliberation_prompt
        
        for i in range(turns):
            try:
                if i > 0:
                    # Previous turn context for multi-turn deliberation
                    current_prompt = f"### 이전 토론 결과 (Turn {i}):\n{consensus_synthesis}\n\n"
                    current_prompt += "### 의장 지시: 위의 토론 결과를 비판적으로 검토하십시오. "
                    current_prompt += "엔진의 수치 데이터(Tension Score 등)와 어긋나는 논리가 있는지 확인하고, "
                    current_prompt += "각 거장의 철학적 일관성을 유지하며 최종적인 창발적 합의안을 완성하십시오."

                llm_response = llm.invoke(current_prompt)
                content = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                consensus_synthesis = content
                deliberation_history.append(consensus_synthesis)
                
            except Exception as llm_e:
                logger.warning(f"[Stage 4] LLM Deliberation Turn {i+1} 실패: {llm_e}")
                if not consensus_synthesis:
                    consensus_synthesis = f"Deliberation failed. Numeric outputs remain stable.\nTension Score: {context_data['tension_score']:.4f}"
                break

        state_updates = {
            "consensus_summary": "", # To be filled below
            "deliberation_logs": deliberation_history
        }

        # [v8.9] Enhanced Summary with Tension Metric
        final_summary = "### Stage 4: Philosophy-based Deliberation Sandbox Report\n\n"
        t_val = context_data['tension_score']
        t_display = f"{t_val:.4f}" if isinstance(t_val, (float, int)) else str(t_val)
        final_summary += f"> **Worldview Tension Score: `{t_display}`** (Turns: {turns})\n"
        final_summary += f"> (Tension이 높을수록 거장들 간의 철학적 충돌이 격렬함을 의미합니다.)\n\n"
        final_summary += str(consensus_synthesis) + "\n\n"
        final_summary += "#### [Ground Truth] 최종 수치 득점\n"
        final_summary += f"- **Ensemble Alpha Signal:** `{avg_signal:.3f}`\n\n"
        
        # ... (Table formatting omitted for brevity, keeping existing)
        final_summary += "| 종목코드 | 통합 시그널 | 최종 비중 | 통계적 의견 |\n"
        final_summary += "|---|---|---|---|\n"
        for out in orchestrator_outputs:
             sig = out['ensemble_signal']
             sz = out['final_position_size']
             opinion = "강력 매수" if sig >= 0.7 else "비중 확대" if sig >= 0.55 else "중립 관망" if sig >= 0.4 else "비중 축소" if sig >= 0.25 else "매도 추천"
             final_summary += f"| **{out['ticker']}** | {sig:.3f} | {sz:.3f} | {opinion} |\n"

        audit_entries.append(f"[{ts}] 합의 도출 및 LLM Deliberation 완료.")
        
        return {
            "orchestrator_outputs": orchestrator_outputs,
            "consensus_summary": final_summary,
            "deliberation_logs": state_updates.get("deliberation_logs", []),
            "audit_log": audit_entries,
            "errors": errors,
        }

    except Exception as e:
        logger.error("[Stage 4] 오류 발생: %s", e, exc_info=True)
        errors.append(f"Stage 4 error: {e}")
        return {
            "orchestrator_outputs": [],
            "consensus_summary": f"[ERROR] Stage 4 오류: {e}",
            "audit_log": audit_entries,
            "errors": errors,
        }



# ══════════════════════════════════════════════════════════
# 8.5 Stage 4.5: Strategic Policy Governor (SPG)
# ══════════════════════════════════════════════════════════
# 10. Strategic Policy Governor (Stage 4.5)
# ══════════════════════════════════════════════════════════

def strategic_policy_governor(state: AgentState) -> Dict[str, Any]:
    """Stage 4.5: Strategic Policy Governor — harmonizing quant signals with investment strategy.
    
    Acts as the 'investment committee chair' that overlays the user's
    investment purpose (DCA, long-term hold, etc.) onto the 12-master quant
    engine (Stage 4) results to establish strategic floors.
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("[Stage 4.5] Strategic Policy Governor 시작")
    logger.info("=" * 60)

    # ── [Experimental] SPG deactivation logic (Ablation Study) ──
    if state.get("_variant_no_spg", False) or state.get("bypass_spg", False):
        logger.warning("[Stage 4.5] SPG 바이패스 모드 활성화 (No Filters Applied)")
        return {
            "spg_outputs": {out["ticker"]: {"strategic_floor": 0.0, "quality_gate_score": 0, "reasons": ["SPG_BYPASSED"]} for out in state.get("orchestrator_outputs", [])},
            "spg_report": "### Stage 4.5: SPG 바이패스 모드\n\n현재 실험을 위해 모든 전략적 품질 게이트 및 하한선이 비활성화되었습니다.",
            "audit_log": [f"[{ts}] Stage 4.5: SPG BYPASSED (Ablation Test)"]
        }

    regime = state.get("market_regime", "Normal_Market")
    parsed_objective = state.get("parsed_objective", {})
    orchestrator_outputs = state.get("orchestrator_outputs", [])
    market_data_payloads = {p["ticker"]: p for p in state.get("market_data_payloads", [])}
    
    strategy = parsed_objective.get("strategy", "value_accumulation")
    horizon_days = parsed_objective.get("horizon_days", 252)
    
    spg_outputs = {}
    audit_entries = [f"[{ts}] Stage 4.5: SPG 시작 (Strategy={strategy}, Horizon={horizon_days}d)"]
    
    # ── Module 3: Regime Attenuation (upward-limiting coefficient) ──
    regime_multipliers = {
        "Normal_Market": 1.0,
        "High_Volatility": 0.7,
        "Bear_Market": 0.5,
        "Market_Crash": 0.2,
        "Systemic_Crash": 0.0
    }
    regime_mult = regime_multipliers.get(regime, 0.5)

    # ── Module 4: Horizon Harmonizer (horizon adjustment coefficient) ──
    if horizon_days < 60: horizon_mult = 0.5
    elif horizon_days < 120: horizon_mult = 0.8
    elif horizon_days < 252: horizon_mult = 1.0
    elif horizon_days < 504: horizon_mult = 1.2
    else: horizon_mult = 1.4

    # ── Module 1: Strategy Resonator (base floor) ──
    base_floors = {
        "value_accumulation": 0.12,
        "income_generation": 0.08,
        "tactical_rebalancing": 0.10,
        "momentum_trading": 0.0,
        "speculative": 0.0
    }
    strategy_base = base_floors.get(strategy, 0.0)
    
    report_lines = [f"### Stage 4.5: Strategic Policy Governor 브리핑\n"]
    report_lines.append(f"- **시장 국면 보정계수:** `{regime_mult}` ({regime})")
    report_lines.append(f"- **투자 기간 보정계수:** `{horizon_mult}` ({horizon_days}일)")
    report_lines.append(f"- **전략적 기본 하한선:** `{strategy_base:.2f}` ({strategy})\n")
    report_lines.append("| 종목 | Contrarian | Quality | 전략적 하한 (Floor) | 비고 |")
    report_lines.append("|---|---|---|---|---|")

    for out in orchestrator_outputs:
        ticker = out["ticker"]
        payload = market_data_payloads.get(ticker, {})
        metrics = payload.get("metrics", {})
        
        # ── Module 2: Contrarian Opportunity (contrarian bonus) ──
        contrarian_bonus = 0.0
        c_reasons = []
        # Graham Deep Value (P/E < 10 or NCAV < 0.7)
        pe = metrics.get("P/E_Ratio", 999)
        ncav = metrics.get("Price_to_NCAV", 999)
        if pe < 10 or ncav < 0.7:
            contrarian_bonus += 0.03
            c_reasons.append("DeepValue")
        # Debt Safety
        if metrics.get("Debt_to_Equity", 999) < 0.5:
            contrarian_bonus += 0.01
            c_reasons.append("LowDebt")
        # Cash Flow / Div
        if metrics.get("FCF_Yield", 0) > 0.03 or metrics.get("Dividend_Yield", 0) > 0.03:
            contrarian_bonus += 0.01
            c_reasons.append("CashFlow")
            
        # ── Module 5: Quality Gate ──
        q_count = 0
        if metrics.get("Debt_to_Equity", 2.0) < 1.0: q_count += 1
        if metrics.get("FCF_Yield", -1) > 0 or metrics.get("Dividend_Yield", 0) > 0.02: q_count += 1
        if 0 < metrics.get("P/E_Ratio", 0) < 30: q_count += 1
        if metrics.get("Operating_Margin", -1) > 0: q_count += 1
        if metrics.get("Price_Momentum_1yr_Percentile", 0) > 0.1: q_count += 1 # Exclude bottom 10%
        
        quality_mult = 0.0
        if q_count >= 5: quality_mult = 1.0
        elif q_count == 4: quality_mult = 0.8
        elif q_count == 3: quality_mult = 0.5
        elif q_count == 2: quality_mult = 0.2
        else: quality_mult = 0.0
        
        # Total Floor Calculation
        adj_floor = (strategy_base + contrarian_bonus) * regime_mult * horizon_mult * quality_mult
        adj_floor = min(0.20, adj_floor)  # max 20% strategic floor per ticker
        
        spg_outputs[ticker] = {
            "strategic_floor": adj_floor,
            "contrarian_bonus": contrarian_bonus,
            "quality_gate_score": q_count,
            "reasons": c_reasons if adj_floor > 0 else ["Quality Gate Rejected"]
        }
        
        status = "✅ 승인" if adj_floor > 0 else "❌ 차단"
        report_lines.append(f"| {ticker} | +{contrarian_bonus:.2f} | {q_count}/4 | **{adj_floor:.1%}** | {status} ({','.join(c_reasons)}) |")

    total_floor = sum(s["strategic_floor"] for s in spg_outputs.values())
    # Total equity cap 30% (bear market protection)
    if total_floor > 0.30:
        scale = 0.30 / total_floor
        for ticker in spg_outputs:
            spg_outputs[ticker]["strategic_floor"] *= scale
        report_lines.append(f"\n> [!NOTE]\n> 총 전략적 비중({total_floor:.1%})이 하락장 상한(30%)을 초과하여 비례 축소되었습니다.")
    
    report_text = "\n".join(report_lines)
    audit_entries.append(f"[{ts}] SPG 완료. 총 전략적 하한 = {sum(s['strategic_floor'] for s in spg_outputs.values()):.1%}")
    
    return {
        "spg_outputs": spg_outputs,
        "spg_report": report_text,
        "audit_log": audit_entries
    }


# ══════════════════════════════════════════════════════════
# 9. Stage 5: Portfolio Sizer
# ══════════════════════════════════════════════════════════

def portfolio_sizer_agent(state: AgentState) -> Dict[str, Any]:
    """Stage 5: Portfolio Sizer — generate a direct answer to the user's investment objective.

    Prioritizes the user's investment objective (parsed_objective) to produce
    concrete trade plans including per-ticker entry, target, stop-loss prices
    and position sizes.

    Determines position sizing via Thorp Kelly + Shannon filter + Taleb
    tail-risk adjustment and applies final investor risk-tolerance scaling.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary (trade_plans + portfolio_allocation + portfolio_report).
    """
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("[Stage 5] Portfolio Sizer 시작 — %s", ts)
    logger.info("=" * 60)

    regime = state.get("market_regime", "Normal_Market")
    risk_tol = state.get("risk_tolerance", "moderate")
    parsed_objective = state.get("parsed_objective", {})
    top_sectors = state.get("top_sectors", [])
    target_tickers = state.get("target_tickers", [])
    consensus_summary = state.get("consensus_summary", "")
    spg_outputs = state.get("spg_outputs", {})
    spg_report = state.get("spg_report", "")

    # Average ensemble_signal across Stage 4 orchestrator outputs
    orchestrator_outputs = state.get("orchestrator_outputs", [])
    if orchestrator_outputs:
        overall_signals = [out.get("ensemble_signal", 0.5) for out in orchestrator_outputs]
        ensemble_signal = sum(overall_signals) / len(overall_signals)
    else:
        ensemble_signal = 0.5

    audit_entries: List[str] = [f"[{ts}] Stage 5: Portfolio Sizer 시작"]
    errors: List[str] = []

    # System prompt including Brain Firmware text
    # [Variant Support] NO_FIRMWARE
    bypass_fw = bool(state.get("_variant_no_firmware", False))
    stage5_prompt = _get_stage5_system_prompt(
        regime, ensemble_signal, risk_tol,
        parsed_objective, top_sectors, target_tickers, spg_report,
        bypass_firmware=bypass_fw
    )
    logger.info(
        "[Stage 5] Brain Firmware 주입 완료 — %d chars "
        "(Regime=%s, Signal=%.3f, Risk=%s)",
        len(stage5_prompt),
        regime,
        ensemble_signal,
        risk_tol,
    )

    audit_entries.append(
        f"[{ts}] Brain Firmware 주입: {len(stage5_prompt):,}자"
    )
    logger.info("[Stage 5] 투자 목적: %s", parsed_objective.get("key_question", "N/A"))

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = _init_gemini_llm(temperature=0.15)

        # Append per-ticker signal lines for the LLM
        ticker_details = ""
        for out in orchestrator_outputs:
             ticker_details += f"  - {out['ticker']}: 통합시그널={out['ensemble_signal']:.3f}, 추천비중={out['final_position_size']:.3f}\n"

        # Assemble human message: objective + pipeline context (KO body unchanged)
        human_content = (
            f"현재 UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}\n\n"
            "━━ 투자 목적 (최우선 반영) ━━\n"
            f"핵심 질문:   {parsed_objective.get('key_question', '투자 방향 제시')}\n"
            f"투자 기간:   {parsed_objective.get('horizon', 'long')} "
            f"({parsed_objective.get('horizon_days', 252)}일)\n"
            f"전략 유형:   {parsed_objective.get('strategy', 'value_accumulation')}\n"
            f"대상 시장:   {parsed_objective.get('target_markets', [])}\n"
            f"원하는 답:   {parsed_objective.get('desired_outputs', [])}\n"
            f"원문 목적:   {state.get('investment_objective', '(없음)')[:300]}\n\n"
            "━━ 파이프라인 분석 결과 ━━\n"
            f"MarketRegime:      {regime}\n"
            f"전체 평균 Ensemble Signal: {ensemble_signal:.3f}\n"
            f"위험 성향:         {risk_tol}\n"
            f"선정 섹터:         {[s.get('sector_name','') if isinstance(s, dict) else s for s in top_sectors[:3]]}\n"
            f"분석 종목:         {target_tickers}\n\n"
            f"개별 종목 평가 (Stage 4):\n{ticker_details}\n"
            f"전략적 보정 리포트 (Stage 4.5):\n{spg_report[:800]}\n"
            f"거장 합의 요약:\n{consensus_summary[:800]}\n\n"
            "━━ Brain Firmware 기반 판단 지시 ━━\n"
            "위 모든 정보를 바탕으로 사용자의 핵심 질문에 직접 답하십시오.\n"
            "섹션 A(직접 답변)에서 종목별 구체적 수치(진입가·목표가·손절가·비중)를 제시하십시오.\n"
            "Firmware의 Thorp Kelly → Shannon 필터 → Taleb 보정 순서를 엄수하십시오.\n"
        )

        response = llm.invoke([
            SystemMessage(content=stage5_prompt),
            HumanMessage(content=human_content),
        ])

        report_text = response.content if hasattr(response, "content") else str(response)
        if isinstance(report_text, list):
            report_text = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in report_text
            )

        logger.info(
            "[Stage 5] 보고서 생성 완료 (%d자)",
            len(report_text),
        )
        audit_entries.append(
            f"[{ts}] Stage 5: 매매 계획 보고서 생성 완료 "
            f"({len(report_text):,}자)"
        )

        # Attempt lightweight JSON scrape for portfolio_allocation
        import json
        import re
        allocation: Dict[str, float] = {}
        try:
            alloc_match = re.search(
                r"portfolio_allocation[^\{]*(\{[^}]+\})", report_text, re.DOTALL
            )
            if alloc_match:
                allocation = json.loads(alloc_match.group(1))
        except Exception:
            pass

        # Fallback allocation when regex/JSON parse fails
        if not allocation and orchestrator_outputs:
             for out in orchestrator_outputs:
                 ticker = out["ticker"]
                 kelly_size = out.get("final_position_size", 0.0)
                 spg_floor = spg_outputs.get(ticker, {}).get("strategic_floor", 0.0)
                 # Stage 4.5 rule: max(engine Kelly, SPG floor)
                 final_target = max(kelly_size, spg_floor)
                 allocation[ticker] = round(final_target, 3)
                 logger.info(f"DEBUG: Stage 5 ticker {ticker}: Kelly={kelly_size}, SPG_Floor={spg_floor} -> Final={final_target}")
             
             total_stock = sum(allocation.values())
             if total_stock > 1.0:  # renormalize if weights sum above 1
                 for k in allocation:
                     allocation[k] = round(allocation[k] / total_stock, 3)
             allocation["CASH"] = round(max(0.0, 1.0 - sum(allocation.values())), 3)
        elif not allocation and top_sectors:
            n = min(len(top_sectors), 3)
            per_sector = round(0.8 / n, 2) if n > 0 else 0.0
            for s in top_sectors[:n]:
                sname = s.get("sector_name", f"Sector_{n}") if isinstance(s, dict) else s
                allocation[sname] = per_sector
            allocation["CASH"] = round(1.0 - per_sector * n, 2)

        logger.info(f"DEBUG: Stage 5 final allocation: {allocation}")
        return {
            "portfolio_allocation": allocation,
            "trade_plans": [],   # reserved for future TradePlanItem parsing
            "portfolio_report": report_text,
            "audit_log": audit_entries,
            "errors": errors,
        }

    except Exception as e:
        logger.error("[Stage 5] LLM 호출 실패: %s", e, exc_info=True)
        errors.append(f"Stage 5 LLM error: {e}")

        fallback_report = (
            f"[Stage 5 오류 — 안전 폴백]\n"
            f"  오류: {e}\n"
            f"  MarketRegime: {regime}\n"
            f"  Ensemble Signal: {ensemble_signal:.3f}\n"
            f"  핵심 질문: {parsed_objective.get('key_question', '(없음)')}\n"
            f"  조치: 시스템을 재시작하거나 오류를 수정한 후 재실행하십시오."
        )

        return {
            "portfolio_allocation": {"CASH": 1.0},
            "trade_plans": [],
            "portfolio_report": fallback_report,
            "audit_log": audit_entries,
            "errors": errors,
        }


# ══════════════════════════════════════════════════════════
# 10. Emergency Shutdown & Conditional Routing
# ══════════════════════════════════════════════════════════

def route_after_macro_pulse(
    state: AgentState,
) -> Literal["proceed", "critical_risk"]:
    """Post-Stage-1 routing: immediately terminate on critical risk, otherwise proceed.

    When a critical regime (Market_Crash, Systemic_Crash, etc.) is detected,
    skips subsequent stages and generates an emergency report.

    Args:
        state: Current AgentState.

    Returns:
        "proceed" or "critical_risk".
    """
    regime = state.get("market_regime", "Normal_Market")
    confidence = state.get("regime_confidence", 0.0)

    if regime in CRITICAL_SHUTDOWN_REGIMES and confidence >= 0.6:
        logger.warning(
            "CRITICAL RISK DETECTED: %s (confidence=%.2f) → 긴급 셧다운",
            regime,
            confidence,
        )
        return "critical_risk"

    return "proceed"


def emergency_shutdown_node(state: AgentState) -> Dict[str, Any]:
    """Emergency shutdown node.

    Executed upon detection of critical systemic risk.
    Halts all investment activity and generates an emergency report,
    following Taleb's black-swan protocol for immediate response.

    Args:
        state: Current AgentState.

    Returns:
        State update dictionary.
    """
    ts = datetime.now(timezone.utc).isoformat()
    regime = state.get("market_regime", "UNKNOWN")
    confidence = state.get("regime_confidence", 0.0)
    reasoning = state.get("macro_reasoning", "N/A")

    logger.critical(
        "EMERGENCY SHUTDOWN — Regime: %s, Confidence: %.2f",
        regime,
        confidence,
    )

    emergency_report = (
        "=" * 60 + "\n"
        "  H-PIOS v8.5 — EMERGENCY SHUTDOWN REPORT\n"
        "=" * 60 + "\n"
        f"  Timestamp   : {ts}\n"
        f"  Regime      : {regime}\n"
        f"  Confidence  : {confidence:.2f}\n"
        f"  Reasoning   : {reasoning}\n\n"
        "  ACTION REQUIRED:\n"
        "    1. 모든 신규 매수 포지션 즉시 중단\n"
        "    2. 기존 포지션 헤지 검토 (풋옵션 / 인버스 ETF)\n"
        "    3. 현금 비중 최대화\n"
        "    4. 지휘관의 수동 승인 없이 재개 금지\n"
        "=" * 60
    )

    return {
        "portfolio_allocation": {"CASH": 1.0},
        "trade_plans": [],
        "portfolio_report": emergency_report,
        "audit_log": [
            f"[{ts}] EMERGENCY SHUTDOWN: {regime} (conf={confidence:.2f})"
        ],
        "errors": [],
    }


# ══════════════════════════════════════════════════════════
# 11. Graph Construction & Compilation
# ══════════════════════════════════════════════════════════

def stage3_mock_node(state: AgentState) -> Dict[str, Any]:
    """[Experimental] Force-inject Stage 3 micro-analysis results.
    Metric names are aligned 100% with Master JSON to guarantee Stage 4 signal generation.
    """
    if state.get("market_data_payloads"):
        print(f"DEBUG: Stage 3 Mock Node - Using injected {len(state['market_data_payloads'])} payloads")
        res = {"market_data_payloads": state["market_data_payloads"]}
        if state.get("target_tickers"):
            res["target_tickers"] = state["target_tickers"]
        else:
            res["target_tickers"] = [p["ticker"] for p in state["market_data_payloads"]]
        return res

    payloads = [
        {
            "ticker": "005930.KS",  # Samsung Electronics — quality value mock
            "metrics": {
                "P/E_Ratio": 12.0, "Price_to_NCAV": 0.5, "ROIC_10yr_Avg": 0.22,
                "Debt_to_Equity": 0.25, "FCF_Yield": 0.09, "Gross_Margin_Volatility": 0.03,
                "PEG_Ratio": 0.8, "EPS_Growth_TTM": 0.15,
                "Expected_Value_of_Signal": 0.7, "Historical_Win_Rate_of_Signal": 0.6,
                "Price_Trend_R_Squared": 0.75, "SKEW_Index": 120, "Beta": 0.9
            }
        },
        {
            "ticker": "051910.KS",  # LG Chem — high-growth / distressed mock
            "metrics": {
                "P/E_Ratio": 45.0, "Price_to_NCAV": 2.5, "ROIC_10yr_Avg": 0.04,
                "Debt_to_Equity": 5.5, # SPG Safety Block Trigger
                "FCF_Yield": -0.12,    # SPG Safety Block Trigger
                "Gross_Margin_Volatility": 0.15,
                "PEG_Ratio": 3.5, "EPS_Growth_TTM": 0.45,
                "Expected_Value_of_Signal": 0.4, "Historical_Win_Rate_of_Signal": 0.45,
                "Price_Trend_R_Squared": 0.3, "SKEW_Index": 155, "Beta": 1.8
            }
        },
        {
            "ticker": "RISK_STOCK",  # strong quant signal but weak fundamentals (trap mock)
            "metrics": {
                "P/E_Ratio": 150.0, "Price_to_NCAV": 15.0, "ROIC_10yr_Avg": -0.05,
                "Debt_to_Equity": 12.5, # SPG Safety Block Trigger
                "FCF_Yield": -0.25,    # SPG Safety Block Trigger
                "Gross_Margin_Volatility": 0.35,
                "PEG_Ratio": 15.5, "EPS_Growth_TTM": 0.05,
                "Expected_Value_of_Signal": 0.95,  # best-looking quant signal (intentional trap)
                "Historical_Win_Rate_of_Signal": 0.85,
                "Price_Trend_R_Squared": 0.9, "SKEW_Index": 180, "Beta": 2.5
            }
        }
    ]
    print(f"DEBUG: Stage 3 Mock Node - Returning {len(payloads)} payloads")
    return {"market_data_payloads": payloads, "target_tickers": ["005930.KS", "051910.KS", "RISK_STOCK"]}

def build_graph():
    """Build the LangGraph for the H-PIOS v8.5 Field Command (5-stage) pipeline."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("stage3_mock", stage3_mock_node)
    workflow.add_node("consensus_engine", consensus_engine_node)
    workflow.add_node("strategic_policy_governor", strategic_policy_governor)
    workflow.add_node("portfolio_sizer", portfolio_sizer_agent)
    workflow.add_node("emergency_shutdown", emergency_shutdown_node)

    # Wire edges (START → mock inject for faster experiments)
    workflow.add_edge(START, "stage3_mock")
    workflow.add_edge("stage3_mock", "consensus_engine")
    workflow.add_edge("consensus_engine", "strategic_policy_governor")
    workflow.add_edge("strategic_policy_governor", "portfolio_sizer")

    workflow.add_edge("portfolio_sizer", END)
    workflow.add_edge("emergency_shutdown", END)

    # KO log: "graph build complete"
    logger.info("H-PIOS v8.5 Field Command 그래프 구성 완료")
    return workflow.compile()


# ══════════════════════════════════════════════════════════
# 12. Module-level graph instance (LangGraph Studio entrypoint)
# ══════════════════════════════════════════════════════════

# LangGraph Studio / `langgraph dev` discovers this `app` export automatically.
# Must match langgraph.json: "graphs": { "agent": "./CORE_GRAPH_agent_flow.py:app" }.
app = build_graph()


# ══════════════════════════════════════════════════════════
# 13. Entry Point
# ══════════════════════════════════════════════════════════

def run_pipeline(
    investment_objective: str = "",
    macro_context: str = "",
    risk_tolerance: str = "moderate",
) -> Dict[str, Any]:
    """Run the H-PIOS v8.5 Field Command pipeline end-to-end.

    Args:
        investment_objective: User goal / question in natural language (e.g. KOSPI long-term plan).
        macro_context: Optional user-supplied macro narrative (e.g. Fed decision, oil price).
        risk_tolerance: One of ``"conservative"``, ``"moderate"``, ``"aggressive"``.

    Returns:
        Final ``AgentState`` dictionary after graph execution.
    """
    logger.info("=" * 60)
    logger.info("H-PIOS v8.5 — FIELD COMMAND PIPELINE 시작")
    logger.info("=" * 60)
    logger.info(
        "투자 목적: %s",
        investment_objective[:200] if investment_objective else "(없음)",
    )
    logger.info(
        "매크로 컨텍스트: %s",
        macro_context[:200] if macro_context else "(없음)",
    )
    logger.info("위험 성향: %s", risk_tolerance)

    graph = build_graph()

    # Initial state
    initial_state: Dict[str, Any] = {
        "investment_objective": investment_objective,
        "macro_context": macro_context,
        "risk_tolerance": risk_tolerance,
        "parsed_objective": {},
        "market_regime": None,
        "regime_confidence": 0.0,
        "macro_risk_level": "",
        "macro_reasoning": "",
        "macro_key_factors": [],
        "macro_search_results": [],
        "live_macro_snapshot": {},
        "macro_pulse_report": "",
        "top_sectors": [],
        "sector_analysis": "",
        "target_tickers": [],
        "market_data_payloads": [],
        "micro_analysis": "",
        "orchestrator_outputs": [],
        "consensus_summary": "",
        "spg_outputs": {},
        "spg_report": "",
        "portfolio_allocation": {},
        "trade_plans": [],
        "portfolio_report": "",
        "errors": [],
        "audit_log": [],

        # Default experiment / ablation variant flags
        "bypass_spg": False,
        "_variant_no_firmware": False,
        "_variant_no_spg": False,
        "_variant_no_edges": False,
        "_variant_no_taleb": False,
        "_variant_no_risk": False,
        "_variant_no_regime": False,
        "_variant_flat_ensemble": False,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        # KO log: "pipeline execution failed"
        logger.critical("파이프라인 실행 실패: %s", e, exc_info=True)
        initial_state["errors"] = [f"Pipeline execution failed: {e}"]
        return initial_state

    # Summary logs (KO messages for operator readability)
    logger.info("=" * 60)
    logger.info("H-PIOS v8.5 — PIPELINE 완료")
    logger.info("=" * 60)
    parsed = final_state.get("parsed_objective", {})
    logger.info("  투자 목적     : %s", parsed.get("key_question", "(없음)"))
    logger.info("  투자 기간     : %s (%s일)", parsed.get("horizon"), parsed.get("horizon_days"))
    logger.info("  Market Regime : %s", final_state.get("market_regime"))
    logger.info("  Confidence    : %.2f", final_state.get("regime_confidence", 0))
    logger.info("  Risk Level    : %s", final_state.get("macro_risk_level", ""))
    logger.info(
        "  Portfolio     : %s", final_state.get("portfolio_allocation")
    )
    logger.info("  Trade Plans   : %d건", len(final_state.get("trade_plans", [])))
    logger.info("  Macro Report  : %d자", len(final_state.get("macro_pulse_report", "")))
    logger.info("  Errors        : %d건", len(final_state.get("errors", [])))

    return final_state


# ══════════════════════════════════════════════════════════
# 13. CLI entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="H-PIOS v8.5 Field Command Pipeline",
    )
    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="",
        help=(
            "투자 목적/질문 (자연어). "
            "예: '1년 이상 장투할건데 코스피 유망 종목을 얼마일때 몇주 사서 언제 익절/손절할까?'"
        ),
    )
    # Back-compat: map --briefing / -b to the same field as --objective
    parser.add_argument(
        "--briefing",
        "-b",
        type=str,
        default="",
        help="(레거시) --objective 와 동일. 하위 호환을 위해 유지.",
    )
    parser.add_argument(
        "--macro_context",
        "-m",
        type=str,
        default="",
        help=(
            "현재 매크로 상황 텍스트 입력 (선택). "
            "예: '연준이 금리를 동결했고 유가가 배럴당 $95를 돌파했다.'"
        ),
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=str,
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="위험 성향 (기본: moderate)",
    )

    args = parser.parse_args()

    # Prefer --objective; fall back to legacy --briefing
    objective = args.objective or args.briefing

    result = run_pipeline(
        investment_objective=objective,
        macro_context=args.macro_context,
        risk_tolerance=args.risk,
    )

    # Full report is in result dict; CLI only logs completion status
    logger.info("CLI Pipeline Execution Finished.")
    if result.get("errors"):
        logger.error("Errors encountered: %d", len(result["errors"]))
