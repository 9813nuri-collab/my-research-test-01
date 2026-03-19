"""
H-PIOS v8.5 — Data Models (Pydantic v2 Strict)
=================================================
12인 투자 거장의 '지식 유전자' JSON을 100 % 수용하는 불변 스키마 계층과,
실전 헤지펀드 수준의 상태 기반(Stateful) 추론을 지원하는 런타임 모델을 정의합니다.

설계 원칙
---------
1. **Strict Mapping** — JSON의 모든 필드를 Optional/Union 없이 정밀 매핑.
2. **Dynamic Prep** — 런타임 상태(EngineState)로 연속적 시장 흐름을 기억.
3. **Input Isolation** — MarketDataPayload / NLPContextPayload 로 데이터 소스 분리.
4. **Semantic NLP Ready** — embedding_vectors, cosine_similarity_scores 뼈대 마련.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────
# 0. 공통 열거형 (Common Enumerations)
# ──────────────────────────────────────────────────────────

class RelationshipType(str, Enum):
    """LogicalEdge 관계 유형.

    JSON에 명시된 5가지 엣지 유형을 엄밀하게 열거합니다.
    """
    OVERRIDE = "Override"
    SYNERGIZE_WITH = "Synergize_With"
    SUPPRESS = "Suppress"
    GLOBAL_WEIGHT_ADJUSTER = "Global_Weight_Adjuster"
    MASTER_OVERRIDE = "Master_Override"
    # New continuous tension types:
    CONTINUOUS_DISCOUNT = "Continuous_Discount"
    CONTINUOUS_SYNERGY = "Continuous_Synergy"
    CONTINUOUS_MACRO_ADJUST = "Continuous_Macro_Adjust"
    CONTINUOUS_MASTER_DISCOUNT = "Continuous_Master_Discount"


class MarketRegime(str, Enum):
    """시장 국면(Regime) 열거.

    12인 거장 JSON의 regime_performance 및 Condition_Regime에서
    참조되는 모든 국면을 포함합니다.
    """
    BEAR_MARKET = "Bear_Market"
    BULL_MARKET = "Bull_Market"
    LOW_INFLATION = "Low_Inflation"
    HIGH_INFLATION = "High_Inflation"
    ALL_WEATHER = "All_Weather"
    INNOVATION_CYCLE = "Innovation_Cycle"
    STAGFLATION = "Stagflation"
    EARLY_BULL_MARKET = "Early_Bull_Market"
    RECESSION = "Recession"
    LATE_BULL_MARKET = "Late_Bull_Market"
    BUBBLE_BURST = "Bubble_Burst"
    DELEVERAGING = "Deleveraging"
    REFLATION = "Reflation"
    MARKET_BOTTOM = "Market_Bottom"
    MARKET_TOP = "Market_Top"
    HIGH_VOLATILITY = "High_Volatility"
    SIDEWAYS_MARKET = "Sideways_Market"
    TAIL_RISK_EVENT = "Tail_Risk_Event"
    NORMAL_MARKET = "Normal_Market"
    HIGH_NOISE = "High_Noise"
    STRONG_TREND = "Strong_Trend"
    # Condition_Regime에서만 등장하는 국면들
    MARKET_CRASH = "Market_Crash"
    LIQUIDITY_CRISIS = "Liquidity_Crisis"
    HIGH_FRAUD_RISK_SECTOR = "High_Fraud_Risk_Sector"
    LIQUIDITY_EXCESS = "Liquidity_Excess"
    TIGHTENING = "Tightening"
    PANIC_SELLING = "Panic_Selling"
    CAPITULATION = "Capitulation"
    SYSTEMIC_CRASH = "Systemic_Crash"
    HIGH_NOISE_MARKET = "High_Noise_Market"
    SIDEWAYS = "Sideways"
    CREDIT_CRUNCH = "Credit_Crunch"


# ──────────────────────────────────────────────────────────
# 1. Intelligence Structure 하위 모델
# ──────────────────────────────────────────────────────────

class QuantMetric(BaseModel):
    """Step 1에서 사용되는 개별 정량 지표.

    Attributes:
        metric: 지표명 (예: Price_to_NCAV, ROIC_10yr_Avg)
        operator: 비교 연산자 (예: '<', '>', 'Trend', 'Volatility')
        threshold: 조건 임계값
        weight: 해당 지표의 가중치 (0.0 ~ 1.0)
        logic_formula: 조건부 로직 수식 문자열
        unit: 단위 (옵셔널, 예: 'Sigma')
    """
    metric: str = Field(..., description="정량 지표 식별자 (예: Price_to_NCAV)")
    operator: str = Field(..., description="비교 연산자 (예: '<', '>', 'Trend', 'Volatility')")
    threshold: float = Field(..., description="조건 임계값")
    weight: float = Field(..., ge=0.0, le=1.0, description="지표 가중치")
    logic_formula: Optional[str] = Field(
        None,
        description="조건부 로직 수식 문자열. Safe Evaluator로 해석됨."
    )
    unit: Optional[str] = Field(None, description="지표 단위 (예: 'Sigma')")


class QualitativeScenario(BaseModel):
    """Step 2에서 사용되는 정성 시나리오 단위.

    NLP 기반으로 뉴스/리포트에서 조건을 탐지하고,
    score_modifier를 적용합니다.

    Attributes:
        condition: 시나리오 발동 조건 (자연어 기술)
        score_modifier: 점수 보정치 (음수 가능)
        context: 시나리오의 금융 공학적 맥락 설명
        keywords: 시나리오 탐지용 키워드 목록
        action: 특수 동작 플래그 (예: 'CRITICAL_INHIBIT')
    """
    condition: str = Field(..., description="시나리오 발동 조건 (자연어)")
    score_modifier: float = Field(..., description="점수 보정치 (음수 = 억제)")
    context: str = Field(..., description="시나리오의 금융 공학적 맥락")
    keywords: List[str] = Field(default_factory=list, description="탐지용 키워드 리스트")
    action: Optional[str] = Field(
        None,
        description="특수 동작 플래그 (예: 'CRITICAL_INHIBIT')"
    )


class Step1QuantitativeAnalysis(BaseModel):
    """정량 분석 단계.

    마스터 JSON의 Intelligence_Structure.Step_1_Quantitative_Analysis를
    완전히 매핑합니다. 복수의 정량 지표를 조건부 로직으로 평가합니다.

    Attributes:
        metrics: 정량 지표 리스트
        output_variable: 이 단계의 출력 변수명
    """
    metrics: List[QuantMetric] = Field(..., min_length=1, description="정량 지표 리스트")
    output_variable: str = Field(..., description="출력 변수명 (예: L_quant_graham)")


class Step2QualitativeContext(BaseModel):
    """정성 평가 단계.

    NLP 신뢰도 임계값과 시나리오 목록으로 구성되며,
    Sentence-Transformers를 통한 코사인 유사도 기반 평가를 지원합니다.

    Attributes:
        nlp_confidence_threshold: NLP 모델 신뢰도 하한
        scenarios: 정성 시나리오 리스트
        output_variable: 이 단계의 출력 변수명
    """
    nlp_confidence_threshold: float = Field(
        ..., ge=0.0, le=1.0,
        description="NLP 모델 신뢰도 하한 (이하이면 시나리오 미반영)"
    )
    scenarios: List[QualitativeScenario] = Field(..., description="정성 시나리오 리스트")
    output_variable: str = Field(..., description="출력 변수명 (예: L_qual_graham)")


class Step3StatisticalCorrection(BaseModel):
    """통계 보정 단계.

    Step 1 + Step 2의 결과를 수식으로 결합하고,
    historical_win_rate, decay_factor, regime_performance로 보정합니다.

    수식(formula)은 Safe Expression Evaluator로 런타임에 동적 해석됩니다.

    Attributes:
        formula: 보정 수식 문자열 (Safe Eval 대상)
        constants: 수식에 사용되는 상수 딕셔너리
    """
    formula: str = Field(..., description="통계 보정 수식 (런타임 Safe Eval)")
    constants: Dict[str, Any] = Field(
        ...,
        description=(
            "수식 상수. 최소 historical_win_rate, decay_factor를 포함. "
            "regime_performance는 국면별 승률 딕셔너리."
        )
    )

    @property
    def historical_win_rate(self) -> float:
        """JSON에 명시된 초기 historical_win_rate를 반환.

        이 값은 default로만 취급되며, 런타임에 update_rolling_win_rate()로 갱신 가능.
        """
        return float(self.constants.get("historical_win_rate", 0.5))

    @property
    def decay_factor(self) -> float:
        """감쇠 계수(decay_factor)를 반환."""
        return float(self.constants.get("decay_factor", 1.0))

    @property
    def regime_performance(self) -> Dict[str, Any]:
        """국면별 성과 딕셔너리를 반환."""
        return self.constants.get("regime_performance", {})


# ──────────────────────────────────────────────────────────
# 2. Intelligence Structure & Final Output
# ──────────────────────────────────────────────────────────

class IntelligenceStructure(BaseModel):
    """3-Step 지능형 분석 파이프라인.

    각 마스터 노드의 핵심 추론 구조를 캡슐화합니다:
    1. 정량 분석 → 2. 정성 맥락 평가 → 3. 통계적 보정

    Attributes:
        Step_1_Quantitative_Analysis: 정량 분석 단계
        Step_2_Qualitative_Context: 정성 평가 단계
        Step_3_Statistical_Correction: 통계 보정 단계
    """
    Step_1_Quantitative_Analysis: Step1QuantitativeAnalysis = Field(
        ..., description="1단계: 정량 분석"
    )
    Step_2_Qualitative_Context: Step2QualitativeContext = Field(
        ..., description="2단계: 정성 맥락 평가"
    )
    Step_3_Statistical_Correction: Step3StatisticalCorrection = Field(
        ..., description="3단계: 통계적 보정"
    )


class FinalOutput(BaseModel):
    """마스터 노드의 최종 출력 메타데이터.

    Attributes:
        Score_Variable: 최종 점수 변수명
        Signal_Type: 시그널 유형 (예: 'Accumulate_Deep_Value')
        State_Flags: 상태 플래그 목록
        Range: 출력 범위 문자열 (참조용)
    """
    Score_Variable: str = Field(..., description="최종 점수 변수명")
    Signal_Type: str = Field(..., description="시그널 유형")
    State_Flags: List[str] = Field(default_factory=list, description="상태 플래그 목록")
    Range: str = Field(..., description="출력 범위 (참조용 문자열)")


# ──────────────────────────────────────────────────────────
# 3. MasterNode — 12인 거장 개별 노드
# ──────────────────────────────────────────────────────────

class MasterNode(BaseModel):
    """12인 투자 거장 마스터 노드.

    JSON의 Nodes 배열 내 개별 객체 구조를 **100 %** 수용합니다.
    각 노드는 독립적인 3-Step 추론 파이프라인과 최종 출력 메타를 포함합니다.

    Attributes:
        Node_ID: 고유 식별자 (예: VAL_GRAHAM_001)
        Master: 거장 이름
        Core_Concept: 핵심 투자 철학
        Full_Description: 상세 설명 (한글)
        Intelligence_Structure: 3-Step 분석 구조
        Final_Output: 최종 출력 메타
    """
    Node_ID: str = Field(..., description="고유 노드 식별자 (예: VAL_GRAHAM_001)")
    Master: str = Field(..., description="투자 거장 이름 (예: Benjamin Graham)")
    Core_Concept: str = Field(..., description="핵심 투자 철학 (예: Margin_of_Safety)")
    Full_Description: str = Field(..., description="상세 설명")
    Intelligence_Structure: IntelligenceStructure = Field(
        ..., description="3-Step 지능형 분석 파이프라인"
    )
    Final_Output: FinalOutput = Field(..., description="최종 출력 메타데이터")


# ──────────────────────────────────────────────────────────
# 4. LogicalEdge — 노드 간 논리적 연결
# ──────────────────────────────────────────────────────────

class LogicalEdge(BaseModel):
    """노드 간 논리적 연결(시냅스).

    JSON의 Logical_Edges 배열 내 객체 구조를 수용합니다.
    GraphOrchestrator가 위상 정렬 및 시그널 해소 시 참조합니다.

    Attributes:
        Edge_ID: 엣지 고유 식별자
        Source: 소스 노드 ID
        Target: 타겟 노드 ID (또는 'ALL_*' 와일드카드)
        Relationship_Type: 관계 유형 Enum (Override, Synergize 등)
        Condition_Regime: 조건 국면 Enum 목록
        Logic: 논리 수식 문자열
    """
    Edge_ID: str = Field(..., description="엣지 고유 식별자")
    Source: str = Field(..., description="소스 노드 ID")
    Target: str = Field(
        ...,
        description="타겟 노드 ID 또는 와일드카드 (예: ALL_OTHER_ENGINES)"
    )
    Relationship_Type: RelationshipType = Field(
        ...,
        description="관계 유형 Enum (Override, Synergize_With, Suppress, Master_Override, Global_Weight_Adjuster)"
    )
    Condition_Regime: List[MarketRegime] = Field(
        default_factory=list,
        description="이 엣지가 활성화되는 시장 국면 Enum 목록"
    )
    Logic: str = Field(..., description="논리 수식 문자열 (Safe Eval)")


# ──────────────────────────────────────────────────────────
# 5. MasterEngineConfig — JSON 최상위 래퍼
# ──────────────────────────────────────────────────────────

class MasterEngineConfig(BaseModel):
    """마스터 JSON 파일의 최상위 구조를 수용하는 래퍼 모델.

    4개의 엔진 도메인(Value, Growth, Macro, Risk) 각각의
    JSON 파일 전체를 이 모델 하나로 파싱합니다.

    Attributes:
        Ontology_Schema: 스키마 버전 식별자
        Engine_Domain: 엔진 도메인 (예: Value_Investment)
        Last_Update: 최종 갱신 일자
        Nodes: 마스터 노드 리스트
        Logical_Edges: 논리적 엣지 리스트
    """
    Ontology_Schema: str = Field(..., description="스키마 버전 (예: H-PIOS_v8.5_GraphRAG_Master)")
    Engine_Domain: str = Field(..., description="엔진 도메인 (예: Value_Investment)")
    Last_Update: str = Field(..., description="최종 갱신 일자 (YYYY-MM-DD)")
    Nodes: List[MasterNode] = Field(..., min_length=1, description="마스터 노드 리스트")
    Logical_Edges: List[LogicalEdge] = Field(
        default_factory=list, description="논리적 엣지 리스트"
    )


# ──────────────────────────────────────────────────────────
# 6. EngineState — 상태 기반(Stateful) 추론 지원
# ──────────────────────────────────────────────────────────

class EngineState(BaseModel):
    """엔진 런타임 상태 (Stateful Inference Support).

    시장의 연속적 흐름을 기억하기 위해 롤링 통계량과 이전 국면 정보를 저장합니다.
    각 거장 노드(또는 전역)에 대해 독립적인 인스턴스가 유지됩니다.

    Attributes:
        node_id: 연관 노드 식별자 (없으면 글로벌 상태)
        rolling_volatility: 롤링 변동성 (예: 20일 이동 표준편차)
        signal_persistence: 시그널 지속 카운터 (연속 N회 동일 방향)
        previous_regime: 직전 시장 국면 (MarketRegime Enum)
        rolling_drawdown: 롤링 최대 낙폭
        rolling_win_rate: 동적으로 갱신되는 롤링 승률
        cumulative_pnl: 누적 손익
        signal_history: 최근 시그널 이력 (시계열)
        last_score: 최근 산출 점수
        last_updated: 마지막 상태 갱신 시각
        metadata: 확장용 메타데이터 딕셔너리
    """
    node_id: Optional[str] = Field(None, description="연관 노드 ID (없으면 글로벌)")
    rolling_volatility: float = Field(0.0, ge=0.0, description="롤링 변동성")
    signal_persistence: int = Field(0, ge=0, description="시그널 지속 카운터")
    previous_regime: Optional[MarketRegime] = Field(None, description="직전 시장 국면 (MarketRegime Enum)")
    rolling_drawdown: float = Field(0.0, le=0.0, description="롤링 최대 낙폭 (≤ 0)")
    rolling_win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="동적 롤링 승률 (None이면 JSON 초기값 사용)"
    )
    cumulative_pnl: float = Field(0.0, description="누적 손익")
    signal_history: List[float] = Field(
        default_factory=list, description="최근 시그널 점수 이력"
    )
    last_score: Optional[float] = Field(None, description="최근 산출 점수")
    last_updated: Optional[datetime] = Field(None, description="마지막 갱신 시각")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="확장용 메타데이터")


class PerformanceState(BaseModel):
    """백테스트/라이브 트레이딩 성과 추적 상태.

    update_rolling_win_rate() 메서드에 주입될 성과 데이터를 구조화합니다.

    Attributes:
        total_trades: 총 거래 수
        winning_trades: 승리 거래 수
        losing_trades: 패배 거래 수
        avg_win_return: 평균 수익률 (승리 시)
        avg_loss_return: 평균 손실률 (패배 시)
        sharpe_ratio: 샤프 비율
        max_drawdown: 최대 낙폭
        calmar_ratio: 칼마 비율
        win_loss_ratio: 손익비 (avg_win / avg_loss)
    """
    total_trades: int = Field(0, ge=0, description="총 거래 수")
    winning_trades: int = Field(0, ge=0, description="승리 거래 수")
    losing_trades: int = Field(0, ge=0, description="패배 거래 수")
    avg_win_return: float = Field(0.0, description="평균 수익률 (승리)")
    avg_loss_return: float = Field(0.0, description="평균 손실률 (패배)")
    sharpe_ratio: Optional[float] = Field(None, description="샤프 비율")
    max_drawdown: float = Field(0.0, le=0.0, description="최대 낙폭 (≤ 0)")
    calmar_ratio: Optional[float] = Field(None, description="칼마 비율")
    win_loss_ratio: Optional[float] = Field(None, ge=0.0, description="손익비")

    @property
    def calculated_win_rate(self) -> Optional[float]:
        """거래 이력으로부터 승률을 산출.

        Returns:
            총 거래 > 0이면 승률, 아니면 None
        """
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return None


# ──────────────────────────────────────────────────────────
# 7. Input Interface — 데이터 소스 분리 페이로드
# ──────────────────────────────────────────────────────────

class MarketDataPayload(BaseModel):
    """재무/매크로 정량 데이터 입력 페이로드.

    데이터 소스(API, DB, CSV 등)로부터 엔진을 분리합니다.
    모든 정량 지표를 metric_name → float 매핑으로 전달합니다.

    Attributes:
        ticker: 종목 코드 (옵셔널)
        timestamp: 데이터 시점
        metrics: 지표명 → 수치값 매핑 딕셔너리
        current_regime: 현재 시장 국면
        regime_confidence: 국면 판별 신뢰도
        metadata: 확장용 메타데이터
    """
    ticker: Optional[str] = Field(None, description="종목 코드 (예: AAPL, 005930.KS)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="데이터 시점 (UTC)"
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="지표명 → 수치값 매핑 (예: {'Price_to_NCAV': 0.55, 'P/E_Ratio': 12.3})"
    )
    current_regime: Optional[MarketRegime] = Field(None, description="현재 시장 국면 (MarketRegime Enum)")
    regime_confidence: float = Field(
        0.5, ge=0.0, le=1.0,
        description="국면 판별 신뢰도"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="확장용 메타데이터")


class NLPContextPayload(BaseModel):
    """자연어 처리(NLP) 기반 정성 데이터 입력 페이로드.

    단순 키워드 매칭을 넘어, Sentence-Transformers 등으로
    JSON의 keywords와 뉴스 문맥 간 코사인 유사도를 계산하여
    Step 2 정성 평가에 반영할 수 있는 뼈대를 제공합니다.

    Attributes:
        source: 데이터 출처 (예: 'Reuters', 'Bloomberg')
        timestamp: 데이터 시점
        raw_texts: 원시 텍스트 리스트 (뉴스, 리포트 등)
        detected_keywords: 탐지된 키워드 매핑 (keyword → 빈도/신뢰도)
        embedding_vectors: 텍스트 임베딩 벡터 리스트
            각 벡터는 Sentence-Transformers 등의 출력.
            raw_texts와 1:1 대응.
        semantic_similarity_scores: 시나리오별 코사인 유사도 매핑
            scenario_condition → 유사도 점수.
            향후 각 시나리오의 keywords와 뉴스 임베딩 간
            코사인 유사도를 사전 계산하여 주입합니다.
        nlp_model_confidence: NLP 모델의 전역 신뢰도
        sentiment_scores: 감성 분석 점수 매핑 (텍스트 인덱스 → 점수)
        metadata: 확장용 메타데이터
    """
    source: Optional[str] = Field(None, description="데이터 출처 (예: Reuters)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="데이터 시점 (UTC)"
    )
    raw_texts: List[str] = Field(
        default_factory=list,
        description="원시 텍스트 리스트 (뉴스, 리포트)"
    )
    detected_keywords: Dict[str, float] = Field(
        default_factory=dict,
        description="탐지된 키워드 → 빈도 또는 신뢰도 매핑"
    )
    embedding_vectors: List[List[float]] = Field(
        default_factory=list,
        description=(
            "텍스트 임베딩 벡터 리스트. "
            "Sentence-Transformers 등의 출력. raw_texts와 1:1 대응."
        )
    )
    semantic_similarity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "시나리오 condition → 코사인 유사도 점수 매핑. "
            "JSON keywords와 뉴스 임베딩 간 유사도를 사전 계산하여 주입."
        )
    )
    nlp_model_confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="NLP 모델 전역 신뢰도"
    )
    sentiment_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="텍스트 식별자 → 감성 분석 점수 매핑 (-1.0 ~ 1.0)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="확장용 메타데이터")

    @field_validator("embedding_vectors")
    @classmethod
    def validate_embedding_dimensions(cls, v: List[List[float]]) -> List[List[float]]:
        """모든 임베딩 벡터의 차원이 동일한지 검증합니다."""
        if len(v) > 1:
            dim = len(v[0])
            for i, vec in enumerate(v[1:], start=1):
                if len(vec) != dim:
                    raise ValueError(
                        f"임베딩 벡터 차원 불일치: v[0]={dim}, v[{i}]={len(vec)}"
                    )
        return v


# ──────────────────────────────────────────────────────────
# 8. Orchestrator 입출력 모델
# ──────────────────────────────────────────────────────────

class NodeExecutionResult(BaseModel):
    """개별 노드 실행 결과.

    AbstractMasterEngine.execute()의 반환 타입입니다.

    Attributes:
        node_id: 실행된 노드 ID
        master: 거장 이름
        raw_score: 정규화 전 원시 점수
        normalized_score: 0.0 ~ 1.0 정규화 점수
        signal_type: 시그널 유형
        active_state_flag: 활성 상태 플래그
        step1_output: Step 1 정량 분석 결과
        step2_output: Step 2 정성 평가 결과
        step3_output: Step 3 보정 결과
        used_win_rate: 실제 적용된 승률 (동적 갱신값 또는 초기값)
        metadata: 확장용 메타데이터
    """
    node_id: str = Field(..., description="실행된 노드 ID")
    master: str = Field(..., description="거장 이름")
    raw_score: float = Field(..., description="정규화 전 원시 점수")
    normalized_score: float = Field(
        ..., ge=0.0, le=1.0, description="0.0 ~ 1.0 정규화 점수"
    )
    signal_type: str = Field(..., description="시그널 유형")
    active_state_flag: Optional[str] = Field(None, description="활성 상태 플래그")
    step1_output: float = Field(..., description="Step 1 정량 분석 결과값")
    step2_output: float = Field(..., description="Step 2 정성 평가 결과값")
    step3_output: float = Field(..., description="Step 3 보정 결과값")
    used_win_rate: float = Field(..., description="적용된 승률")
    internal_monologue: Optional[str] = Field(None, description="거장의 내적 독백/추론 논리 (샌드박스용)")
    philosophical_weight: float = Field(1.0, description="토론 내 발언 비중/권위")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="확장용 메타데이터")


class OrchestratorOutput(BaseModel):
    """GraphOrchestrator의 최종 출력.

    12인 거장의 앙상블 결과와 최종 포지션 사이징을 포함합니다.

    Attributes:
        timestamp: 실행 시각
        ticker: 대상 종목 코드
        node_results: 노드별 실행 결과 딕셔너리
        ensemble_signal: 앙상블 종합 시그널 (0.0 ~ 1.0)
        final_position_size: Thorp Kelly 기준 최종 포지션 비율
        override_active: 마스터 오버라이드 활성 여부
        override_source: 오버라이드 소스 노드 ID
        active_regime: 현재 활성 국면 (MarketRegime Enum)
        confidence: 전체 시그널 신뢰도
        audit_log: 실행 감사 로그
    """
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="실행 시각 (UTC)"
    )
    ticker: Optional[str] = Field(None, description="대상 종목 코드")
    node_results: Dict[str, NodeExecutionResult] = Field(
        ..., description="노드 ID → 실행 결과 매핑"
    )
    ensemble_signal: float = Field(
        ..., ge=0.0, le=1.0, description="앙상블 종합 시그널"
    )
    final_position_size: float = Field(
        ..., ge=0.0, le=1.0, description="최종 포지션 비율 (0.0 ~ 1.0)"
    )
    override_active: bool = Field(False, description="마스터 오버라이드 활성 여부")
    override_source: Optional[str] = Field(None, description="오버라이드 소스 노드 ID")
    active_regime: Optional[MarketRegime] = Field(None, description="현재 활성 국면 (MarketRegime Enum)")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="전체 시그널 신뢰도"
    )
    spg_veto_active: bool = Field(False, description="SPG 거부권 발동 여부")
    spg_report: Dict[str, Any] = Field(default_factory=dict, description="SPG 상세 판정 리포트")
    deliberation_logs: List[str] = Field(default_factory=list, description="샌드박스 토론 로그")
    tension_score: float = Field(0.0, description="관점 간 충돌/긴장도 (0.0~1.0)")
    audit_log: List[str] = Field(
        default_factory=list, description="실행 감사 로그 (단계별 추적)"
    )
