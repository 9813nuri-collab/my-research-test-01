"""
H-PIOS v8.5 — Data Models (Pydantic v2 Strict)
=================================================
Defines immutable schema layers that fully accommodate the 'knowledge gene' JSON
of 12 investment masters, along with runtime models supporting hedge-fund-grade
stateful inference.

Design Principles
---------
1. **Strict Mapping** — Precise mapping of every JSON field without Optional/Union.
2. **Dynamic Prep** — Remembers continuous market flow via runtime state (EngineState).
3. **Input Isolation** — Separates data sources into MarketDataPayload / NLPContextPayload.
4. **Semantic NLP Ready** — Provides scaffolding for embedding_vectors and cosine_similarity_scores.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────
# 0. Common Enumerations
# ──────────────────────────────────────────────────────────

class RelationshipType(str, Enum):
    """LogicalEdge relationship types.

    Strictly enumerates the five edge types specified in the JSON.
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
    """Market regime enumeration.

    Covers all regimes referenced in the 12-master JSON's
    regime_performance and Condition_Regime fields.
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
    # Regimes appearing only in Condition_Regime
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
# 1. Intelligence Structure Sub-models
# ──────────────────────────────────────────────────────────

class QuantMetric(BaseModel):
    """Individual quantitative metric used in Step 1.

    Attributes:
        metric: Metric name (e.g., Price_to_NCAV, ROIC_10yr_Avg)
        operator: Comparison operator (e.g., '<', '>', 'Trend', 'Volatility')
        threshold: Condition threshold value
        weight: Weight of this metric (0.0–1.0)
        logic_formula: Conditional logic formula string
        unit: Unit (optional, e.g., 'Sigma')
    """
    metric: str = Field(..., description="Quantitative metric identifier (e.g., Price_to_NCAV)")
    operator: str = Field(..., description="Comparison operator (e.g., '<', '>', 'Trend', 'Volatility')")
    threshold: float = Field(..., description="Condition threshold value")
    weight: float = Field(..., ge=0.0, le=1.0, description="Metric weight")
    logic_formula: Optional[str] = Field(
        None,
        description="Conditional logic formula string; interpreted by the Safe Evaluator."
    )
    unit: Optional[str] = Field(None, description="Metric unit (e.g., 'Sigma')")


class QualitativeScenario(BaseModel):
    """Qualitative scenario unit used in Step 2.

    Detects conditions from news/reports via NLP and applies
    a score_modifier accordingly.

    Attributes:
        condition: Scenario trigger condition (natural-language description)
        score_modifier: Score adjustment value (can be negative)
        context: Financial-engineering context of the scenario
        keywords: List of keywords for scenario detection
        action: Special action flag (e.g., 'CRITICAL_INHIBIT')
    """
    condition: str = Field(..., description="Scenario trigger condition (natural language)")
    score_modifier: float = Field(..., description="Score adjustment value (negative = suppression)")
    context: str = Field(..., description="Financial-engineering context of the scenario")
    keywords: List[str] = Field(default_factory=list, description="Detection keyword list")
    action: Optional[str] = Field(
        None,
        description="Special action flag (e.g., 'CRITICAL_INHIBIT')"
    )


class Step1QuantitativeAnalysis(BaseModel):
    """Quantitative analysis stage.

    Fully maps the master JSON's Intelligence_Structure.Step_1_Quantitative_Analysis.
    Evaluates multiple quantitative metrics via conditional logic.

    Attributes:
        metrics: List of quantitative metrics
        output_variable: Output variable name for this stage
    """
    metrics: List[QuantMetric] = Field(..., min_length=1, description="List of quantitative metrics")
    output_variable: str = Field(..., description="Output variable name (e.g., L_quant_graham)")


class Step2QualitativeContext(BaseModel):
    """Qualitative evaluation stage.

    Composed of an NLP confidence threshold and a scenario list;
    supports cosine-similarity-based evaluation via Sentence-Transformers.

    Attributes:
        nlp_confidence_threshold: Lower bound for NLP model confidence
        scenarios: List of qualitative scenarios
        output_variable: Output variable name for this stage
    """
    nlp_confidence_threshold: float = Field(
        ..., ge=0.0, le=1.0,
        description="NLP model confidence lower bound (scenarios ignored below this threshold)"
    )
    scenarios: List[QualitativeScenario] = Field(..., description="List of qualitative scenarios")
    output_variable: str = Field(..., description="Output variable name (e.g., L_qual_graham)")


class Step3StatisticalCorrection(BaseModel):
    """Statistical correction stage.

    Combines Step 1 and Step 2 results via a formula and applies corrections
    using historical_win_rate, decay_factor, and regime_performance.

    The formula is dynamically interpreted at runtime by the Safe Expression Evaluator.

    Attributes:
        formula: Correction formula string (target of Safe Eval)
        constants: Dictionary of constants used in the formula
    """
    formula: str = Field(..., description="Statistical correction formula (runtime Safe Eval)")
    constants: Dict[str, Any] = Field(
        ...,
        description=(
            "Formula constants. Must include at least historical_win_rate and decay_factor. "
            "regime_performance is a per-regime win-rate dictionary."
        )
    )

    @property
    def historical_win_rate(self) -> float:
        """Return the initial historical_win_rate specified in the JSON.

        This value is treated as a default and can be updated at runtime
        via update_rolling_win_rate().
        """
        return float(self.constants.get("historical_win_rate", 0.5))

    @property
    def decay_factor(self) -> float:
        """Return the decay factor."""
        return float(self.constants.get("decay_factor", 1.0))

    @property
    def regime_performance(self) -> Dict[str, Any]:
        """Return the per-regime performance dictionary."""
        return self.constants.get("regime_performance", {})


# ──────────────────────────────────────────────────────────
# 2. Intelligence Structure & Final Output
# ──────────────────────────────────────────────────────────

class IntelligenceStructure(BaseModel):
    """Three-step intelligent analysis pipeline.

    Encapsulates the core inference structure of each master node:
    1. Quantitative analysis -> 2. Qualitative context evaluation -> 3. Statistical correction

    Attributes:
        Step_1_Quantitative_Analysis: Quantitative analysis stage
        Step_2_Qualitative_Context: Qualitative evaluation stage
        Step_3_Statistical_Correction: Statistical correction stage
    """
    Step_1_Quantitative_Analysis: Step1QuantitativeAnalysis = Field(
        ..., description="Stage 1: Quantitative analysis"
    )
    Step_2_Qualitative_Context: Step2QualitativeContext = Field(
        ..., description="Stage 2: Qualitative context evaluation"
    )
    Step_3_Statistical_Correction: Step3StatisticalCorrection = Field(
        ..., description="Stage 3: Statistical correction"
    )


class FinalOutput(BaseModel):
    """Final output metadata of a master node.

    Attributes:
        Score_Variable: Final score variable name
        Signal_Type: Signal type (e.g., 'Accumulate_Deep_Value')
        State_Flags: List of state flags
        Range: Output range string (for reference)
    """
    Score_Variable: str = Field(..., description="Final score variable name")
    Signal_Type: str = Field(..., description="Signal type")
    State_Flags: List[str] = Field(default_factory=list, description="List of state flags")
    Range: str = Field(..., description="Output range (reference string)")


# ──────────────────────────────────────────────────────────
# 3. MasterNode — Individual Node for Each of the 12 Masters
# ──────────────────────────────────────────────────────────

class MasterNode(BaseModel):
    """Master node representing one of the 12 investment masters.

    Fully accommodates the structure of each object in the JSON Nodes array.
    Each node contains an independent three-step inference pipeline and
    final output metadata.

    Attributes:
        Node_ID: Unique identifier (e.g., VAL_GRAHAM_001)
        Master: Master investor name
        Core_Concept: Core investment philosophy
        Full_Description: Detailed description
        Intelligence_Structure: Three-step analysis structure
        Final_Output: Final output metadata
    """
    Node_ID: str = Field(..., description="Unique node identifier (e.g., VAL_GRAHAM_001)")
    Master: str = Field(..., description="Master investor name (e.g., Benjamin Graham)")
    Core_Concept: str = Field(..., description="Core investment philosophy (e.g., Margin_of_Safety)")
    Full_Description: str = Field(..., description="Detailed description")
    Intelligence_Structure: IntelligenceStructure = Field(
        ..., description="Three-step intelligent analysis pipeline"
    )
    Final_Output: FinalOutput = Field(..., description="Final output metadata")


# ──────────────────────────────────────────────────────────
# 4. LogicalEdge — Logical Connections Between Nodes
# ──────────────────────────────────────────────────────────

class LogicalEdge(BaseModel):
    """Logical connection (synapse) between nodes.

    Accommodates the object structure in the JSON Logical_Edges array.
    Referenced by GraphOrchestrator during topological sorting and signal resolution.

    Attributes:
        Edge_ID: Unique edge identifier
        Source: Source node ID
        Target: Target node ID (or 'ALL_*' wildcard)
        Relationship_Type: Relationship type enum (Override, Synergize, etc.)
        Condition_Regime: List of condition regime enums
        Logic: Logic formula string
    """
    Edge_ID: str = Field(..., description="Unique edge identifier")
    Source: str = Field(..., description="Source node ID")
    Target: str = Field(
        ...,
        description="Target node ID or wildcard (e.g., ALL_OTHER_ENGINES)"
    )
    Relationship_Type: RelationshipType = Field(
        ...,
        description="Relationship type enum (Override, Synergize_With, Suppress, Master_Override, Global_Weight_Adjuster)"
    )
    Condition_Regime: List[MarketRegime] = Field(
        default_factory=list,
        description="List of market regime enums under which this edge is activated"
    )
    Logic: str = Field(..., description="Logic formula string (Safe Eval)")


# ──────────────────────────────────────────────────────────
# 5. MasterEngineConfig — Top-level JSON Wrapper
# ──────────────────────────────────────────────────────────

class MasterEngineConfig(BaseModel):
    """Wrapper model accommodating the top-level structure of the master JSON file.

    Parses the entire JSON file for each of the four engine domains
    (Value, Growth, Macro, Risk) into a single model.

    Attributes:
        Ontology_Schema: Schema version identifier
        Engine_Domain: Engine domain (e.g., Value_Investment)
        Last_Update: Last update date
        Nodes: List of master nodes
        Logical_Edges: List of logical edges
    """
    Ontology_Schema: str = Field(..., description="Schema version (e.g., H-PIOS_v8.5_GraphRAG_Master)")
    Engine_Domain: str = Field(..., description="Engine domain (e.g., Value_Investment)")
    Last_Update: str = Field(..., description="Last update date (YYYY-MM-DD)")
    Nodes: List[MasterNode] = Field(..., min_length=1, description="List of master nodes")
    Logical_Edges: List[LogicalEdge] = Field(
        default_factory=list, description="List of logical edges"
    )


# ──────────────────────────────────────────────────────────
# 6. EngineState — Stateful Inference Support
# ──────────────────────────────────────────────────────────

class EngineState(BaseModel):
    """Engine runtime state (Stateful Inference Support).

    Stores rolling statistics and previous regime information to retain
    continuous market flow. An independent instance is maintained for each
    master node (or globally).

    Attributes:
        node_id: Associated node identifier (None for global state)
        rolling_volatility: Rolling volatility (e.g., 20-day moving standard deviation)
        signal_persistence: Signal persistence counter (N consecutive same-direction signals)
        previous_regime: Previous market regime (MarketRegime enum)
        rolling_drawdown: Rolling maximum drawdown
        rolling_win_rate: Dynamically updated rolling win rate
        cumulative_pnl: Cumulative profit and loss
        signal_history: Recent signal history (time series)
        last_score: Most recent computed score
        last_updated: Last state update timestamp
        metadata: Extensible metadata dictionary
    """
    node_id: Optional[str] = Field(None, description="Associated node ID (None for global)")
    rolling_volatility: float = Field(0.0, ge=0.0, description="Rolling volatility")
    signal_persistence: int = Field(0, ge=0, description="Signal persistence counter")
    previous_regime: Optional[MarketRegime] = Field(None, description="Previous market regime (MarketRegime enum)")
    rolling_drawdown: float = Field(0.0, le=0.0, description="Rolling maximum drawdown (≤ 0)")
    rolling_win_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Dynamic rolling win rate (None falls back to JSON initial value)"
    )
    cumulative_pnl: float = Field(0.0, description="Cumulative profit and loss")
    signal_history: List[float] = Field(
        default_factory=list, description="Recent signal score history"
    )
    last_score: Optional[float] = Field(None, description="Most recent computed score")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")


class PerformanceState(BaseModel):
    """Backtest / live-trading performance tracking state.

    Structures performance data to be injected into update_rolling_win_rate().

    Attributes:
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        avg_win_return: Average return on winning trades
        avg_loss_return: Average return on losing trades
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown
        calmar_ratio: Calmar ratio
        win_loss_ratio: Win/loss ratio (avg_win / avg_loss)
    """
    total_trades: int = Field(0, ge=0, description="Total number of trades")
    winning_trades: int = Field(0, ge=0, description="Number of winning trades")
    losing_trades: int = Field(0, ge=0, description="Number of losing trades")
    avg_win_return: float = Field(0.0, description="Average return (winning trades)")
    avg_loss_return: float = Field(0.0, description="Average return (losing trades)")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    max_drawdown: float = Field(0.0, le=0.0, description="Maximum drawdown (≤ 0)")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    win_loss_ratio: Optional[float] = Field(None, ge=0.0, description="Win/loss ratio")

    @property
    def calculated_win_rate(self) -> Optional[float]:
        """Compute win rate from trade history.

        Returns:
            Win rate if total_trades > 0, otherwise None.
        """
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return None


# ──────────────────────────────────────────────────────────
# 7. Input Interface — Data Source Isolation Payloads
# ──────────────────────────────────────────────────────────

class MarketDataPayload(BaseModel):
    """Financial / macro quantitative data input payload.

    Decouples the engine from data sources (API, DB, CSV, etc.).
    All quantitative metrics are passed as metric_name -> float mappings.

    Attributes:
        ticker: Ticker symbol (optional)
        timestamp: Data timestamp
        metrics: Metric name -> numeric value mapping dictionary
        current_regime: Current market regime
        regime_confidence: Regime detection confidence
        metadata: Extensible metadata
    """
    ticker: Optional[str] = Field(None, description="Ticker symbol (e.g., AAPL, 005930.KS)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Data timestamp (UTC)"
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="Metric name -> numeric value mapping (e.g., {'Price_to_NCAV': 0.55, 'P/E_Ratio': 12.3})"
    )
    current_regime: Optional[MarketRegime] = Field(None, description="Current market regime (MarketRegime enum)")
    regime_confidence: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Regime detection confidence"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")


class NLPContextPayload(BaseModel):
    """NLP-based qualitative data input payload.

    Goes beyond simple keyword matching by providing scaffolding to compute
    cosine similarity between JSON keywords and news context using
    Sentence-Transformers, feeding into the Step 2 qualitative evaluation.

    Attributes:
        source: Data source (e.g., 'Reuters', 'Bloomberg')
        timestamp: Data timestamp
        raw_texts: List of raw texts (news articles, reports, etc.)
        detected_keywords: Detected keyword mapping (keyword -> frequency/confidence)
        embedding_vectors: List of text embedding vectors.
            Each vector is the output of Sentence-Transformers or similar models.
            Corresponds 1:1 with raw_texts.
        semantic_similarity_scores: Per-scenario cosine similarity mapping.
            scenario_condition -> similarity score.
            Pre-computed cosine similarity between each scenario's keywords
            and news embeddings is injected here.
        nlp_model_confidence: Global NLP model confidence
        sentiment_scores: Sentiment analysis score mapping (text identifier -> score)
        metadata: Extensible metadata
    """
    source: Optional[str] = Field(None, description="Data source (e.g., Reuters)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Data timestamp (UTC)"
    )
    raw_texts: List[str] = Field(
        default_factory=list,
        description="List of raw texts (news articles, reports)"
    )
    detected_keywords: Dict[str, float] = Field(
        default_factory=dict,
        description="Detected keyword -> frequency or confidence mapping"
    )
    embedding_vectors: List[List[float]] = Field(
        default_factory=list,
        description=(
            "List of text embedding vectors. "
            "Output of Sentence-Transformers or similar; corresponds 1:1 with raw_texts."
        )
    )
    semantic_similarity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Scenario condition -> cosine similarity score mapping. "
            "Pre-computed similarity between JSON keywords and news embeddings."
        )
    )
    nlp_model_confidence: float = Field(
        1.0, ge=0.0, le=1.0,
        description="Global NLP model confidence"
    )
    sentiment_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Text identifier -> sentiment analysis score mapping (-1.0 to 1.0)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")

    @field_validator("embedding_vectors")
    @classmethod
    def validate_embedding_dimensions(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate that all embedding vectors have identical dimensions."""
        if len(v) > 1:
            dim = len(v[0])
            for i, vec in enumerate(v[1:], start=1):
                if len(vec) != dim:
                    # KO message kept for stability; EN: embedding dimension mismatch
                    raise ValueError(
                        f"임베딩 벡터 차원 불일치: v[0]={dim}, v[{i}]={len(vec)}"
                    )
        return v


# ──────────────────────────────────────────────────────────
# 8. Orchestrator Input/Output Models
# ──────────────────────────────────────────────────────────

class NodeExecutionResult(BaseModel):
    """Individual node execution result.

    Return type of AbstractMasterEngine.execute().

    Attributes:
        node_id: Executed node ID
        master: Master investor name
        raw_score: Raw score before normalization
        normalized_score: Normalized score (0.0–1.0)
        signal_type: Signal type
        active_state_flag: Active state flag
        step1_output: Step 1 quantitative analysis result
        step2_output: Step 2 qualitative evaluation result
        step3_output: Step 3 correction result
        used_win_rate: Actually applied win rate (dynamic or initial value)
        metadata: Extensible metadata
    """
    node_id: str = Field(..., description="Executed node ID")
    master: str = Field(..., description="Master investor name")
    raw_score: float = Field(..., description="Raw score before normalization")
    normalized_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized score (0.0–1.0)"
    )
    signal_type: str = Field(..., description="Signal type")
    active_state_flag: Optional[str] = Field(None, description="Active state flag")
    step1_output: float = Field(..., description="Step 1 quantitative analysis result")
    step2_output: float = Field(..., description="Step 2 qualitative evaluation result")
    step3_output: float = Field(..., description="Step 3 correction result")
    used_win_rate: float = Field(..., description="Applied win rate")
    internal_monologue: Optional[str] = Field(None, description="Master's internal monologue / reasoning logic (sandbox use)")
    philosophical_weight: float = Field(1.0, description="Deliberation speaking weight / authority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata")


class OrchestratorOutput(BaseModel):
    """Final output of the GraphOrchestrator.

    Contains the ensemble result of the 12 masters and final position sizing.

    Attributes:
        timestamp: Execution timestamp
        ticker: Target ticker symbol
        node_results: Per-node execution result dictionary
        ensemble_signal: Ensemble composite signal (0.0–1.0)
        final_position_size: Final position ratio based on Thorp Kelly criterion
        override_active: Whether master override is active
        override_source: Override source node ID
        active_regime: Current active regime (MarketRegime enum)
        confidence: Overall signal confidence
        audit_log: Execution audit log
    """
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Execution timestamp (UTC)"
    )
    ticker: Optional[str] = Field(None, description="Target ticker symbol")
    node_results: Dict[str, NodeExecutionResult] = Field(
        ..., description="Node ID -> execution result mapping"
    )
    ensemble_signal: float = Field(
        ..., ge=0.0, le=1.0, description="Ensemble composite signal"
    )
    final_position_size: float = Field(
        ..., ge=0.0, le=1.0, description="Final position ratio (0.0–1.0)"
    )
    override_active: bool = Field(False, description="Whether master override is active")
    override_source: Optional[str] = Field(None, description="Override source node ID")
    active_regime: Optional[MarketRegime] = Field(None, description="Current active regime (MarketRegime enum)")
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Overall signal confidence"
    )
    spg_veto_active: bool = Field(False, description="Whether SPG veto is active")
    spg_report: Dict[str, Any] = Field(default_factory=dict, description="SPG detailed adjudication report")
    deliberation_logs: List[str] = Field(default_factory=list, description="Sandbox deliberation logs")
    tension_score: float = Field(0.0, description="Inter-perspective conflict / tension score (0.0–1.0)")
    audit_log: List[str] = Field(
        default_factory=list, description="Execution audit log (step-by-step trace)"
    )
