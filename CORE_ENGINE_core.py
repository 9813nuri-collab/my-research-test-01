"""
H-PIOS v8.6 — Core Execution Engine (Balanced Intelligence Protocol)
======================================================================
Core engine driving dynamic reasoning logic from the Master JSON
of 12 investment maestros, at production hedge-fund grade.

[v8.6 Update Note]
------------------
1. **Pessimism Bias Systemic Fix**: 
   - Sets the Step 1 Quantitative Analysis base score floor to 0.2,
     eliminating the bias where missing data immediately triggers a strong sell signal.
2. **Collective Intelligence (Weighted Ensemble)**:
   - Applies exponential weighting proportional to each expert's conviction
     (distance from the 0.18 neutral point).
   - Prioritizes the voices of a few outspoken experts over the noise of a silent majority.
3. **Organic Tension Tuning**:
   - Raises the Synergy Boost coefficient from 0.5 to 0.8 to strengthen
     coupling between high-quality stocks.
   - Ensures flexibility in risk discount amplitude and macro correction coefficients.
"""

from __future__ import annotations

import ast
import math
import operator
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from CORE_MODELS_models import (
    EngineState,
    FinalOutput,
    LogicalEdge,
    MarketDataPayload,
    MarketRegime,
    MasterEngineConfig,
    MasterNode,
    NLPContextPayload,
    NodeExecutionResult,
    OrchestratorOutput,
    PerformanceState,
    QualitativeScenario,
    QuantMetric,
    RelationshipType,
    Step1QuantitativeAnalysis,
    Step2QualitativeContext,
    Step3StatisticalCorrection,
)

logger = logging.getLogger("H-PIOS.engine_core")

# -----------------------------------------------------------------------------
# Korean in logs, ValueError messages, and audit_log lines is intentional
# (unchanged for stable tooling / local ops). English gloss for reviewers:
#   수식 평가 깊이 초과 — expression evaluation depth exceeded
#   알 수 없는 변수 — unknown variable; 사용 가능한 변수 — available variables
#   허용되지 않은 … 연산 — disallowed binary/unary/compare/logic operation
#   호출 불가능한 객체 — not callable
#   수식 파싱 실패 — formula parse failed; 원인 — cause
#   유효한 거래 이력이 없어 승률 갱신 불가 — cannot update win rate (no trades)
#   롤링 승률 갱신 — rolling win-rate update
#   지표 … 데이터 없음 → 0점 — metric missing → score 0
#   수식 평가 실패 … → 0점 — formula eval failed → score 0
#   발동 — triggered; 위상 정렬 — topological sort; 순환 참조 — cycle detected
#   Soft-Shutdown 적용 — soft shutdown applied; 시너지/억제/연속전이 — synergy / suppress / continuous transfer
#   앙상블 대상 노드 없음 — no ensemble nodes; 신뢰도 필터 — confidence filter
#   등록되지 않은 노드 — unregistered node; SPG 판정/ VETO / Floor — policy governor messages
# -----------------------------------------------------------------------------


# ══════════════════════════════════════════════════════════
# 1. Safe Formula Evaluator — AST-Based Secure Expression Evaluator
# ══════════════════════════════════════════════════════════

class SafeFormulaEvaluator:
    """AST-based restricted expression evaluator.

    For security and runtime stability, this evaluator avoids Python's
    built-in ``eval()`` and instead parses expressions via the ``ast`` module,
    executing only whitelisted operations and functions.

    Permitted operations:
        - Arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``, ``%``
        - Comparison: ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``
        - Logical: ``and``, ``or``, ``not``
        - Unary: ``-``, ``+``
        - Built-in functions: ``MAX``, ``MIN``, ``ABS``, ``ROUND``, ``LOG``, ``EXP``, ``SQRT``
        - Ternary (IF-THEN-ELSE) patterns are supported via preprocessing

    Usage::

        evaluator = SafeFormulaEvaluator()
        ctx = {"L_quant_graham": 0.6, "L_qual_graham": 0.5,
               "historical_win_rate": 0.62, "decay_factor": 0.85}
        result = evaluator.evaluate(
            "(L_quant_graham + L_qual_graham) * historical_win_rate * decay_factor",
            ctx
        )
    """

    # Permitted binary operator mapping
    _BINARY_OPS: Dict[type, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    # Permitted comparison operator mapping
    _COMPARE_OPS: Dict[type, Callable] = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }

    # Permitted unary operators
    _UNARY_OPS: Dict[type, Callable] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }

    # Permitted built-in functions (case-insensitive)
    _SAFE_FUNCTIONS: Dict[str, Callable] = {
        "MAX": max,
        "MIN": min,
        "ABS": abs,
        "ROUND": round,
        "LOG": math.log,
        "EXP": math.exp,
        "SQRT": math.sqrt,
        "max": max,
        "min": min,
        "abs": abs,
        "round": round,
        "log": math.log,
        "exp": math.exp,
        "sqrt": math.sqrt,
    }

    # Maximum AST node depth (guards against infinite recursion)
    _MAX_DEPTH: int = 50

    def __init__(self, extra_functions: Optional[Dict[str, Callable]] = None) -> None:
        """Initialize the SafeFormulaEvaluator.

        Args:
            extra_functions: Additional permitted function mapping (optional).
                e.g., ``{"SIGMOID": lambda x: 1/(1+math.exp(-x))}``
        """
        self._functions = dict(self._SAFE_FUNCTIONS)
        if extra_functions:
            self._functions.update(extra_functions)

    # ── Preprocessing: Convert JSON IF-THEN-ELSE to Python ternary ──

    @staticmethod
    def _preprocess_formula(formula: str) -> str:
        """Preprocess non-standard JSON formula patterns into Python AST-compatible form.

        Supported transformations:
            - ``IF <cond> THEN <val1> ELSE <val2>``
              → ``(<val1>) if (<cond>) else (<val2>)``
            - Normalizes special characters in variable names (``/``, ``-``, etc.)
              to ``_`` for Python AST-compatible identifiers

        Args:
            formula: Original formula string

        Returns:
            Python AST-compatible formula string
        """
        import re

        # Convert IF ... THEN ... ELSE ... pattern (single level, no nesting)
        pattern = r"IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)(?:\s*$|\s*\))"
        
        def _replace_if(match: re.Match) -> str:
            cond = match.group(1).strip()
            then_val = match.group(2).strip()
            else_val = match.group(3).strip()
            return f"(({then_val}) if ({cond}) else ({else_val}))"

        result = formula
        # Iterative substitution (handles nested cases)
        for _ in range(5):
            new_result = re.sub(pattern, _replace_if, result, flags=re.IGNORECASE)
            if new_result == result:
                break
            result = new_result

        # Normalize special characters in variable names: P/E_Ratio → P_E_Ratio
        # Replace '/' between alphanumeric/_ characters with '_'
        result = re.sub(r'(?<=[A-Za-z0-9_])/(?=[A-Za-z0-9_])', '_', result)
        # Replace '-' within identifiers with '_' (e.g., Debt-to-Equity → Debt_to_Equity)
        result = re.sub(r'(?<=[A-Za-z0-9_])-(?=[A-Za-z])', '_', result)

        return result

    # ── AST node evaluation (recursive tree walk) ──

    def _eval_node(self, node: ast.AST, context: Dict[str, Any], depth: int = 0) -> Any:
        """Recursively evaluate an AST node in a safe manner.

        Args:
            node: AST node
            context: Variable name → value mapping
            depth: Current recursion depth

        Returns:
            Evaluation result (float, bool, etc.)

        Raises:
            RecursionError: When maximum depth is exceeded
            ValueError: When a disallowed AST node is encountered
        """
        if depth > self._MAX_DEPTH:
            raise RecursionError(
                f"수식 평가 깊이 초과 (최대 {self._MAX_DEPTH}). "
                "수식이 너무 복잡하거나 순환 참조가 있을 수 있습니다."
            )

        # ── Constant ──
        if isinstance(node, ast.Constant):
            return node.value

        # ── Variable (Name) ──
        if isinstance(node, ast.Name):
            name = node.id
            # Return callable if it matches a built-in function name
            if name in self._functions:
                return self._functions[name]
            if name in context:
                return context[name]
            raise ValueError(
                f"알 수 없는 변수: '{name}'. "
                f"사용 가능한 변수: {list(context.keys())}"
            )

        # ── Binary operation (BinOp) ──
        if isinstance(node, ast.BinOp):
            op_func = self._BINARY_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"허용되지 않은 이항 연산: {type(node.op).__name__}")
            left = self._eval_node(node.left, context, depth + 1)
            right = self._eval_node(node.right, context, depth + 1)
            return op_func(left, right)

        # ── Unary operation (UnaryOp) ──
        if isinstance(node, ast.UnaryOp):
            op_func = self._UNARY_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"허용되지 않은 단항 연산: {type(node.op).__name__}")
            operand = self._eval_node(node.operand, context, depth + 1)
            return op_func(operand)

        # ── Comparison (Compare) ──
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context, depth + 1)
            for op_node, comparator in zip(node.ops, node.comparators):
                op_func = self._COMPARE_OPS.get(type(op_node))
                if op_func is None:
                    raise ValueError(f"허용되지 않은 비교 연산: {type(op_node).__name__}")
                right = self._eval_node(comparator, context, depth + 1)
                if not op_func(left, right):
                    return False
                left = right
            return True

        # ── Boolean operation (BoolOp) ──
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(
                    self._eval_node(v, context, depth + 1) for v in node.values
                )
            if isinstance(node.op, ast.Or):
                return any(
                    self._eval_node(v, context, depth + 1) for v in node.values
                )
            raise ValueError(f"허용되지 않은 논리 연산: {type(node.op).__name__}")

        # ── Function call (Call) ──
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func, context, depth + 1)
            if not callable(func):
                raise ValueError(f"호출 불가능한 객체: {func}")
            args = [self._eval_node(a, context, depth + 1) for a in node.args]
            return func(*args)

        # ── Ternary expression (IfExp) ──
        if isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, context, depth + 1)
            if test:
                return self._eval_node(node.body, context, depth + 1)
            return self._eval_node(node.orelse, context, depth + 1)

        # ── Expression wrapper ──
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, context, depth + 1)

        raise ValueError(
            f"허용되지 않은 AST 노드 유형: {type(node).__name__}. "
            "보안상 제한된 수식만 평가 가능합니다."
        )

    # ── Public API ──

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize special characters in variable names to underscores.

        Applies the same rules as ``_preprocess_formula`` to normalize
        context keys so they match variable names within formulas.

        Args:
            key: Original variable name

        Returns:
            Normalized variable name
        """
        import re
        result = re.sub(r'(?<=[A-Za-z0-9_])/(?=[A-Za-z0-9_])', '_', key)
        result = re.sub(r'(?<=[A-Za-z0-9_])-(?=[A-Za-z])', '_', result)
        return result

    def evaluate(self, formula: str, context: Dict[str, Any]) -> float:
        """Safely evaluate a formula string.

        Args:
            formula: Formula string (loaded from JSON)
            context: Variable name → value mapping (Step 1/2 outputs, constants, etc.)

        Returns:
            Evaluation result (float)

        Raises:
            ValueError: When formula parsing/evaluation fails
            RecursionError: When maximum depth is exceeded
        """
        preprocessed = self._preprocess_formula(formula)

        # Normalize context keys with the same rules (P/E_Ratio → P_E_Ratio)
        normalized_ctx: Dict[str, Any] = {}
        for k, v in context.items():
            normalized_ctx[self._normalize_key(k)] = v
            # Retain the original key as well (in case it is referenced pre-normalization)
            if k != self._normalize_key(k):
                normalized_ctx[k] = v

        try:
            tree = ast.parse(preprocessed, mode="eval")
        except SyntaxError as e:
            raise ValueError(
                f"수식 파싱 실패: '{formula}' → 전처리: '{preprocessed}'. "
                f"원인: {e}"
            ) from e

        result = self._eval_node(tree, normalized_ctx)

        # Convert bool → float (True=1.0, False=0.0)
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        return float(result)


# ══════════════════════════════════════════════════════════
# 2. Score Normalizer — Pluggable Normalization Strategies
# ══════════════════════════════════════════════════════════

@runtime_checkable
class ScoreNormalizer(Protocol):
    """Score normalization protocol (pluggable design).

    All normalization strategies must implement this protocol.
    Returns a normalized score in the range [0.0, 1.0].
    """

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Normalize a raw score to the [0.0, 1.0] range.

        Args:
            raw_score: Pre-normalization raw score
            **kwargs: Strategy-specific additional parameters

        Returns:
            Normalized score in [0.0, 1.0]
        """
        ...


class SigmoidNormalizer:
    """Sigmoid normalizer (default strategy).

    Formula: f(x) = 1 / (1 + e^(-k * (x - x0)))

    Parameters:
        k: Slope coefficient (larger values yield sharper transitions). Default 5.0
        x0: Center point (the 0.5 inflection point of the sigmoid). Default 0.5
    """

    def __init__(self, k: float = 5.0, x0: float = 0.5) -> None:
        """Initialize the SigmoidNormalizer.

        Args:
            k: Slope coefficient (default 5.0)
            x0: Center point (default 0.5)
        """
        self.k = k
        # [Calibration] x0 set to 0.21 for academic zero-point (Baseline maps to ~0.55)
        self.x0 = 0.21

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Normalize using the sigmoid function.

        Args:
            raw_score: Raw score
            **kwargs: Ignored (for protocol compatibility)

        Returns:
            Normalized score in [0.0, 1.0]
        """
        try:
            return 1.0 / (1.0 + math.exp(-self.k * (raw_score - self.x0)))
        except OverflowError:
            # Handle exp overflow for extreme values
            return 0.0 if raw_score < self.x0 else 1.0


class ZScoreNormalizer:
    """Z-Score-based normalizer.

    Standardizes raw scores against a moving mean/standard deviation,
    then maps the result to [0, 1] via a sigmoid.

    Parameters:
        mean: Moving average (default 0.0)
        std: Moving standard deviation (default 1.0)
        sigmoid_k: Post-stage sigmoid slope (default 1.5)
    """

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, sigmoid_k: float = 1.5
    ) -> None:
        self.mean = mean
        self.std = std
        self._sigmoid = SigmoidNormalizer(k=sigmoid_k, x0=0.0)

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Normalize via Z-Score → Sigmoid.

        Args:
            raw_score: Raw score
            **kwargs:
                mean (float): Dynamic mean (optional)
                std (float): Dynamic standard deviation (optional)

        Returns:
            Normalized score in [0.0, 1.0]
        """
        mean = kwargs.get("mean", self.mean)
        std = kwargs.get("std", self.std)
        if std == 0:
            std = 1e-8  # Guard against zero variance
        z = (raw_score - mean) / std
        return self._sigmoid.normalize(z)


class MinMaxNormalizer:
    """Min-Max normalizer.

    Performs linear normalization based on historical minimum/maximum values.

    Parameters:
        min_val: Historical minimum (default 0.0)
        max_val: Historical maximum (default 2.0)
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 2.0) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, raw_score: float, **kwargs: Any) -> float:
        """Perform Min-Max normalization.

        Args:
            raw_score: Raw score
            **kwargs:
                min_val (float): Dynamic minimum (optional)
                max_val (float): Dynamic maximum (optional)

        Returns:
            Normalized score in [0.0, 1.0] (clipped)
        """
        lo = kwargs.get("min_val", self.min_val)
        hi = kwargs.get("max_val", self.max_val)
        if hi == lo:
            return 0.5
        normalized = (raw_score - lo) / (hi - lo)
        return max(0.0, min(1.0, normalized))


# ══════════════════════════════════════════════════════════
# 3. AbstractMasterEngine — Individual Maestro Node Execution Engine
# ══════════════════════════════════════════════════════════

class AbstractMasterEngine:
    """Execution engine for an individual investment maestro node (one of 12).

    Runs a 3-step reasoning pipeline based on the MasterNode JSON structure
    and produces a normalized final score.

    Core capabilities:
        - Step 1: Quantitative analysis (dynamic evaluation of logic_formula)
        - Step 2: Qualitative assessment (NLP keyword / cosine similarity based)
        - Step 3: Statistical correction (dynamic evaluation of formula)
        - Normalization: Pluggable normalizer (default: Sigmoid)
        - Win-rate update: Dynamically overrides via update_rolling_win_rate()

    Args:
        node: MasterNode Pydantic model
        normalizer: ScoreNormalizer protocol implementation (default: SigmoidNormalizer)
        evaluator: SafeFormulaEvaluator instance (shareable)
    """

    def __init__(
        self,
        node: MasterNode,
        normalizer: Optional[ScoreNormalizer] = None,
        evaluator: Optional[SafeFormulaEvaluator] = None,
    ) -> None:
        self.node = node
        self.normalizer: ScoreNormalizer = normalizer or SigmoidNormalizer()
        self.evaluator = evaluator or SafeFormulaEvaluator()

        # Store the JSON initial value as the default win rate
        self._default_win_rate: float = (
            node.Intelligence_Structure
            .Step_3_Statistical_Correction
            .historical_win_rate
        )
        # Dynamic rolling win rate (None falls back to default)
        self._rolling_win_rate: Optional[float] = None

        # Engine state
        self.state = EngineState(node_id=node.Node_ID)

    # ── Properties ──

    @property
    def node_id(self) -> str:
        """Unique node identifier."""
        return self.node.Node_ID

    @property
    def effective_win_rate(self) -> float:
        """Current effective win rate.

        Returns the dynamic rolling win rate if set; otherwise returns
        the JSON-defined initial value.
        """
        if self._rolling_win_rate is not None:
            return self._rolling_win_rate
        if self.state.rolling_win_rate is not None:
            return self.state.rolling_win_rate
        return self._default_win_rate

    # ── Win-rate update ──

    def update_rolling_win_rate(
        self, performance: PerformanceState
    ) -> float:
        """Dynamically update the rolling win rate from external backtest/live performance data.

        The JSON ``historical_win_rate`` is treated only as the initial default;
        once this method is called, it is dynamically overridden.

        Args:
            performance: Performance tracking state (PerformanceState)

        Returns:
            Updated rolling win rate

        Raises:
            ValueError: When no valid trade history is available
        """
        calculated = performance.calculated_win_rate
        if calculated is None:
            raise ValueError(
                f"노드 {self.node_id}: 유효한 거래 이력이 없어 승률 갱신 불가. "
                f"total_trades={performance.total_trades}"
            )

        self._rolling_win_rate = calculated
        self.state.rolling_win_rate = calculated
        self.state.last_updated = datetime.utcnow()

        logger.info(
            "노드 %s 롤링 승률 갱신: %.4f → %.4f (trades=%d)",
            self.node_id, self._default_win_rate, calculated,
            performance.total_trades,
        )
        return calculated

    # ── Step 1: Quantitative Analysis ──

    def _execute_step1(
        self, market_data: MarketDataPayload
    ) -> float:
        """Execute Step 1: Quantitative Analysis.

        Evaluates each QuantMetric's logic_formula via SafeFormulaEvaluator
        and sums the results.

        Args:
            market_data: Financial/macro quantitative data payload

        Returns:
            Step 1 quantitative score (sum)
        """
        step1 = self.node.Intelligence_Structure.Step_1_Quantitative_Analysis
        # [v8.6] Pessimism Bias Systemic Fix — base score floor set to 0.2
        # (Changed from additive to max to prevent Bullish Bias)
        total_score = 0.0

        for metric_def in step1.metrics:
            # Look up by both original and normalized keys (handles P/E_Ratio, etc.)
            metric_value = market_data.metrics.get(metric_def.metric)
            if metric_value is None:
                normalized_key = self.evaluator._normalize_key(metric_def.metric)
                metric_value = market_data.metrics.get(normalized_key)

            if metric_value is None:
                logger.debug(
                    "노드 %s Step1: 지표 '%s' 데이터 없음 → 0점 처리",
                    self.node_id, metric_def.metric,
                )
                continue

            if metric_def.logic_formula:
                # Dynamically interpret logic_formula via SafeEval
                context = {metric_def.metric: metric_value}
                try:
                    score = self.evaluator.evaluate(metric_def.logic_formula, context)
                except (ValueError, RecursionError) as e:
                    logger.warning(
                        "노드 %s Step1: 수식 평가 실패 '%s' → 0점. 원인: %s",
                        self.node_id, metric_def.logic_formula, e,
                    )
                    score = 0.0
            else:
                # Fall back to direct operator/threshold comparison when no logic_formula
                score = self._evaluate_metric_direct(metric_def, metric_value)

            total_score += score
            
        return max(0.40, total_score)

    @staticmethod
    def _evaluate_metric_direct(metric_def: QuantMetric, value: float) -> float:
        """Evaluate directly via operator/threshold when no logic_formula is provided.

        Args:
            metric_def: Quantitative metric definition
            value: Actual metric value

        Returns:
            The metric weight if the condition is met; 0.0 otherwise
        """
        op_map = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        op_func = op_map.get(metric_def.operator)
        if op_func and op_func(value, metric_def.threshold):
            return metric_def.weight
        # Special operators (Trend, Volatility) use simple threshold comparison
        if metric_def.operator in ("Trend", "Volatility"):
            if value > metric_def.threshold:
                return metric_def.weight
        return 0.0

    # ── Step 2: Qualitative Assessment ──

    def _execute_step2(
        self, nlp_data: Optional[NLPContextPayload]
    ) -> Tuple[float, Optional[str]]:
        """Execute Step 2: Qualitative Assessment.

        Computes scenario scores based on NLP keyword matching and cosine
        similarity. Returns a special flag if a ``CRITICAL_INHIBIT`` action
        is triggered.

        Args:
            nlp_data: NLP context payload (returns 0.0 if None)

        Returns:
            (Step 2 qualitative score, active special action or None)
        """
        if nlp_data is None:
            return 0.0, None

        step2 = self.node.Intelligence_Structure.Step_2_Qualitative_Context
        total_score = 0.0
        critical_action: Optional[str] = None

        for scenario in step2.scenarios:
            activation_score = self._evaluate_scenario(
                scenario, nlp_data, step2.nlp_confidence_threshold
            )

            if activation_score > 0.0 or (
                activation_score == 0.0
                and scenario.score_modifier < 0
                and self._check_keyword_match(scenario, nlp_data)
            ):
                # Scenario triggered
                effective_modifier = scenario.score_modifier * activation_score
                total_score += effective_modifier

                if scenario.action == "CRITICAL_INHIBIT":
                    critical_action = "CRITICAL_INHIBIT"
                    total_score += scenario.score_modifier  # Forced inhibition
                    logger.warning(
                        "노드 %s Step2: CRITICAL_INHIBIT 발동 — %s",
                        self.node_id, scenario.condition,
                    )

        return total_score, critical_action

    def _evaluate_scenario(
        self,
        scenario: QualitativeScenario,
        nlp_data: NLPContextPayload,
        confidence_threshold: float,
    ) -> float:
        """Compute the activation score for an individual scenario.

        Priority:
        1. Use cosine similarity from semantic_similarity_scores if available
        2. Otherwise, fall back to keyword-matching heuristic

        Args:
            scenario: Qualitative scenario definition
            nlp_data: NLP data payload
            confidence_threshold: NLP confidence lower bound

        Returns:
            Activation score (0.0–1.0; 0.0 if not triggered)
        """
        # Global NLP model confidence check
        if nlp_data.nlp_model_confidence < confidence_threshold:
            return 0.0

        # 1) Cosine similarity-based evaluation (Semantic NLP backbone)
        similarity = nlp_data.semantic_similarity_scores.get(scenario.condition)
        if similarity is not None:
            # Trigger if similarity ≥ confidence_threshold
            if similarity >= confidence_threshold:
                return similarity
            return 0.0

        # 2) Keyword matching fallback
        if self._check_keyword_match(scenario, nlp_data):
            return 1.0

        return 0.0

    @staticmethod
    def _check_keyword_match(
        scenario: QualitativeScenario,
        nlp_data: NLPContextPayload,
    ) -> bool:
        """Determine scenario activation via keyword matching.

        A scenario is triggered if its keywords appear in detected_keywords
        or are contained within the raw_texts.

        Args:
            scenario: Scenario definition
            nlp_data: NLP data

        Returns:
            Whether keywords matched
        """
        if not scenario.keywords:
            return False

        # Check detected_keywords mapping
        for kw in scenario.keywords:
            if kw in nlp_data.detected_keywords:
                return True

        # Full-text search fallback on raw_texts
        combined_text = " ".join(nlp_data.raw_texts).lower()
        for kw in scenario.keywords:
            if kw.lower() in combined_text:
                return True

        return False

    # ── Step 3: Statistical Correction ──

    def _execute_step3(
        self,
        step1_score: float,
        step2_score: float,
        current_regime: Optional[MarketRegime] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Execute Step 3: Statistical Correction.

        Combines Step 1 + Step 2 results using the JSON-defined formula,
        corrected by historical_win_rate (dynamically updatable) and decay_factor.

        Args:
            step1_score: Step 1 quantitative score
            step2_score: Step 2 qualitative score
            current_regime: Current market regime (optional)
            extra_context: Additional context variables (e.g., other node scores)

        Returns:
            Corrected final score
        """
        step3 = self.node.Intelligence_Structure.Step_3_Statistical_Correction
        output_var_s1 = (
            self.node.Intelligence_Structure
            .Step_1_Quantitative_Analysis.output_variable
        )
        output_var_s2 = (
            self.node.Intelligence_Structure
            .Step_2_Qualitative_Context.output_variable
        )

        # Construct formula evaluation context
        context: Dict[str, Any] = {
            output_var_s1: step1_score,
            output_var_s2: step2_score,
            "historical_win_rate": self.effective_win_rate,
            "decay_factor": step3.decay_factor,
        }

        # Regime-specific performance correction (v8.6: added N/A and type-error guards)
        if current_regime:
            regime_key = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
            if regime_key in step3.regime_performance:
                regime_perf = step3.regime_performance[regime_key]
                try:
                    if regime_perf != "N/A":
                        context["historical_win_rate"] = (
                            self.effective_win_rate * float(regime_perf)
                        )
                except (ValueError, TypeError):
                    pass

        # Additional context (e.g., scores from other nodes)
        if extra_context:
            context.update(extra_context)

        try:
            result = self.evaluator.evaluate(step3.formula, context)
        except (ValueError, RecursionError) as e:
            logger.error(
                "노드 %s Step3: 수식 평가 실패 '%s' → 0.0. 원인: %s",
                self.node_id, step3.formula, e,
            )
            result = 0.0

        return result

    # ── State flag determination ──

    def _determine_state_flag(self, normalized_score: float) -> Optional[str]:
        """Determine the active state flag based on the normalized score.

        Maps State_Flags list indices to score intervals:
        - High score → first flag (positive)
        - Low score → last flag (warning)

        Args:
            normalized_score: Normalized score in [0.0, 1.0]

        Returns:
            Active state flag string, or None
        """
        flags = self.node.Final_Output.State_Flags
        if not flags:
            return None
        n = len(flags)
        # Partition the score into n intervals
        idx = min(int(normalized_score * n), n - 1)
        # High score → first flag (reverse mapping)
        return flags[n - 1 - idx]

    # ── Integrated execution ──

    def execute(
        self,
        market_data: MarketDataPayload,
        nlp_data: Optional[NLPContextPayload] = None,
        current_regime: Optional[MarketRegime] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> NodeExecutionResult:
        """Execute the full 3-step pipeline and return the normalized result.

        Args:
            market_data: Quantitative data payload
            nlp_data: NLP qualitative data payload (optional)
            current_regime: Current market regime (optional)
            extra_context: Additional context variable dictionary (optional)

        Returns:
            NodeExecutionResult containing the normalized score
        """
        # Step 1: Quantitative Analysis
        step1_score = self._execute_step1(market_data)

        # Step 2: Qualitative Assessment
        step2_score, critical_action = self._execute_step2(nlp_data)

        # Immediate zero score on CRITICAL_INHIBIT activation
        if critical_action == "CRITICAL_INHIBIT":
            logger.warning(
                "노드 %s: CRITICAL_INHIBIT 발동 → 강제 0점 할당", self.node_id
            )
            result = NodeExecutionResult(
                node_id=self.node_id,
                master=self.node.Master,
                raw_score=0.0,
                normalized_score=0.0,
                signal_type=self.node.Final_Output.Signal_Type,
                active_state_flag="CRITICAL_INHIBIT",
                step1_output=step1_score,
                step2_output=step2_score,
                step3_output=0.0,
                used_win_rate=self.effective_win_rate,
                metadata={"critical_action": critical_action},
            )
            self._update_state(result)
            return result

        # Step 3: Statistical Correction
        step3_score = self._execute_step3(
            step1_score, step2_score, current_regime, extra_context
        )

        # Normalization (0.0–1.0)
        normalized = self.normalizer.normalize(step3_score)

        # Determine state flag
        state_flag = self._determine_state_flag(normalized)

        result = NodeExecutionResult(
            node_id=self.node_id,
            master=self.node.Master,
            raw_score=step3_score,
            normalized_score=normalized,
            signal_type=self.node.Final_Output.Signal_Type,
            active_state_flag=state_flag,
            step1_output=step1_score,
            step2_output=step2_score,
            step3_output=step3_score,
            used_win_rate=self.effective_win_rate,
        )

        self._update_state(result)
        return result

    def _update_state(self, result: NodeExecutionResult) -> None:
        """Update engine state from the execution result.

        Args:
            result: Node execution result
        """
        self.state.last_score = result.normalized_score
        self.state.signal_history.append(result.normalized_score)
        # Retain only the most recent 100 entries
        if len(self.state.signal_history) > 100:
            self.state.signal_history = self.state.signal_history[-100:]
        self.state.last_updated = datetime.utcnow()

        # Update signal persistence
        if len(self.state.signal_history) >= 2:
            prev = self.state.signal_history[-2]
            curr = self.state.signal_history[-1]
            if (prev >= 0.5 and curr >= 0.5) or (prev < 0.5 and curr < 0.5):
                self.state.signal_persistence += 1
            else:
                self.state.signal_persistence = 0


# ══════════════════════════════════════════════════════════
# 4. GraphOrchestrator — 12-Maestro DAG Signal Resolution System
# ══════════════════════════════════════════════════════════

class GraphOrchestrator:
    """Synapse system that orchestrates the 12 investment maestro nodes.

    Core capabilities:
        1. **Topological Sort** — Analyzes inter-node dependencies (edges)
           to guarantee a safe execution order. Raises an exception on
           circular dependency detection.

        2. **Signal Resolution (resolve_signals)** — Four-tier hierarchy:
           (a) Priority Override: Taleb black-swan etc. → forced shutdown on critical risk
           (b) Global Adjustment: Dalio macro score → continuous weight adjustment
           (c) Synergy/Suppress: Inter-node interactions (e.g., Munger–Buffett)
           (d) Position Sizing: Thorp Kelly-based final allocation chaining

        3. **Continuous Weight Adjustment** — Replaces binary threshold logic
           with continuous functions (linear interpolation / exponential decay)
           for smooth attenuation/amplification

    Args:
        configs: List of MasterEngineConfig (four domain JSONs)
        normalizer: Shared ScoreNormalizer (default: SigmoidNormalizer)
        evaluator: Shared SafeFormulaEvaluator (default: new instance)
    """

    # ── Priority Override precedence (higher = higher priority) ──
    _OVERRIDE_PRIORITY: Dict[str, int] = {
        "Master_Override": 100,   # Taleb: black swan, liquidity freeze
        "Override": 80,           # Munger: fraud/accounting risk
        "Global_Weight_Adjuster": 60,  # Dalio: macro regime
        "Suppress": 40,           # General suppression
        "Synergize_With": 20,     # Synergy
    }

    # Taleb threshold: master override triggers above this level
    _TALEB_RUIN_THRESHOLD: float = 0.75  # normalized basis

    def __init__(
        self,
        configs: List[MasterEngineConfig],
        normalizer: Optional[ScoreNormalizer] = None,
        evaluator: Optional[SafeFormulaEvaluator] = None,
    ) -> None:
        self._normalizer = normalizer or SigmoidNormalizer()
        self._evaluator = evaluator or SafeFormulaEvaluator()

        # Node engine registry: node_id → AbstractMasterEngine
        self._engines: Dict[str, AbstractMasterEngine] = {}
        # Complete list of logical edges
        self._edges: List[LogicalEdge] = []
        # Cached topological sort result
        self._sorted_order: Optional[List[str]] = None

        # Global engine state
        self.global_state = EngineState(node_id="GLOBAL")

        self._load_configs(configs)

    # ── Initialization ──

    def _load_configs(self, configs: List[MasterEngineConfig]) -> None:
        """Load engines and edges from MasterEngineConfig list.

        Args:
            configs: MasterEngineConfig instances for four domains
        """
        for config in configs:
            for node in config.Nodes:
                engine = AbstractMasterEngine(
                    node=node,
                    normalizer=self._normalizer,
                    evaluator=self._evaluator,
                )
                self._engines[node.Node_ID] = engine
                logger.info(
                    "노드 등록: %s (%s — %s)",
                    node.Node_ID, node.Master, config.Engine_Domain,
                )

            self._edges.extend(config.Logical_Edges)

        # Perform topological sort
        self._sorted_order = self._topological_sort()
        logger.info("위상 정렬 완료: %s", self._sorted_order)

    # ── Topological Sort (DAG) ──

    def _topological_sort(self) -> List[str]:
        """Perform topological sort by analyzing inter-node dependencies.

        Builds a dependency graph from Source → Target edge directions
        and guarantees a safe execution order via Kahn's algorithm.

        Wildcard targets (ALL_*) are expanded to all registered nodes.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: When a circular dependency is detected
        """
        all_node_ids = set(self._engines.keys())

        # Build adjacency list and in-degree map
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = {nid: 0 for nid in all_node_ids}

        for edge in self._edges:
            source = edge.Source
            if source not in all_node_ids:
                continue

            # Expand targets (wildcard handling)
            targets = self._resolve_targets(edge.Target, all_node_ids, source)

            for target in targets:
                if target in all_node_ids and target != source:
                    # Source must execute before Target to influence it
                    if target not in adjacency[source]:
                        adjacency[source].add(target)
                        in_degree[target] += 1

        # Kahn's algorithm
        queue: deque[str] = deque(
            nid for nid in all_node_ids if in_degree[nid] == 0
        )
        sorted_order: List[str] = []

        while queue:
            node_id = queue.popleft()
            sorted_order.append(node_id)
            for neighbor in adjacency.get(node_id, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(all_node_ids):
            missing = all_node_ids - set(sorted_order)
            raise ValueError(
                f"순환 참조 감지: 위상 정렬 불가. 미처리 노드: {missing}"
            )

        return sorted_order

    @staticmethod
    def _resolve_targets(
        target: str, all_node_ids: Set[str], source: str
    ) -> List[str]:
        """Expand wildcard targets into a list of actual node IDs.

        Args:
            target: Target string (node ID or 'ALL_*')
            all_node_ids: Set of all registered node IDs
            source: Source node ID (to prevent self-reference)

        Returns:
            List of target node IDs
        """
        if target.startswith("ALL_"):
            if target == "ALL_OTHER_ENGINES":
                return [nid for nid in all_node_ids if nid != source]
            elif target == "ALL_GROWTH_AND_VALUE_NODES":
                return [
                    nid for nid in all_node_ids
                    if nid.startswith("GRO_") or nid.startswith("VAL_")
                ]
            else:
                # Other ALL_ patterns: all nodes except source
                return [nid for nid in all_node_ids if nid != source]
        return [target]

    # ── Public API: Signal Resolution ──

    def resolve_signals(
        self,
        market_data: MarketDataPayload,
        nlp_data: Optional[NLPContextPayload] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorOutput:
        """Resolve the 12-maestro signals and compute the final position.

        Execution hierarchy:
            1. Priority Override — Forced shutdown on critical systemic risk
            2. Global Adjustment — Continuous weight adjustment based on macro regime
            3. Synergy/Suppress — Reflect inter-node interactions
            4. Position Sizing — Final allocation via Thorp Kelly criterion

        Args:
            market_data: Quantitative data payload
            nlp_data: NLP qualitative data (optional)

        Returns:
            OrchestratorOutput — per-node results, ensemble signal, final position
        """
        if self._sorted_order is None:
            self._sorted_order = self._topological_sort()

        current_regime = market_data.current_regime
        audit_log: List[str] = []
        node_results: Dict[str, NodeExecutionResult] = {}

        # ── Phase 0: Execute nodes in topological order ──
        audit_log.append(f"[Phase 0] 위상 정렬 순서: {self._sorted_order}")

        for node_id in self._sorted_order:
            engine = self._engines[node_id]

            # Pass other node scores as extra_context
            node_context: Dict[str, Any] = {}
            if extra_context:
                node_context.update(extra_context)

            for prev_id, prev_result in node_results.items():
                node_context[
                    self._engines[prev_id].node.Final_Output.Score_Variable
                ] = prev_result.normalized_score

            result = engine.execute(
                market_data=market_data,
                nlp_data=nlp_data,
                current_regime=current_regime,
                extra_context=node_context,
            )
            node_results[node_id] = result
            audit_log.append(
                f"  노드 {node_id} ({engine.node.Master}): "
                f"raw={result.raw_score:.4f}, norm={result.normalized_score:.4f}, "
                f"flag={result.active_state_flag}"
            )

        # ── Phase 0.5: Raw Tension Calculation (Sandbox Metric) ──
        # Calculate standard deviation of expert signals BEFORE any discounting or overrides
        expert_scores = [
            r.normalized_score for nid, r in node_results.items() 
            if nid.startswith(("VAL_", "GRO_", "MAC_"))
        ]
        raw_tension = 0.0
        if len(expert_scores) > 1:
            mean = sum(expert_scores) / len(expert_scores)
            var = sum((x - mean) ** 2 for x in expert_scores) / len(expert_scores)
            raw_tension = math.sqrt(var)
        
        audit_log.append(f"[Phase 0.5] Raw Philosophy Tension Score: {raw_tension:.4f}")

        # ── Phase 1: Priority Override / Master Influence — risk attenuation & adjustment ──
        # [v8.8] Soft-Shutdown: exponential decay instead of hard shutdown
        # [v8.9.8] Variant Support: skip Phase 1 override entirely for controlled experiments
        skip_override = bool(extra_context.get("_variant_no_override", False)) if extra_context else False

        if skip_override:
            influence_active, influence_source, influence_factor = False, None, 1.0
            audit_log.append("[Phase 1] Override SKIPPED (_variant_no_override=True)")
        else:
            override_k = extra_context.get("_variant_override_k", None) if extra_context else None
            influence_active, influence_source, influence_factor = self._apply_master_influences(
                node_results, current_regime, audit_log, override_k=override_k
            )

        # [v8.9.6] Variant Support: HARD_OVERRIDE (Retro-design for ablation)
        is_hard_override = bool(extra_context.get("_variant_hard_override", False)) if extra_context else False

        if influence_active and (influence_factor < 0.15 or is_hard_override):
            if is_hard_override:
                # Traditional H-PIOS v8.5 behavior: Hard-Zero
                for nid in node_results:
                    if nid != influence_source:
                        node_results[nid] = node_results[nid].model_copy(
                            update={"normalized_score": 0.0}
                        )
                audit_log.append("[Phase 1] TRADITIONAL HARD-ZERO applied (Ablation Variant).")
                return OrchestratorOutput(
                    ticker=market_data.ticker,
                    node_results=node_results,
                    ensemble_signal=0.0,
                    final_position_size=0.0,
                    override_active=True,
                    override_source=influence_source,
                    active_regime=current_regime,
                    confidence=node_results.get("RSK_SHANNON_001").normalized_score if "RSK_SHANNON_001" in node_results else 1.0,
                    audit_log=audit_log,
                    tension_score=raw_tension
                )
            else:
                # [v8.9] Reduced Dictatorship: Soft-Shutdown with non-zero floor
                for nid in node_results:
                    if nid != influence_source:
                        new_val = max(0.01, node_results[nid].normalized_score * influence_factor)
                        node_results[nid] = node_results[nid].model_copy(
                            update={"normalized_score": new_val}
                        )
                audit_log.append(
                    f"[Phase 1] Extreme Risk Discount applied (Factor={influence_factor:.4f}) - Non-zero floor retained for deliberation."
                )
        elif influence_active:
            # Soft-Shutdown: exponentially attenuate all node scores
            for nid in node_results:
                if nid != influence_source:
                    new_val = node_results[nid].normalized_score * influence_factor
                    node_results[nid] = node_results[nid].model_copy(
                        update={"normalized_score": new_val}
                    )
            audit_log.append(
                f"[Phase 1] Soft-Shutdown 적용 (Source={influence_source}, Factor={influence_factor:.4f})"
            )

        # ── Phase 2: Global Adjustment — macro continuous weight adjustment ──
        adjusted_scores = self._apply_global_adjustments(
            node_results, current_regime, audit_log
        )

        # ── Phase 3: Synergy/Suppress — inter-node interactions ──
        synergized_scores = self._apply_synergy_suppress(
            adjusted_scores, node_results, current_regime, audit_log
        )

        # ── Phase 4: Ensemble & Position Sizing ──
        ensemble_signal, final_position = self._compute_ensemble_and_sizing(
            synergized_scores, node_results, audit_log
        )

        # ── Phase 5: Strategic Policy Governor (SPG) ──
        # [v8.5] Deterministic Guardrail for Hype/Toxic Asset Filtering
        # [v8.9.7] Variant Support: skip SPG entirely for controlled experiments
        skip_spg = bool(extra_context.get("_variant_no_spg", False)) if extra_context else False

        spg_veto = False
        spg_report: Dict[str, Any] = {}

        if skip_spg:
            audit_log.append("[Phase 5] SPG SKIPPED (_variant_no_spg=True)")
        else:
            spg = StrategicPolicyGovernor()
            spg_veto, spg_report = spg.evaluate(market_data, ensemble_signal, audit_log)

            if spg_veto:
                ensemble_signal = 0.0
                final_position = 0.0
                # SPG VETO triggered → final signal and position set to 0.0
                audit_log.append("[Phase 5] SPG VETO 발동 → 최종 시그널 및 비중 0.0 처리")
            elif spg_report.get("apply_floor"):
                floor_val = spg_report.get("strategic_floor", 0.0)
                ensemble_signal = max(ensemble_signal, floor_val)
                audit_log.append(f"[Phase 5] SPG Floor 적용 → Ensemble Signal {ensemble_signal:.4f}")

        # Shannon confidence
        shannon_confidence = 1.0
        if "RSK_SHANNON_001" in node_results:
            shannon_confidence = node_results["RSK_SHANNON_001"].normalized_score

        # Update global state
        self.global_state.last_score = ensemble_signal
        self.global_state.previous_regime = current_regime
        self.global_state.last_updated = datetime.utcnow()

        # ── Phase 6: Final Return Preparation ──

        return OrchestratorOutput(
            ticker=market_data.ticker,
            node_results=node_results,
            ensemble_signal=ensemble_signal,
            final_position_size=final_position,
            override_active=False,
            spg_veto_active=spg_veto,
            spg_report=spg_report,
            active_regime=current_regime,
            confidence=shannon_confidence,
            tension_score=raw_tension,
            audit_log=audit_log,
        )

    # ── Phase 1: Priority Override ──

    def _apply_master_influences(
        self,
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
        override_k: Optional[float] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """Process Master/Priority influence via an exponential decay function.

        Instead of a hard-coded binary shutdown, this attenuates other nodes'
        signals exponentially in proportion to the risk score.

        Formula: factor = exp(-k * Ruin_Risk)
        - With k=3.0: Risk 0.5 → factor 0.22, Risk 0.8 → factor 0.09, Risk 1.0 → factor 0.05

        Args:
            override_k: If provided, use this as the decay coefficient instead of default 3.5.

        Returns:
            (influence active flag, source node ID, decay factor)
        """
        k = override_k if override_k is not None else 3.5
        audit_log.append(f"[Phase 1] Master Influence (Soft-Shutdown) 평가 시작 (k={k})")

        # Direct Taleb check
        taleb_result = results.get("RSK_TALEB_001")
        if taleb_result:
            risk_score = taleb_result.normalized_score
            influence_factor = math.exp(-k * risk_score)
            
            if risk_score > 0.4: # Log when a significant risk level is detected
                audit_log.append(
                    f"  ⚠ Taleb Master Influence: Risk={risk_score:.4f} → "
                    f"Decay Factor={influence_factor:.4f}"
                )
            
            # Treat as effective shutdown when risk is extremely high (e.g., > 1.2)
            if risk_score > 1.5:
                return True, "RSK_TALEB_001", 0.0
                
            if risk_score > 0.1: # Meaningful risk present
                return True, "RSK_TALEB_001", influence_factor

        return False, None, 1.0

    # ── Phase 2: Global Adjustment (continuous function) ──

    def _apply_global_adjustments(
        self,
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
    ) -> Dict[str, float]:
        """Adjust global weights as a continuous function of macro node scores.

        Instead of hard-coded ``if > 0.8`` binary logic, weights are smoothly
        attenuated/amplified as a continuous function of the score.

        Applied functions:
            - growth_multiplier = linear_interp(dalio_score, [0.3, 0.8], [1.0, 0.5])
            - value_multiplier  = linear_interp(dalio_score, [0.3, 0.8], [1.0, 1.5])

        Args:
            results: Per-node execution results
            regime: Current market regime
            audit_log: Audit log

        Returns:
            Dictionary of node ID → adjusted score
        """
        audit_log.append("[Phase 2] Global Adjustment (연속 가중치) 적용 시작")
        adjusted: Dict[str, float] = {
            nid: r.normalized_score for nid, r in results.items()
        }

        for edge in self._edges:
            if edge.Relationship_Type != RelationshipType.GLOBAL_WEIGHT_ADJUSTER:
                continue

            source_result = results.get(edge.Source)
            if source_result is None:
                continue

            # Regime condition check
            if edge.Condition_Regime and regime:
                if regime not in edge.Condition_Regime:
                    continue

            source_score = source_result.normalized_score
            targets = self._resolve_targets(
                edge.Target, set(self._engines.keys()), edge.Source
            )

            for target_id in targets:
                if target_id not in adjusted:
                    continue

                original = adjusted[target_id]

                # Apply continuous weight function
                multiplier = self._continuous_weight_function(
                    source_score, target_id, edge
                )
                adjusted[target_id] = max(0.0, min(1.0, original * multiplier))

                audit_log.append(
                    f"  {edge.Source}({source_score:.3f}) → {target_id}: "
                    f"multiplier={multiplier:.3f}, "
                    f"{original:.3f} → {adjusted[target_id]:.3f}"
                )

        return adjusted

    def _continuous_weight_function(
        self, source_score: float, target_id: str, edge: LogicalEdge
    ) -> float:
        """Compute a continuous weight multiplier based on the source score.

        Uses linear interpolation instead of binary ``if > threshold``
        for smooth attenuation/amplification.

        Growth nodes → attenuated as macro risk increases
        Value nodes → amplified as macro risk increases

        Args:
            source_score: Source node normalized score (0–1)
            target_id: Target node ID
            edge: Logical edge

        Returns:
            Weight multiplier (0.3–2.0)
        """
        # Determine whether the target is a growth or value node
        if target_id.startswith("GRO_"):
            # Growth: attenuated as Dalio risk increases
            # source_score 0.0→1.0, multiplier 1.0→0.3
            return self._linear_interpolation(
                source_score,
                x_range=(0.2, 0.9),
                y_range=(1.0, 0.3),
            )
        elif target_id.startswith("VAL_") and "Graham" in (
            self._engines.get(target_id, None) and
            self._engines[target_id].node.Master or ""
        ):
            # Graham Value: amplified as Dalio risk increases (contrarian opportunity)
            return self._linear_interpolation(
                source_score,
                x_range=(0.2, 0.9),
                y_range=(1.0, 1.8),
            )
        else:
            # Other nodes: mild attenuation
            return self._exponential_decay(source_score, decay_rate=0.5)

    @staticmethod
    def _linear_interpolation(
        x: float,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (1.0, 0.5),
    ) -> float:
        """Compute a continuous weight via linear interpolation.

        Clips when x is outside x_range.

        Args:
            x: Input value (0–1)
            x_range: (x_min, x_max) interpolation domain
            y_range: (y_at_x_min, y_at_x_max) output range

        Returns:
            Interpolated weight
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        if x <= x_min:
            return y_min
        if x >= x_max:
            return y_max

        t = (x - x_min) / (x_max - x_min)
        return y_min + t * (y_max - y_min)

    @staticmethod
    def _exponential_decay(
        x: float, decay_rate: float = 0.5, base: float = 1.0
    ) -> float:
        """Compute a continuous weight via exponential decay.

        Formula: base * exp(-decay_rate * x)

        Args:
            x: Input value (0–1)
            decay_rate: Decay rate
            base: Base value

        Returns:
            Decayed weight (> 0)
        """
        return base * math.exp(-decay_rate * x)

    # ── Phase 3: Synergy / Suppress ──

    def _apply_synergy_suppress(
        self,
        adjusted_scores: Dict[str, float],
        results: Dict[str, NodeExecutionResult],
        regime: Optional[MarketRegime],
        audit_log: List[str],
    ) -> Dict[str, float]:
        """Apply inter-node synergy/suppress interactions.

        Based on the edge Relationship_Type:
        - Synergize_With: Amplify when both source and target are high
        - Suppress: Attenuate target when source is high

        All interactions are processed via continuous functions.

        Args:
            adjusted_scores: Post-Phase 2 adjusted scores
            results: Original node results (for state flag reference)
            regime: Current market regime
            audit_log: Audit log

        Returns:
            Final score dictionary with synergy/suppress applied
        """
        audit_log.append("[Phase 3] Synergy/Suppress 상호작용 적용 시작")
        final_scores = dict(adjusted_scores)

        # Sort by priority (higher priority first)
        interaction_edges = [
            e for e in self._edges
            if e.Relationship_Type in (
                RelationshipType.SYNERGIZE_WITH,
                RelationshipType.SUPPRESS,
                RelationshipType.CONTINUOUS_DISCOUNT,
                RelationshipType.CONTINUOUS_SYNERGY,
            )
        ]
        interaction_edges.sort(
            key=lambda e: self._OVERRIDE_PRIORITY.get(e.Relationship_Type, 0),
            reverse=True,
        )

        for edge in interaction_edges:
            source_score = final_scores.get(edge.Source)
            if source_score is None:
                continue

            # Regime condition check
            if edge.Condition_Regime and regime:
                if regime not in edge.Condition_Regime:
                    continue

            targets = self._resolve_targets(
                edge.Target, set(self._engines.keys()), edge.Source
            )

            for target_id in targets:
                if target_id not in final_scores:
                    continue

                target_score = final_scores[target_id]

                if edge.Relationship_Type == RelationshipType.SYNERGIZE_WITH:
                    # Synergy: amplify proportionally to the geometric mean of source × target
                    synergy_factor = self._compute_synergy_multiplier(
                        source_score, target_score
                    )
                    new_score = min(1.0, target_score * synergy_factor)
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  시너지: {edge.Source}({source_score:.3f}) ↔ "
                        f"{target_id}({target_score:.3f}) → "
                        f"factor={synergy_factor:.3f}, new={new_score:.3f}"
                    )

                elif edge.Relationship_Type == RelationshipType.SUPPRESS:
                    # Suppress: attenuate target proportionally to source score
                    suppress_factor = self._compute_suppress_multiplier(
                        source_score, results.get(edge.Source)
                    )
                    new_score = max(0.0, target_score * suppress_factor)
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  억제: {edge.Source}({source_score:.3f}) → "
                        f"{target_id}({target_score:.3f}) → "
                        f"factor={suppress_factor:.3f}, new={new_score:.3f}"
                    )

                elif edge.Relationship_Type in (RelationshipType.CONTINUOUS_DISCOUNT, RelationshipType.CONTINUOUS_SYNERGY):
                    # Continuous weight transfer: source score directly influences target proportionally/inversely
                    if edge.Relationship_Type == RelationshipType.CONTINUOUS_DISCOUNT:
                        # [Calibration] Sharper suppression: 0.05 -> 0.02
                        multiplier = source_score + 0.02
                    else:
                        # [Calibration] Stronger synergy: 0.5 -> 0.8 boost
                        multiplier = 1.0 + (source_score * 0.8)
                        
                    new_score = max(0.0, min(1.0, target_score * multiplier))
                    final_scores[target_id] = new_score
                    audit_log.append(
                        f"  연속전이: {edge.Source}({source_score:.3f}) → {target_id}: "
                        f"multiplier={multiplier:.3f}, new={new_score:.3f}"
                    )

        return final_scores

    @staticmethod
    def _compute_synergy_multiplier(
        source_score: float, target_score: float
    ) -> float:
        """Compute the synergy multiplier.

        Amplifies proportionally to the geometric mean when both nodes score
        high. Maximum amplification is capped at 1.5x.

        Formula: 1.0 + 0.5 * sqrt(source * target)

        Args:
            source_score: Source node score
            target_score: Target node score

        Returns:
            Synergy multiplier (1.0–1.5)
        """
        geometric_mean = math.sqrt(max(0.0, source_score * target_score))
        # [v8.6] Synergy Boost Coefficient: 0.8 -> 0.95
        return 1.0 + 0.95 * geometric_mean

    @staticmethod
    def _compute_suppress_multiplier(
        source_score: float,
        source_result: Optional[NodeExecutionResult] = None,
    ) -> float:
        """Compute the suppress multiplier.

        Higher source scores attenuate the target more strongly.
        A CRITICAL_INHIBIT flag triggers full suppression (0.0).

        Formula: max(0.0, 1.0 - source_score)

        Args:
            source_score: Source node score
            source_result: Source node execution result (for state flag reference)

        Returns:
            Suppress multiplier (0.0–1.0)
        """
        # CRITICAL_INHIBIT → full suppression
        if source_result and source_result.active_state_flag == "CRITICAL_INHIBIT":
            return 0.0

        # Continuous attenuation proportional to source score
        return max(0.0, 1.0 - source_score)

    # ── Phase 4: Ensemble & Position Sizing Pipeline ──

    def _compute_ensemble_and_sizing(
        self,
        final_scores: Dict[str, float],
        results: Dict[str, NodeExecutionResult],
        audit_log: List[str],
    ) -> Tuple[float, float]:
        """Compute the ensemble signal and final position size.

        Ensemble:
            - Weighted average of Value, Growth, and Macro domain nodes
            - Thorp/Taleb/Shannon are excluded from the ensemble

        Position Sizing (Kelly Criterion):
            - Ensemble signal × Shannon confidence → Thorp Kelly Fraction
            - Conservative allocation via Half-Kelly

        Args:
            final_scores: Post-Phase 3 final scores
            results: Node execution results
            audit_log: Audit log

        Returns:
            (ensemble signal, final position size)
        """
        audit_log.append("[Phase 4] 앙상블 & Position Sizing 파이프라인 시작")

        # ── Ensemble (excluding Risk domain) ──
        ensemble_nodes = {
            nid: score for nid, score in final_scores.items()
            if not nid.startswith("RSK_")
        }

        if not ensemble_nodes:
            # No nodes eligible for ensemble → 0.0
            audit_log.append("  앙상블 대상 노드 없음 → 0.0")
            return 0.0, 0.0

        # [v8.6.2] Collective Intelligence: symmetric conviction-weighted ensemble
        # Calibration: Neutral Point shifted to 0.55 (A2_BALANCED Target)
        NEUTRAL_POINT = 0.55
        weighted_sum = 0.0
        total_confidence = 0.0
        
        for nid, score in ensemble_nodes.items():
            # [Refinement] Symmetric weighting: ensures that a bearish expert (0.075)
            # and a bullish expert (0.925) have equal voice (weight) relative
            # to the neutral point (0.18)
            dist = abs(score - NEUTRAL_POINT)
            if score >= NEUTRAL_POINT:
                # Upside weight scaling (0.18–1.0)
                relative_dist = dist / (1.0 - NEUTRAL_POINT + 1e-8)
            else:
                # Downside weight scaling (0.0–0.18)
                relative_dist = dist / (NEUTRAL_POINT + 1e-8)
                
            # [Calibration] Exponent 2.8 + floor 0.05 for balanced transition
            confidence = (relative_dist ** 2.8) + 0.05
            weighted_sum += score * confidence
            total_confidence += confidence
            
        ensemble_signal = weighted_sum / total_confidence if total_confidence > 0 else 0.18
        ensemble_signal = max(0.0, min(1.0, ensemble_signal))
        
        audit_log.append(
            f"  [v8.6] symmetric_weighted_ensemble = {ensemble_signal:.4f} "
            f"(대상 {len(ensemble_nodes)}개 노드)"
        )

        # ── Shannon Confidence Filter ──
        shannon_factor = 1.0
        shannon_result = results.get("RSK_SHANNON_001")
        if shannon_result:
            shannon_factor = shannon_result.normalized_score
            audit_log.append(f"  Shannon 신뢰도 필터 = {shannon_factor:.4f}")

        # ── Thorp Kelly Criterion Position Sizing ──
        final_position = self._kelly_position_sizing(
            ensemble_signal, shannon_factor, results, audit_log
        )

        return ensemble_signal, final_position

    def _kelly_position_sizing(
        self,
        ensemble_signal: float,
        shannon_factor: float,
        results: Dict[str, NodeExecutionResult],
        audit_log: List[str],
    ) -> float:
        """Compute the final position size based on the Kelly Criterion.

        Formula: f* = (bp - q) / b
        - b: Win/loss ratio
        - p: Win rate (ensemble signal-based)
        - q: Loss rate (1 - p)

        Applies Half-Kelly for conservative allocation.
        Shannon confidence is applied as a direct multiplier.

        Args:
            ensemble_signal: Ensemble signal (0–1)
            shannon_factor: Shannon confidence (0.1–1.0)
            results: Node execution results
            audit_log: Audit log

        Returns:
            Final position size (0.0–1.0)
        """
        # Incorporate Edge Conviction Premium from Thorp node result
        thorp_result = results.get("RSK_THORP_001")
        edge_premium = 0.0
        if thorp_result:
            edge_premium = thorp_result.step2_output  # Edge_Conviction_Premium
            audit_log.append(
                f"  Thorp Edge Conviction Premium = {edge_premium:.4f}"
            )

        # Win rate estimate based on ensemble signal
        estimated_win_rate = max(0.01, min(0.99, ensemble_signal))
        estimated_loss_rate = 1.0 - estimated_win_rate

        # Win/loss ratio (default 1.5:1, adjusted by edge_premium)
        win_loss_ratio = 1.5 * (1.0 + max(0.0, edge_premium))

        # Kelly Fraction: f* = (bp - q) / b
        kelly_fraction = (
            (win_loss_ratio * estimated_win_rate - estimated_loss_rate)
            / win_loss_ratio
        )

        # Half-Kelly (conservative)
        half_kelly = max(0.0, kelly_fraction * 0.5)

        # Apply Shannon confidence
        final_size = half_kelly * shannon_factor

        # Cap at maximum 100%
        final_size = max(0.0, min(1.0, final_size))

        audit_log.append(
            f"  Kelly: p={estimated_win_rate:.3f}, b={win_loss_ratio:.3f}, "
            f"f*={kelly_fraction:.4f}, half_kelly={half_kelly:.4f}, "
            f"× shannon={shannon_factor:.3f} → final={final_size:.4f}"
        )

        return final_size

    # ── Utilities ──

    def get_engine(self, node_id: str) -> Optional[AbstractMasterEngine]:
        """Retrieve an engine instance by node ID.

        Args:
            node_id: Node identifier

        Returns:
            AbstractMasterEngine instance, or None
        """
        return self._engines.get(node_id)

    def update_node_win_rate(
        self, node_id: str, performance: PerformanceState
    ) -> float:
        """Update the rolling win rate of a specific node from external performance data.

        Args:
            node_id: Target node ID
            performance: Performance data

        Returns:
            Updated win rate

        Raises:
            KeyError: When the node ID does not exist
        """
        engine = self._engines.get(node_id)
        if engine is None:
            raise KeyError(f"등록되지 않은 노드 ID: {node_id}")
        return engine.update_rolling_win_rate(performance)

    @property
    def registered_nodes(self) -> List[str]:
        """List of all registered node IDs."""
        return list(self._engines.keys())

    @property
    def execution_order(self) -> List[str]:
        """Topologically sorted execution order."""
        return list(self._sorted_order) if self._sorted_order else []


# ══════════════════════════════════════════════════════════
# 5. Strategic Policy Governor (SPG) — System-2 Guardrail
# ══════════════════════════════════════════════════════════

class StrategicPolicyGovernor:
    """Deterministic guardrail engine that exercises strategic control and veto power.

    Independent of any agent or engine 'conviction', this governor filters
    'toxic' stocks based on fundamental financial soundness and commercial
    common sense.
    """
    
    def evaluate(
        self, 
        market_data: MarketDataPayload, 
        ensemble_signal: float, 
        audit_log: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Exercise final veto power over the ensemble signal.
        
        Args:
            market_data: Payload containing financial metrics (Debt, FCF, P/E, etc.)
            ensemble_signal: Primary signal produced by the engine
            audit_log: System audit log
            
        Returns:
            (Veto flag, detailed adjudication report)
        """
        audit_log.append("[Phase 5] Strategic Policy Governor (SPG) 판정 시작")
        
        metrics = market_data.metrics
        report: Dict[str, Any] = {
            "checks": {},
            "score": 0,
            "threshold": 3,
            "status": "PASS"
        }
        
        # 1. Debt Safety
        debt_to_equity = metrics.get("Debt_to_Equity", metrics.get("Debt/Equity", 0.0))
        debt_pass = debt_to_equity < 1.5
        report["checks"]["debt_safety"] = {"value": debt_to_equity, "pass": debt_pass}
        
        # 2. Cash Flow Health
        fcf_yield = metrics.get("FCF_Yield", 0.0)
        cash_pass = fcf_yield > 0.01  # Minimum 1% FCF yield
        report["checks"]["cash_flow"] = {"fcf": fcf_yield, "pass": cash_pass}
        
        # 3. Valuation Sanity
        pe_ratio = metrics.get("P/E_Ratio", 15.0)
        pe_pass = 0 < pe_ratio < 60
        report["checks"]["valuation"] = {"value": pe_ratio, "pass": pe_pass}
        
        # 4. Profitability
        roic = metrics.get("ROIC_10yr_Avg", 0.0)
        roic_pass = roic > 0.08  # Minimum 8% ROIC
        report["checks"]["profitability"] = {"value": roic, "pass": roic_pass}
        
        # 5. Asset Quality
        ncav_ratio = metrics.get("Price_to_NCAV", 1.0)
        ncav_pass = ncav_ratio < 15.0  # Guard against extreme bubbles
        report["checks"]["asset_quality"] = {"value": ncav_ratio, "pass": ncav_pass}
        
        # Compute final score
        report["score"] = sum(1 for c in report["checks"].values() if c["pass"])
        
        # [NEW] Strategic Floor Logic: downside protection for quality stocks
        # adj_floor = (strategy_base + contrarian_bonus) * regime_mult * quality_mult
        score = report["score"]
        strategy_base = 0.05
        contrarian_bonus = 0.02 if (pe_ratio < 10 and pe_ratio > 0) else 0.0
        quality_mult = 1.0 + (score / 5.0)  # 1.0 ~ 2.0
        
        # Regime correction (e.g., increase quality allocation in bear markets)
        regime_mult = 1.2 if market_data.current_regime == MarketRegime.BEAR_MARKET else 1.0
        
        strategic_floor = (strategy_base + contrarian_bonus) * regime_mult * quality_mult
        report["strategic_floor"] = strategic_floor
        
        # Veto Logic
        is_veto = False
        if ensemble_signal > 0.6 and score < 3:
            is_veto = True
            report["status"] = "VETO_BY_TOXIC_FUNDAMENTALS"
            audit_log.append(f"  ⚠ SPG VETO: Conviction={ensemble_signal:.3f}이나 Fundamentals({score}/5) 부실")
        elif ensemble_signal > 0.8 and score < 4:
            is_veto = True
            report["status"] = "VETO_BY_HIGH_HYPE_RISK"
            audit_log.append(f"  ⚠ SPG VETO: Extreme Hype({ensemble_signal:.3f}) 대비 재무 점수({score}/5) 부족")
        else:
            # When no veto, consider applying floor if quality is perfect (5/5)
            if ensemble_signal < strategic_floor and score >= 5:
                report["apply_floor"] = True
                audit_log.append(f"  SPG Floor 적용 대상 탐지: {ensemble_signal:.3f} < Floor {strategic_floor:.3f}")
            else:
                report["apply_floor"] = False
            
            audit_log.append(f"  SPG PASS: Fundamental Score = {score}/5, Floor = {strategic_floor:.3f}")
            
        return is_veto, report
